"""
DynamicFactorNeutralMarkowitz
=============================

Ce module implémente une stratégie de portefeuille de type Markowitz dynamique
avec contrainte de neutralité factorielle.

Idée générale :
- On dispose d'un signal alpha_t par actif.
- On dispose éventuellement d'un terme c_t par actif.
- On dispose de betas dynamiques par actif, c'est-à-dire les expositions aux facteurs.
- On construit une matrice de covariance Sigma_t des résidus.
- On choisit les poids w_t qui maximisent le signal attendu tout en neutralisant
  les expositions factorielles : beta_t.T @ w_t ≈ 0.
- On calibre ensuite le risque soit via une volatilité cible, soit via une aversion
  au risque explicite.

Le code accepte trois méthodes de covariance :
1. constant : covariance empirique constante sur toute la période.
2. rolling  : covariance glissante sur une fenêtre passée.
3. dcc      : covariance dynamique via GARCH univariés + corrélation DCC.
"""

# NumPy sert aux calculs matriciels, produits scalaires, pseudo-inverses, etc.
import numpy as np
# Pandas sert à manipuler les séries temporelles : DataFrame, Series, index de dates.
import pandas as pd
# Matplotlib sert uniquement aux graphiques de diagnostic.
import matplotlib.pyplot as plt
# Le package arch est nécessaire uniquement pour la méthode DCC-GARCH.
# Si le package n'est pas installé, on garde arch_model = None.
# Le code continuera à fonctionner pour sigma_method="constant" ou "rolling".
try:
    from arch import arch_model
except ImportError:
    arch_model = None


class DynamicFactorNeutralMarkowitz:
    """
    Classe principale qui :
    - construit les matrices de covariance Sigma_t ;
    - calcule les poids factor-neutral de Markowitz ;
    - stocke les poids, les expositions factorielles et la volatilité ex-ante ;
    - fournit des méthodes de backtest, performance, diagnostics et graphiques.
    """

    def __init__(
        self,
        target_vol=0.02,        # Volatilité cible ex-ante du portefeuille par période, ex : 0.02 = 2%.
        risk_aversion=None,     # Si fourni, les poids sont divisés par ce lambda au lieu d'être calibrés sur target_vol.
        sigma_method="constant",   # Méthode de covariance : "constant", "rolling" ou "dcc".
        rolling_window=60,      # Taille de la fenêtre utilisée pour la covariance rolling.
        dcc_a=0.03,             # Paramètre a du DCC : poids donné au choc de corrélation récent.
        dcc_b=0.95,             # Paramètre b du DCC : persistance de la corrélation dynamique.
        garch_scale=100,        # Facteur d'échelle pour stabiliser l'estimation GARCH sur des rendements petits.
        ridge=1e-8              # Petite régularisation ajoutée à la diagonale de Sigma pour éviter les matrices singulières.
    ):
        # On stocke tous les hyperparamètres dans l'objet.
        self.target_vol = target_vol
        self.risk_aversion = risk_aversion
        self.sigma_method = sigma_method
        self.rolling_window = rolling_window
        self.dcc_a = dcc_a
        self.dcc_b = dcc_b
        self.garch_scale = garch_scale
        self.ridge = ridge

        # Attributs remplis après fit(). Le suffixe _ indique un résultat estimé.
        self.weights_ = None
        self.factor_exposures_ = None
        self.portfolio_vol_ = None
        self.sigmas_ = None
        self.used_dates_ = None
        self.assets_ = None
        self.factor_names_ = None

    # ======================================================
    # Helpers
    # ======================================================

    def _as_dataframe(self, x):
        """
        Convertit une entrée en DataFrame.

        Pourquoi ?
        - Les méthodes suivantes attendent des DataFrames avec index de dates.
        - Si x est une Series, on la transforme en DataFrame à une seule colonne.
        - Si x est déjà un DataFrame, on en fait une copie pour éviter de modifier l'objet original.
        """
        if x is None:
            return None
        if isinstance(x, pd.Series):
            return x.to_frame()
        return x.copy()

    def _normalize_index(self, df):
        """
        Normalise l'index temporel :
        - conversion en datetime ;
        - tri chronologique.
        Cela évite les problèmes d'alignement entre returns, alphas, c, betas et residuals.
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _normalize_betas(self, betas, assets):
        """
        Nettoie le dictionnaire des betas.

        Format attendu :
            betas[asset] = DataFrame indexé par date
                           colonnes = facteurs
                           valeurs = beta de l'actif aux facteurs.

        La méthode conserve uniquement les actifs présents dans assets.
        """
        clean_betas = {}

        for asset in assets:
            if asset not in betas:
                continue

            beta_df = betas[asset].copy()
            beta_df.index = pd.to_datetime(beta_df.index)
            beta_df = beta_df.sort_index()

            clean_betas[asset] = beta_df

        if len(clean_betas) == 0:
            raise ValueError("Aucun actif commun trouvé dans betas.")

        return clean_betas

    def _get_common_beta_dates(self, betas):
        """
        Cherche les dates communes à tous les DataFrames de betas.
        On ne peut optimiser que sur les dates où tous les actifs ont un beta disponible.
        """
        common_dates = None

        for _, beta_df in betas.items():
            if common_dates is None:
                common_dates = beta_df.index
            else:
                common_dates = common_dates.intersection(beta_df.index)

        return common_dates

    def _get_beta_t(self, betas, date, assets):
        """
        Construit la matrice beta_t à une date donnée.

        Résultat :
            beta_t.shape = (N actifs, K facteurs)

        Chaque ligne correspond à un actif.
        Chaque colonne correspond à un facteur.
        """
        beta_matrix = []

        for asset in assets:
            beta_matrix.append(betas[asset].loc[date].values)

        return np.asarray(beta_matrix)

    # ======================================================
    # Construction de Sigma_t
    # ======================================================

    def _build_constant_sigmas(self, residuals, dates, assets):
        """
        Construit une covariance constante.

        Sigma est calculée une seule fois à partir des résidus historiques,
        puis la même matrice est utilisée pour toutes les dates.
        """
        Sigma = residuals[assets].dropna().cov().values
        Sigma = Sigma + self.ridge * np.eye(len(assets))

        return {
            pd.Timestamp(date): Sigma
            for date in dates
        }

    def _build_rolling_sigmas(self, residuals, dates, assets):
        """
        Construit une covariance dynamique par fenêtre glissante.

        À chaque date t :
        - on prend les rolling_window observations précédentes ;
        - on calcule leur matrice de covariance ;
        - on ajoute une petite régularisation ridge sur la diagonale.

        Attention :
        - La covariance à la date t utilise uniquement le passé, pas la valeur de t.
        - Les premières dates sont ignorées car il n'y a pas assez d'historique.
        """
        residuals = residuals[assets].dropna()

        sigmas = {}

        for date in dates:
            if date not in residuals.index:
                continue

            pos = residuals.index.get_loc(date)

            if pos < self.rolling_window:
                continue

            window_resid = residuals.iloc[pos - self.rolling_window:pos]

            Sigma_t = window_resid.cov().values
            Sigma_t = Sigma_t + self.ridge * np.eye(len(assets))

            sigmas[pd.Timestamp(date)] = Sigma_t

        return sigmas

    def _build_dcc_sigmas(self, residuals, dates, assets):
        """
        Construit des matrices Sigma_t via une approche DCC-GARCH simplifiée.

        Étapes :
        1. Pour chaque actif, on estime un GARCH(1,1) sur les résidus.
        2. On récupère :
           - les volatilités conditionnelles h_t^{1/2} ;
           - les résidus standardisés z_t.
        3. On estime une corrélation dynamique DCC :
           Q_t = (1-a-b) Q_bar + a z_{t-1} z_{t-1}' + b Q_{t-1}
        4. On transforme Q_t en matrice de corrélation R_t.
        5. On reconstruit :
           Sigma_t = D_t R_t D_t
           avec D_t = diag(volatilités conditionnelles).

        Remarque :
        - Ce DCC utilise dcc_a et dcc_b fixés par l'utilisateur.
        - Il ne calibre pas a et b par maximum de vraisemblance.
        """
        if arch_model is None:
            raise ImportError(
                "Le package arch n'est pas installé. "
                "Installe-le avec : pip install arch"
            )

        residuals = residuals[assets].dropna()

        T, N = residuals.shape

        cond_vols = pd.DataFrame(
            index=residuals.index,
            columns=assets,
            dtype=float
        )

        std_resids = pd.DataFrame(
            index=residuals.index,
            columns=assets,
            dtype=float
        )

        for asset in assets:
            x = residuals[asset].dropna() * self.garch_scale

            am = arch_model(
                x,
                mean="Zero",
                vol="GARCH",
                p=1,
                q=1,
                dist="normal"
            )

            res = am.fit(disp="off")

            cond_vols.loc[x.index, asset] = res.conditional_volatility / self.garch_scale
            std_resids.loc[x.index, asset] = res.std_resid

        cond_vols = cond_vols.dropna()
        std_resids = std_resids.dropna()

        common_dates = cond_vols.index.intersection(std_resids.index)
        common_dates = common_dates.intersection(pd.Index(dates))
        common_dates = common_dates.sort_values()

        cond_vols = cond_vols.loc[common_dates]
        std_resids = std_resids.loc[common_dates]

        Z = std_resids.values

        Q_bar = np.cov(Z.T)
        Q_t = Q_bar.copy()

        sigmas = {}

        for i, date in enumerate(common_dates):
            if i > 0:
                z_prev = Z[i - 1].reshape(-1, 1)

                Q_t = (
                    (1 - self.dcc_a - self.dcc_b) * Q_bar
                    + self.dcc_a * (z_prev @ z_prev.T)
                    + self.dcc_b * Q_t
                )

            diag_q = np.sqrt(np.diag(Q_t))
            R_t = Q_t / np.outer(diag_q, diag_q)

            D_t = np.diag(cond_vols.loc[date].values)

            Sigma_t = D_t @ R_t @ D_t
            Sigma_t = Sigma_t + self.ridge * np.eye(N)

            sigmas[pd.Timestamp(date)] = Sigma_t

        return sigmas

    def _build_sigmas(self, residuals, dates, assets):
        """
        Routeur qui choisit la méthode de construction de Sigma_t
        selon self.sigma_method.
        """
        residuals = self._normalize_index(self._as_dataframe(residuals))

        if self.sigma_method == "constant":
            return self._build_constant_sigmas(residuals, dates, assets)

        if self.sigma_method == "rolling":
            return self._build_rolling_sigmas(residuals, dates, assets)

        if self.sigma_method == "dcc":
            return self._build_dcc_sigmas(residuals, dates, assets)

        raise ValueError("sigma_method doit être 'constant', 'rolling' ou 'dcc'.")

    # ======================================================
    # Optimisation
    # ======================================================

    def _compute_weights_one_date(self, alpha_t, c_t, beta_t, sigma_t):
        """
        Calcule les poids optimaux à une date donnée.

        Objectif intuitif :
            exploiter le signal mu_t = alpha_t + c_t

        Contrainte :
            neutralité factorielle : beta_t.T @ w_t = 0

        Formule utilisée :
            P_t = Sigma^{-1}
                  - Sigma^{-1} B (B' Sigma^{-1} B)^{-1} B' Sigma^{-1}

        P_t est le projecteur qui retire la composante exposée aux facteurs.
        Le signal brut est :
            raw_signal = P_t @ mu_t

        Ensuite :
        - si risk_aversion est fourni : w = raw_signal / risk_aversion
        - sinon : les poids sont rescalés pour atteindre target_vol.
        """
        alpha_t = np.asarray(alpha_t).reshape(-1)
        c_t = np.asarray(c_t).reshape(-1)
        beta_t = np.asarray(beta_t)
        sigma_t = np.asarray(sigma_t)

        mu_t = alpha_t + c_t
        N = len(mu_t)

        if beta_t.ndim == 1:
            beta_t = beta_t.reshape(-1, 1)

        if beta_t.shape[0] != N:
            raise ValueError(
                f"beta_t a {beta_t.shape[0]} lignes, mais mu_t a {N} actifs."
            )

        if sigma_t.shape != (N, N):
            raise ValueError(
                f"sigma_t doit être de taille {(N, N)}, mais vaut {sigma_t.shape}."
            )

        sigma_t = sigma_t + self.ridge * np.eye(N)

        sigma_inv = np.linalg.pinv(sigma_t)

        middle = beta_t.T @ sigma_inv @ beta_t
        middle_inv = np.linalg.pinv(middle)

        P_t = sigma_inv - sigma_inv @ beta_t @ middle_inv @ beta_t.T @ sigma_inv

        raw_signal = P_t @ mu_t
        signal_strength = mu_t.T @ P_t @ mu_t

        if signal_strength <= 1e-12:
            return np.zeros(N)

        if self.risk_aversion is not None:
            return raw_signal / self.risk_aversion

        return self.target_vol * raw_signal / np.sqrt(signal_strength)

    # ======================================================
    # Fit principal
    # ======================================================

    def fit(self, returns, alphas, betas, residuals, c=None):
        """
        Méthode principale d'estimation.

        Paramètres attendus :
        - returns   : DataFrame des rendements réalisés par actif.
        - alphas    : DataFrame des signaux alpha_t par actif.
        - betas     : dictionnaire {asset: DataFrame de betas dynamiques}.
        - residuals : DataFrame des résidus utilisés pour estimer Sigma_t.
        - c         : DataFrame optionnel du terme c_t. Si absent, c_t = 0.

        Résultats stockés :
        - self.weights_           : poids optimaux par date et actif.
        - self.factor_exposures_  : exposition factorielle beta_t.T @ w_t.
        - self.portfolio_vol_     : volatilité ex-ante sqrt(w' Sigma w).
        - self.sigmas_            : dictionnaire des matrices Sigma_t.
        """
        returns = self._normalize_index(self._as_dataframe(returns))
        alphas = self._normalize_index(self._as_dataframe(alphas))

        if c is None:
            c = pd.DataFrame(
                0.0,
                index=alphas.index,
                columns=alphas.columns
            )
        else:
            c = self._normalize_index(self._as_dataframe(c))

        assets = returns.columns.intersection(alphas.columns)
        assets = assets.intersection(c.columns)

        if len(assets) == 0:
            raise ValueError("Aucun actif commun entre returns, alphas et c.")

        returns = returns[assets].dropna(how="any")
        alphas = alphas[assets].dropna(how="any")
        c = c[assets].dropna(how="any")

        betas = self._normalize_betas(betas, assets)

        assets = pd.Index([asset for asset in assets if asset in betas])

        returns = returns[assets]
        alphas = alphas[assets]
        c = c[assets]

        beta_dates = self._get_common_beta_dates(betas)

        preliminary_dates = returns.index.intersection(alphas.index)
        preliminary_dates = preliminary_dates.intersection(c.index)
        preliminary_dates = preliminary_dates.intersection(beta_dates)
        preliminary_dates = preliminary_dates.sort_values()

        sigmas = self._build_sigmas(
            residuals=residuals,
            dates=preliminary_dates,
            assets=assets
        )

        sigma_dates = pd.Index(sigmas.keys())

        common_dates = preliminary_dates.intersection(sigma_dates)
        common_dates = common_dates.sort_values()

        print("Méthode Sigma :", self.sigma_method)
        print("Nombre d'actifs utilisés :", len(assets))
        print("Dates returns :", len(returns))
        print("Dates alphas  :", len(alphas))
        print("Dates c       :", len(c))
        print("Dates betas   :", len(beta_dates))
        print("Dates sigmas  :", len(sigma_dates))
        print("Dates communes:", len(common_dates))

        if len(common_dates) == 0:
            raise ValueError("Aucune date commune après construction de Sigma_t.")

        self.assets_ = list(assets)
        self.used_dates_ = common_dates
        self.sigmas_ = sigmas
        self.factor_names_ = list(betas[assets[0]].columns)

        weights = []
        exposures = []
        vols = []

        for date in common_dates:
            alpha_t = alphas.loc[date, assets].values
            c_t = c.loc[date, assets].values
            beta_t = self._get_beta_t(betas, date, assets)
            sigma_t = sigmas[date]

            w_t = self._compute_weights_one_date(
                alpha_t=alpha_t,
                c_t=c_t,
                beta_t=beta_t,
                sigma_t=sigma_t
            )

            weights.append(w_t)
            exposures.append(beta_t.T @ w_t)
            vols.append(np.sqrt(w_t.T @ sigma_t @ w_t))

        self.weights_ = pd.DataFrame(
            weights,
            index=common_dates,
            columns=assets
        )

        self.factor_exposures_ = pd.DataFrame(
            exposures,
            index=common_dates,
            columns=self.factor_names_
        )

        self.portfolio_vol_ = pd.Series(
            vols,
            index=common_dates,
            name="portfolio_vol"
        )

        return self

    # ======================================================
    # PnL / Backtest
    # ======================================================

    def get_pnl(self, returns):
        """
        Calcule le PnL de la stratégie.

        Important :
        - Les poids w_t sont décalés d'une période avec shift(1).
        - Cela évite le look-ahead bias :
          le poids décidé à t-1 est appliqué au rendement réalisé à t.
        """
        returns = self._normalize_index(self._as_dataframe(returns))
        returns = returns[self.weights_.columns]

        common_dates = self.weights_.index.intersection(returns.index)

        w = self.weights_.loc[common_dates]
        r = returns.loc[common_dates]

        pnl = (w.shift(1) * r).sum(axis=1).dropna()
        pnl.name = "pnl"

        return pnl

    def get_cumulative_pnl(self, returns):
        """
        Renvoie le PnL cumulé, c'est-à-dire la somme des PnL période par période.
        """
        return self.get_pnl(returns).cumsum().rename("cumulative_pnl")
    """
    def get_wealth(self, returns):
        """
        Renvoie une mesure de richesse cumulée.

        Remarque importante :
        - Ici, wealth = cumsum(PnL), donc c'est une richesse additive.
        - Si le PnL est un rendement de portefeuille, la richesse classique serait :
              (1 + pnl).cumprod()
        """
        return (1 + self.get_pnl(returns)).cumprod().rename("wealth")
    """
    def get_wealth(self, returns):
        return (self.get_pnl(returns)).cumsum().rename("wealth")

    def backtest(self, returns):
        """
        Produit un DataFrame simple de backtest avec :
        - pnl ;
        - cumulative_pnl ;
        - wealth.

        Attention :
        - Dans le code actuel, wealth = pnl.cumprod(), ce qui est inhabituel
          si pnl contient des rendements.
        - Pour des rendements, on utilise généralement (1 + pnl).cumprod().
        """
        pnl = self.get_pnl(returns)

        return pd.DataFrame({
            "pnl": pnl,
            "cumulative_pnl": pnl.cumsum(),
            #"wealth": (1 + pnl).cumprod()
            "wealth": (pnl).cumprod()
        })

    def performance(self, returns, periods_per_year=252):
        """
        Calcule quelques métriques de performance annualisées.

        periods_per_year=252 correspond aux marchés actions quotidiens.
        Pour les cryptos, on peut plutôt utiliser 365 si les données sont journalières.
        """
        pnl = self.get_pnl(returns)
        #wealth = (1 + pnl).cumprod()
        wealth = (pnl).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        vol = pnl.std()

        return {
            "mean_daily_pnl": pnl.mean(),
            "daily_volatility": vol,
            "annualized_return": pnl.mean() * periods_per_year,
            "annualized_volatility": vol * np.sqrt(periods_per_year),
            "annualized_sharpe": pnl.mean() / vol * np.sqrt(periods_per_year) if vol > 0 else np.nan,
            "max_drawdown": drawdown.min(),
            "final_wealth": wealth.iloc[-1]
        }

    # ======================================================
    # Diagnostics
    # ======================================================

    def check_neutrality(self, betas, date=None):
        """
        Vérifie la neutralité factorielle à une date donnée.

        Renvoie :
            beta_t.T @ w_t

        Si la neutralité est bien respectée, les valeurs doivent être proches de zéro.
        """
        if date is None:
            date = self.used_dates_[0]

        date = pd.Timestamp(date)

        betas = self._normalize_betas(betas, self.assets_)

        beta_t = self._get_beta_t(betas, date, self.assets_)
        w_t = self.weights_.loc[date].values

        exposure = beta_t.T @ w_t

        return pd.Series(
            exposure,
            index=self.factor_names_,
            name=date
        )

    def get_sigma(self, date):
        """
        Renvoie la matrice Sigma_t à une date donnée sous forme de DataFrame,
        avec les noms des actifs en lignes et en colonnes.
        """
        date = pd.Timestamp(date)

        return pd.DataFrame(
            self.sigmas_[date],
            index=self.assets_,
            columns=self.assets_
        )
    def get_variance_series(self, asset):
        """
        Extrait la variance dynamique Sigma_t[i, i] d'un actif donné.
        Utile pour tracer l'évolution du risque idiosyncratique d'un actif.
        """
        i = self.assets_.index(asset)

        return pd.Series(
            {
                date: Sigma[i, i]
                for date, Sigma in self.sigmas_.items()
            },
            name=f"variance_{asset}"
        ).sort_index()


    def get_volatility_series(self, asset):
        """
        Extrait la volatilité dynamique d'un actif :
            volatilité = sqrt(variance)
        """
        return np.sqrt(
            self.get_variance_series(asset)
        ).rename(f"volatility_{asset}")


    def get_covariance_series(self, asset_i, asset_j):
        """
        Extrait la covariance dynamique Sigma_t[i, j] entre deux actifs.
        """
        i = self.assets_.index(asset_i)
        j = self.assets_.index(asset_j)

        return pd.Series(
            {
                date: Sigma[i, j]
                for date, Sigma in self.sigmas_.items()
            },
            name=f"covariance_{asset_i}_{asset_j}"
        ).sort_index()


    def get_correlation_series(self, asset_i, asset_j):
        """
        Extrait la corrélation dynamique entre deux actifs :

            corr_ij,t = Sigma_ij,t / sqrt(Sigma_ii,t * Sigma_jj,t)
        """
        i = self.assets_.index(asset_i)
        j = self.assets_.index(asset_j)

        return pd.Series(
            {
                date: Sigma[i, j] / np.sqrt(Sigma[i, i] * Sigma[j, j])
                for date, Sigma in self.sigmas_.items()
            },
            name=f"correlation_{asset_i}_{asset_j}"
        ).sort_index()
    
    # ======================================================
    # Plots
    # ======================================================

    def plot_weights(self):
        """
        Trace les poids dynamiques par actif.
        """
        self.weights_.plot(figsize=(12, 5))
        plt.title("Poids dynamiques")
        plt.axhline(0)
        plt.show()

    def plot_factor_exposures(self):
        """
        Trace les expositions factorielles du portefeuille.
        Elles doivent idéalement rester proches de zéro.
        """
        self.factor_exposures_.plot(figsize=(12, 5))
        plt.title("Expositions factorielles")
        plt.axhline(0)
        plt.show()

    def plot_portfolio_vol(self):
        """
        Trace la volatilité ex-ante du portefeuille et la compare à target_vol.
        """
        self.portfolio_vol_.plot(figsize=(12, 5))
        plt.title("Volatilité ex-ante")
        plt.axhline(self.target_vol, linestyle="--")
        plt.show()

    def plot_wealth(self, returns):
        """
        Trace la richesse cumulée renvoyée par get_wealth().
        """
        self.get_wealth(returns).plot(figsize=(12, 5))
        plt.title("Richesse cumulée")
        plt.show()
