# ============================================================
# 1. IMPORTS
# ============================================================

# dataclass permet de créer facilement une classe de configuration.
from dataclasses import dataclass

# Optional, List et Dict servent à typer les arguments et retours de fonctions.
from typing import Optional, List, Dict

# Bibliothèques classiques pour les calculs numériques, les données et les graphiques.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
# yfinance permet de télécharger automatiquement les données Yahoo Finance.
import yfinance as yf

# statsmodels est utilisé ici pour faire les régressions OLS.
import statsmodels.api as sm

# PCA sert à construire des facteurs principaux à partir des rendements.
from sklearn.decomposition import PCA


# ============================================================
# 2. FACTOR CONFIG
# ============================================================

@dataclass
class FactorConfig:
    """
    Classe de configuration d'un facteur.

    Chaque facteur est défini par :
    - name : le nom du facteur dans le DataFrame final ;
    - factor_type : le type de facteur à construire ;
    - method : méthode éventuelle, par exemple equal_weight ;
    - window : fenêtre de calcul pour momentum ou volatilité ;
    - n_components : nombre de composantes pour la PCA.
    """

    name: str
    factor_type: str
    method: Optional[str] = None
    window: Optional[int] = None
    n_components: Optional[int] = None


# ============================================================
# 3. DATA LOADER
# ============================================================

class DataLoader:
    """
    Classe responsable du téléchargement des prix et du calcul des rendements.
    Elle isole toute la partie data afin de ne pas mélanger téléchargement,
    construction des facteurs et régression.
    """

    def __init__(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: Optional[str] = None,
        return_method: str = "log"
    ):
        # Liste des actifs à télécharger, par exemple BTC-USD, ETH-USD, etc.
        self.tickers = tickers

        # Date de début et date de fin de l'échantillon.
        self.start = start
        self.end = end

        # Méthode de calcul des rendements : "log" ou "simple".
        self.return_method = return_method

    def load_prices(self) -> pd.DataFrame:
        """Télécharge les prix ajustés de clôture depuis Yahoo Finance."""

        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False
        )

        # Avec auto_adjust=True, la colonne Close correspond aux prix ajustés.
        prices = data["Close"]

        # Si un seul ticker est téléchargé, yfinance renvoie parfois une Series.
        # On la convertit en DataFrame pour garder une structure homogène.
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(self.tickers[0])

        # On retire uniquement les lignes totalement vides.
        return prices.dropna(how="all")

    def compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calcule les rendements à partir des prix."""

        if self.return_method == "log":
            # Rendement logarithmique : log(P_t / P_{t-1}).
            returns = np.log(prices / prices.shift(1))

        elif self.return_method == "simple":
            # Rendement simple : P_t / P_{t-1} - 1.
            returns = prices.pct_change()

        else:
            raise ValueError("return_method must be 'log' or 'simple'")

        return returns.dropna(how="all")


# ============================================================
# 4. FACTOR STORE
# ============================================================

class FactorStore:
    """
    Classe qui stocke les facteurs construits ou ajoutés manuellement.
    Elle joue le rôle de base centrale des facteurs disponibles.
    """

    def __init__(self):
        # DataFrame contenant tous les facteurs en colonnes.
        self.factors = pd.DataFrame()

        # Dictionnaire optionnel pour documenter chaque facteur.
        self.descriptions = {}
        
    def add_factors(
        self,
        factor_df: pd.DataFrame,
        descriptions: Optional[Dict[str, str]] = None
    ):
        """Ajoute plusieurs facteurs en une seule fois."""

        self.factors = pd.concat([self.factors, factor_df], axis=1)

        for col in factor_df.columns:
            self.descriptions[col] = (
                descriptions.get(col) if descriptions else None
            )

    def get_factors(self, names: Optional[List[str]] = None) -> pd.DataFrame:
        """Retourne tous les facteurs ou seulement une sélection."""

        if names is None:
            return self.factors.copy()

        # Vérification de sécurité : on évite de demander un facteur inexistant.
        missing = [name for name in names if name not in self.factors.columns]

        if missing:
            raise ValueError(f"Unknown factors: {missing}")

        return self.factors[names].copy()

    def list_factors(self) -> List[str]:
        """Liste les noms des facteurs disponibles."""
        return list(self.factors.columns)


# ============================================================
# 5. FACTOR BUILDER
# ============================================================

class FactorBuilder:

    """
    Classe chargée de construire les facteurs à partir des rendements.
    Chaque méthode correspond à un type de facteur.
    """

    def __init__(self, returns: pd.DataFrame):
        # On garde une copie pour éviter de modifier l'objet d'origine.
        self.returns = returns.copy()

    def build_market_factor(self, config: FactorConfig) -> pd.DataFrame:
        """Construit un facteur marché équipondéré."""

        if config.method == "equal_weight":
            # Facteur marché = moyenne des rendements des actifs à chaque date.
            factor = self.returns.mean(axis=1)

        else:
            raise ValueError("Only method='equal_weight' is currently supported.")

        return pd.DataFrame({config.name: factor})

    def build_pca_factor(self, config: FactorConfig) -> pd.DataFrame:
        """Construit des facteurs PCA à partir des rendements."""

        if config.n_components is None:
            raise ValueError("n_components must be provided for PCA factors.")

        # La PCA ne supporte pas les valeurs manquantes.
        clean_returns = self.returns.dropna()

        pca = PCA(n_components=config.n_components)
        pcs = pca.fit_transform(clean_returns)

        # Exemple : si name="PC" et n_components=3, colonnes PC1, PC2, PC3.
        columns = [
            f"{config.name}{i + 1}"
            for i in range(config.n_components)
        ]

        return pd.DataFrame(
            pcs,
            index=clean_returns.index,
            columns=columns
        )

    def build_momentum_factor(self, config: FactorConfig) -> pd.DataFrame:
        """Construit un facteur momentum moyen sur une fenêtre glissante."""

        if config.window is None:
            raise ValueError("window must be provided for momentum factors.")

        # Moyenne rolling par actif, puis moyenne transversale entre actifs.
        factor = self.returns.rolling(config.window).mean().mean(axis=1)

        return pd.DataFrame({config.name: factor})

    def build_volatility_factor(self, config: FactorConfig) -> pd.DataFrame:
        """Construit un facteur volatilité moyenne sur une fenêtre glissante."""

        if config.window is None:
            raise ValueError("window must be provided for volatility factors.")

        # Volatilité rolling par actif, puis moyenne transversale.
        factor = self.returns.rolling(config.window).std().mean(axis=1)

        return pd.DataFrame({config.name: factor})

    def build_dispersion_factor(self, config: FactorConfig) -> pd.DataFrame:
        """Construit un facteur de dispersion cross-sectionnelle."""

        # Dispersion = écart-type des rendements des actifs à une même date.
        factor = self.returns.std(axis=1)

        return pd.DataFrame({config.name: factor})

    def build(self, configs: List[FactorConfig]) -> pd.DataFrame:
        """Construit tous les facteurs demandés dans la liste de configurations."""

        factor_list = []

        for config in configs:

            if config.factor_type == "market":
                factor = self.build_market_factor(config)

            elif config.factor_type == "pca":
                factor = self.build_pca_factor(config)

            elif config.factor_type == "momentum":
                factor = self.build_momentum_factor(config)

            elif config.factor_type == "volatility":
                factor = self.build_volatility_factor(config)

            elif config.factor_type == "dispersion":
                factor = self.build_dispersion_factor(config)

            else:
                raise ValueError(f"Unknown factor_type: {config.factor_type}")

            factor_list.append(factor)

        # On concatène tous les facteurs en colonnes.
        return pd.concat(factor_list, axis=1).dropna(how="all")


# ============================================================
# 6. REGRESSION RESULTS
# ============================================================

class RegressionResults:
    """
    Classe de stockage des résultats de régression.

    Elle permet d'avoir une interface commune pour :
    - OLS statique ;
    - rolling window ;
    - Kalman.
    """

    def __init__(self, method: str):
        # Méthode utilisée : OLS, ROLLING ou KALMAN.
        self.method = method.upper()

        # Modèles statsmodels pour l'OLS statique.
        self.models = {}

        # Alphas et betas estimés.
        self.alphas = {}
        self.betas = {}

        # Résidus et valeurs ajustées.
        self.residuals = {}
        self.fitted_values = {}

        # Statistiques disponibles surtout pour OLS.
        self.r2 = {}
        self.tstats = {}
        self.pvalues = {}

    def _filter_assets(self, assets=None):
        """Vérifie et normalise la liste des actifs demandés."""

        if assets is None:
            return list(self.alphas.keys())

        if isinstance(assets, str):
            assets = [assets]

        missing = [asset for asset in assets if asset not in self.alphas]

        if missing:
            raise ValueError(f"Unknown assets: {missing}")

        return assets

    def add_static_result(self, asset: str, model):
        """Ajoute les résultats d'une régression OLS statique."""

        params = model.params

        self.models[asset] = model

        # L'alpha correspond à la constante.
        self.alphas[asset] = params["const"]

        # Les betas correspondent aux coefficients des facteurs.
        self.betas[asset] = params.drop("const")

        self.residuals[asset] = model.resid
        self.fitted_values[asset] = model.fittedvalues

        self.r2[asset] = model.rsquared
        self.tstats[asset] = model.tvalues
        self.pvalues[asset] = model.pvalues

    def add_dynamic_result(
        self,
        asset: str,
        alphas: pd.Series,
        betas: pd.DataFrame,
        residuals: pd.Series,
        fitted_values: pd.Series
    ):
        """Ajoute les résultats d'une estimation dynamique."""

        self.alphas[asset] = alphas
        self.betas[asset] = betas
        self.residuals[asset] = residuals
        self.fitted_values[asset] = fitted_values

    def get_alphas(self, assets=None):
        """Retourne les alphas pour un ou plusieurs actifs."""

        assets = self._filter_assets(assets)

        if self.method == "OLS":
            # En OLS, un seul alpha par actif.
            return pd.Series(
                {asset: self.alphas[asset] for asset in assets},
                name="alpha"
            )

        # En rolling/Kalman, l'alpha varie dans le temps.
        return pd.DataFrame(
            {asset: self.alphas[asset] for asset in assets}
        )

    def get_betas(self, assets=None):
        """Retourne les betas pour un ou plusieurs actifs."""

        assets = self._filter_assets(assets)

        if self.method == "OLS":
            # En OLS, une ligne par actif et une colonne par facteur.
            return pd.DataFrame(
                {asset: self.betas[asset] for asset in assets}
            ).T

        if len(assets) == 1:
            # Pour un seul actif dynamique, on retourne directement un DataFrame.
            return self.betas[assets[0]]

        # Pour plusieurs actifs dynamiques, on retourne un dictionnaire.
        return {
            asset: self.betas[asset]
            for asset in assets
        }

    def get_residuals(self, assets=None):
        """Retourne les résidus de régression."""

        assets = self._filter_assets(assets)

        return pd.DataFrame(
            {asset: self.residuals[asset] for asset in assets}
        )

    def get_fitted_values(self, assets=None):
        """Retourne les valeurs prédites par le modèle."""

        assets = self._filter_assets(assets)

        return pd.DataFrame(
            {asset: self.fitted_values[asset] for asset in assets}
        )

    def get_r2(self, assets=None):
        """Retourne les R², uniquement disponibles pour l'OLS statique."""

        assets = self._filter_assets(assets)

        if self.method != "OLS":
            raise ValueError("R2 is only available for static OLS.")

        return pd.Series(
            {asset: self.r2[asset] for asset in assets},
            name="R2"
        )

    def get_tstats(self, assets=None):
        """Retourne les t-statistiques, uniquement pour l'OLS statique."""

        assets = self._filter_assets(assets)

        if self.method != "OLS":
            raise ValueError("t-stats are only available for static OLS.")

        return pd.DataFrame(
            {asset: self.tstats[asset] for asset in assets}
        ).T

    def get_pvalues(self, assets=None):
        """Retourne les p-values, uniquement pour l'OLS statique."""

        assets = self._filter_assets(assets)

        if self.method != "OLS":
            raise ValueError("p-values are only available for static OLS.")

        return pd.DataFrame(
            {asset: self.pvalues[asset] for asset in assets}
        ).T


# ============================================================
# 7. FACTOR REGRESSION
# ============================================================

class FactorRegression:
    """
    Classe qui estime les modèles factoriels.

    Modèle général :
        r_i,t = alpha_i + beta_i' F_t + epsilon_i,t

    où :
    - r_i,t est le rendement de l'actif i ;
    - F_t est le vecteur des facteurs ;
    - alpha_i est la performance non expliquée par les facteurs ;
    - beta_i mesure l'exposition aux facteurs.
    """

    def __init__(self, returns: pd.DataFrame, factors: pd.DataFrame):
        self.returns = returns.copy()
        self.factors = factors.copy()

    def _prepare_data(
        self,
        asset: str,
        factors_to_use: List[str]
    ):
        """Prépare y et X propres pour une régression donnée."""

        if asset not in self.returns.columns:
            raise ValueError(f"Unknown asset: {asset}")

        missing_factors = [
            factor for factor in factors_to_use
            if factor not in self.factors.columns
        ]

        if missing_factors:
            raise ValueError(f"Unknown factors: {missing_factors}")

        # Matrice explicative X : facteurs sélectionnés.
        X = self.factors[factors_to_use]

        # Ajout de la constante pour estimer l'alpha.
        X = sm.add_constant(X)

        # Variable expliquée y : rendement de l'actif.
        y = self.returns[asset]

        # Alignement temporel de y et X, puis suppression des NA.
        df = pd.concat([y, X], axis=1).dropna()

        y_clean = df[asset]
        X_clean = df.drop(columns=[asset])

        return y_clean, X_clean, df

    def run_ols(
        self,
        assets: List[str],
        factors_to_use: List[str]
    ) -> RegressionResults:
        """Estime une régression OLS statique pour chaque actif."""

        results = RegressionResults(method="OLS")

        for asset in assets:

            y_clean, X_clean, _ = self._prepare_data(
                asset=asset,
                factors_to_use=factors_to_use
            )

            model = sm.OLS(y_clean, X_clean).fit()

            results.add_static_result(asset, model)

        return results

    def run_rolling_window(
        self,
        assets: List[str],
        factors_to_use: List[str],
        window: int = 90
    ) -> RegressionResults:
        """
        Estime des alphas et betas dynamiques par fenêtre glissante.

        À chaque date t, le modèle est estimé sur les window observations
        précédentes, puis utilisé pour prédire y_t.
        """

        results = RegressionResults(method="ROLLING")

        for asset in assets:

            y_clean, X_clean, df = self._prepare_data(
                asset=asset,
                factors_to_use=factors_to_use
            )

            if len(df) <= window:
                raise ValueError(
                    f"Not enough observations for asset {asset}. "
                    f"Need more than window={window}."
                )

            alpha_values = {}
            beta_values = {}
            residual_values = {}
            fitted_values = {}

            for i in range(window, len(df)):

                # Fenêtre d'estimation : observations [i-window, ..., i-1].
                X_window = X_clean.iloc[i - window:i]
                y_window = y_clean.iloc[i - window:i]

                model = sm.OLS(y_window, X_window).fit()

                current_date = df.index[i]

                # Observation courante à prédire.
                x_t = X_clean.iloc[i]
                y_t = y_clean.iloc[i]

                y_hat = float(x_t @ model.params)
                residual = y_t - y_hat

                alpha_values[current_date] = model.params["const"]
                beta_values[current_date] = model.params.drop("const")

                fitted_values[current_date] = y_hat
                residual_values[current_date] = residual

            alphas = pd.Series(alpha_values, name=asset)
            betas = pd.DataFrame(beta_values).T
            residuals = pd.Series(residual_values, name=asset)
            fitted = pd.Series(fitted_values, name=asset)

            results.add_dynamic_result(
                asset=asset,
                alphas=alphas,
                betas=betas,
                residuals=residuals,
                fitted_values=fitted
            )

        return results

    def run_kalman(
        self,
        assets: List[str],
        factors_to_use: List[str],
        delta: float = 1e-4,
        R: float = 1e-3
    ) -> RegressionResults:
        """
        Estime des coefficients dynamiques avec un filtre de Kalman simple.

        Interprétation :
        - theta contient alpha et les betas ;
        - P représente l'incertitude sur theta ;
        - Q contrôle la vitesse de variation des coefficients ;
        - R représente le bruit d'observation.
        """

        results = RegressionResults(method="KALMAN")

        for asset in assets:

            y_clean, X_clean, df = self._prepare_data(
                asset=asset,
                factors_to_use=factors_to_use
            )

            n_params = X_clean.shape[1]

            # Initialisation des paramètres alpha/beta à zéro.
            theta = np.zeros(n_params)

            # Matrice de covariance initiale des paramètres.
            P = np.eye(n_params)

            # Bruit d'état : autorise les coefficients à évoluer dans le temps.
            Q = delta * np.eye(n_params)

            theta_history = []
            fitted_values = []
            residual_values = []

            for t in range(len(df)):

                x_t = X_clean.iloc[t].values.reshape(-1, 1)
                y_t = y_clean.iloc[t]

                # Étape de prédiction.
                theta_pred = theta.copy()
                P_pred = P + Q

                # Prédiction du rendement courant.
                #y_hat = float(x_t.T @ theta_pred.reshape(-1, 1))
                y_hat = (x_t.T @ theta_pred.reshape(-1, 1)).item()
                error = y_t - y_hat

                # Variance de l'erreur de prédiction.
                #S = float(x_t.T @ P_pred @ x_t + R)
                S = (x_t.T @ P_pred @ x_t + R).item()
                # Gain de Kalman : poids donné à la nouvelle information.
                K = P_pred @ x_t / S

                # Mise à jour des paramètres et de leur covariance.
                theta = theta_pred + K.flatten() * error
                P = P_pred - K @ x_t.T @ P_pred

                theta_history.append(theta.copy())
                fitted_values.append(y_hat)
                residual_values.append(error)

            theta_df = pd.DataFrame(
                theta_history,
                index=df.index,
                columns=X_clean.columns
            )

            alphas = theta_df["const"].rename(asset)
            betas = theta_df.drop(columns=["const"])

            fitted = pd.Series(
                fitted_values,
                index=df.index,
                name=asset
            )

            residuals = pd.Series(
                residual_values,
                index=df.index,
                name=asset
            )

            results.add_dynamic_result(
                asset=asset,
                alphas=alphas,
                betas=betas,
                residuals=residuals,
                fitted_values=fitted
            )

        return results


    def run(
        self,
        assets: List[str],
        factors_to_use: List[str],
        method: str = "OLS",
        **kwargs
    ) -> RegressionResults:
        """Interface unique pour lancer OLS, rolling window ou Kalman."""

        method = method.upper()

        if method == "OLS":
            return self.run_ols(
                assets=assets,
                factors_to_use=factors_to_use
            )

        elif method in ["ROLLING", "ROLLING_WINDOW"]:
            return self.run_rolling_window(
                assets=assets,
                factors_to_use=factors_to_use,
                **kwargs
            )

        elif method == "KALMAN":
            return self.run_kalman(
                assets=assets,
                factors_to_use=factors_to_use,
                **kwargs
            )

        else:
            raise ValueError(
                "method must be 'OLS', 'ROLLING', or 'KALMAN'"
            )


# ============================================================
# 8. FACTOR ANALYSIS
# ============================================================

class FactorAnalysis:
    """
    Classe principale qui orchestre tout le workflow :
    - chargement des données ;
    - construction des facteurs ;
    - ajout de facteurs externes ;
    - estimation des modèles factoriels.
    """

    def __init__(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: Optional[str] = None,
        return_method: str = "log"
    ):
        self.tickers = tickers

        self.loader = DataLoader(
            tickers=tickers,
            start=start,
            end=end,
            return_method=return_method
        )

        self.prices = None
        self.returns = None
        self.factor_store = FactorStore()

    def load_data(self):
        """Télécharge les prix puis calcule les rendements."""

        self.prices = self.loader.load_prices()
        self.returns = self.loader.compute_returns(self.prices)

    def build_factors(self, configs: List[FactorConfig]):
        """Construit les facteurs définis dans configs."""

        if self.returns is None:
            raise ValueError("You must call load_data() before build_factors().")

        builder = FactorBuilder(self.returns)
        factors = builder.build(configs)

        self.factor_store.add_factors(factors)

    def add_external_factor(
        self,
        name: str,
        factor_series: pd.Series,
        description: Optional[str] = None
    ):
        """Ajoute un facteur externe, aligné sur les dates des rendements."""

        if self.returns is None:
            raise ValueError("You must call load_data() before adding factors.")

        # Réindexation pour s'assurer que le facteur a les mêmes dates que les rendements.
        factor_series = factor_series.reindex(self.returns.index)

        self.factor_store.add_factor(
            name=name,
            factor_series=factor_series,
            description=description
        )

    def list_factors(self) -> List[str]:
        """Liste les facteurs disponibles."""
        return self.factor_store.list_factors()

    def get_factors(
        self,
        names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Récupère tous les facteurs ou une sélection."""
        return self.factor_store.get_factors(names)

    def run_regression(
        self,
        assets: List[str],
        factors_to_use: List[str],
        method: str = "OLS",
        **kwargs
    ) -> RegressionResults:
        """Lance une régression factorielle sur les actifs choisis."""

        if self.returns is None:
            raise ValueError("You must call load_data() before run_regression().")

        factors = self.factor_store.get_factors(factors_to_use)

        regression = FactorRegression(
            returns=self.returns,
            factors=factors
        )

        return regression.run(
            assets=assets,
            factors_to_use=factors_to_use,
            method=method,
            **kwargs
        )
