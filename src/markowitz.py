import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install arch
try:
    from arch import arch_model
except ImportError:
    arch_model = None


class DynamicFactorNeutralMarkowitz:

    def __init__(
        self,
        target_vol=0.02,
        risk_aversion=None,
        sigma_method="constant",   # "constant", "rolling", "dcc"
        rolling_window=60,
        dcc_a=0.03,
        dcc_b=0.95,
        garch_scale=100,
        ridge=1e-8
    ):
        self.target_vol = target_vol
        self.risk_aversion = risk_aversion
        self.sigma_method = sigma_method
        self.rolling_window = rolling_window
        self.dcc_a = dcc_a
        self.dcc_b = dcc_b
        self.garch_scale = garch_scale
        self.ridge = ridge

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
        if x is None:
            return None
        if isinstance(x, pd.Series):
            return x.to_frame()
        return x.copy()

    def _normalize_index(self, df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _normalize_betas(self, betas, assets):
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
        common_dates = None

        for _, beta_df in betas.items():
            if common_dates is None:
                common_dates = beta_df.index
            else:
                common_dates = common_dates.intersection(beta_df.index)

        return common_dates

    def _get_beta_t(self, betas, date, assets):
        beta_matrix = []

        for asset in assets:
            beta_matrix.append(betas[asset].loc[date].values)

        return np.asarray(beta_matrix)

    # ======================================================
    # Construction de Sigma_t
    # ======================================================

    def _build_constant_sigmas(self, residuals, dates, assets):
        Sigma = residuals[assets].dropna().cov().values
        Sigma = Sigma + self.ridge * np.eye(len(assets))

        return {
            pd.Timestamp(date): Sigma
            for date in dates
        }

    def _build_rolling_sigmas(self, residuals, dates, assets):
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
        returns = self._normalize_index(self._as_dataframe(returns))
        returns = returns[self.weights_.columns]

        common_dates = self.weights_.index.intersection(returns.index)

        w = self.weights_.loc[common_dates]
        r = returns.loc[common_dates]

        pnl = (w.shift(1) * r).sum(axis=1).dropna()
        pnl.name = "pnl"

        return pnl

    def get_cumulative_pnl(self, returns):
        return self.get_pnl(returns).cumsum().rename("cumulative_pnl")
    """
    def get_wealth(self, returns):
        return (1 + self.get_pnl(returns)).cumprod().rename("wealth")
    """
    def get_wealth(self, returns):
        return (self.get_pnl(returns)).cumsum().rename("wealth")

    def backtest(self, returns):
        pnl = self.get_pnl(returns)

        return pd.DataFrame({
            "pnl": pnl,
            "cumulative_pnl": pnl.cumsum(),
            #"wealth": (1 + pnl).cumprod()
            "wealth": (pnl).cumprod()
        })

    def performance(self, returns, periods_per_year=252):
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
        date = pd.Timestamp(date)

        return pd.DataFrame(
            self.sigmas_[date],
            index=self.assets_,
            columns=self.assets_
        )

    # ======================================================
    # Plots
    # ======================================================

    def plot_weights(self):
        self.weights_.plot(figsize=(12, 5))
        plt.title("Poids dynamiques")
        plt.axhline(0)
        plt.show()

    def plot_factor_exposures(self):
        self.factor_exposures_.plot(figsize=(12, 5))
        plt.title("Expositions factorielles")
        plt.axhline(0)
        plt.show()

    def plot_portfolio_vol(self):
        self.portfolio_vol_.plot(figsize=(12, 5))
        plt.title("Volatilité ex-ante")
        plt.axhline(self.target_vol, linestyle="--")
        plt.show()

    def plot_wealth(self, returns):
        self.get_wealth(returns).plot(figsize=(12, 5))
        plt.title("Richesse cumulée")
        plt.show()
