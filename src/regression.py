from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regression_results import RegressionResults


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
