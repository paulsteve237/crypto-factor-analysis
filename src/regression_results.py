import pandas as pd


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
