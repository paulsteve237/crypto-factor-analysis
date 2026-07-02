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
