# ============================================================
# 5. FACTOR BUILDER
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
