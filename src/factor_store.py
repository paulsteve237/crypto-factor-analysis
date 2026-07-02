from typing import Optional, List, Dict

import pandas as pd


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
        

    def add_factor(
        self,
        name: str,
        factor_series: pd.Series,
        description: Optional[str] = None
    ):
        """Ajoute un facteur individuel dans le store."""

        factor_df = pd.DataFrame({name: factor_series})
        self.add_factors(
            factor_df=factor_df,
            descriptions={name: description}
        )

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
