from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf


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
