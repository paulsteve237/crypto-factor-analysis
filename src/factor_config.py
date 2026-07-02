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

