from dataclasses import dataclass
from typing import Optional


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
