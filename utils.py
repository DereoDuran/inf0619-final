from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import hashlib

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"
SEED = 42

CAT = (
    [f"Product_Info_{i}" for i in (1, 2, 3, 5, 6, 7)]
    + [f"Employment_Info_{i}" for i in (2, 3, 5)]
    + [f"InsuredInfo_{i}" for i in range(1, 8)]
    + [f"Insurance_History_{i}" for i in (1, 2, 3, 4, 7, 8, 9)]
    + ["Family_Hist_1"]
    + [f"Medical_History_{i}" for i in range(2, 42) if i not in (10, 15, 24, 32)]
)
CONT = (
    ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI"]
    + [f"Employment_Info_{i}" for i in (1, 4, 6)]
    + ["Insurance_History_5"]
    + [f"Family_Hist_{i}" for i in (2, 3, 4, 5)]
)
DISC = [f"Medical_History_{i}" for i in (1, 10, 15, 24, 32)]
DUMMY = [f"Medical_Keyword_{i}" for i in range(1, 49)]
FEATS = CAT + CONT + DISC + DUMMY

HASH_TRAIN = "b9d33ee3792f"
HASH_TEST = "beeff8a276c3"
HASH_FOLDS = "9ea76b467f2a"

def sha256(caminho, n=12):
    """Primeiros n caracteres do sha256 do arquivo."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:n]

AZUL = "#2a78d6"     # série principal
CINZA = "#b5b4ae"    # contexto e linhas de referência
TINTA = "#0b0b0b"    # texto principal
TINTA_2 = "#52514e"  # texto secundário

def estilo():
    """Visual padrão dos gráficos: grade leve, sem bordas sobrando."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })

VERMELHO = "#e34948"
# divergente: azul (negativo), cinza neutro no meio, vermelho (positivo)
DIVERGENTE = LinearSegmentedColormap.from_list("divergente", [AZUL, "#f0efec", VERMELHO])