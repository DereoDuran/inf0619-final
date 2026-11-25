from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import TargetEncoder
from sklearn.decomposition import PCA
import hashlib
import numpy as np
import pandas as pd

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

LARANJA = "#eb6834"  # segunda série
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

# códigos fixos da Product_Info_2 (as 19 combinações vistas na EDA; o teste não tem nenhuma nova)
CODIGO_PI2 = {v: i for i, v in enumerate(
    ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "B1", "B2",
     "C1", "C2", "C3", "C4", "D1", "D2", "D3", "D4", "E1"]
)}


def carrega_dados():
    """Lê treino, teste e folds, conferindo os hashes. Devolve (treino com a coluna fold, teste)."""
    for nome, esperado in [("train.csv", HASH_TRAIN), ("test.csv", HASH_TEST), ("folds.csv", HASH_FOLDS)]:
        obtido = sha256(DADOS / nome)
        assert obtido == esperado, f"{nome}: hash {obtido}, esperado {esperado}"
    treino = pd.read_csv(DADOS / "train.csv").merge(
        pd.read_csv(DADOS / "folds.csv"), on="Id", validate="one_to_one"
    )
    teste = pd.read_csv(DADOS / "test.csv")
    return treino, teste


def remove_duplicadas(treino):
    """Tira as linhas repetidas nas features (fica a primeira). Só para o treino."""
    return treino.drop_duplicates(subset=FEATS).reset_index(drop=True)

VERSAO_PREP = 1

def bloco_keywords(df):
    """Bloco B: quantas keywords a linha tem, e se tem alguma (3.6)."""
    soma = df[DUMMY].sum(axis=1)
    return {"kw_soma": soma, "kw_alguma": (soma > 0).astype(int)}

def bloco_familia(df):
    """Bloco E: pares Family_Hist unidos com a origem (vivo, falecido, desconhecido) e número de pais vivos (3.7)."""
    novas = {}
    for a, b, nome in [("Family_Hist_2", "Family_Hist_3", "fh_23"), ("Family_Hist_4", "Family_Hist_5", "fh_45")]:
        novas[f"{nome}_idade"] = df[a].fillna(df[b])
        novas[f"{nome}_status"] = np.select([df[a].notna(), df[b].notna()], [1, 0], default=-1)
    novas["fh_pais_vivos"] = df["Family_Hist_2"].notna().astype(int) + df["Family_Hist_4"].notna().astype(int)
    return novas

COLS_NA = [  # as 13 colunas com NA no treino (3.2)
    "Employment_Info_1", "Employment_Info_4", "Employment_Info_6", "Insurance_History_5",
    "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5",
    "Medical_History_1", "Medical_History_10", "Medical_History_15", "Medical_History_24", "Medical_History_32",
]


def bloco_ausencia(df):
    """Bloco A: um indicador 0/1 por coluna com NA (3.3)."""
    return {f"na_{c}": df[c].isna().astype(int) for c in COLS_NA}


def bloco_contagem_na(df):
    """Bloco N: quantos NA a linha tem, no total e por grupo de colunas (3.3)."""
    na = df[COLS_NA].isna()
    novas = {"na_total": na.sum(axis=1)}
    for grupo in ("Employment_Info", "Insurance_History", "Family_Hist", "Medical_History"):
        novas[f"na_{grupo}"] = na[[c for c in COLS_NA if c.startswith(grupo)]].sum(axis=1)
    return novas


LETRA_PI2 = {l: i for i, l in enumerate("ABCDE")}
COMBINACAO_PI2 = {v: k for k, v in CODIGO_PI2.items()}  # código → "D3"


def bloco_interacoes(df):
    """Bloco D (2º lugar): interações das medidas com a idade."""
    return {"bmi_x_idade": df["BMI"] * df["Ins_Age"], "wt_x_idade": df["Wt"] * df["Ins_Age"]}


def bloco_pi2(df):
    """Bloco P (2º lugar): letra e número da Product_Info_2 separados."""
    comb = df["Product_Info_2"].map(COMBINACAO_PI2)  # a prepara já trocou o texto pelo código
    return {"pi2_letra": comb.str[0].map(LETRA_PI2), "pi2_numero": comb.str[1].astype(int)}


BLOCOS = {"B": bloco_keywords, "E": bloco_familia, "A": bloco_ausencia, "N": bloco_contagem_na,
          "D": bloco_interacoes, "P": bloco_pi2}


def prepara(df, blocos=()):
    """Transformações determinísticas, que não aprendem nada dos dados: base + blocos de features opcionais."""
    df = df.copy()
    df["Product_Info_2"] = df["Product_Info_2"].map(CODIGO_PI2)
    novas = {}
    for b in blocos:
        novas.update(BLOCOS[b](df))
    return pd.concat([df[FEATS], pd.DataFrame(novas, index=df.index)], axis=1) if novas else df[FEATS]

SEQUENCIAL = LinearSegmentedColormap.from_list("sequencial", ["#ffffff", "#9ec5f4", AZUL, "#104281"])

def plota_confusao(y_true, y_pred, titulo, classes=range(1, 9)):
    """Matriz de confusão normalizada por linha (% de cada classe real), com o valor em todas as células."""
    cm = confusion_matrix(y_true, y_pred, labels=list(classes), normalize="true") * 100
    fig, ax = plt.subplots(figsize=(6, 5.2), layout="constrained")
    im = ax.imshow(cm, cmap=SEQUENCIAL, vmin=0, vmax=100)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            rotulo = f"{v:.0f}" if v >= 1 or v == 0 else f"{v:.1f}"
            ax.text(j, i, rotulo, ha="center", va="center", fontsize=8,
                    color="white" if v > 50 else TINTA)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("classe prevista")
    ax.set_ylabel("classe real")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% da classe real")
    fig.suptitle(titulo, x=0.02, ha="left", fontsize=10, color=TINTA)
    plt.show()
    return cm

CAT_NATIVAS = [c for c in CAT if c != "Medical_History_2"]  # a MH2 passa do limite de 255 categorias do HGB

class KeywordAlvo(BaseEstimator, TransformerMixin):
    """1º lugar: média do Response para cada keyword-valor (com cross-fitting no treino);
    por linha, a média e o mínimo dessas 48 médias."""

    def __init__(self, colunas, seed=0):
        self.colunas = colunas
        self.seed = seed

    def fit_transform(self, X, y):
        self.te_ = TargetEncoder(target_type="continuous", random_state=self.seed)
        return self._junta(X, self.te_.fit_transform(X[self.colunas], y))

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        return self._junta(X, self.te_.transform(X[self.colunas]))

    def _junta(self, X, enc):
        X = X.reset_index(drop=True).copy()
        X["kw_alvo_media"] = enc.mean(axis=1)
        X["kw_alvo_min"] = enc.min(axis=1)
        return X

class FrequenciaEncoder(BaseEstimator, TransformerMixin):
    """Acrescenta, para cada coluna, a fração de linhas do treino com aquela categoria (categoria nova vira 0)."""

    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        self.freq_ = {c: X[c].value_counts(normalize=True) for c in self.colunas}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.colunas:
            X[f"freq_{c}"] = X[c].map(self.freq_[c]).fillna(0.0).to_numpy()
        return X


class AlvoEncoder(BaseEstimator, TransformerMixin):
    """Acrescenta target encoding multiclasse (uma coluna por classe), com cross-fitting no treino."""

    def __init__(self, colunas, seed=0):
        self.colunas = colunas
        self.seed = seed

    def fit_transform(self, X, y):
        self.te_ = TargetEncoder(target_type="multiclass", random_state=self.seed)
        return self._junta(X, self.te_.fit_transform(X[self.colunas], y))

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        return self._junta(X, self.te_.transform(X[self.colunas]))

    def _junta(self, X, novas):
        nomes = [f"te_{n}" for n in self.te_.get_feature_names_out()]
        return pd.concat([X.reset_index(drop=True), pd.DataFrame(novas, columns=nomes)], axis=1)

class PCAColunas(BaseEstimator, TransformerMixin):
    """Acrescenta k componentes principais de um grupo de colunas, ajustados no treino."""

    def __init__(self, colunas, k, prefixo="pca", seed=0):
        self.colunas = colunas
        self.k = k
        self.prefixo = prefixo
        self.seed = seed

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.k, random_state=self.seed).fit(X[self.colunas])
        return self

    def transform(self, X):
        X = X.copy()
        comp = self.pca_.transform(X[self.colunas])
        for i in range(self.k):
            X[f"{self.prefixo}_{i + 1}"] = comp[:, i]
        return X