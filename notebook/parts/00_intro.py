"""Parte 00 - capa, objetivo, aviso, configuracao (seeds, threads, versoes) e helpers comuns a todas as partes.

Variaveis/funcoes compartilhadas: SEED, DATA_DIR, THREADS, part_times, show(), set_seed(), HW.
"""

# %% [markdown]
# # TriNet · Três redes neurais para risco financeiro
#
# **T2 · Aprendizado Profundo (CIC407) · Inteligência Artificial e Aplicações**
#
# **Integrantes (nome completo e RA):** *a preencher*
#
# > Trabalho de curso, feito só para fins educacionais. Nada aqui é recomendação de investimento, e o modelo de crédito não serve para decidir crédito de verdade: os dados são de Taiwan, de 2005, e os custos usados no limiar são só um exemplo.
#
# ## Objetivo
#
# Neste trabalho aplicamos redes neurais em TensorFlow/Keras a três problemas supervisionados da área de risco financeiro. Montamos cada rede camada a camada com `tf.keras.layers` (sem `keras.applications` e sem transfer learning), escolhemos os hiperparâmetros com testes curtos e justificamos cada escolha em células de texto.
#
# | # | Problema | Tipo | Dados | O que a rede prevê |
# |---|---|---|---|---|
# | 1 | Risco de crédito | Classificação binária | Default of Credit Card Clients (UCI, 30.000 clientes) | se o cliente fica inadimplente no mês seguinte |
# | 2 | Risco de mercado, com números | Regressão | S&P 500, 5 anos (505 ações) | volatilidade dos próximos 5 pregões |
# | 3 | Risco de mercado, com imagens | CNN (3 classes) | gráficos de candlestick feitos com as mesmas ações | regime de volatilidade: baixa, média ou alta |
#
# As redes 2 e 3 preveem a mesma coisa, uma com números e outra com imagem. No fim comparamos as duas para ver se o gráfico ajuda além dos números.
#
# ## Organização do notebook
#
# 0. Capa e configuração.
# 1. Crédito (classificação): EDA, preparação dos dados, normalização, ablação, rede final, justificativas, treino, teste (matriz de confusão e relatório), equidade e novas instâncias.
# 2. Volatilidade (regressão): dados do S&P 500, features, divisão por data, normalização, ablação, rede final, RMSE e MAE com a margem relativa, erro por período e novas instâncias.
# 3. Candlestick (CNN): geração das imagens, ablação, CNN final, acurácia e comparação com a rede 2.
# 4. Conclusões, limitações e considerações éticas.
#
# ## Dados e licenças
#
# - Default of Credit Card Clients (UCI #350): Yeh, I-C. (2009), DOI 10.24432/C55S3H, licença CC BY 4.0. Baixado do repositório da UCI.
# - S&P 500, 5 anos (Kaggle `camnugent/sandp500`): preços diários de fev/2013 a fev/2018. Não conseguimos confirmar a licença do arquivo, por isso não colocamos esses dados no repositório.
# - Os dois downloads são anônimos, sem conta.

# %% [markdown]
# ## Configuração: versões, hardware, seeds e threads
#
# - Reprodutibilidade: chamamos `keras.utils.set_random_seed(seed)` antes de cada modelo e, nas ablações, usamos a média e o desvio de 3 seeds.
# - O treino roda em CPU (TensorFlow 2.21, sem GPU). A variável de ambiente `TRINET_TF_THREADS` limita as threads (0 = todas).
# - Os dados brutos ficam na pasta indicada por `TRINET_DATA_DIR` (padrão: `data/`, ao lado do notebook).

# %%
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido do TensorFlow
NOTEBOOK_T0 = time.time()
part_times: dict[str, float] = {}  # segundos por parte, preenchido ao fim de cada uma

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf

THREADS = int(os.environ.get("TRINET_TF_THREADS", "0"))
if THREADS > 0:  # precisa vir antes de qualquer operacao do TensorFlow
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)

SEED = 42
DATA_DIR = Path(os.environ.get("TRINET_DATA_DIR", "data")).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 100, "figure.autolayout": True, "axes.titlesize": 11})
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def set_seed(seed: int = SEED) -> None:
    """Fixa as sementes do Python, NumPy e TensorFlow (via Keras) de uma vez."""
    random.seed(seed)
    keras.utils.set_random_seed(seed)


def show(obj) -> None:
    """Mostra um DataFrame/Series no notebook (``display``) ou imprime, se nao houver IPython."""
    try:
        from IPython.display import display

        display(obj)
    except Exception:  # noqa: BLE001 - fallback para execucao como script
        print(obj)


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return float("nan")


HW = f"{_cpu_name()} | {os.cpu_count()} threads | {_ram_gb():.0f} GB RAM | GPU: {'sim' if tf.config.list_physical_devices('GPU') else 'nao (CPU)'}"
set_seed(SEED)
print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__} | Keras {keras.__version__} | NumPy {np.__version__} | pandas {pd.__version__} | scikit-learn {sklearn.__version__}")
print(f"Hardware: {HW}")
print(f"Threads do TensorFlow: {'todas' if THREADS == 0 else THREADS} | seed base: {SEED} | dados em: {DATA_DIR}")
