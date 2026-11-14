"""Parte 30 - Rede 3: CNN em graficos de preco (regime de volatilidade dos proximos 5 pregoes, 3 classes).

Roda depois da parte 15 (``sp``, ``split``, ``regime_edges``, ``regime_labels``, ``pick_new_tickers``, ``REGIME_NAMES``...).
Variaveis expostas (todas as da parte levam prefixo ``cnn_`` / ``img_`` / ``CNN_`` para nao colidir com as redes 1 e 2):

    cnn_test_acc, cnn_val_acc, cnn_confusion, cnn_epoch_seconds       metricas principais
    cnn_model, cnn_cfg, cnn_edges, cnn_history, cnn_ablation          modelo final, configuracao, cortes dos tercis, curvas, tabela
    img_x_train/val/test, img_y_train/val/test, cnn_idx_train/val/test   imagens uint8 (N,64,64,3), rotulos e linhas de ``sp``
    render_windows, render_chart, png_bytes, cnn_ohlcv                rasterizacao (copia inline de tools/render_chart.py)
    cnn_direction_acc, cnn_new_predictions, cnn_har_acc_test          sanidade de direcao, previsoes fora da amostra, baseline HAR
    cnn_pred_test, cnn_probs_test, cnn_vs_har_pvalue, cnn_comparison  previsoes no teste, McNemar CNN x HAR, tabela de baselines
"""

# %% [markdown]
# # Parte 3 · Rede 3: CNN em gráficos de preço (regime de volatilidade)
#
# > **Aviso:** conteúdo educacional (trabalho de curso). Nada aqui é recomendação de investimento.
#
# ## Motivação
#
# Jiang, Kelly e Xiu (*(Re-)Imag(in)ing Price Trends*, **Journal of Finance**, 2023) mostram que uma rede convolucional
# consegue extrair padrões de **imagens de preço**: barras OHLC, média móvel e volume desenhados como pixels, sem nenhuma
# feature calculada à mão. Aqui seguimos a ideia, com uma diferença importante de **alvo**.
#
# - **Direção (sobe/desce) é quase imprevisível** em horizonte curto: o ganho reportado no artigo aparece em carteiras
#   (muitas ações, custos, rebalanceamento), e réplicas menores concluem que a CNN não passa de "chute" em acerto direcional.
#   Mostramos isso numa célula de sanidade ao final da seção.
# - **Volatilidade é persistente** (agrupamentos de volatilidade: dias agitados seguem dias agitados) e, por isso, **previsível**.
#   O alvo é o **regime de volatilidade dos próximos 5 pregões**: `baixa`, `media` ou `alta`, definidos pelos **tercis** de
#   `y = sqrt(252 · média(r²_{t+1..t+5}))` calculados **só no treino**.
#
# Esta é a **mesma grandeza** da Rede 2 (volatilidade, regressão), agora lida **a partir da imagem**. A pergunta da conclusão é:
# *o gráfico acrescenta algo além das features numéricas?*
#
# **Tarefa:** imagem 64×64×3 (20 pregões) → 3 classes. **Métrica exigida:** acurácia (acaso = 1/3 ≈ 33,3%).
# **Restrições do enunciado:** a rede é montada **do zero**, camada a camada, com `tf.keras.layers` (nada de `keras.applications`
# nem transfer learning).

# %%
import gc
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from IPython.display import Markdown, display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

keras = tf.keras
layers = tf.keras.layers

_t_part = time.time()
# TRINET_TF_THREADS limita as threads do TensorFlow (a parte 0 ja faz isso; aqui so vale na execucao isolada desta parte).
_threads = 0 if "THREADS" in globals() else int(os.environ.get("TRINET_TF_THREADS", "0"))
if _threads:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        print("TensorFlow ja inicializado: TRINET_TF_THREADS ignorado.")

CNN_SEED = 42
keras.utils.set_random_seed(CNN_SEED)
print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | threads intra-op: {tf.config.threading.get_intra_op_parallelism_threads() or 'auto'}")

# %% [markdown]
# ## 3.1 Do preço à imagem: rasterização determinística em NumPy
#
# Cada amostra é uma **janela de 20 pregões** desenhada em **64×64×3 pixels** (`uint8`, valores 0 ou 255, **sem antialiasing**
# e sem matplotlib, para que o mesmo gráfico dê exatamente os mesmos pixels no treino e na API):
#
# | canal | conteúdo | desenho |
# |---|---|---|
# | 0 (R) | **barras OHLC** | traço vertical *high-low* no pixel central do dia; *tick* de abertura à esquerda, de fechamento à direita; 3 px por dia |
# | 1 (G) | **média móvel** de 20 pregões do fechamento | linha contínua (as 19 linhas anteriores à janela só "aquecem" a média) |
# | 2 (B) | **volume** | barras de 1 px nas 16 linhas de baixo, normalizadas pelo maior volume da janela |
#
# Os três canais são independentes (R, G e B de uma imagem PNG); assim o preço usa a altura toda e o volume não "cobre" as
# barras. Os preços entram em **log-preço** (retornos iguais viram deslocamentos iguais). O que resta decidir é a **escala
# vertical**: (a) **por janela**, mínimo e máximo de cada janela viram o topo e a base (como em Jiang-Kelly-Xiu), ou (b) **fixa**,
# com um intervalo constante de log-preço para as 64 linhas, centrado no ponto médio da janela. A escolha afeta justamente o
# que a rede consegue ver da **volatilidade**, então ela sai de um **piloto curto** (mais abaixo), não de opinião.
#
# O código abaixo é a cópia inline de `tools/render_chart.py` (o apêndice do notebook testa que as duas cópias dão arrays idênticos).

# %%
import struct
import zlib
from collections.abc import Sequence
from typing import Literal

from numpy.typing import NDArray

CHART_WINDOW = 20  # pregoes desenhados
# linhas usadas no maximo (as 19 anteriores aquecem a media movel)
CHART_CONTEXT = 2 * CHART_WINDOW - 1
CHART_SIZE = 64  # altura = largura em pixels
CHART_DAY_PX = 3  # pixels por pregao (tick de abertura, traco central, tick de fechamento)
CHART_LEFT = 2  # margem esquerda em pixels (60 px de dados + 2 + 2 = 64)
CHART_VOL_ROWS = 16  # altura maxima das barras de volume
CHART_SPAN = 0.30  # log-preco que cabe nas 64 linhas quando scale="fixed"
CHART_SCALE: Literal["fixed", "window"] = "fixed"
_EPS_SPAN = 1e-9

_F64 = NDArray[np.float64]
_U8 = NDArray[np.uint8]
_I64 = NDArray[np.int64]

_X_OPEN: _I64 = CHART_LEFT + CHART_DAY_PX * np.arange(CHART_WINDOW, dtype=np.int64)
_X_MID: _I64 = _X_OPEN + 1
_X_CLOSE: _I64 = _X_OPEN + 2
_ROWS: _I64 = np.arange(CHART_SIZE, dtype=np.int64)
# colunas da media movel: de x_mid[0] ate x_mid[-1], com interpolacao linear entre pregoes
_MA_COLS: _I64 = np.arange(_X_MID[0], _X_MID[-1] + 1, dtype=np.int64)
_MA_DAY: _I64 = (_MA_COLS - _X_MID[0]) // CHART_DAY_PX
_MA_NEXT: _I64 = np.minimum(_MA_DAY + 1, CHART_WINDOW - 1)
_MA_FRAC: _F64 = ((_MA_COLS - _X_MID[0]) % CHART_DAY_PX) / CHART_DAY_PX


def _validate(arr: _F64) -> None:
    if arr.ndim != 3 or arr.shape[2] != 5 or arr.shape[1] < CHART_WINDOW:
        raise ValueError(
            f"esperava (N, T>={CHART_WINDOW}, 5) = open, high, low, close, volume; "
            f"obtive {arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("a janela tem valores NaN ou infinitos")
    if (arr[:, :, :4] <= 0).any():
        raise ValueError("precos precisam ser positivos")
    if (arr[:, :, 4] < 0).any():
        raise ValueError("volume nao pode ser negativo")


def _moving_average(close: _F64) -> _F64:
    """Media movel de 20 pregoes do fechamento nos ultimos 20 dias (expansiva sem contexto)."""
    n, t = close.shape
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(close, axis=1)], axis=1)
    idx = np.arange(t - CHART_WINDOW, t)
    start = np.maximum(idx - (CHART_WINDOW - 1), 0)
    return np.asarray((csum[:, idx + 1] - csum[:, start]) / (idx + 1 - start), dtype=np.float64)


def _frame(lo: _F64, hi: _F64, scale: Literal["fixed", "window"], span: float) -> tuple[_F64, _F64]:
    """Topo do eixo vertical e extensao (em log-preco) para cada janela, shape (N,)."""
    if scale == "window":
        return hi, np.maximum(hi - lo, _EPS_SPAN)
    return (lo + hi) / 2 + span / 2, np.full_like(lo, span)


def _or_columns(channel: _U8, columns: _I64, mask: NDArray[np.bool_]) -> None:
    """``channel[:, :, columns] |= 255`` onde ``mask`` (N, len(columns), altura) e verdadeiro."""
    channel[:, :, columns] |= np.where(mask.transpose(0, 2, 1), np.uint8(255), np.uint8(0))


def _draw_ohlc(channel: _U8, rows_o: _I64, rows_h: _I64, rows_l: _I64, rows_c: _I64) -> None:
    grid = _ROWS[None, None, :]
    stem = (grid >= rows_h[:, :, None]) & (grid <= rows_l[:, :, None])  # high-low (linha 0 = topo)
    _or_columns(channel, _X_MID, stem)
    n_idx = np.arange(channel.shape[0])[:, None]
    channel[n_idx, rows_o, _X_OPEN[None, :]] = 255
    channel[n_idx, rows_c, _X_CLOSE[None, :]] = 255


def _draw_ma(channel: _U8, rowf_ma: _F64) -> None:
    per_col = rowf_ma[:, _MA_DAY] * (1 - _MA_FRAC) + rowf_ma[:, _MA_NEXT] * _MA_FRAC
    rows = np.rint(per_col).astype(np.int64)
    prev = np.concatenate([rows[:, :1], rows[:, :-1]], axis=1)
    grid = _ROWS[None, None, :]
    line = (grid >= np.minimum(prev, rows)[:, :, None]) & (
        grid <= np.maximum(prev, rows)[:, :, None]
    )
    _or_columns(channel, _MA_COLS, line)


def _draw_volume(channel: _U8, volume: _F64) -> None:
    vmax = volume.max(axis=1, keepdims=True)
    ratio = np.divide(volume, vmax, out=np.zeros_like(volume), where=vmax > 0)
    heights = np.maximum(np.rint(ratio * CHART_VOL_ROWS), (volume > 0)).astype(np.int64)
    bars = _ROWS[None, None, :] >= (CHART_SIZE - heights)[:, :, None]
    _or_columns(channel, _X_MID, bars)


def render_windows(
    ohlcv: _F64 | Sequence[Sequence[Sequence[float]]],
    *,
    scale: Literal["fixed", "window"] = CHART_SCALE,
    span: float = CHART_SPAN,
) -> _U8:
    """Rasteriza janelas OHLCV em lote: ``(N, T, 5)`` -> ``(N, 64, 64, 3)`` uint8.

    Colunas: open, high, low, close, volume. Desenha os ultimos ``CHART_WINDOW`` (20) pregoes; as
    linhas anteriores (ate 19) so aquecem a media movel. Aceita tambem uma janela ``(T, 5)``.
    """
    arr = np.asarray(ohlcv, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None]
    _validate(arr)
    n, t, _ = arr.shape
    shown = arr[:, t - CHART_WINDOW :, :]
    ma = _moving_average(arr[:, :, 3])
    log_ohlc = np.log(shown[:, :, :4])
    log_ma = np.log(ma)
    lows = np.minimum(log_ohlc[:, :, 2].min(axis=1), log_ma.min(axis=1))
    highs = np.maximum(log_ohlc[:, :, 1].max(axis=1), log_ma.max(axis=1))
    top, extent = _frame(lows, highs, scale, span)
    factor = ((CHART_SIZE - 1) / extent)[:, None]

    def to_rows(log_price: _F64) -> _F64:
        return np.clip((top[:, None] - log_price) * factor, 0, CHART_SIZE - 1)

    rows = [np.rint(to_rows(log_ohlc[:, :, k])).astype(np.int64) for k in range(4)]
    img = np.zeros((n, CHART_SIZE, CHART_SIZE, 3), dtype=np.uint8)
    _draw_ohlc(img[..., 0], *rows)
    _draw_ma(img[..., 1], to_rows(log_ma))
    _draw_volume(img[..., 2], shown[:, :, 4])
    return img


def render_chart(
    window: _F64 | Sequence[Sequence[float]],
    *,
    scale: Literal["fixed", "window"] = CHART_SCALE,
    span: float = CHART_SPAN,
) -> _U8:
    """Uma janela ``(T, 5)`` -> imagem ``(64, 64, 3)``; mesmo codigo do lote."""
    batch = render_windows(np.asarray(window, dtype=np.float64)[None], scale=scale, span=span)
    image: _U8 = batch[0]
    return image


# --------------------------------------------------------------------------- PNG (sem Pillow)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def png_bytes(image: _U8) -> bytes:
    """Codifica ``(H, W, 3)`` uint8 em PNG RGB de 8 bits, sem filtro (bytes deterministicos)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("esperava um array uint8 (H, W, 3)")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )

# %% [markdown]
# ### Como é uma janela
#
# Um exemplo real (ação do treino) decomposto nos três canais, e o composto RGB que a rede recebe.

# %%
CNN_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
cnn_ohlcv = sp[CNN_OHLCV_COLS].to_numpy(dtype=np.float64)  # (linhas, 5), na ordem de ``sp`` (ticker, data)


def cnn_gather_windows(rows: np.ndarray) -> np.ndarray:
    """Janelas (N, 39, 5) terminando nas linhas ``rows`` de ``sp`` (a regra V3 garante >= 60 linhas antes)."""
    rows = np.asarray(rows, dtype=np.int64)
    return cnn_ohlcv[rows[:, None] + np.arange(-(CHART_CONTEXT - 1), 1)[None, :]]


_valid = sp["valid"].to_numpy()
_part = sp["part"].to_numpy()
_demo_row = int(np.flatnonzero(_valid & (_part == "train"))[40_000])
_demo = render_windows(cnn_gather_windows(np.array([_demo_row])))[0]
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
for ax, (title, img) in zip(
    axes,
    [("canal 0: OHLC", _demo[..., [0, 0, 0]]), ("canal 1: média móvel", _demo[..., [1, 1, 1]]), ("canal 2: volume", _demo[..., [2, 2, 2]]), ("RGB (entrada da CNN)", _demo)],
    strict=True,
):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.suptitle(f"{sp.at[_demo_row, 'ticker']}, 20 pregões até {sp.at[_demo_row, 'date']:%d/%m/%Y} (y = {sp.at[_demo_row, 'y']:.1%} a.a.)", fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.2 Piloto: qual escala vertical preserva a informação de volatilidade?
#
# Com a escala **por janela**, uma janela calma e uma agitada com a mesma forma viram a mesma imagem (o zoom é automático) e a
# amplitude absoluta, que é justamente a volatilidade, se perde. Com a escala **fixa**, barras longas = ação agitada, mas
# uma janela muito calma ocupa poucas linhas (resolução) e uma muito agitada pode sair da tela (recorte). O intervalo fixo
# também precisa ser escolhido. O piloto abaixo treina uma CNN **pequena** (8/16/32 filtros) por poucas épocas em
# **6 mil imagens de treino** e mede a acurácia em **3 mil imagens de validação** (o teste não é tocado), para
# `scale="window"` e quatro intervalos fixos. Roda em poucos minutos; `CNN_RUN_PILOT = False` pula a execução.

# %%
CNN_RUN_PILOT = True
CNN_PILOT_TRAIN, CNN_PILOT_VAL, CNN_PILOT_EPOCHS, CNN_PILOT_SEEDS = 6_000, 3_000, 8, (0, 1)


def cnn_pilot_model() -> keras.Model:
    """CNN do piloto: mesma família da rede final (3 blocos conv-pool com 8/16/32 filtros)."""
    return keras.Sequential(
        [
            layers.Input((CHART_SIZE, CHART_SIZE, 3)),
            layers.Rescaling(1 / 255),
            layers.Conv2D(8, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(3, activation="softmax"),
        ]
    )


_y_all = sp["y"].to_numpy()
cnn_edges = regime_edges(_y_all[_valid & (_part == "train")])  # tercis de y calculados SO no treino (todas as amostras validas)
_labels_all = regime_labels(np.nan_to_num(_y_all), cnn_edges)

cnn_pilot_table = pd.DataFrame()
if CNN_RUN_PILOT:
    _rng = np.random.default_rng(0)
    _tr = _rng.choice(np.flatnonzero(_valid & (_part == "train")), CNN_PILOT_TRAIN, replace=False)
    _va = _rng.choice(np.flatnonzero(_valid & (_part == "val")), CNN_PILOT_VAL, replace=False)
    _w_tr, _w_va = cnn_gather_windows(_tr), cnn_gather_windows(_va)
    _log_range = np.log(_w_tr[:, -CHART_WINDOW:, 1].max(axis=1)) - np.log(_w_tr[:, -CHART_WINDOW:, 2].min(axis=1))  # amplitude (log) de cada janela
    _rows = []
    for _name, _kw in {
        "por janela (min/max)": {"scale": "window"},
        "fixa, 0,15": {"scale": "fixed", "span": 0.15},
        "fixa, 0,20": {"scale": "fixed", "span": 0.20},
        "fixa, 0,30": {"scale": "fixed", "span": 0.30},
        "fixa, 0,45": {"scale": "fixed", "span": 0.45},
    }.items():
        _xtr, _xva = render_windows(_w_tr, **_kw), render_windows(_w_va, **_kw)
        _accs = []
        for _seed in CNN_PILOT_SEEDS:
            keras.utils.set_random_seed(_seed)
            _m = cnn_pilot_model()
            _m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            _h = _m.fit(_xtr, _labels_all[_tr], validation_data=(_xva, _labels_all[_va]), epochs=CNN_PILOT_EPOCHS, batch_size=128, verbose=0)
            _accs.append(max(_h.history["val_accuracy"]))
        _span = _kw.get("span")
        _rows.append(
            {
                "escala vertical": _name,
                "val acc (melhor época)": np.mean(_accs),
                "desvio": np.std(_accs),
                "janelas recortadas": float((_log_range > _span).mean()) if _span else 0.0,  # amplitude maior que o intervalo fixo
                "px por 1% de preço": (CHART_SIZE - 1) / (100 * _span) if _span else float("nan"),
            }
        )
    cnn_pilot_table = pd.DataFrame(_rows).set_index("escala vertical")
    keras.utils.set_random_seed(CNN_SEED)
    del _xtr, _xva, _w_tr, _w_va
cnn_pilot_table.style.format({"val acc (melhor época)": "{:.3f}", "desvio": "{:.3f}", "janelas recortadas": "{:.1%}", "px por 1% de preço": "{:.1f}"}, na_rep="-") if len(cnn_pilot_table) else print("piloto não executado")

# %% [markdown]
# ### Decisão do piloto
#
# **Como ler a tabela.** Com 6 mil imagens, 8 épocas e 2 seeds, o desvio entre seeds (coluna `desvio`) é da mesma ordem que as diferenças entre
# os intervalos fixos: **não dá para eleger um vencedor entre 0,15 e 0,45 só pela acurácia**. O que se repete, nesta tabela e em execuções
# exploratórias anteriores com outros tamanhos de amostra, é que **a escala por janela fica abaixo das escalas fixas**, que é o que a
# teoria prevê (ela apaga a amplitude, que é o próprio sinal de volatilidade). Por isso a decisão entre os intervalos fixos usa também as
# duas colunas determinísticas: a fração de janelas **recortadas** (saem da tela) e a **resolução** (px por 1% de preço).
#
# - `0,15` recorta cerca de **15% das janelas** (e quase 3 em cada 10 das janelas de volatilidade `alta`), justamente a classe que queremos
#   reconhecer: o gráfico "satura" e uma janela agitada vira outra igualmente cortada. Descartado.
# - `0,45` quase nunca recorta (≈ 0,1–0,2%), mas gasta só ≈ 1,4 px por 1% de preço: um dia calmo (±0,5%) fica quase invisível. Descartado.
# - `0,20` recorta ≈ 6% e `0,30` pouco mais de 1%. **Escolha: `0,30`** (ln da razão máx/mín da janela, ≈ 35% de amplitude): o recorte é raro (≈ 1–1,5% das janelas), a resolução
#   (≈ 2,1 px por 1%) ainda separa os regimes e a janela típica (mediana ≈ 9% de amplitude) ocupa umas 19 linhas de 64.
#
# `CHART_SPAN = 0.30` e `CHART_SCALE = "fixed"` ficam como padrão do renderizador (`tools/render_chart.py`), o mesmo código que a API de
# serving e o gerador dos *golden samples* usam. O piloto usa 6 mil imagens, então a escolha é uma **decisão de projeto fundamentada em
# propriedades da imagem**, não um *grid search* de acurácia.

# %% [markdown]
# ## 3.3 Construção do conjunto de dados
#
# Regras (as mesmas da Rede 2: V3.1 a V3.8, aplicadas pelas funções da parte 15):
#
# 1. **Amostra válida em `t`:** janela `[t-59, t+5]` contígua e sem `|r| > 0,4` (`sp["valid"]`); assim cada imagem tem o alvo
#    definido e as mesmas amostras podem ser comparadas com a Rede 2.
# 2. **Divisão cronológica** com embargo de 5 pregões (`split`): treino até ago/2016, validação até mai/2017, teste de
#    mai/2017 a jan/2018; os 5 últimos pregões ficam para as previsões "fora da amostra" no fim da seção.
# 3. **Classes:** tercis de `y` calculados **só nas amostras válidas de treino** (`cnn_edges`): `baixa` (< 14,4% a.a.),
#    `media` (14,4% a 22,9%) e `alta` (> 22,9%). O treino fica ~balanceado por construção; a validação e o teste **não** (o
#    mercado mudou de regime), e isso aparece na tabela de balanceamento.
# 4. **Amostragem com passo:** janelas de pregões vizinhos quase não diferem (19 de 20 dias em comum), e ~585 mil imagens não
#    cabem na CPU em minutos. Mantemos 1 de cada `CNN_STRIDE` pregões de cada ticker (com deslocamento por ticker), o mesmo passo em
#    treino, validação e teste, para chegar a **~60 mil imagens de treino**. Cada imagem é guardada em `uint8` (12 KB): ~1 GB no total.

# %%
CNN_TRAIN_TARGET = 60_000
_n_train_valid = int((_valid & (_part == "train")).sum())
CNN_STRIDE = max(1, round(_n_train_valid / CNN_TRAIN_TARGET))
_day_idx = np.searchsorted(split.days, sp["date"].to_numpy())
_ticker_code = sp["ticker"].astype("category").cat.codes.to_numpy()
_keep = ((_day_idx + _ticker_code) % CNN_STRIDE) == 0

cnn_idx_train, cnn_idx_val, cnn_idx_test = (np.flatnonzero(_valid & _keep & (_part == p)) for p in ("train", "val", "test"))
print(f"passo = {CNN_STRIDE} | amostras válidas por partição: treino {_n_train_valid:,d} | usadas: treino {len(cnn_idx_train):,d}, validação {len(cnn_idx_val):,d}, teste {len(cnn_idx_test):,d}")


def cnn_render_rows(rows: np.ndarray, chunk: int = 4_000) -> np.ndarray:
    """Renderiza as linhas de ``sp`` em lotes e devolve um array ``uint8`` (N, 64, 64, 3)."""
    out = np.empty((len(rows), CHART_SIZE, CHART_SIZE, 3), dtype=np.uint8)
    for start in range(0, len(rows), chunk):
        out[start : start + chunk] = render_windows(cnn_gather_windows(rows[start : start + chunk]))
    return out


_t0 = time.time()
img_x_train, img_x_val, img_x_test = (cnn_render_rows(r) for r in (cnn_idx_train, cnn_idx_val, cnn_idx_test))
img_y_train, img_y_val, img_y_test = (_labels_all[r] for r in (cnn_idx_train, cnn_idx_val, cnn_idx_test))
cnn_render_seconds = time.time() - _t0
print(f"{len(img_x_train) + len(img_x_val) + len(img_x_test):,d} imagens renderizadas em {cnn_render_seconds:.0f}s | memória: {(img_x_train.nbytes + img_x_val.nbytes + img_x_test.nbytes) / 1e9:.2f} GB (uint8)")

# Conferencia com a Rede 2: os cortes dos tercis devem ser os mesmos (mesma funcao, mesmas amostras de treino).
if "vol_tercile_edges" in globals():
    assert np.allclose(np.asarray(vol_tercile_edges, dtype="float64"), cnn_edges), (vol_tercile_edges, cnn_edges)
    print("cortes dos tercis idênticos aos da Rede 2:", np.round(cnn_edges, 4))

# %% [markdown]
# ### Balanceamento das classes e exemplos por classe

# %%
cnn_balance = pd.DataFrame(
    {name: np.bincount(y, minlength=3) for name, y in (("treino", img_y_train), ("validação", img_y_val), ("teste", img_y_test))},
    index=list(REGIME_NAMES),
).T
cnn_balance_pct = cnn_balance.div(cnn_balance.sum(axis=1), axis=0) * 100
display(pd.concat({"imagens": cnn_balance, "%": cnn_balance_pct.round(1)}, axis=1))

fig, ax = plt.subplots(figsize=(5.5, 2.6))
cnn_balance_pct.plot.bar(ax=ax, color=["#4C78A8", "#9AA5B1", "#E45756"], rot=0)
ax.axhline(100 / 3, color="k", ls=":", lw=1)
ax.set_ylabel("% das imagens")
ax.set_title("Classes por partição (pontilhado = 1/3)")
ax.legend(title="regime", fontsize=8)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 8, figsize=(13, 5.4))
_rng = np.random.default_rng(1)
for c, name in enumerate(REGIME_NAMES):
    picks = _rng.choice(np.flatnonzero(img_y_train == c), 8, replace=False)
    for ax, k in zip(axes[c], picks, strict=True):
        ax.imshow(img_x_train[k], interpolation="nearest")
        ax.set_title(f"{sp.at[cnn_idx_train[k], 'ticker']} · y={sp.at[cnn_idx_train[k], 'y']:.0%}", fontsize=7)
        ax.axis("off")
    axes[c, 0].text(-0.15, 0.5, name, transform=axes[c, 0].transAxes, rotation=90, ha="right", va="center", fontsize=11, fontweight="bold")
fig.suptitle("Exemplos de treino por classe (linhas: baixa, média, alta volatilidade futura)", y=1.0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.4 A CNN, camada a camada
#
# **Reescala visível.** A API entrega os pixels como `float32` em 0–255 e a **primeira camada do modelo é `Rescaling(1/255)`**:
# a normalização (exigida pelo enunciado, item 3.1.1) fica *dentro* do grafo e, portanto, vai junto na exportação.
#
# **Ponto de partida** (blocos conv → pool, depois uma cabeça densa):
#
# `Input(64,64,3)` → `Rescaling(1/255)` → 3 × [`Conv2D(3×3, same)` → `BatchNormalization` → `ReLU` → `MaxPooling2D(2)`] →
# `Flatten` → `Dense(64, relu)` → `Dropout(0,5)` → `Dense(3, softmax)`
#
# Tamanho dos mapas: 64×64 → 32×32 → 16×16 → 8×8; o `Flatten` vê 8·8·32 = 2.048 valores. A largura (8/16/32 filtros) sai
# da medição do tempo de época na CPU (célula abaixo): o objetivo é ≲ 60 s por época com ~60 mil imagens.
#
# A **única camada própria** é `VerticalJitter`, usada só na ablação de *augmentation*: desloca verticalmente, de até ±3 linhas,
# os canais de preço (OHLC e média móvel) **sem tocar no volume**. Somar uma constante ao log-preço **não altera a volatilidade**,
# então o rótulo continua válido. Já espelhar horizontalmente (`RandomFlip`) inverteria o eixo do tempo e trocaria a abertura
# pelo fechamento, o que **quebra a semântica do gráfico**; por isso não é usado.

# %%
class VerticalJitter(layers.Layer):
    """Desloca os canais de preco (0 e 1) por ``s`` linhas, ``s`` inteiro uniforme em [-max_shift, max_shift], so no treino."""

    trinet_augmentation = True  # o export remove a camada do clone de inferencia

    def __init__(self, max_shift: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = int(max_shift)

    def call(self, x, training=None):
        if not training or self.max_shift == 0:
            return x
        shift = tf.random.uniform([tf.shape(x)[0]], -self.max_shift, self.max_shift + 1, dtype=tf.int32)
        src = tf.range(CHART_SIZE)[None, :] + shift[:, None]  # (N, 64): linha de origem de cada linha
        inside = tf.cast((src >= 0) & (src < CHART_SIZE), x.dtype)[:, :, None, None]
        moved = tf.gather(x, tf.clip_by_value(src, 0, CHART_SIZE - 1), axis=1, batch_dims=1) * inside
        return tf.concat([moved[..., :2], x[..., 2:]], axis=-1)  # volume (canal 2) fica no lugar

    def get_config(self):
        return {**super().get_config(), "max_shift": self.max_shift}


def build_cnn(filters=(8, 16, 32), dense=64, dropout=0.5, jitter=0, n_classes=3, name="candlestick_cnn"):
    """Monta a CNN camada a camada. ``filters`` = um bloco conv por entrada; ``jitter`` > 0 liga a augmentation."""
    stack = [layers.Input((CHART_SIZE, CHART_SIZE, 3), name="chart"), layers.Rescaling(1 / 255, name="rescaling")]
    if jitter:
        stack.append(VerticalJitter(jitter, name="vertical_jitter"))
    for i, f in enumerate(filters, start=1):
        stack += [
            layers.Conv2D(f, 3, padding="same", name=f"conv{i}"),
            layers.BatchNormalization(name=f"bn{i}"),
            layers.ReLU(name=f"relu{i}"),
            layers.MaxPooling2D(2, name=f"pool{i}"),
        ]
    stack += [layers.Flatten(name="flatten"), layers.Dense(dense, activation="relu", name="dense")]
    if dropout:
        stack.append(layers.Dropout(dropout, name="dropout"))
    stack.append(layers.Dense(n_classes, activation="softmax", name="output"))
    return keras.Sequential(stack, name=name)


class EpochTimer(keras.callbacks.Callback):
    """Segundos de cada época."""

    def on_train_begin(self, logs=None):
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(time.time() - self._t)


class ImageBatches(keras.utils.PyDataset):
    """Lotes fatiados direto do array ``uint8`` (embaralha a cada época se ``shuffle``), opcionalmente só nas linhas ``rows``.

    Passar o array inteiro ao ``fit`` faz o Keras copiá-lo para um tensor (dobra a memória de ~1 GB de imagens), e ``x[rows]`` copiaria
    o subconjunto a cada treino; fatiar por lote evita as duas cópias.
    """

    def __init__(self, x, y, *, batch_size, rows=None, shuffle=False, seed=0):
        super().__init__()
        self.x, self.y, self.batch_size, self.shuffle = x, y, batch_size, shuffle
        self.rows = np.arange(len(x)) if rows is None else np.asarray(rows)
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(self.rows)) if shuffle else np.arange(len(self.rows))

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.batch_size))

    def __getitem__(self, i):
        idx = self.rows[np.sort(self._order[i * self.batch_size : (i + 1) * self.batch_size])]
        return self.x[idx], self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            self._order = self._rng.permutation(len(self.rows))


def fit_cnn(cfg, x_tr, y_tr, x_va, y_va, *, epochs, seed, lr=1e-3, patience=None, plateau=False, batch_size=128, rows=None, verbose=0):
    """Treina uma CNN do zero (loss e metrica do enunciado); com ``patience``, EarlyStopping(restore_best_weights).

    ``rows`` restringe o treino a essas linhas de ``x_tr``/``y_tr`` (sem copiar as imagens).
    """
    gc.collect()  # libera o modelo e os grafos do treino anterior (ciclos Keras/TF) antes de criar outro
    keras.utils.set_random_seed(seed)
    model = build_cnn(**cfg)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    timer = EpochTimer()
    callbacks = [timer]
    if patience:
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))
    if plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5))
    train = ImageBatches(x_tr, y_tr, batch_size=batch_size, rows=rows, shuffle=True, seed=seed)
    valid = ImageBatches(x_va, y_va, batch_size=512)
    hist = model.fit(train, validation_data=valid, epochs=epochs, callbacks=callbacks, verbose=verbose)
    return model, hist.history, timer.seconds


CNN_BASE_CFG = {"filters": (8, 16, 32), "dense": 64, "dropout": 0.5, "jitter": 0}
build_cnn(**CNN_BASE_CFG).summary()

# %% [markdown]
# ### Dimensionando para a CPU: tempo da primeira época
#
# O orçamento é **≲ 60 s por época** com ~60 mil imagens. Para escolher a largura, cronometramos uma época de 6 mil imagens
# (depois de uma época de aquecimento) para 4 larguras do mesmo desenho e extrapolamos para o treino inteiro. Depois, a 1ª época
# **real** da configuração base no treino completo.

# %%
_probe = slice(0, 6_000)
_rows = []
for _filters in [(8, 16, 32), (12, 24, 48), (16, 32, 64), (32, 64, 128)]:
    _cfg = {**CNN_BASE_CFG, "filters": _filters}
    _m, _h, _sec = fit_cnn(_cfg, img_x_train[_probe], img_y_train[_probe], img_x_val[:512], img_y_val[:512], epochs=2, seed=0)
    _rows.append({"filtros": "/".join(map(str, _filters)), "parâmetros": _m.count_params(), "s por época (extrapolado p/ o treino todo)": _sec[1] / 6_000 * len(img_x_train)})
cnn_sizing = pd.DataFrame(_rows).set_index("filtros")
display(cnn_sizing.style.format({"parâmetros": "{:,d}", "s por época (extrapolado p/ o treino todo)": "{:.0f}"}))

_m, _h, _sec = fit_cnn(CNN_BASE_CFG, img_x_train, img_y_train, img_x_val, img_y_val, epochs=1, seed=CNN_SEED)
cnn_first_epoch_seconds = _sec[0]
print(f"1ª época da base (8/16/32), treino completo ({len(img_x_train):,d} imagens): {cnn_first_epoch_seconds:.1f}s | parâmetros: {_m.count_params():,d} | val acc {_h['val_accuracy'][0]:.3f}")
del _m

# %% [markdown]
# ## 3.5 Justificativas das escolhas
#
# ### Justificativa: Arquitetura (Rede 3)
#
# - **Ativação das ocultas: ReLU** (nas convoluções e na densa). Não satura para entradas positivas (gradiente 1), é barata na CPU e produz
#   mapas **esparsos**, o que combina com imagens de pixels 0/255. `sigmoid` e `tanh` saturam e atrasam o treino de redes mais fundas; `softmax`
#   só faz sentido na saída. A `BatchNormalization` antes de cada ReLU estabiliza a escala dos mapas (o gráfico tem muitos pixels apagados e
#   poucos acesos, então a distribuição das ativações é bem assimétrica).
# - **Profundidade × largura das convoluções:** 3 blocos `Conv2D(3×3)` com **8, 16 e 32 filtros** (a largura dobra a cada `MaxPooling2D`, que divide
#   a área por 4, então o custo por camada fica parecido). Cada bloco aumenta o **campo receptivo**: ao fim do terceiro, um neurônio vê
#   ~22 px (≈ 7 pregões de 3 px); o `Flatten` + `Dense(64)` integram a janela inteira de 20 pregões. A largura foi limitada pelo **orçamento de
#   CPU** (≲ 60 s por época): a tabela de dimensionamento acima mostra o tempo por época subindo de ≈ 30–40 s (8/16/32) para ≈ 70 s (16/32/64) e
#   ≈ 160–180 s (32/64/128) no treino todo (os segundos variam com a carga da máquina). A **ablação** (próxima seção) compara a base com uma versão
#   **mais rasa** (2 blocos) e uma **mais larga** (16/32/64).
# - **Cabeça densa:** `Flatten` (8·8·32 = 2.048 valores) → `Dense(64, relu)` → `Dropout`. Essa camada concentra ~95% dos parâmetros da base e é o
#   principal vetor de *overfitting*, por isso o `Dropout(0,5)` (e a variação 0,25 na ablação).
# - **Saída: `Dense(3, softmax)`.** As três classes (`baixa`, `media`, `alta`) são **mutuamente exclusivas**; o `softmax` devolve uma distribuição de
#   probabilidade que soma 1, e é essa distribuição que a API devolve.
#
# ### Justificativa: Compilação (Rede 3)
#
# - **Loss: `sparse_categorical_crossentropy`.** É a entropia cruzada (log-verossimilhança) para o `softmax`; "sparse" porque os rótulos são os inteiros
#   0/1/2, sem precisar de one-hot. Penaliza com força apostas confiantes e erradas, e dá probabilidades utilizáveis.
# - **Otimizador: Adam** (taxa 1e-3), com `ReduceLROnPlateau` (×0,5 após 2 épocas sem melhora da loss de validação) e `EarlyStopping`
#   (paciência 4, `restore_best_weights`). O Adam adapta o passo por parâmetro, o que ajuda com a escala irregular dos mapas de uma CNN do zero;
#   a redução por platô troca "quantas épocas treinar" por um critério medido na validação. A varredura curta da seção seguinte compara 3e-4, 1e-3 e
#   3e-3 (3 seeds cada) e é lida com cuidado: o desvio entre seeds é grande nesse tamanho de amostra.
# - **Métrica: acurácia** (exigida pelo enunciado). O treino tem as 3 classes quase balanceadas (por construção dos tercis), então a acurácia é
#   informativa e o piso é 1/3. Como a validação e o teste **não** são balanceados (o mercado mudou de regime), a avaliação final também mostra a
#   **matriz de confusão** e o relatório por classe (precisão, revocação e F1), além da comparação com baselines.

# %% [markdown]
# ## 3.6 Ablação de configurações
#
# Em vez de um *tuner*, uma **ablação manual** com 5 configurações, avaliadas por **acurácia de validação** (média ± desvio de **3 seeds**), em
# **12 mil imagens de treino** e até 8 épocas (`EarlyStopping` com paciência 3 e `restore_best_weights`), para caber no orçamento de CPU. O
# resultado é um **ranking de configurações**, não a acurácia final (a escolhida é re-treinada no treino inteiro). A regra de escolha é fixada
# antes de olhar o teste: entre as configurações que **cabem no orçamento de CPU** (≲ 60 s por época no treino todo, pela tabela da seção 3.4), a de maior
# acurácia média de validação; se houver empate dentro de 0,5 pp, a de menos parâmetros.
#
# | config | o que muda em relação à base (`8/16/32`, `Dropout(0,5)`) |
# |---|---|
# | **base** | ponto de partida |
# | **dropout 0,25** | menos regularização na cabeça densa |
# | **augmentation (jitter ±3 linhas)** | `VerticalJitter` só no treino: desloca o preço verticalmente (preserva a volatilidade) |
# | **mais rasa (2 blocos)** | `8/16` filtros; o `Flatten` passa a ver 16·16·16 = 4.096 valores (mais parâmetros na cabeça densa, menos convoluções) |
# | **mais larga (16/32/64)** | dobra os filtros; **referência** para ver se mais capacidade ajuda. Custa ≈ 2× o tempo por época e estoura o orçamento, então **não concorre** à escolha |

# %%
CNN_ABL_TRAIN, CNN_ABL_EPOCHS, CNN_ABL_SEEDS = 12_000, 8, (0, 1, 2)
CNN_ABLATION = {
    "base": CNN_BASE_CFG,
    "dropout 0,25": {**CNN_BASE_CFG, "dropout": 0.25},
    "augmentation (jitter ±3 linhas)": {**CNN_BASE_CFG, "jitter": 3},
    "mais rasa (2 blocos)": {**CNN_BASE_CFG, "filters": (8, 16)},
    "mais larga (16/32/64, extra)": {**CNN_BASE_CFG, "filters": (16, 32, 64)},
}
_sub = np.random.default_rng(CNN_SEED).permutation(len(img_x_train))[:CNN_ABL_TRAIN]
_rows, ablation_runs, ablation_scores = [], {}, {}
_t0 = time.time()
for _name, _cfg in CNN_ABLATION.items():
    _accs, _losses, _secs = [], [], []
    for _seed in CNN_ABL_SEEDS:
        _m, _h, _sec = fit_cnn(_cfg, img_x_train, img_y_train, img_x_val, img_y_val, epochs=CNN_ABL_EPOCHS, seed=_seed, patience=3, rows=_sub)
        _loss, _acc = _m.evaluate(img_x_val, img_y_val, batch_size=512, verbose=0)  # pesos restaurados (melhor val_loss)
        _accs.append(_acc)
        _losses.append(_loss)
        _secs.append(np.mean(_sec))
        ablation_runs[(_name, _seed)] = _h
        ablation_scores[(_name, _seed)] = (_loss, _acc)
    _rows.append(
        {
            "configuração": _name,
            "parâmetros": _m.count_params(),
            "val acc (média)": np.mean(_accs),
            "val acc (desvio)": np.std(_accs),
            "val loss (média)": np.mean(_losses),
            "s/época": np.mean(_secs),
        }
    )
cnn_ablation = pd.DataFrame(_rows).set_index("configuração")
print(f"ablação concluída em {(time.time() - _t0) / 60:.1f} min ({len(CNN_ABLATION)} configurações × {len(CNN_ABL_SEEDS)} seeds, {CNN_ABL_TRAIN:,d} imagens, até {CNN_ABL_EPOCHS} épocas)")
cnn_ablation.style.format({"parâmetros": "{:,d}", "val acc (média)": "{:.3f}", "val acc (desvio)": "{:.3f}", "val loss (média)": "{:.3f}", "s/época": "{:.1f}"})

# %% [markdown]
# ### Taxa de aprendizado (varredura curta na configuração base)
#
# Mesmo subconjunto, mesmas épocas e **as mesmas 3 seeds** da ablação, para dar **evidência** (e não só convenção) à escolha da taxa do
# Adam. A linha de `1e-3` reaproveita as execuções da configuração `base` da ablação (idênticas em tudo), então só 2 taxas são treinadas.

# %%
CNN_LR_VALUES = (3e-4, 1e-3, 3e-3)
_rows = []
for _lr in CNN_LR_VALUES:
    _res = []
    for _seed in CNN_ABL_SEEDS:
        if _lr == 1e-3:  # = configuracao "base" da ablacao (mesmos dados, seed e callbacks)
            _loss, _acc = ablation_scores[("base", _seed)]
            _best = int(np.argmin(ablation_runs[("base", _seed)]["val_loss"])) + 1
        else:
            _m, _h, _sec = fit_cnn(CNN_BASE_CFG, img_x_train, img_y_train, img_x_val, img_y_val, epochs=CNN_ABL_EPOCHS, seed=_seed, lr=_lr, patience=3, rows=_sub)
            _loss, _acc = _m.evaluate(img_x_val, img_y_val, batch_size=512, verbose=0)
            _best = int(np.argmin(_h["val_loss"])) + 1
        _res.append((_acc, _loss, _best))
    _a, _l, _b = np.array(_res).T
    _rows.append({"taxa de aprendizado": _lr, "val acc (média)": _a.mean(), "val acc (desvio)": _a.std(), "val loss (média)": _l.mean(), "melhor época (média)": _b.mean()})
cnn_lr_table = pd.DataFrame(_rows).set_index("taxa de aprendizado")
cnn_lr_table.style.format({"val acc (média)": "{:.3f}", "val acc (desvio)": "{:.3f}", "val loss (média)": "{:.3f}", "melhor época (média)": "{:.1f}"})

# %% [markdown]
# ### Leitura da ablação e da taxa de aprendizado
#
# *(Números da execução de referência; ao reexecutar, mudam na segunda casa decimal. Seeds e ordem dos lotes mexem alguns pontos percentuais com 12 mil imagens.)*
#
# - **Quase tudo empata.** `base`, `augmentation` e `mais rasa` ficam em ≈ 0,507–0,512 de acurácia de validação (desvios de até ≈ 0,7 pp entre seeds) e a `mais larga`
#   em ≈ 0,516: diferenças menores que o ruído que a própria ablação mede. A única diferença que parece sinal é a de `dropout 0,25`
#   (≈ 2,5 pp abaixo, com um desvio de ≈ 4 pp: alguma seed oscilou muito), compatível com "menos regularização desestabiliza uma cabeça densa com
#   131 mil pesos", mas não conclusiva com 3 seeds.
# - **Capacidade não é o gargalo.** Dobrar os filtros não ajuda de forma relevante e custa ≈ 2× o tempo por época (por isso está fora da escolha). O sinal
#   de volatilidade no gráfico é grosseiro (amplitude das barras, tamanho do volume) e já cabe em ~140 mil parâmetros.
# - **Escolha.** Pela regra (dentro do orçamento de tempo, empate de 0,5 pp resolvido por menos parâmetros), fica a `base` (8/16/32, `Dropout(0,5)`).
#   Em execuções anteriores, com outra ordem de embaralhamento, a média da `base` chegou a cair para ≈ 0,48 (desvio ≈ 4 pp) e a `mais rasa` ficou
#   na frente; por isso o notebook usa 3 seeds e uma regra de desempate explícita, em vez de apostar numa única corrida.
# - **Taxa de aprendizado.** `3e-4` e `1e-3` empatam (≈ 0,508 e 0,509). `3e-3` tem média menor e um desvio ≈ 10× maior (instável, a loss de
#   validação oscila de época a época). `3e-4` só chegaria ao mínimo perto da 8ª época (a "melhor época" média fica em ≈ 7–8 de 8), então custaria mais
#   épocas para o mesmo resultado. Mantemos **`1e-3` com `ReduceLROnPlateau`**: é o ponto estável e mais rápido da varredura.

# %% [markdown]
# ## 3.7 Treino final e curvas
#
# A configuração escolhida é re-treinada **do zero no treino inteiro** (~60 mil imagens), com `EarlyStopping(monitor="val_loss", patience=4,
# restore_best_weights=True)` e `ReduceLROnPlateau`, até 20 épocas. As curvas mostram treino × validação; a linha pontilhada marca a melhor
# época (os pesos dessa época são os que seguem para o teste).

# %%
# Regra de escolha (fixada antes de olhar o teste): entre as configurações que cabem no orçamento de CPU (tempo por época no treino todo,
# tabela de dimensionamento da seção 3.4), a de maior acurácia média de validação; empate dentro de 0,5 pp -> menos parâmetros.
CNN_EPOCH_BUDGET_S = 60
_cost_col = "s por época (extrapolado p/ o treino todo)"


def cnn_epoch_cost(cfg: dict) -> float:
    """Segundos por época (treino todo) da largura ``cfg['filters']``; sem linha na tabela de dimensionamento, o custo da base (limite superior)."""
    return float(cnn_sizing[_cost_col].get("/".join(map(str, cfg["filters"])), cnn_sizing[_cost_col].iloc[0]))


cnn_ablation["cabe no orçamento"] = [cnn_epoch_cost(CNN_ABLATION[n]) <= CNN_EPOCH_BUDGET_S for n in cnn_ablation.index]
_pool = cnn_ablation[cnn_ablation["cabe no orçamento"]]
_near = _pool[_pool["val acc (média)"] >= _pool["val acc (média)"].max() - 0.005]
cnn_choice = str(_near["parâmetros"].idxmin())
cnn_cfg = CNN_ABLATION[cnn_choice]
_out = ", ".join(f"`{n}` ({cnn_epoch_cost(CNN_ABLATION[n]):.0f} s/época)" for n in cnn_ablation.index[~cnn_ablation["cabe no orçamento"]])
display(Markdown(f"**Configuração escolhida pela ablação:** `{cnn_choice}` → `{cnn_cfg}`" + (f"  \nFora do orçamento de {CNN_EPOCH_BUDGET_S} s/época (não concorre): {_out}" if _out else "")))

CNN_FINAL_EPOCHS = 20
cnn_model, cnn_history, _secs = fit_cnn(
    cnn_cfg, img_x_train, img_y_train, img_x_val, img_y_val, epochs=CNN_FINAL_EPOCHS, seed=CNN_SEED, patience=4, plateau=True, verbose=2
)
cnn_epoch_seconds = float(np.mean(_secs))
cnn_best_epoch = int(np.argmin(cnn_history["val_loss"])) + 1
cnn_val_loss, cnn_val_acc = cnn_model.evaluate(img_x_val, img_y_val, batch_size=512, verbose=0)  # pesos restaurados (melhor val_loss)
print(
    f"\nparâmetros {cnn_model.count_params():,d} | {len(_secs)} épocas, {cnn_epoch_seconds:.1f}s/época "
    f"({sum(_secs) / 60:.1f} min no total) | melhor época {cnn_best_epoch} | val acc {cnn_val_acc:.3f}, val loss {cnn_val_loss:.3f}"
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
_epochs = np.arange(1, len(cnn_history["loss"]) + 1)
for ax, key, title in zip(axes, ("loss", "accuracy"), ("Loss (entropia cruzada)", "Acurácia"), strict=True):
    ax.plot(_epochs, cnn_history[key], label="treino")
    ax.plot(_epochs, cnn_history[f"val_{key}"], label="validação")
    ax.axvline(cnn_best_epoch, color="gray", ls=":", label="melhor época")
    ax.set_xlabel("época")
    ax.set_title(title)
axes[1].axhline(1 / 3, color="k", ls="--", lw=0.8, label="acaso (1/3)")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.8 Avaliação no teste (tocado uma única vez)

# %%
cnn_probs_test = cnn_model.predict(img_x_test, batch_size=512, verbose=0)
cnn_pred_test = cnn_probs_test.argmax(axis=1)
cnn_test_acc = float((cnn_pred_test == img_y_test).mean())
cnn_confusion = confusion_matrix(img_y_test, cnn_pred_test, labels=[0, 1, 2])
_half = 1.96 * np.sqrt(cnn_test_acc * (1 - cnn_test_acc) / len(img_y_test))
_major = int(np.bincount(img_y_train).argmax())
print(f"ACURÁCIA NO TESTE: {cnn_test_acc:.3%}  (IC 95% ≈ ±{_half:.1%}; {len(img_y_test):,d} imagens)")
print(f"acaso (1/3): 33,3% | classe do treino mais frequente ('{REGIME_NAMES[_major]}') no teste: {(img_y_test == _major).mean():.1%} | validação: {cnn_val_acc:.1%}")

fig, ax = plt.subplots(figsize=(5.2, 4.4))
_pct = cnn_confusion / cnn_confusion.sum(axis=1, keepdims=True)
sns.heatmap(
    _pct,
    annot=np.array([[f"{n:,d}\n{p:.0%}" for n, p in zip(rn, rp, strict=True)] for rn, rp in zip(cnn_confusion, _pct, strict=True)]),
    fmt="",
    cmap="Blues",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "fração da classe real"},
    xticklabels=REGIME_NAMES,
    yticklabels=REGIME_NAMES,
    linewidths=0.5,
    ax=ax,
)
for i in range(3):  # destaca os acertos (diagonal)
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="#D62728", lw=3))
ax.set_xlabel("classe prevista")
ax.set_ylabel("classe real")
ax.set_title(f"Matriz de confusão no teste (acurácia {cnn_test_acc:.1%})")
plt.tight_layout()
plt.show()

print(classification_report(img_y_test, cnn_pred_test, target_names=REGIME_NAMES, digits=3))

# Erros mais "confiantes": imagens em que a CNN apostou alto na classe errada.
_wrong = np.flatnonzero(cnn_pred_test != img_y_test)
_worst = _wrong[np.argsort(-cnn_probs_test[_wrong, cnn_pred_test[_wrong]])[:8]]
fig, axes = plt.subplots(1, 8, figsize=(13, 2.5))
for ax, k in zip(axes, _worst, strict=True):
    ax.imshow(img_x_test[k], interpolation="nearest")
    ax.set_title(
        f"{sp.at[cnn_idx_test[k], 'ticker']} {sp.at[cnn_idx_test[k], 'date']:%d/%m/%y}\nreal {REGIME_NAMES[img_y_test[k]]}, previu {REGIME_NAMES[cnn_pred_test[k]]} ({cnn_probs_test[k].max():.0%})",
        fontsize=6.5,
    )
    ax.axis("off")
fig.suptitle("Erros em que a CNN mais confiou", y=1.04, fontsize=10)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.9 Comparação com a Rede 2 (números × imagem) para o mesmo alvo de 3 classes

# %%
from scipy.stats import binomtest

# Baseline HAR (regressão logística em ln rv_5, ln rv_20, ln rv_60) para o MESMO alvo de 3 classes e as MESMAS imagens
# (mesmo subconjunto com passo). As variaveis ``vol_baseline_acc_*`` vem da Rede 2 (teste completo, sem passo).
_r2 = sp["logret"] ** 2
_rv = {k: np.sqrt(ANN * _r2.groupby(sp["ticker"], sort=False).transform(lambda s, k=k: s.rolling(k).mean())).to_numpy() for k in (5, 20, 60)}
_x_har = np.log(np.maximum(np.column_stack([_rv[k] for k in (5, 20, 60)]), VOL_FLOOR))
_har = LogisticRegression(max_iter=1000).fit(_x_har[cnn_idx_train], img_y_train)
cnn_har_acc_val = float(_har.score(_x_har[cnn_idx_val], img_y_val))
cnn_har_acc_test = float(_har.score(_x_har[cnn_idx_test], img_y_test))
_har_pred = _har.predict(_x_har[cnn_idx_test])
cnn_rede2_available = "vol_baseline_acc_har" in globals()  # False = execucao isolada desta parte (sem a parte 2)
if "vol_baseline_acc_har" not in globals():  # execucao isolada da parte 3: usa o calculo local (a Rede 2 sobrescreve quando existe)
    vol_baseline_acc_har = cnn_har_acc_test
if "vol_baseline_acc_mlp" not in globals():
    vol_baseline_acc_mlp = float("nan")

_cnn_ok, _har_ok = cnn_pred_test == img_y_test, _har_pred == img_y_test
_b, _c = int((_cnn_ok & ~_har_ok).sum()), int((~_cnn_ok & _har_ok).sum())  # discordâncias (McNemar exato)
cnn_vs_har_pvalue = float(binomtest(_b, _b + _c, 0.5).pvalue) if _b + _c else 1.0

cnn_comparison = pd.DataFrame(
    {
        "acurácia no teste": [1 / 3, float((img_y_test == _major).mean()), cnn_har_acc_test, float(vol_baseline_acc_har), float(vol_baseline_acc_mlp), cnn_test_acc],
        "base": ["-", "teste da CNN", "teste da CNN (mesmas imagens)", "Rede 2, teste completo", "Rede 2, teste completo", "teste da CNN"],
    },
    index=["acaso (1/3)", "classe majoritária do treino", "HAR logístico (3 features de vol.)", "Rede 2: HAR (variável vol_baseline_acc_har)", "Rede 2: MLP (variável vol_baseline_acc_mlp)", "CNN em gráficos"],
)
display(cnn_comparison.style.format({"acurácia no teste": "{:.3f}"}, na_rep="n/d"))
print(f"CNN × HAR nas mesmas imagens: CNN acerta e HAR erra {_b}; HAR acerta e CNN erra {_c} (McNemar exato, p = {cnn_vs_har_pvalue:.3g})")

fig, ax = plt.subplots(figsize=(7, 2.8))
_plot = cnn_comparison["acurácia no teste"].dropna()
ax.barh(_plot.index[::-1], _plot.to_numpy()[::-1], color=["#E45756" if n.startswith("CNN") else "#9AA5B1" for n in _plot.index[::-1]])
ax.axvline(1 / 3, color="k", ls=":", lw=1)
ax.set_xlim(0.25, max(0.7, float(_plot.max()) + 0.05))
ax.set_xlabel("acurácia (3 classes)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3.10 Sanidade: e a direção (sobe/desce)?

# %% [markdown]
# O mesmo desenho de CNN (e as mesmas imagens), agora com o alvo **direção**: `1` se `ln(close_{t+5} / close_t) > 0`, `0` caso
# contrário (2 classes, `Dense(2, softmax)`). Treino curto em 12 mil imagens, com os mesmos passos de validação. Se a direção
# fosse previsível pelo gráfico, a acurácia ficaria bem acima da classe majoritária. O esperado é "perto do acaso".

# %%
_ret5 = np.log(sp.groupby("ticker", sort=False)["close"].shift(-HORIZON) / sp["close"]).to_numpy()
_dir = (_ret5 > 0).astype("int64")
cnn_dir_train, cnn_dir_val, cnn_dir_test = _dir[cnn_idx_train], _dir[cnn_idx_val], _dir[cnn_idx_test]
CNN_DIR_TRAIN, CNN_DIR_EPOCHS = 12_000, 6
_sub = np.random.default_rng(CNN_SEED).permutation(len(img_x_train))[:CNN_DIR_TRAIN]
_m, _h, _ = fit_cnn({**cnn_cfg, "n_classes": 2}, img_x_train, cnn_dir_train, img_x_val, cnn_dir_val, epochs=CNN_DIR_EPOCHS, seed=CNN_SEED, patience=3, rows=_sub)
_prior = int(np.bincount(cnn_dir_train).argmax())  # classe mais frequente no treino
cnn_direction_acc = {
    "val": float(_m.evaluate(img_x_val, cnn_dir_val, batch_size=512, verbose=0)[1]),
    "test": float(_m.evaluate(img_x_test, cnn_dir_test, batch_size=512, verbose=0)[1]),
    "test_always_prior_class": float((cnn_dir_test == _prior).mean()),
    "test_share_up": float(cnn_dir_test.mean()),
}
_half = 1.96 * np.sqrt(0.25 / len(cnn_dir_test))
display(
    pd.DataFrame(
        {
            "validação": [cnn_direction_acc["val"], float((cnn_dir_val == _prior).mean())],
            "teste": [cnn_direction_acc["test"], cnn_direction_acc["test_always_prior_class"]],
        },
        index=["CNN (direção)", f"sempre '{'sobe' if _prior else 'desce'}' (classe majoritária do treino)"],
    ).style.format("{:.3f}")
)
print(f"Direção no teste: CNN {cnn_direction_acc['test']:.1%} | acaso 50% (IC 95% ≈ ±{_half:.1%}) | fração de dias de alta no teste: {cnn_direction_acc['test_share_up']:.1%}")
del _m

# %% [markdown]
# ## 3.11 Previsões em novas instâncias (os 5 últimos pregões)

# %% [markdown]
# Os 5 últimos pregões do dataset (`split.new`) ainda **não têm alvo** (faltam os 5 dias futuros) e não pertencem ao teste:
# são previsões de verdade "fora da amostra". Sorteamos 12 tickers (seed fixa, `pick_new_tickers`) entre os que têm a janela
# completa nos 5 dias, e mostramos o gráfico, o regime previsto e as probabilidades.

# %%
_new_mask = (sp["part"].eq("new") & sp["valid_feat"]).to_numpy()
_cand = sp.loc[_new_mask].groupby("ticker").size()
cnn_new_tickers = pick_new_tickers(_cand[_cand == HORIZON].index, k=12, seed=7)
cnn_new_rows = np.flatnonzero(_new_mask & sp["ticker"].isin(cnn_new_tickers).to_numpy())
img_x_new = cnn_render_rows(cnn_new_rows)
cnn_p_new = cnn_model.predict(img_x_new, verbose=0)
cnn_new_predictions = pd.DataFrame(
    {
        "ticker": sp.loc[cnn_new_rows, "ticker"].to_numpy(),
        "date": sp.loc[cnn_new_rows, "date"].to_numpy(),
        "regime previsto": [REGIME_NAMES[k] for k in cnn_p_new.argmax(axis=1)],
        **{f"p({name})": cnn_p_new[:, i] for i, name in enumerate(REGIME_NAMES)},
    }
)
_last = cnn_new_predictions["date"] == cnn_new_predictions["date"].max()
fig, axes = plt.subplots(3, 4, figsize=(11, 8.2))
for ax, k in zip(axes.ravel(), np.flatnonzero(_last.to_numpy()), strict=True):
    row = cnn_new_predictions.iloc[k]
    ax.imshow(img_x_new[k], interpolation="nearest")
    ax.set_title(f"{row['ticker']}: {row['regime previsto']}", fontsize=10, fontweight="bold")
    ax.set_xlabel("  ".join(f"{n} {cnn_p_new[k, i]:.0%}" for i, n in enumerate(REGIME_NAMES)), fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle(f"Previsão do regime de volatilidade dos 5 pregões seguintes a {cnn_new_predictions['date'].max():%d/%m/%Y} (sem rótulo ainda)", fontsize=11)
plt.tight_layout()
plt.show()
display(cnn_new_predictions.pivot(index="ticker", columns="date", values="regime previsto").rename(columns=lambda d: f"{d:%d/%m}"))

# %% [markdown]
# ## 3.12 Conclusões da Rede 3
#
# - **Resultado.** A CNN que lê só o gráfico (20 pregões, 64×64×3) acerta o regime de volatilidade dos 5 pregões seguintes em ≈ 52% das imagens de teste
#   (acaso = 33,3%; sempre prever `baixa` = 43,8%). A acurácia de teste acompanha a de validação, sem sinal de excesso de ajuste ao conjunto de validação.
#   O intervalo de 95% (≈ ± 0,9 pp) é **otimista**: as janelas se sobrepõem (passo de 7 pregões, janelas de 20) e as ações do mesmo dia são correlacionadas.
# - **O gráfico acrescenta pouco além dos números.** Contra a regressão logística com as 3 features de volatilidade (HAR) nas **mesmas imagens**, a CNN fica
#   ≈ 1 pp acima, com McNemar exato p < 0,01. É uma diferença real nesta amostra, mas pequena: boa parte do que a rede "vê" é a amplitude recente das barras,
#   que o `ln rv_5`, `ln rv_20` e `ln rv_60` já resumem. Não é um argumento para trocar o modelo tabular pela imagem.
# - **Onde erra.** A matriz de confusão mostra um classificador quase **binário**: `baixa` é recuperada em ≈ 87% dos casos, `alta` em ≈ 40% e a classe `media`,
#   que fica entre as duas, em apenas ≈ 14% (a rede a empurra para `baixa`). Parte do erro vem do deslocamento de regime: o treino é balanceado por construção
#   (tercis), mas validação e teste têm mais dias calmos (≈ 43–44% `baixa`), e o modelo herda um viés para a classe mais calma.
# - **Direção continua imprevisível.** Com o mesmo desenho e o alvo sobe/desce, a acurácia de teste coincide com a fração de dias de alta (≈ 57,6%): a rede aprende a
#   responder "sobe" sempre. O ganho existe para a **volatilidade**, não para a direção, como na literatura.
# - **Limitações.** Um único corte temporal (2013-2018, mercado em geral calmo); uma imagem em baixa resolução, sem antialiasing e com escala vertical fixa; ≈ 57 mil
#   imagens de treino por limite de CPU (os trabalhos de referência usam milhões); sem custos de transação ou qualquer estratégia de negociação. Nada disto é recomendação
#   de investimento.
# - **Para a API.** O modelo exportado (`candlestick-cnn`, ONNX) recebe o **PNG** do gráfico renderizado por `tools/render_chart.py` e devolve as 3 probabilidades.
#   O apêndice de exportação confere que o caminho de serving (PNG → Pillow → ONNX Runtime) reproduz as previsões do Keras em todo o teste.

# %%
try:
    import resource

    cnn_peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB no Linux
except (ImportError, OSError):
    cnn_peak_rss_gb = float("nan")
if "part_times" in globals():
    part_times["30_cnn"] = time.time() - _t_part
print(f"Parte 3 concluída em {(time.time() - _t_part) / 60:.1f} min | pico de memória do processo: {cnn_peak_rss_gb:.1f} GB")
