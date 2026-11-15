"""Parte 15 - dados de mercado (S&P 500), compartilhada pelas redes 2 (volatilidade) e 3 (CNN de graficos).

API estavel (PLANO v1.2, V3). Rodar esta parte sozinha basta para usar tudo abaixo:

    DATA_DIR, SP500_FILE            caminhos (TRINET_DATA_DIR; padrao ``data`` ao lado do notebook)
    ANN, LOOKBACK, HORIZON, MAX_GAP_DAYS, MAX_ABS_RET     constantes (252, 60, 5, 5, 0.4)
    load_sp500() -> DataFrame       colunas date, ticker, open, high, low, close, volume, logret
                                    (ordenado por ticker/date, RangeIndex; ``sp500_report`` guarda as contagens)
    add_positions(df)               acrescenta ``pos`` (indice no ticker) e ``rem`` (linhas restantes no ticker)
    forward_vol(df) -> Series       y_t = sqrt(252 * mean(r_{t+1..t+5}^2)) (NaN sem 5 retornos futuros)
    sample_valid_mask(df, need_target=True) -> bool ndarray
                                    janela contigua [t-59, t+5] (ou [t-59, t] se need_target=False),
                                    sem salto > 5 dias corridos e sem |r| > 0.4
    chronological_split(dates)      -> ChronoSplit (mascaras train/val/test/new + fronteiras em data)
    new_instance_dates(dates, n=5)  os n ultimos pregoes (sem alvo ainda)
    pick_new_tickers(candidates, k=12, seed=7)   sorteio com seed fixa
    regime_edges(y_train) / regime_labels(y, edges)   tercis do treino -> classes 0/1/2 (baixa/media/alta)
"""

# %% [markdown]
# # Parte 2A · Dados de mercado: S&P 500 (usados nas redes 2 e 3)
#
# As redes de volatilidade (regressão) e de candlestick (CNN) preveem a mesma coisa: a volatilidade dos próximos 5 pregões de ações do S&P 500. Por isso o carregamento, a limpeza, o alvo e a divisão em treino, validação e teste ficam numa parte só, e as duas redes usam exatamente as mesmas amostras. Assim a comparação entre números e imagem é justa.
#
# **Fonte:** Kaggle `camnugent/sandp500` (preços diários ajustados a splits, 505 tickers, fev/2013 a fev/2018). Não conseguimos confirmar a licença do arquivo, então o CSV não vai para o repositório. O download é anônimo (`kagglehub`).
#
# > Conteúdo educacional. Nada aqui é recomendação de investimento.

# %%
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("TRINET_DATA_DIR", "data")).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR / "kagglehub")  # cache do kagglehub dentro de DATA_DIR

ANN = 252  # pregões por ano (anualização da volatilidade)
LOOKBACK = 60  # janela de passado das features (rv_60, mom_60, drawdown_60)
HORIZON = 5  # janela de futuro do alvo
MAX_GAP_DAYS = 5  # salto máximo (dias corridos) entre dois pregões consecutivos da janela
MAX_ABS_RET = 0.4  # |log-retorno| acima disso = artefato de ação corporativa (split/spin-off sem ajuste)
VOL_FLOOR = 0.005  # piso do alvo antes do log

sp500_report: dict[str, int] = {}


def load_sp500() -> pd.DataFrame:
    """Baixa (anonimamente) e limpa o S&P 500 conforme a regra V3.1 do plano.

    Remove linhas com NaN em open/high/low/close, volume 0, ``high < low`` ou ``close`` fora de [low, high]
    (contagens independentes por regra no dado bruto; ``linhas_removidas`` e a uniao, pois uma linha pode violar varias).
    Devolve date, ticker, open, high, low, close, volume e ``logret = ln(close_t / close_{t-1})`` dentro do ticker,
    ordenado por (ticker, date). As contagens ficam em ``sp500_report``.
    """
    import kagglehub

    folder = Path(kagglehub.dataset_download("camnugent/sandp500"))
    csv = next(folder.rglob("all_stocks_5yr.csv"))
    raw = pd.read_csv(
        csv,
        dtype={"Name": "string"},
        keep_default_na=False,
        na_values={c: [""] for c in ("open", "high", "low", "close", "volume")},
    )
    raw = raw.rename(columns={"Name": "ticker"})
    raw["date"] = pd.to_datetime(raw["date"])
    sp500_report.clear()
    sp500_report["linhas_brutas"] = len(raw)
    sp500_report["tickers_brutos"] = int(raw["ticker"].nunique())
    raw = raw.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)

    ohlc = ["open", "high", "low", "close"]
    rules = {
        "nan_ohlc": raw[ohlc].isna().any(axis=1),
        "volume_zero": raw["volume"].fillna(0) <= 0,
        "high_menor_que_low": raw["high"] < raw["low"],
        "close_fora_de_low_high": (raw["close"] > raw["high"]) | (raw["close"] < raw["low"]),
    }
    keep = pd.Series(True, index=raw.index)
    for name, bad in rules.items():  # contagem independente (cada regra olha o dado bruto); linhas podem violar varias
        sp500_report[name] = int(bad.sum())
        keep &= ~bad
    sp500_report["linhas_removidas"] = int((~keep).sum())
    df = raw.loc[keep].reset_index(drop=True)
    df["logret"] = np.log(df["close"]).groupby(df["ticker"]).diff()
    df = df[["date", "ticker", "open", "high", "low", "close", "volume", "logret"]]
    sp500_report["linhas_limpas"] = len(df)
    sp500_report["tickers_limpos"] = int(df["ticker"].nunique())
    sp500_report["pregoes"] = int(df["date"].nunique())
    return df


def add_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta ``pos`` (0 = primeira linha do ticker) e ``rem`` (linhas que ainda restam depois desta)."""
    g = df.groupby("ticker", sort=False)
    df["pos"] = g.cumcount().to_numpy()
    df["rem"] = g.cumcount(ascending=False).to_numpy()
    return df


def forward_vol(df: pd.DataFrame) -> pd.Series:
    """Alvo ``y_t = sqrt(252 * mean(r_{t+1..t+5}^2))`` (volatilidade anualizada, fração; 0.20 = 20% a.a.).

    Usa os 5 retornos **futuros** do mesmo ticker; fica NaN nas 5 ultimas linhas de cada ticker.
    Aplica o piso ``VOL_FLOOR`` (0.005), para o log nunca ver zero.
    """
    r2 = df["logret"] ** 2
    fwd = r2.groupby(df["ticker"], sort=False).transform(lambda s: s.rolling(HORIZON).mean().shift(-HORIZON))
    return np.sqrt(ANN * fwd).clip(lower=VOL_FLOOR).rename("y")


def sample_valid_mask(df: pd.DataFrame, need_target: bool = True) -> np.ndarray:
    """Amostra valida em ``t`` (regra V3.4): janela contigua [t-59, t+5] (ou [t-59, t] se ``need_target=False``).

    Invalida se algum pregao da janela tem salto > 5 dias corridos desde o anterior, ``|r| > 0.4`` ou r indefinido
    (primeira linha do ticker), ou se o ticker nao tem linhas suficientes antes (60) e depois (5) de ``t``.
    """
    if "pos" not in df:
        df = add_positions(df.copy())
    gap = df.groupby("ticker", sort=False)["date"].diff().dt.days
    bad = (gap > MAX_GAP_DAYS) | gap.isna() | (df["logret"].abs() > MAX_ABS_RET) | df["logret"].isna()
    n = len(df)
    fwd = HORIZON if need_target else 0
    c = np.concatenate([[0], np.cumsum(bad.to_numpy(dtype=np.int64))])
    i = np.arange(n)
    inside = (df["pos"].to_numpy() >= LOOKBACK - 1) & (df["rem"].to_numpy() >= fwd)
    hi = np.clip(i + fwd, 0, n - 1) + 1
    lo = np.clip(i - (LOOKBACK - 1), 0, n)
    window_bad = np.where(inside, c[hi] - c[lo], 1)
    return inside & (window_bad == 0)


@dataclass
class ChronoSplit:
    """Divisao cronologica por data (todos os tickers juntos), com embargo (purge) de ``purge`` pregoes."""

    days: np.ndarray  # pregões únicos, ordenados
    train: np.ndarray  # máscaras booleanas alinhadas ao vetor de datas recebido
    val: np.ndarray
    test: np.ndarray
    new: np.ndarray  # ultimos pregoes (sem alvo)
    bounds: dict[str, pd.Timestamp] = field(default_factory=dict)
    purge: int = 5

    def summary(self) -> str:
        b = self.bounds
        f = lambda k: b[k].strftime("%Y-%m-%d")  # noqa: E731
        return (
            f"treino    {f('train_start')} -> {f('train_end')}\n"
            f"embargo   {self.purge} pregoes\n"
            f"validacao {f('val_start')} -> {f('val_end')}\n"
            f"embargo   {self.purge} pregoes\n"
            f"teste     {f('test_start')} -> {f('test_end')}\n"
            f"novas instancias (sem alvo): {f('new_start')} -> {f('new_end')}"
        )


def chronological_split(dates, fractions=(0.70, 0.15, 0.15), purge: int = HORIZON, n_new: int = HORIZON) -> ChronoSplit:
    """Regra V3.6: 70/15/15 dos pregões (por indice de dia), embargo de ``purge`` pregoes em cada fronteira.

    Com N pregoes: treino = dias [0, b1), validacao = [b1+purge, b2), teste = [b2+purge, N-n_new), onde
    b1 = round(0.70 N) e b2 = round(0.85 N). Os ``n_new`` ultimos pregoes nao tem alvo e viram as novas instancias.
    O embargo existe porque o alvo olha 5 pregoes a frente: sem ele, o alvo da ultima amostra de treino cobriria dias
    da validacao.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    days = np.sort(dates.unique().to_numpy())
    n = len(days)
    b1 = round(fractions[0] * n)
    b2 = round((fractions[0] + fractions[1]) * n)
    d = pd.Series(np.searchsorted(days, dates.to_numpy()))  # indice do pregao de cada linha
    d = d.to_numpy()
    train = d < b1
    val = (d >= b1 + purge) & (d < b2)
    test = (d >= b2 + purge) & (d < n - n_new)
    new = d >= n - n_new
    ts = lambda k: pd.Timestamp(days[k])  # noqa: E731
    bounds = {
        "train_start": ts(0),
        "train_end": ts(b1 - 1),
        "val_start": ts(b1 + purge),
        "val_end": ts(b2 - 1),
        "test_start": ts(b2 + purge),
        "test_end": ts(n - n_new - 1),
        "new_start": ts(n - n_new),
        "new_end": ts(n - 1),
    }
    return ChronoSplit(days=days, train=train, val=val, test=test, new=new, bounds=bounds, purge=purge)


def new_instance_dates(dates, n: int = HORIZON) -> np.ndarray:
    """Os ``n`` ultimos pregoes do dataset: ainda sem alvo (faltam os 5 dias futuros), logo previsoes de verdade."""
    days = np.sort(pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).unique().to_numpy())
    return days[-n:]


def pick_new_tickers(candidates, k: int = 12, seed: int = 7) -> list[str]:
    """Sorteio (seed fixa) de ``k`` tickers entre os candidatos, em ordem alfabetica."""
    cand = sorted(set(map(str, candidates)))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(cand, size=min(k, len(cand)), replace=False).tolist())


def regime_edges(y_train) -> np.ndarray:
    """Tercis de ``y`` calculados SO no treino (cortes 1/3 e 2/3)."""
    return np.quantile(np.asarray(y_train, dtype="float64"), [1 / 3, 2 / 3])


def regime_labels(y, edges) -> np.ndarray:
    """0 = baixa, 1 = media, 2 = alta volatilidade, a partir dos cortes do treino."""
    return np.digitize(np.asarray(y, dtype="float64"), edges).astype("int64")


REGIME_NAMES = ("baixa", "media", "alta")

# %% [markdown]
# ## Carregamento e limpeza
#
# Ordenamos por `(ticker, date)` e removemos as linhas com NaN em open/high/low/close, volume 0, `high < low` ou `close` fora de `[low, high]` (as contagens são independentes: uma linha pode quebrar mais de uma regra, e as 4 linhas de volume 0 também têm OHLC em NaN). O retorno é `r_t = ln(close_t / close_{t-1})`, calculado dentro de cada ticker.

# %%
_t0 = time.time()
sp = load_sp500()
sp500_load_seconds = time.time() - _t0
print(f"S&P 500 carregado em {sp500_load_seconds:.1f}s")
for _k, _v in sp500_report.items():
    print(f"  {_k:<24s}{_v:>10,d}")
sp.head()

# %% [markdown]
# ## Alvo, janela válida e divisão por data
#
# - **Alvo:** `y_t = sqrt(252 · média(r²_{t+1..t+5}))`, a volatilidade anualizada dos próximos 5 pregões (em fração: 0,20 = 20% ao ano).
# - **Amostra válida em `t`:** exige a janela `[t-59, t+5]` contínua (sem saltos de mais de 5 dias corridos entre pregões) e nenhum |r| > 0,4 nela. Retornos desse tamanho quase sempre são splits ou spin-offs que a fonte não ajustou (BAX, DISCA/DISCK, EBAY, LNT, NI). A regra também descarta alguns eventos reais (AMD, CHK, WMB), e deixamos isso registrado.
# - **Divisão por data:** 70% / 15% / 15% dos pregões, com todos os tickers juntos e um intervalo de 5 pregões entre as partes (o alvo olha 5 dias para frente, então sem esse intervalo haveria vazamento). Nenhuma amostra é embaralhada entre períodos.
# - **Novas instâncias:** os 5 últimos pregões (que ainda não têm alvo) ficam fora do teste, que termina 5 pregões antes do fim.

# %%
sp = add_positions(sp)
sp["y"] = forward_vol(sp)
sp["valid"] = sample_valid_mask(sp)  # janela [t-59, t+5]
sp["valid_feat"] = sample_valid_mask(sp, need_target=False)  # janela [t-59, t] (novas instâncias)

_gap = sp.groupby("ticker", sort=False)["date"].diff().dt.days
split = chronological_split(sp["date"])
sp["part"] = np.select([split.train, split.val, split.test, split.new], ["train", "val", "test", "new"], default="purge")

_big = sp["logret"].abs() > MAX_ABS_RET
print(f"retornos |log| > {MAX_ABS_RET}: {int(_big.sum())} | saltos > {MAX_GAP_DAYS} dias corridos: {int((_gap > MAX_GAP_DAYS).sum())}")
print(f"amostras com alvo definido: {int(sp['y'].notna().sum()):,d} | válidas (janela contígua e sem artefato): {int(sp['valid'].sum()):,d}")
print(split.summary())
print(sp[sp["valid"]].groupby("part").size().to_string())

# %%
# ``sp`` (DataFrame), ``split`` (ChronoSplit) e ``sp500_report`` ficam disponíveis para as redes 2 e 3.
assert sp["valid"].sum() > 300_000
assert (sp.loc[sp["valid"], "y"] > 0).all()
print("OK: parte 15 pronta.")
