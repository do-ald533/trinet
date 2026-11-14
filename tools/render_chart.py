"""Renderiza uma janela de 20 pregoes OHLCV como imagem 64x64x3 (NumPy puro, deterministico).

Esta e a referencia da funcao de rasterizacao usada pelo notebook (secao "Rede 3: CNN em graficos de
preco", onde o mesmo codigo aparece inline) e pela ferramenta de linha de comando. Um teste do
apendice do notebook garante que as duas copias produzem arrays identicos bit a bit.

Sem matplotlib, sem antialiasing: cada pixel e 0 ou 255. Tres canais independentes:

* canal 0 (R): barras OHLC (traco vertical high-low no pixel central do dia, tick de abertura a
  esquerda, tick de fechamento a direita; 3 px por dia, 2 px de margem em cada lado);
* canal 1 (G): media movel de 20 dias do fechamento (linha continua);
* canal 2 (B): volume (barras de 1 px de largura nas ``CHART_VOL_ROWS`` linhas de baixo,
  normalizadas pelo maior volume da janela).

Os precos entram em **espaco logaritmico**; o eixo vertical e fixo em ``CHART_SPAN`` (unidades de
log-preco para as 64 linhas), centrado no ponto medio da faixa [min, max] da janela
(``scale="fixed"``), ou ajustado ao min/max de cada janela (``scale="window"``, como em Jiang,
Kelly e Xiu, 2023).

Uso:
    python tools/render_chart.py acao.csv -o grafico.png
        (CSV com cabecalho ``date,open,high,low,close,volume``; usa as ultimas linhas)
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
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


# --------------------------------------------------------------------------- CLI

_COLUMNS = ("open", "high", "low", "close", "volume")


def read_ohlcv(path: Path) -> _F64:
    """CSV de um ticker (``date,open,...,volume``) -> ``(linhas, 5)`` em ordem de data."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV vazio")
        names = {name.strip().lower(): name for name in reader.fieldnames}
        missing = [c for c in ("date", *_COLUMNS) if c not in names]
        if missing:
            raise ValueError(f"colunas ausentes no CSV: {missing}")
        rows = sorted(reader, key=lambda r: r[names["date"]])
    return np.asarray([[float(r[names[c]]) for c in _COLUMNS] for r in rows], dtype=np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Renderiza as ultimas linhas de um CSV OHLCV como PNG 64x64."
    )
    parser.add_argument(
        "csv", type=Path, help="CSV com date,open,high,low,close,volume (um ticker)"
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("chart.png"))
    parser.add_argument("--scale", choices=("fixed", "window"), default=CHART_SCALE)
    parser.add_argument(
        "--span", type=float, default=CHART_SPAN, help="log-preco nas 64 linhas (scale=fixed)"
    )
    args = parser.parse_args(argv)
    try:
        data = read_ohlcv(args.csv)
        if len(data) < CHART_WINDOW:
            raise ValueError(f"preciso de pelo menos {CHART_WINDOW} linhas; o CSV tem {len(data)}")
        image = render_windows(data[-CHART_CONTEXT:][None], scale=args.scale, span=args.span)[0]
    except (OSError, ValueError, KeyError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    args.output.write_bytes(png_bytes(image))
    print(
        f"{args.output} ({image.shape[1]}x{image.shape[0]}, {int((image > 0).sum())} pixels acesos)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
