"""Auxiliares do apendice de exportacao da CNN ``candlestick-cnn`` (fora da area avaliada).

Complementa ``trinet_export`` (helpers genericos, dono: parte tabular) com o que e especifico da rede de imagens:
clone de inferencia sem augmentation, graficos sinteticos para os golden samples, medicao do *serving skew*
(caminho Keras x caminho de producao da API) e a comparacao entre a rasterizacao do notebook e ``tools/render_chart.py``.

So e importado pelo apendice (``96_export_cnn.py``); o notebook continua funcionando sem ele.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from trinet_export import pillow_preprocess

IMAGE_MODEL_NAME = "candlestick-cnn"
INPUT_NAME = "chart"
SIZE = 64


# --------------------------------------------------------------------------- ambiente


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_string() -> str:
    """Modelo da CPU (so o nome do processador, nada identificavel do usuario)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return f"{line.split(':', 1)[1].strip()} ({platform.machine()}, CPU)"
    except OSError:
        pass
    return f"{platform.processor() or platform.machine()} (CPU)"


def git_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)  # noqa: S607 - git do PATH, so para registrar o commit
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if len(sha) >= 7 else None


# --------------------------------------------------------------------------- rasterizacao: notebook x tools/render_chart.py


def load_tool_renderer(repo_root: Path) -> ModuleType:
    """Importa ``tools/render_chart.py`` pelo caminho (``tools`` nao e um pacote)."""
    path = repo_root / "tools" / "render_chart.py"
    spec = importlib.util.spec_from_file_location("trinet_tools_render_chart", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"nao foi possivel importar {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_same_renderer(nb: dict[str, Callable[..., Any]], tool: ModuleType, windows: np.ndarray) -> dict[str, Any]:
    """A copia inline do notebook e ``tools/render_chart.py`` devem dar arrays e PNGs identicos bit a bit."""
    checks: dict[str, bool] = {}
    for scale, span in (("fixed", 0.30), ("fixed", 0.18), ("window", 0.30)):
        checks[f"render_windows[{scale},{span}]"] = bool(np.array_equal(nb["render_windows"](windows, scale=scale, span=span), tool.render_windows(windows, scale=scale, span=span)))
    checks["render_chart"] = bool(np.array_equal(nb["render_chart"](windows[0]), tool.render_chart(windows[0])))
    checks["lote == unitario"] = bool(np.array_equal(nb["render_windows"](windows)[1], nb["render_chart"](windows[1])))
    image = tool.render_chart(windows[0])
    checks["png_bytes"] = nb["png_bytes"](image) == tool.png_bytes(image)
    checks["png decodificado == array"] = bool(np.array_equal(pillow_preprocess(tool.png_bytes(image), SIZE, SIZE), image.astype(np.float32)))
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise AssertionError(f"rasterizacao do notebook difere de tools/render_chart.py: {bad}")
    return {"identical": True, "checks": checks}


def check_cli(tool: ModuleType, window: np.ndarray) -> bool:
    """Roda a CLI da ferramenta sobre um CSV sintetico e confere que o PNG e o ultimo grafico da janela."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path, png_path = Path(tmp) / "SYN.csv", Path(tmp) / "SYN.png"
        lines = ["date,open,high,low,close,volume"]
        lines += [f"2020-01-{i + 1:02d},{o!r},{h!r},{lo!r},{c!r},{v!r}" for i, (o, h, lo, c, v) in enumerate(window.tolist())]
        csv_path.write_text("\n".join(lines) + "\n")
        if tool.main([str(csv_path), "-o", str(png_path)]) != 0:
            return False
        return bool(np.array_equal(pillow_preprocess(png_path.read_bytes(), SIZE, SIZE), tool.render_chart(window).astype(np.float32)))


# --------------------------------------------------------------------------- series sinteticas (golden samples)


def synthetic_ohlcv(
    count: int, seed: int = 2026, rows: int = 39, edges: Sequence[float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Janelas OHLCV de passeio aleatorio geometrico (``(count, rows, 5)``) e a volatilidade anual usada em cada uma.

    Nada vem do CSV do Kaggle: so serve para exercitar a API e o *serving skew* sem redistribuir dados de terceiros.
    Com ``edges`` (cortes dos tercis, ex.: 14,4% e 22,9% a.a.) as janelas se alternam entre as 3 classes, com a
    volatilidade sorteada longe dos cortes; sem ``edges``, a volatilidade varia de 8% a 90% a.a. (log-uniforme).
    """
    rng = np.random.default_rng(seed)
    if edges is None:
        sigma = np.exp(rng.uniform(np.log(0.08), np.log(0.9), count))
    else:
        low, high = float(edges[0]), float(edges[1])
        brackets = np.array([[0.07, 0.9 * low], [1.1 * low, 0.9 * high], [1.1 * high, 0.70]])
        regime = rng.permutation(np.arange(count) % 3)
        sigma = rng.uniform(brackets[regime, 0], brackets[regime, 1])
    daily = (sigma / np.sqrt(252))[:, None]
    ret = rng.standard_normal((count, rows)) * daily
    close = 100.0 * np.exp(np.cumsum(ret, axis=1))
    prev = np.concatenate([np.full((count, 1), 100.0), close[:, :-1]], axis=1)
    open_ = prev * np.exp(rng.standard_normal((count, rows)) * daily * 0.3)
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal((count, rows))) * daily * 0.5)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal((count, rows))) * daily * 0.5)
    volume = np.exp(rng.normal(15.0, 0.4, (count, rows))) * (1.0 + 0.5 * np.abs(ret) / daily)
    return np.stack([open_, high, low, close, volume], axis=2), sigma


# --------------------------------------------------------------------------- clone de inferencia


def rebuild_inference_cnn(model: Any, input_name: str = INPUT_NAME) -> Any:
    """Clone float32 de inferencia da CNN: sem ``Dropout`` e sem camadas de augmentation, com os pesos copiados."""
    import tensorflow as tf

    layers = tf.keras.layers
    skip = (layers.Dropout, layers.InputLayer, layers.RandomFlip, layers.RandomRotation, layers.RandomZoom, layers.RandomTranslation)
    x = inp = layers.Input(shape=tuple(model.input_shape[1:]), name=input_name, dtype="float32")
    for layer in model.layers:
        if isinstance(layer, skip) or getattr(layer, "trinet_augmentation", False):
            continue
        new = layer.__class__.from_config(layer.get_config())
        x = new(x)
        if layer.weights:
            new.set_weights(layer.get_weights())
    return tf.keras.Model(inp, x, name=f"{model.name}-inference")


# --------------------------------------------------------------------------- caminho de serving


def encode_pngs(images: np.ndarray, png_bytes: Callable[[np.ndarray], bytes]) -> list[bytes]:
    return [png_bytes(img) for img in images]


def serving_batch(pngs: Sequence[bytes], height: int = SIZE, width: int = SIZE) -> np.ndarray:
    """Pillow -> RGB -> (resize BILINEAR se preciso) -> float32 0-255: o preprocessamento da API, imagem a imagem."""
    return np.stack([pillow_preprocess(data, height, width) for data in pngs]).astype(np.float32)


def serving_path_probs(
    onnx_bytes: bytes,
    images: np.ndarray,
    png_bytes: Callable[[np.ndarray], bytes],
    chunk: int = 2000,
) -> np.ndarray:
    """Probabilidades pelo caminho de producao: PNG -> Pillow (RGB) -> float32 0-255 -> ONNX Runtime."""
    from trinet_export import ort_predict

    parts = []
    for start in range(0, len(images), chunk):
        batch = serving_batch(encode_pngs(images[start : start + chunk], png_bytes))
        parts.append(ort_predict(onnx_bytes, batch))
    return np.concatenate(parts)


def skew_report(keras_probs: np.ndarray, served_probs: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float | int]:
    """Concordancia entre o caminho de treino (Keras) e o de servico (Pillow + runtime exportado)."""
    k, s = keras_probs.argmax(axis=1), served_probs.argmax(axis=1)
    out: dict[str, float | int] = {
        "samples": len(k),
        "max_abs_diff": float(np.abs(keras_probs - served_probs).max()),
        "top1_agreement": float((k == s).mean()),
    }
    if labels is not None:
        out["accuracy_delta_pp"] = float(abs((k == labels).mean() - (s == labels).mean()) * 100.0)
    return out


def encode_resized_png(image: np.ndarray, size: tuple[int, int]) -> bytes:
    """PNG do grafico em outro tamanho (largura, altura), para exercitar o resize BILINEAR da API."""
    from PIL import Image

    resample = Image.Resampling.NEAREST if size[0] >= image.shape[1] else Image.Resampling.BOX
    buf = io.BytesIO()
    Image.fromarray(image, "RGB").resize(size, resample).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
