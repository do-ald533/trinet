"""Apendice de exportacao do notebook ``T2_TriNet.ipynb`` (fora da area avaliada).

Helpers genericos que transformam as redes treinadas em artefatos consumidos pela API de inferencia:
``artifacts/<modelo>/<versao>/{manifest.json, model.onnx, weights.npz, golden.json, golden/*.png, results.json}``.

- MLPs tabulares (``credit-default-mlp``, ``volatility-mlp``): ``rebuild_inference``, ``export_onnx`` (ou, se o
  ``tf2onnx`` falhar, ``onnx_from_weights``), ``extract_mlp``/``numpy_mlp_forward`` (runtime NumPy do contrato, com
  ``exp`` para o alvo em log), ``records_to_features``, ``tabular_golden``.
- Comuns a todos: ``check_gate``, ``ArtifactBundle``, ``write_artifact`` (escrita atomica + validacao com ``trinet-contract``).
- A CNN (``candlestick-cnn``) usa ``trinet_export_image`` por cima destes helpers (``pillow_preprocess``, ``ort_predict`` ...).

Este modulo so e importado pelo apendice. O notebook continua funcionando (e avaliavel) sem ele.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ONNX_OPSET = 17
_NORM_EPS = 1e-7  # epsilon do Normalization do Keras: (x - mean) / max(sqrt(var), eps)


# --------------------------------------------------------------------------- utilidades


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_default(obj: Any) -> Any:
    """Serializa tipos NumPy/pandas que aparecem nas tabelas de resultados."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"nao serializavel: {type(obj)!r}")


def dumps(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, default=json_default, allow_nan=False) + "\n").encode()


# --------------------------------------------------------------------------- clone de inferencia


def rebuild_inference(model: Any, input_name: str) -> Any:
    """Clone float32 de inferencia, com pesos copiados, pronto para ``model.export``.

    - remove ``Dropout`` e as camadas de aumento de dados (``Random*``);
    - troca cada ``Normalization`` por ``Rescaling`` com constantes (``(x-mean)/std``; invertida: ``x*std+mean``).
      Motivo: o ``Normalization`` do Keras 3 guarda media/variancia como tensores soltos, e o ``tf2onnx`` os
      exporta como *entradas* extras do grafo; com ``Rescaling`` elas viram constantes (initializers).
    """
    import tensorflow as tf

    layers = tf.keras.layers
    dropped = (
        layers.Dropout,
        layers.RandomFlip,
        layers.RandomRotation,
        layers.RandomZoom,
        layers.RandomTranslation,
        layers.RandomContrast,
        layers.RandomBrightness,
        layers.InputLayer,
    )
    x = inp = layers.Input(shape=tuple(model.input_shape[1:]), name=input_name, dtype="float32")
    for layer in model.layers:
        if isinstance(layer, dropped):
            continue
        if isinstance(layer, layers.Normalization):
            mean = np.asarray(layer.mean, dtype="float64").reshape(-1)
            std = np.maximum(np.sqrt(np.asarray(layer.variance, dtype="float64").reshape(-1)), _NORM_EPS)
            scale, offset = (std, mean) if layer.invert else (1.0 / std, -mean / std)
            new = layers.Rescaling(
                scale=scale.astype("float32").tolist(), offset=offset.astype("float32").tolist(), name=layer.name
            )
            x = new(x)
            continue
        new = layer.__class__.from_config(layer.get_config())
        x = new(x)
        if layer.weights:
            new.set_weights(layer.get_weights())
    return tf.keras.Model(inp, x, name=f"{model.name}-inference")


def export_onnx(infer_model: Any, path: Path, input_name: str, opset: int = ONNX_OPSET) -> None:
    """``Model.export(format="onnx")`` (tf2onnx por baixo) com lote dinamico e opset fixo."""
    import tensorflow as tf

    spec = tf.TensorSpec((None, *infer_model.input_shape[1:]), tf.float32, name=input_name)
    infer_model.export(str(path), format="onnx", opset_version=opset, verbose=False, input_signature=[spec])


def inspect_onnx(path: Path) -> dict[str, Any]:
    """Le o ONNX com ``onnx`` e ``onnxruntime`` e devolve nomes/formas reais dos tensores e o opset."""
    import onnx
    import onnxruntime as ort

    proto = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(proto)
    external = any(t.data_location == onnx.TensorProto.EXTERNAL for t in proto.graph.initializer)
    if external:
        raise RuntimeError("o ONNX nao pode usar dados externos")
    opset = next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def dims(shape: Sequence[Any]) -> list[int | None]:
        return [d if isinstance(d, int) else None for d in shape]

    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        raise RuntimeError(f"esperava 1 entrada e 1 saida, obtive {len(ins)} e {len(outs)}")
    return {
        "opset": int(opset),
        "ir_version": int(proto.ir_version),
        "input": {"name": ins[0].name, "shape": dims(ins[0].shape), "type": ins[0].type},
        "output": {"name": outs[0].name, "shape": dims(outs[0].shape), "type": outs[0].type},
        "ops": sorted({n.op_type for n in proto.graph.node}),
        "bytes": path.stat().st_size,
    }


def ort_predict(path: Path | bytes, x: np.ndarray, batch: int = 256) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(path if isinstance(path, bytes) else str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = [sess.run(None, {name: np.ascontiguousarray(x[i : i + batch], dtype=np.float32)})[0] for i in range(0, len(x), batch)]
    return np.concatenate(out)


def onnx_from_weights(
    layers: Sequence[dict[str, str]], arrays: dict[str, np.ndarray], n_in: int, n_out: int, input_name: str = "features", opset: int = ONNX_OPSET
) -> bytes:
    """Plano B documentado: monta o ONNX com ``onnx.helper`` a partir dos pesos extraidos (sem ``tf2onnx``).

    Cobre exatamente as camadas do contrato ``numpy_mlp`` (normalize, dense relu/linear/softmax, denormalize, exp).
    Entradas/saidas: ``input_name`` e ``output_0``, lote dinamico.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes: list[Any] = []
    inits: list[Any] = []
    current = input_name

    def const(name: str, value: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(np.asarray(value, dtype="float32"), name=name))
        return name

    for i, spec in enumerate(layers):
        out = f"t{i}"
        op = spec["op"]
        if op == "exp":
            nodes.append(helper.make_node("Exp", [current], [out]))
        elif op in ("normalize", "denormalize"):
            key = spec["key"]
            mean = const(f"{key}.mean", arrays[f"{key}.mean"])
            std = const(f"{key}.std", np.maximum(np.sqrt(arrays[f"{key}.variance"].astype("float64")), _NORM_EPS))
            first, second = ("Sub", "Div") if op == "normalize" else ("Mul", "Add")
            nodes.append(helper.make_node(first, [current, mean if op == "normalize" else std], [f"{out}a"]))
            nodes.append(helper.make_node(second, [f"{out}a", std if op == "normalize" else mean], [out]))
        else:
            key = spec["key"]
            kernel, bias = const(f"{key}.kernel", arrays[f"{key}.kernel"]), const(f"{key}.bias", arrays[f"{key}.bias"])
            nodes.append(helper.make_node("MatMul", [current, kernel], [f"{out}a"]))
            nodes.append(helper.make_node("Add", [f"{out}a", bias], [f"{out}b"]))
            act = {"relu": "Relu", "softmax": "Softmax", "linear": "Identity"}[spec["activation"]]
            extra = {"axis": -1} if act == "Softmax" else {}
            nodes.append(helper.make_node(act, [f"{out}b"], [out], **extra))
        current = out
    nodes.append(helper.make_node("Identity", [current], ["output_0"]))
    graph = helper.make_graph(
        nodes,
        "trinet_mlp",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, ["batch", n_in])],
        [helper.make_tensor_value_info("output_0", TensorProto.FLOAT, ["batch", n_out])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="trinet-notebook", ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    return bytes(model.SerializeToString(deterministic=True))


# --------------------------------------------------------------------------- MLPs: pesos para o runtime NumPy


def extract_mlp(model: Any) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    """Pesos (``<key>.mean/.variance/.kernel/.bias``) e a lista de camadas do contrato ``numpy_mlp``.

    ``Activation("exponential")`` vira ``{"op": "exp"}`` (alvo em log de volta a unidades naturais).
    """
    import tensorflow as tf

    arrays: dict[str, np.ndarray] = {}
    layers: list[dict[str, str]] = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            arrays[f"{layer.name}.mean"] = np.asarray(layer.mean, dtype="float32").reshape(-1)
            arrays[f"{layer.name}.variance"] = np.asarray(layer.variance, dtype="float32").reshape(-1)
            layers.append({"op": "denormalize" if layer.invert else "normalize", "key": layer.name})
        elif isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            arrays[f"{layer.name}.kernel"] = kernel.astype("float32")
            arrays[f"{layer.name}.bias"] = bias.astype("float32")
            layers.append({"op": "dense", "key": layer.name, "activation": layer.activation.__name__})
        elif isinstance(layer, tf.keras.layers.Activation) and layer.activation.__name__ == "exponential":
            layers.append({"op": "exp"})
        elif isinstance(layer, tf.keras.layers.Dropout | tf.keras.layers.InputLayer):
            continue
        else:
            raise TypeError(f"camada nao suportada pelo runtime numpy_mlp: {type(layer).__name__}")
    return arrays, layers


def numpy_mlp_forward(arrays: dict[str, np.ndarray], layers: Sequence[dict[str, str]], x: np.ndarray) -> np.ndarray:
    """Implementacao de referencia do contrato ``numpy_mlp`` (float64), usada para verificar o ``weights.npz``."""
    h = np.asarray(x, dtype="float64")
    for spec in layers:
        op = spec["op"]
        if op == "exp":
            h = np.exp(h)
            continue
        key = spec["key"]
        if op in ("normalize", "denormalize"):
            std = np.maximum(np.sqrt(arrays[f"{key}.variance"].astype("float64")), _NORM_EPS)
            mean = arrays[f"{key}.mean"].astype("float64")
            h = (h - mean) / std if op == "normalize" else h * std + mean
        else:
            h = h @ arrays[f"{key}.kernel"].astype("float64") + arrays[f"{key}.bias"].astype("float64")
            if spec["activation"] == "relu":
                h = np.maximum(h, 0.0)
            elif spec["activation"] == "softmax":
                e = np.exp(h - h.max(axis=1, keepdims=True))
                h = e / e.sum(axis=1, keepdims=True)
    return h


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


# --------------------------------------------------------------------------- tabular: pre-processamento do manifesto


def records_to_features(spec: Any, records: Sequence[dict[str, Any]]) -> np.ndarray:
    """Reimplementa as 3 primitivas do contrato (imputar constante, one-hot, razao A/B).

    Mesma semantica da API: a matematica e feita em float64 e so o resultado final vira float32.
    """
    from trinet_contract.manifest import CategoricalField, OneHotFeature, PassthroughFeature, RatioFeature

    rows = []
    for record in records:
        values: dict[str, Any] = {}
        for field in spec.fields:
            raw = record.get(field.name)
            if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                if field.impute is None:
                    raise ValueError(f"campo {field.name!r} ausente e sem constante de imputacao")
                raw = field.impute
            values[field.name] = raw if isinstance(field, CategoricalField) else float(raw)
        row: list[float] = []
        for feature in spec.features:
            if isinstance(feature, PassthroughFeature):
                row.append(float(values[feature.source]))
            elif isinstance(feature, OneHotFeature):
                row.extend(1.0 if values[feature.source] == v else 0.0 for v in spec.field(feature.source).vocabulary)
            elif isinstance(feature, RatioFeature):
                row.append(float(values[feature.numerator]) / float(values[feature.denominator]))
        rows.append(row)
    return np.asarray(rows, dtype="float32")


def clean_record(record: dict[str, Any]) -> dict[str, float | str | None]:
    """Registro cru serializavel: NaN vira ``None`` (null), numeros viram ``float`` do Python."""
    out: dict[str, float | str | None] = {}
    for key, value in record.items():
        if value is None or (isinstance(value, float | np.floating) and np.isnan(value)):
            out[key] = None
        elif isinstance(value, str):
            out[key] = value
        else:
            out[key] = float(value)
    return out


def tabular_golden(
    *,
    model: str,
    version: str,
    records: Sequence[dict[str, Any]],
    features: np.ndarray,
    outputs: np.ndarray,
    labels: Sequence[str | float | None],
    output_atol: float | None = None,
) -> dict[str, Any]:
    cases = [
        {
            "record": clean_record(rec),
            "features": [float(v) for v in feat],
            "output": [float(v) for v in out],
            "label": lab if lab is None or isinstance(lab, str) else float(lab),
        }
        for rec, feat, out, lab in zip(records, features, outputs, labels, strict=True)
    ]
    payload: dict[str, Any] = {"schema_version": "1.0", "kind": "tabular", "model": model, "version": version, "cases": cases}
    if output_atol is not None:
        payload["tolerance"] = {"features_atol": 1e-6, "output_atol": output_atol, "top1_margin": 1e-3}
    return payload


# --------------------------------------------------------------------------- imagens: caminho de serving (Pillow)


def pillow_preprocess(source: bytes | Path, height: int, width: int) -> np.ndarray:
    """Caminho de producao da API: decodifica -> RGB -> resize BILINEAR (se preciso) -> float32 em 0-255."""
    from PIL import Image

    handle = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(handle) as im:
        im = im.convert("RGB")
        if im.size != (width, height):
            im = im.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32)


# --------------------------------------------------------------------------- manifesto, gate e escrita atomica


def check_gate(metric: str, split: str, value: float, threshold: float, higher_is_better: bool) -> dict[str, Any]:
    """Quality gate de exportacao: recusa escrever o artefato se a metrica ficar fora do limiar."""
    passed = value >= threshold if higher_is_better else value <= threshold
    gate = {
        "metric": metric,
        "split": split,
        "threshold": float(threshold),
        "value": float(value),
        "higher_is_better": higher_is_better,
    }
    if not passed:
        raise RuntimeError(f"QUALITY GATE REPROVADO: {metric} ({split}) = {value:.6g} vs limiar {threshold:.6g}; artefato NAO escrito")
    return gate


@dataclass
class ArtifactBundle:
    name: str
    version: str
    onnx_bytes: bytes
    manifest_fields: dict[str, Any]
    golden: dict[str, Any]
    results: dict[str, Any]
    npz: bytes | None = None
    numpy_layers: list[dict[str, str]] | None = None
    golden_pngs: dict[str, bytes] | None = None


def write_artifact(root: Path, bundle: ArtifactBundle, extra_validate: Callable[[Path], None] | None = None) -> Path:
    """Escreve ``root/<name>/<version>`` de forma atomica e valida tudo com ``trinet-contract``."""
    from trinet_contract.golden import load_golden
    from trinet_contract.manifest import Manifest, load_manifest, sha256_file

    target = root / bundle.name / bundle.version
    staging = root / bundle.name / f".staging-{bundle.version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        (staging / "model.onnx").write_bytes(bundle.onnx_bytes)
        runtimes: list[dict[str, Any]] = [
            {"kind": "onnx", "file": "model.onnx", "sha256": sha256_bytes(bundle.onnx_bytes), "opset": bundle.manifest_fields["onnx_opset"]}
        ]
        if bundle.npz is not None:
            (staging / "weights.npz").write_bytes(bundle.npz)
            runtimes.append({"kind": "numpy_mlp", "file": "weights.npz", "sha256": sha256_bytes(bundle.npz), "layers": bundle.numpy_layers})
        if bundle.golden_pngs:
            (staging / "golden").mkdir()
            for fname, data in bundle.golden_pngs.items():
                (staging / "golden" / fname).write_bytes(data)
        golden_bytes = dumps(bundle.golden)
        (staging / "golden.json").write_bytes(golden_bytes)
        (staging / "results.json").write_bytes(dumps(bundle.results))

        fields = {k: v for k, v in bundle.manifest_fields.items() if k != "onnx_opset"}
        fields["runtimes"] = runtimes
        fields["golden"] = {"file": "golden.json", "sha256": sha256_bytes(golden_bytes)}
        manifest = Manifest.model_validate(fields)
        (staging / "manifest.json").write_bytes(dumps(manifest.model_dump(mode="json")))

        # releitura e validacao com o contrato
        reread = load_manifest(staging / "manifest.json")
        golden = load_golden(staging / "golden.json")
        for runtime in reread.runtimes:
            if sha256_file(staging / runtime.file) != runtime.sha256:
                raise RuntimeError(f"sha256 inconsistente para {runtime.file}")
        if reread.golden is None or sha256_file(staging / reread.golden.file) != reread.golden.sha256:
            raise RuntimeError("sha256 inconsistente para golden.json")
        if golden.model != reread.name or golden.version != reread.version:
            raise RuntimeError("golden.json nao corresponde ao manifesto")
        if extra_validate is not None:
            extra_validate(staging)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def round_up(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step
