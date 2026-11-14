from __future__ import annotations

import io

import make_dummy_artifacts as dummy
import numpy as np
import pytest

from trinet_api.infrastructure.runtimes import RuntimeLoadError
from trinet_api.infrastructure.runtimes.numpy_mlp import NumpyMlpRuntime
from trinet_contract import DenormalizeLayer, DenseLayer, NormalizeLayer, TensorSpec

LAYERS = (
    NormalizeLayer(key="n"),
    DenseLayer(key="d", activation="relu"),
    DenseLayer(key="o", activation="softmax"),
    DenormalizeLayer(key="t"),
)
IN = TensorSpec(name="x", shape=(None, 3))
OUT = TensorSpec(name="y", shape=(None, 2))


def f32(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def good_arrays() -> dict[str, np.ndarray]:
    return {
        "n.mean": f32(1, 2, 3),
        "n.variance": f32(1, 4, 9),
        "d.kernel": np.arange(12, dtype=np.float32).reshape(3, 4) / 10,
        "d.bias": f32(0.1, 0, -0.1, 0),
        "o.kernel": np.arange(8, dtype=np.float32).reshape(4, 2) / 10,
        "o.bias": f32(0, 0.5),
        "t.mean": f32(10, 20),
        "t.variance": f32(1, 4),
    }


def build(
    arrays: dict[str, np.ndarray], layers: tuple = LAYERS, out: TensorSpec = OUT
) -> NumpyMlpRuntime:  # type: ignore[type-arg]
    return NumpyMlpRuntime(dummy.npz_bytes(arrays), layers, IN, out)


def test_matches_a_hand_written_reference() -> None:
    arrays = good_arrays()
    x = np.array([[1, 2, 3], [4, 0, -1]], dtype=np.float32)
    expected = dummy.mlp_reference(x, LAYERS, arrays)
    got = build(arrays).predict(x)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_statistics_may_be_stored_as_row_vectors_like_keras() -> None:
    arrays = good_arrays()
    arrays["n.mean"] = arrays["n.mean"].reshape(1, 3)
    arrays["n.variance"] = arrays["n.variance"].reshape(1, 3)
    assert build(arrays).predict(np.ones((1, 3), np.float32)).shape == (1, 2)


def test_zero_variance_uses_the_keras_epsilon_instead_of_dividing_by_zero() -> None:
    arrays = good_arrays()
    arrays["n.variance"] = f32(0, 4, 9)
    assert np.all(np.isfinite(build(arrays).predict(f32(5, 5, 5).reshape(1, 3))))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.pop("d.bias"), "no array"),
        (lambda a: a.update({"d.kernel": np.ones((5, 4), np.float32)}), "chain"),
        (lambda a: a.update({"d.bias": f32(1, 2)}), "chain"),
        (lambda a: a.update({"n.mean": f32(1, 2)}), "statistics"),
        (lambda a: a.update({"d.kernel": np.ones((3, 4), np.int64)}), "floating"),
        (lambda a: a.update({"d.kernel": np.full((3, 4), np.nan, np.float32)}), "non-finite"),
        (lambda a: a.update({"d.kernel": np.ones((3, 4, 1), np.float32)}), "2-D"),
        (
            lambda a: a.update(
                {"o.kernel": np.ones((4, 5), np.float32), "o.bias": np.ones(5, np.float32)}
            ),
            r"statistics|chain|outputs",
        ),
    ],
)
def test_inconsistent_weights_are_rejected(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    arrays = good_arrays()
    mutate(arrays)
    with pytest.raises(RuntimeLoadError, match=message):
        build(arrays)


def test_output_width_must_match_the_manifest() -> None:
    with pytest.raises(RuntimeLoadError, match="outputs 2 values"):
        build(good_arrays(), out=TensorSpec(name="y", shape=(None, 3)))


def test_pickled_arrays_are_never_loaded() -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **{"n.mean": np.array([object()], dtype=object)})  # savez pickles objects
    with pytest.raises(RuntimeLoadError, match=r"unreadable|not a valid"):
        NumpyMlpRuntime(buffer.getvalue(), LAYERS, IN, OUT)


@pytest.mark.parametrize("blob", [b"", b"not an archive", b"\x93NUMPY\x01\x00"])
def test_garbage_is_not_an_npz(blob: bytes) -> None:
    with pytest.raises(RuntimeLoadError):
        NumpyMlpRuntime(blob, LAYERS, IN, OUT)


def test_a_bare_npy_file_is_not_an_archive() -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.ones(3, np.float32))
    with pytest.raises(RuntimeLoadError, match="npz"):
        NumpyMlpRuntime(buffer.getvalue(), LAYERS, IN, OUT)


def test_only_flat_feature_vectors_are_supported() -> None:
    with pytest.raises(RuntimeLoadError, match="flat"):
        NumpyMlpRuntime(
            dummy.npz_bytes(good_arrays()),
            LAYERS,
            TensorSpec(name="x", shape=(None, 3, 3)),
            OUT,
        )
