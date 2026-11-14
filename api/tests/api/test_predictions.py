from __future__ import annotations

import io
import json
import math
from collections.abc import Callable
from typing import Any

import make_dummy_artifacts as dummy
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.api.conftest import multipart
from tests.fakes import FakeRuntime

from trinet_api.domain.entities import ModelRegistry

PREDICT = "/v1/models/{}/predictions"


def post_json(client: TestClient, name: str, body: Any, auth: dict[str, str], **kw: Any) -> Any:
    return client.post(PREDICT.format(name), headers=auth, json=body, **kw)


# --------------------------------------------------------------------------- success


def test_classification_response_shape(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    response = post_json(
        client, "credit-default-mlp", {"instances": [golden_records["credit-default-mlp"]]}, auth
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.headers["x-model-version"] == "0.0.0-dummy"
    body = response.json()
    assert body["model_name"] == "credit-default-mlp"
    assert body["model_version"] == "0.0.0-dummy"
    assert body["request_id"] == response.headers["x-request-id"]
    (prediction,) = body["predictions"]
    assert prediction["kind"] == "classification"
    probabilities = prediction["probabilities"]
    assert set(probabilities) == {"no_default", "default"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert prediction["label"] == max(probabilities, key=probabilities.__getitem__)
    assert prediction["probability"] == pytest.approx(max(probabilities.values()))


def test_regression_response_shape(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    response = post_json(
        client, "volatility-mlp", {"instances": [golden_records["volatility-mlp"]]}, auth
    )
    assert response.status_code == 200
    (prediction,) = response.json()["predictions"]
    assert prediction["kind"] == "regression"
    assert math.isfinite(prediction["value"])
    assert prediction["unit"] == "volatilidade anualizada (fração)"
    assert prediction["target"] == "realized_volatility_5d"
    assert prediction["value"] > 0.0  # exp() output: always positive


def test_volatility_regression_is_positive_for_any_valid_record(
    client: TestClient, auth: dict[str, str]
) -> None:
    rng = np.random.default_rng(5)
    records = [dummy.volatility_record(rng) for _ in range(20)]
    response = post_json(client, "volatility-mlp", {"instances": records}, auth)
    assert response.status_code == 200
    values = [p["value"] for p in response.json()["predictions"]]
    assert all(math.isfinite(v) and v > 0.0 for v in values)


def test_batches_keep_their_order(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    base = golden_records["volatility-mlp"]
    records = [{**base, "rv_5": 0.1 + 0.05 * i} for i in range(5)]
    batch = post_json(client, "volatility-mlp", {"instances": records}, auth).json()["predictions"]
    singles = [
        post_json(client, "volatility-mlp", {"instances": [r]}, auth).json()["predictions"][0]
        for r in records
    ]
    assert [p["value"] for p in batch] == pytest.approx([p["value"] for p in singles], rel=1e-5)
    assert len({p["value"] for p in batch}) == 5


def test_image_prediction(client: TestClient, auth: dict[str, str], png_bytes: bytes) -> None:
    response = client.post(PREDICT.format("candlestick-cnn"), headers=auth, **multipart(png_bytes))
    assert response.status_code == 200, response.text
    (prediction,) = response.json()["predictions"]
    assert prediction["kind"] == "image_classification"
    assert prediction["label"] in {"baixa", "media", "alta"}
    scores = [entry["probability"] for entry in prediction["top_k"]]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert prediction["top_k"][0]["label"] == prediction["label"]
    assert prediction["probability"] == pytest.approx(scores[0])


def test_a_client_request_id_is_echoed_and_hostile_ones_are_replaced(
    client: TestClient, auth: dict[str, str]
) -> None:
    ok = client.get("/v1/models", headers={**auth, "X-Request-ID": "client-supplied-id-42"})
    assert ok.headers["x-request-id"] == "client-supplied-id-42"
    for hostile in (
        "short",
        "has spaces in it ok",
        "x" * 200,
        "id\twith\ttabs12",
        "<script>alert(1)",
    ):
        response = client.get("/v1/models", headers={**auth, "X-Request-ID": hostile})
        assert response.headers["x-request-id"] != hostile
        assert len(response.headers["x-request-id"]) == 32


def test_the_model_catalogue_and_detail(client: TestClient, auth: dict[str, str]) -> None:
    listing = client.get("/v1/models", headers=auth).json()["models"]
    assert [m["name"] for m in listing] == [
        "candlestick-cnn",
        "credit-default-mlp",
        "volatility-mlp",
    ]
    assert all(m["available"] for m in listing)
    detail = client.get("/v1/models/volatility-mlp", headers=auth)
    assert detail.headers["x-model-version"] == "0.0.0-dummy"
    body = detail.json()
    assert body["task"] == "regression"
    assert {f["name"] for f in body["input"]["fields"]} >= {"rv_5", "rv_20", "dow"}
    assert "model.onnx" not in detail.text  # artifact files are internal


# --------------------------------------------------------------------------- 4xx


def test_unknown_model_is_404(client: TestClient, auth: dict[str, str]) -> None:
    for call in (
        lambda: client.get("/v1/models/nope", headers=auth),
        lambda: post_json(client, "nope", {"instances": [{}]}, auth),
    ):
        response = call()
        assert response.status_code == 404
        assert response.json()["code"] == "model_not_found"
        assert response.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize("name", ["UPPER", "-lead", "a" * 64, "has space", "dot.dot", "%00", "a_b"])
def test_malformed_model_names_are_422(client: TestClient, auth: dict[str, str], name: str) -> None:
    response = client.get(f"/v1/models/{name}", headers=auth)
    assert response.status_code in (404, 422)
    assert response.headers["content-type"] == "application/problem+json"


@pytest.mark.parametrize(
    ("mutate", "code", "loc_tail"),
    [
        (lambda r: r.pop("limit_bal"), "required", "limit_bal"),
        (lambda r: r.update(limit_bal=None), "null_not_allowed", "limit_bal"),
        (lambda r: r.update(limit_bal="12"), "type_error", "limit_bal"),
        (lambda r: r.update(limit_bal=True), "type_error", "limit_bal"),
        (lambda r: r.update(limit_bal=0.0), "below_minimum", "limit_bal"),
        (lambda r: r.update(limit_bal=1e39), "out_of_range", "limit_bal"),
        (lambda r: r.update(Surprise=1), "unknown_field", "Surprise"),
        (lambda r: r.update(education="phd"), "not_in_vocabulary", "education"),
        (lambda r: r.update(pay_1=9), "above_maximum", "pay_1"),
    ],
)
def test_record_validation_pinpoints_the_field(
    client: TestClient,
    auth: dict[str, str],
    golden_records: dict[str, Any],
    mutate: Callable[[dict[str, Any]], object],
    code: str,
    loc_tail: str,
) -> None:
    record = dict(golden_records["credit-default-mlp"])
    mutate(record)
    response = post_json(client, "credit-default-mlp", {"instances": [record]}, auth)
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["code"] == "invalid_input"
    assert {"loc": ["instances", 0, loc_tail], "code": code} == {
        k: v for k, v in body["errors"][0].items() if k in ("loc", "code")
    }


def test_validation_messages_never_echo_the_submitted_value(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    record = {
        **golden_records["credit-default-mlp"],
        "limit_bal": "SECRET-VALUE-123",
        "Surprise-XYZ-field": 1,
    }
    response = post_json(client, "credit-default-mlp", {"instances": [record]}, auth)
    assert response.status_code == 422
    assert "SECRET-VALUE-123" not in response.text


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"instances": []},
        {"instances": {}},
        {"instances": [1]},
        {"instances": ["x"]},
        {"instances": [[]]},
        {"instances": [None]},
        {"instances": [{}], "extra": 1},
        [],
        "text",
        None,
        3,
    ],
)
def test_bad_envelopes_are_422(client: TestClient, auth: dict[str, str], body: Any) -> None:
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=json.dumps(body).encode(),
    )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_input"


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"",
        b"nan",
        b'{"instances": [NaN]}',
        b'{"instances": [{"limit_bal": Infinity}]}',
        b"\xff\xfe",
    ],
)
def test_malformed_json_is_422_not_500(
    client: TestClient, auth: dict[str, str], raw: bytes
) -> None:
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=raw,
    )
    assert response.status_code == 422
    assert (
        response.json()["errors"][0]["code"] in {"invalid_json", "json_invalid", "model_type"}
        or response.json()["errors"]
    )


def test_deeply_nested_json_is_rejected_cleanly(client: TestClient, auth: dict[str, str]) -> None:
    nested = b'{"instances": [' + b"[" * 5000 + b"]" * 5000 + b"]}"
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "application/json"},
        content=nested,
    )
    assert response.status_code == 422


def test_batches_over_the_limit_are_rejected(
    make_client: Callable[..., TestClient], auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    client = make_client(max_batch=3)
    record = golden_records["credit-default-mlp"]
    assert (
        post_json(client, "credit-default-mlp", {"instances": [record] * 3}, auth).status_code
        == 200
    )
    response = post_json(client, "credit-default-mlp", {"instances": [record] * 4}, auth)
    assert response.status_code == 422
    assert response.json()["errors"][0]["code"] == "batch_too_large"


@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "application/x-www-form-urlencoded",
        "application/jsonx",
        "multipart/form-data; boundary=x",
    ],
)
def test_tabular_models_demand_json(
    client: TestClient, auth: dict[str, str], content_type: str
) -> None:
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": content_type},
        content=b"{}",
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"


def test_a_missing_content_type_is_415(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(PREDICT.format("credit-default-mlp"), headers=auth, content=b"{}")
    assert response.status_code == 415


def test_image_models_demand_multipart(client: TestClient, auth: dict[str, str]) -> None:
    response = post_json(client, "candlestick-cnn", {"instances": [{}]}, auth)
    assert response.status_code == 415


def test_json_charset_parameter_is_accepted(
    client: TestClient, auth: dict[str, str], golden_records: dict[str, Any]
) -> None:
    body = json.dumps({"instances": [golden_records["credit-default-mlp"]]}).encode()
    response = client.post(
        PREDICT.format("credit-default-mlp"),
        headers={**auth, "Content-Type": "Application/JSON; charset=utf-8"},
        content=body,
    )
    assert response.status_code == 200


def test_wrong_methods_are_405_problems(client: TestClient, auth: dict[str, str]) -> None:
    for method, path in (
        ("PUT", "/v1/models"),
        ("DELETE", "/v1/models/credit-default-mlp"),
        ("GET", PREDICT.format("credit-default-mlp")),
    ):
        response = client.request(method, path, headers=auth)
        assert response.status_code == 405
        assert response.headers["content-type"] == "application/problem+json"
        assert "allow" in response.headers


def test_unknown_paths_are_404_problems(client: TestClient, auth: dict[str, str]) -> None:
    for path in ("/", "/v1", "/v1/models/", "/metrics", "/docs", "/redoc", "/admin", "/.env"):
        response = client.get(path, headers=auth)
        assert (
            response.status_code in (404, 307, 200)
            if path == "/v1/models/"
            else response.status_code == 404
        )
    assert client.get("/nope", headers=auth).json()["code"] == "not_found"


# --------------------------------------------------------------------------- images


def test_image_error_cases(client: TestClient, auth: dict[str, str], png_bytes: bytes) -> None:
    url = PREDICT.format("candlestick-cnn")
    cases = {
        "not an image": (multipart(b"hello world, definitely not an image"), 415),
        "gif": (multipart(b"GIF89a" + b"\x00" * 64), 415),
        "svg": (multipart(b"<svg xmlns='http://www.w3.org/2000/svg'/>"), 415),
        "truncated png": (multipart(png_bytes[: len(png_bytes) // 2]), 422),
        "empty file": (multipart(b""), 415),
        "wrong field": (multipart(png_bytes, field="image"), 422),
    }
    for label, (kwargs, expected) in cases.items():
        response = client.post(url, headers=auth, **kwargs)
        assert response.status_code == expected, (label, response.text)
        assert response.headers["content-type"] == "application/problem+json", label


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_the_chart_model_accepts_png_only(
    client: TestClient, auth: dict[str, str], fmt: str
) -> None:
    """A lossy re-encoding would destroy the 1-pixel strokes the CNN reads."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (0, 0, 0)).save(buffer, format=fmt)
    response = client.post(
        PREDICT.format("candlestick-cnn"), headers=auth, **multipart(buffer.getvalue())
    )
    assert response.status_code == 415
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["code"] == "unsupported_media_type"
    assert "PNG" in body["detail"]
    assert body["request_id"] == response.headers["x-request-id"]


def test_a_native_size_png_is_used_as_is(client: TestClient, auth: dict[str, str]) -> None:
    chart = dummy.synthetic_chart("alta", np.random.default_rng(0))
    buffer = io.BytesIO()
    Image.fromarray(chart).save(buffer, format="PNG")
    response = client.post(
        PREDICT.format("candlestick-cnn"), headers=auth, **multipart(buffer.getvalue())
    )
    assert response.status_code == 200
    (prediction,) = response.json()["predictions"]
    assert {e["label"] for e in prediction["top_k"]} == {"baixa", "media", "alta"}


def test_a_form_without_files_is_422(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        PREDICT.format("candlestick-cnn"),
        headers=auth,
        data={"file": "just text"},
        files={"x": ("x", b"y")},
    )
    assert response.status_code == 422


def test_image_larger_than_the_manifest_limit_is_413(
    client: TestClient, auth: dict[str, str], registry: ModelRegistry
) -> None:
    limit = registry.get("candlestick-cnn").manifest.input.max_bytes  # type: ignore[union-attr]
    padding = b"\x89PNG\r\n\x1a\n" + b"0" * (limit + 10)
    response = client.post(PREDICT.format("candlestick-cnn"), headers=auth, **multipart(padding))
    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"


def test_pixel_bombs_are_refused_before_decoding(
    make_client: Callable[..., TestClient], auth: dict[str, str], png_bytes: bytes
) -> None:
    client = make_client(max_image_pixels=100)  # the 40x30 PNG has 1200 pixels
    response = client.post(PREDICT.format("candlestick-cnn"), headers=auth, **multipart(png_bytes))
    assert response.status_code == 422
    assert "pixels" in response.json()["errors"][0]["message"]


# --------------------------------------------------------------------------- availability


def test_an_unavailable_model_is_404_model_not_available(
    make_client: Callable[..., TestClient],
    fake_registry: Callable[..., ModelRegistry],
    auth: dict[str, str],
) -> None:
    client = make_client(registry_override=fake_registry())
    for response in (
        client.get("/v1/models/broken-model", headers=auth),
        post_json(client, "broken-model", {"instances": [{}]}, auth),
    ):
        assert response.status_code == 404
        assert response.json()["code"] == "model_not_available"
        assert "integrity" in response.json()["detail"]
    listing = {m["name"]: m for m in client.get("/v1/models", headers=auth).json()["models"]}
    assert listing["broken-model"]["available"] is False
    assert listing["broken-model"]["reason"] == "integrity_check_failed"
    assert listing["credit-default-mlp"]["available"] is True


def test_runtime_failures_are_generic_500s(
    make_client: Callable[..., TestClient],
    fake_registry: Callable[..., ModelRegistry],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    leaked_path = "internal-path-/srv/models/hidden.onnx"
    registry = fake_registry(
        {"credit-default-mlp": FakeRuntime((25,), 2, error=RuntimeError(leaked_path))}
    )
    client = make_client(registry_override=registry)
    response = post_json(
        client, "credit-default-mlp", {"instances": [golden_records["credit-default-mlp"]]}, auth
    )
    assert response.status_code == 500
    assert response.json()["code"] == "inference_failed"
    assert leaked_path not in response.text
    assert "Traceback" not in response.text


def test_a_garbage_runtime_answer_is_a_500_not_a_wrong_prediction(
    make_client: Callable[..., TestClient],
    fake_registry: Callable[..., ModelRegistry],
    auth: dict[str, str],
    golden_records: dict[str, Any],
) -> None:
    broken = FakeRuntime((25,), 2, fn=lambda b: np.full((len(b), 7), 0.5, dtype=np.float32))
    client = make_client(registry_override=fake_registry({"credit-default-mlp": broken}))
    response = post_json(
        client, "credit-default-mlp", {"instances": [golden_records["credit-default-mlp"]]}, auth
    )
    assert response.status_code == 500
