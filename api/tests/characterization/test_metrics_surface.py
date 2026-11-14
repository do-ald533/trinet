"""Characterization (R0): the Prometheus surface of the API, pinned as a SUPERSET contract.

What is pinned (``metrics.surface.json``):

* every metric family (exposed name, type, help text, label names, histogram buckets): the
  test fails when a pinned family disappears or changes, but extra families are allowed
  (the refactor will add some);
* the label VALUES produced by a scripted traffic mix: ``route`` is the route template (or
  ``unmatched``), ``model`` is the path parameter literally named ``name`` filtered through the
  allowlist of registered models (anything else is ``unknown``, so attacker supplied names
  can never create a time series), ``method`` is one of the HTTP verbs or ``OTHER``.

Regenerate the JSON with ``UPDATE_SNAPSHOTS=1 pytest`` (same convention as the OpenAPI snapshot).
Everything here goes through the real app and its dedicated metrics port, so a package move
cannot change the outcome.
"""

from __future__ import annotations

import difflib
import io
import json
import os
import re
import socket
import urllib.request
from collections.abc import Callable, Iterator
from http import HTTPMethod
from pathlib import Path
from typing import Any

import make_dummy_artifacts as dummy
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from tests.conftest import API_KEY, make_settings

# Update these imports if packages move (refactor R1/R2): the only place this file names them.
from trinet_api.domain.entities import ModelRegistry, UnavailableModel
from trinet_api.infrastructure.observability import Metrics
from trinet_api.interface.http.app import create_app
from trinet_contract import ClassificationOutput

PIN = Path(__file__).with_name("metrics.surface.json")
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"
AUTH = {"X-API-Key": API_KEY}
BROKEN = "broken-model"
KNOWN_ROUTES = {
    "/health/live",
    "/health/ready",
    "/openapi.json",
    "/v1/models",
    "/v1/models/{name}",
    "/v1/models/{name}/predictions",
    "unmatched",
}
LINUX_PROCFS = Path("/proc/self/stat").exists()  # ProcessCollector exposes nothing elsewhere


# --------------------------------------------------------------------------- plumbing


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def scrape(port: int) -> tuple[str, str]:
    """``(content_type, body)`` of the dedicated metrics server."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        assert response.status == 200
        body: bytes = response.read()
        return str(response.headers["Content-Type"]), body.decode()


def describe(text: str) -> list[dict[str, Any]]:
    """Families of an exposition text as ``{name, type, documentation, labelnames[, buckets]}``.

    ``*_created`` series are an artefact of the text format (a separate gauge family) and are
    not part of the contract. Counters are named as exposed (``..._total``).
    """
    families = []
    for family in text_string_to_metric_families(text):
        if family.name.endswith("_created"):
            continue
        exposed = f"{family.name}_total" if family.type == "counter" else family.name
        entry: dict[str, Any] = {
            "name": exposed,
            "type": family.type,
            "documentation": family.documentation,
            "labelnames": sorted({k for s in family.samples for k in s.labels} - {"le"}),
        }
        if family.type == "histogram":
            entry["buckets"] = sorted(
                {s.labels["le"] for s in family.samples if "le" in s.labels}, key=float
            )
        families.append(entry)
    return sorted(families, key=lambda f: str(f["name"]))


def sample_rows(text: str, sample_name: str, label_order: tuple[str, ...]) -> list[list[Any]]:
    rows = [
        [*(s.labels[k] for k in label_order), int(s.value)]
        for family in text_string_to_metric_families(text)
        for s in family.samples
        if s.name == sample_name
    ]
    return sorted(rows, key=lambda row: [str(c) for c in row])


def label_values(text: str) -> dict[str, Any]:
    """The label values of every series the scripted traffic creates."""
    info = sorted(
        [
            s.labels["model"],
            s.labels["version"],
            s.labels["status"],
            s.labels["runtime"],
            int(s.value),
        ]
        for family in text_string_to_metric_families(text)
        for s in family.samples
        if s.name == "trinet_model_info"
    )
    predicted: dict[str, int] = {}
    for model, _label, count in sample_rows(
        text, "trinet_predicted_class_total", ("model", "label")
    ):
        predicted[model] = predicted.get(model, 0) + count
    return {
        "http_requests_total": sample_rows(
            text, "trinet_http_requests_total", ("method", "route", "model", "status")
        ),
        "http_request_duration_seconds_count": sample_rows(
            text, "trinet_http_request_duration_seconds_count", ("method", "route", "model")
        ),
        "inference_duration_seconds_count": sample_rows(
            text, "trinet_inference_duration_seconds_count", ("model", "runtime")
        ),
        "model_info": info,
        "predicted_class_total_per_model": dict(sorted(predicted.items())),
    }


def read_pin() -> dict[str, Any]:
    pinned: dict[str, Any] = json.loads(PIN.read_text())
    return pinned


def persist_if_requested(current: dict[str, Any]) -> None:
    if UPDATE:
        PIN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")


def assert_superset(pinned: list[dict[str, Any]], current: list[dict[str, Any]]) -> None:
    by_name = {f["name"]: f for f in current}
    problems = []
    for family in pinned:
        if family["name"].startswith("process_") and not LINUX_PROCFS:
            continue
        found = by_name.get(family["name"])
        if found is None:
            problems.append(f"missing family {family['name']}")
        elif found != family:
            diff = "".join(
                difflib.unified_diff(
                    json.dumps(family, indent=1, sort_keys=True).splitlines(True),
                    json.dumps(found, indent=1, sort_keys=True).splitlines(True),
                    "pinned",
                    "current",
                    n=1,
                )
            )
            problems.append(f"family {family['name']} changed:\n{diff}")
    assert not problems, "\n".join(problems) + "\n(UPDATE_SNAPSHOTS=1 accepts the change)"


# --------------------------------------------------------------------------- scenario


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (20, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def golden_record(root: Path, model: str) -> dict[str, Any]:
    golden = json.loads((root / model / dummy.DUMMY_VERSION / "golden.json").read_text())
    record: dict[str, Any] = golden["cases"][0]["record"]
    return record


def with_broken_model(registry: ModelRegistry) -> ModelRegistry:
    entries = {entry.name: entry for entry in registry}
    entries[BROKEN] = UnavailableModel(BROKEN, "1.0.0", "integrity_check_failed")
    return ModelRegistry(entries)


def serve_and_scrape(
    root: Path, registry: ModelRegistry, drive: Callable[[FastAPI, TestClient], None]
) -> str:
    """Start an app with its metrics server, run ``drive`` and return the scraped text.

    Retries on a fresh port when another test process grabbed the chosen one (xdist).
    """
    failure: Exception | None = None
    for _ in range(5):
        port = free_port()
        settings = make_settings(
            root, metrics_enabled=True, metrics_host="127.0.0.1", metrics_port=port
        )
        app = create_app(settings, registry=registry)
        with TestClient(app, raise_server_exceptions=False) as client:
            drive(app, client)
            try:
                return scrape(port)[1]
            except OSError as exc:  # port taken by someone else: try another one
                failure = exc
    raise AssertionError("the metrics server never came up") from failure


def scripted_traffic(root: Path) -> Callable[[FastAPI, TestClient], None]:
    credit = golden_record(root, "credit-default-mlp")
    volatility = golden_record(root, "volatility-mlp")
    png = png_bytes()
    predict = "/v1/models/{}/predictions"

    def drive(_app: FastAPI, client: TestClient) -> None:
        upload = {"file": ("scene.png", png, "application/octet-stream")}
        calls: list[tuple[int, Callable[[], Any]]] = [
            (200, lambda: client.get("/health/live")),
            (200, lambda: client.get("/health/ready")),  # 200 "degraded": one model is broken
            (200, lambda: client.get("/openapi.json")),
            (200, lambda: client.get("/v1/models", headers=AUTH)),
            (401, lambda: client.get("/v1/models")),
            (200, lambda: client.get("/v1/models/credit-default-mlp", headers=AUTH)),
            (200, lambda: client.get("/v1/models/volatility-mlp", headers=AUTH)),
            (404, lambda: client.get("/v1/models/attacker-supplied-name", headers=AUTH)),
            (422, lambda: client.get("/v1/models/Bad_Name", headers=AUTH)),
            (422, lambda: client.get("/v1/models/%E2%98%83", headers=AUTH)),
            (404, lambda: client.get(f"/v1/models/{BROKEN}", headers=AUTH)),
            (404, lambda: client.get("/nowhere")),
            (404, lambda: client.get("/metrics", headers=AUTH)),
            (404, lambda: client.get("/v1/models/a/b/c", headers=AUTH)),
            (405, lambda: client.delete("/v1/models", headers=AUTH)),
            (405, lambda: client.request("BREW", "/v1/models", headers=AUTH)),
            (405, lambda: client.get(predict.format("credit-default-mlp"), headers=AUTH)),
            (
                200,
                lambda: client.post(
                    predict.format("credit-default-mlp"),
                    headers=AUTH,
                    json={"instances": [credit, credit]},
                ),
            ),
            (
                200,
                lambda: client.post(
                    predict.format("volatility-mlp"), headers=AUTH, json={"instances": [volatility]}
                ),
            ),
            (
                200,
                lambda: client.post(predict.format("candlestick-cnn"), headers=AUTH, files=upload),
            ),
            (
                422,
                lambda: client.post(
                    predict.format("credit-default-mlp"), headers=AUTH, json={"instances": [{}]}
                ),
            ),
            (
                415,
                lambda: client.post(
                    predict.format("credit-default-mlp"), headers=AUTH, content=b"{}"
                ),
            ),
        ]
        for expected, call in calls:
            response = call()
            assert response.status_code == expected, (
                str(response.request.url),
                response.text[:300],
            )

    return drive


@pytest.fixture(scope="module")
def scraped(artifacts_root: Path, registry: ModelRegistry) -> Iterator[str]:
    """The exposition text after the scripted traffic; also (re)writes the pin on request."""
    saved_pixel_limit = Image.MAX_IMAGE_PIXELS  # create_app changes this process-wide value
    try:
        text = serve_and_scrape(
            artifacts_root, with_broken_model(registry), scripted_traffic(artifacts_root)
        )
    finally:
        Image.MAX_IMAGE_PIXELS = saved_pixel_limit
    persist_if_requested({"families": describe(text), "label_values": label_values(text)})
    yield text


def exercised_metrics() -> Metrics:
    metrics = Metrics()
    metrics.register_models([("alpha", "1.0.0", "dummy", "onnx", True)])
    metrics.observe_http(method="GET", route="/x", model="alpha", status=200, seconds=0.01)
    metrics.observe_prediction(model="alpha", runtime="onnx", inference_seconds=0.01, labels=["a"])
    return metrics


# --------------------------------------------------------------------------- families


def test_the_scraped_families_are_a_superset_of_the_pinned_surface(scraped: str) -> None:
    assert_superset(read_pin()["families"], describe(scraped))


@pytest.mark.usefixtures("scraped")  # ordering only: the pin is (re)written before it is read
def test_a_fresh_metrics_registry_keeps_the_pinned_families() -> None:
    text = generate_latest(exercised_metrics().registry).decode()
    assert_superset(read_pin()["families"], describe(text))


def test_the_pin_covers_every_family_the_app_exposes_today(scraped: str) -> None:
    """Guards the pin itself: a family missing from the JSON would not be protected."""
    exposed = {f["name"] for f in describe(scraped) if f["name"].startswith("trinet_")}
    pinned = {f["name"] for f in read_pin()["families"]}
    assert exposed == {n for n in pinned if n.startswith("trinet_")}


def test_the_expected_families_exist_with_their_labels(scraped: str) -> None:
    """Hand-written copy of the essentials, independent of the JSON (so the JSON cannot rot)."""
    families = {f["name"]: f for f in describe(scraped)}
    expected = {
        "trinet_http_requests_total": ("counter", ["method", "model", "route", "status"]),
        "trinet_http_request_duration_seconds": ("histogram", ["method", "model", "route"]),
        "trinet_inference_duration_seconds": ("histogram", ["model", "runtime"]),
        "trinet_predicted_class_total": ("counter", ["label", "model"]),
        "trinet_http_inflight_requests": ("gauge", []),
        "trinet_inflight_inferences": ("gauge", []),
        "trinet_model_info": ("gauge", ["model", "runtime", "status", "version"]),
    }
    for name, (kind, labels) in expected.items():
        assert (families[name]["type"], families[name]["labelnames"]) == (kind, labels), name
    assert families["trinet_http_request_duration_seconds"]["buckets"] == [
        "0.002",
        "0.005",
        "0.01",
        "0.025",
        "0.05",
        "0.1",
        "0.25",
        "0.5",
        "1.0",
        "2.5",
        "5.0",
        "+Inf",
    ]


# --------------------------------------------------------------------------- label values


def test_the_label_values_of_the_scripted_traffic_are_unchanged(scraped: str) -> None:
    pinned = read_pin()["label_values"]
    current = label_values(scraped)
    assert current == pinned, "label values drifted (UPDATE_SNAPSHOTS=1 accepts the change)"


def test_every_label_value_comes_from_a_closed_set(scraped: str, registry: ModelRegistry) -> None:
    allowed_models = {entry.name for entry in registry} | {BROKEN, "unknown"}
    verbs = {m.value for m in HTTPMethod} | {"OTHER"}
    seen_models: set[str] = set()
    for family in text_string_to_metric_families(scraped):
        for sample in family.samples:
            labels = sample.labels
            if not sample.name.startswith("trinet_"):
                continue
            if "model" in labels:
                seen_models.add(labels["model"])
                assert labels["model"] in allowed_models, sample
            if "route" in labels:
                assert labels["route"] in KNOWN_ROUTES, sample
            if "method" in labels:
                assert labels["method"] in verbs, sample
            if "status" in labels and sample.name.startswith("trinet_http_requests"):
                assert re.fullmatch(r"[1-5][0-9]{2}", labels["status"]), sample
    assert {"unknown", BROKEN, "credit-default-mlp"} <= seen_models


def test_unknown_models_become_unknown_and_never_reach_a_label(scraped: str) -> None:
    for attacker_value in ("attacker-supplied-name", "Bad_Name", "☃", "%E2%98%83", "a/b/c"):
        assert attacker_value not in scraped
    assert 'model="unknown",route="/v1/models/{name}",status="404"' in scraped
    assert 'model="unknown",route="/v1/models/{name}",status="422"' in scraped
    # an unavailable model is registered: the allowlist is "every registered name", not "served"
    assert f'model="{BROKEN}",route="/v1/models/{{name}}",status="404"' in scraped
    # unmatched paths never reflect the path into the route label
    assert 'route="unmatched",status="404"' in scraped
    assert "nowhere" not in scraped


def test_predicted_class_labels_come_from_the_manifest(
    scraped: str, registry: ModelRegistry
) -> None:
    rows = sample_rows(scraped, "trinet_predicted_class_total", ("model", "label"))
    assert {model for model, _label, _count in rows} == {"credit-default-mlp", "candlestick-cnn"}
    for model, label, _count in rows:
        output = registry.get(model).manifest.output
        assert isinstance(output, ClassificationOutput)
        assert label in output.labels, (model, label)  # regression models produce no class series


def test_only_the_path_parameter_named_name_feeds_the_model_label(
    artifacts_root: Path, registry: ModelRegistry
) -> None:
    """A rename of ``{name}`` in a route (e.g. when routers are split) would silently turn
    the model label into ``unknown``: the middleware reads ``path_params["name"]`` only."""

    def by_name(name: str) -> dict[str, str]:
        return {"name": name}

    def by_model(model: str) -> dict[str, str]:
        return {"model": model}

    def drive(app: FastAPI, client: TestClient) -> None:
        app.add_api_route("/probe/{name}", by_name, methods=["GET"])
        app.add_api_route("/probe-other/{model}", by_model, methods=["GET"])
        for path in (
            "/probe/credit-default-mlp",
            "/probe/not-registered",
            "/probe-other/credit-default-mlp",
        ):
            assert client.get(path).status_code == 200

    saved = Image.MAX_IMAGE_PIXELS
    try:
        text = serve_and_scrape(artifacts_root, registry, drive)
    finally:
        Image.MAX_IMAGE_PIXELS = saved
    rows = sample_rows(text, "trinet_http_requests_total", ("route", "model", "status"))
    assert ["/probe/{name}", "credit-default-mlp", "200", 1] in rows
    assert ["/probe/{name}", "unknown", "200", 1] in rows
    assert ["/probe-other/{model}", "unknown", "200", 1] in rows
    assert "not-registered" not in text


def test_model_label_allowlist_of_a_bare_metrics_object() -> None:
    metrics = Metrics()
    assert metrics.model_label("alpha") == "unknown"  # nothing registered yet
    metrics.register_models(
        [("alpha", "1.0.0", "dummy", "onnx", True), ("beta", "", "unavailable", "none", False)]
    )
    assert [metrics.model_label(n) for n in ("alpha", "beta", "gamma", None, "")] == [
        "alpha",
        "beta",
        "unknown",
        "unknown",
        "unknown",
    ]
    metrics.observe_http(method="BREW", route="unmatched", model="gamma", status=404, seconds=0.0)
    assert (
        'method="OTHER",model="unknown",route="unmatched",status="404"'
        in generate_latest(metrics.registry).decode()
    )


# --------------------------------------------------------------------------- where it is served


@pytest.mark.parametrize("metrics_enabled", [True, False])
def test_metrics_are_not_served_on_the_public_app(
    make_client: Callable[..., TestClient], metrics_enabled: bool
) -> None:
    client = make_client(
        metrics_enabled=metrics_enabled, metrics_host="127.0.0.1", metrics_port=free_port()
    )
    assert client.get("/metrics", headers=AUTH).status_code == 404
    assert client.get("/metrics").status_code == 404  # not even a 401: the path does not exist
    assert client.get("/v1/metrics", headers=AUTH).status_code == 404


def test_metrics_are_served_on_the_dedicated_port(make_client: Callable[..., TestClient]) -> None:
    for _ in range(5):
        port = free_port()
        make_client(metrics_enabled=True, metrics_host="127.0.0.1", metrics_port=port)
        try:
            content_type, body = scrape(port)
        except OSError:
            continue
        assert content_type.startswith("text/plain")
        assert "# TYPE trinet_http_requests_total counter" in body
        assert "# TYPE trinet_model_info gauge" in body
        return
    pytest.fail("the metrics server never came up")


def test_nothing_listens_on_the_metrics_port_when_metrics_are_disabled(
    make_client: Callable[..., TestClient],
) -> None:
    port = free_port()
    make_client(metrics_enabled=False, metrics_host="127.0.0.1", metrics_port=port)
    with pytest.raises(OSError):
        scrape(port)
