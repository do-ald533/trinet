"""Black-box smoke test of the *built image* (Testcontainers), or of any running instance.

Skipped unless ``TRINET_IMAGE`` names an image (the tests start it with Testcontainers, so Docker
is required) or ``TRINET_BASE_URL`` + ``TRINET_API_KEY`` point at an instance that is already
running (this is what ``.github/scripts/smoke-image.sh`` exports for its post-command, and how
the suite is checked without Docker). Run it with ``make integration IMAGE=...`` or::

    TRINET_IMAGE=trinet-api:local uv run pytest -m integration api/tests/integration

The container gets the same hardening as ``deploy/compose.yaml`` (read-only root, tmpfs ``/tmp``,
no capabilities) and a random one-off API key mounted as a secret file, exactly like production.
The key is never printed or logged.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import pytest

from trinet_api.infrastructure.loading import latest_versions
from trinet_contract import (
    ImageGolden,
    Manifest,
    TabularGolden,
    load_golden,
)

IMAGE = os.environ.get("TRINET_IMAGE", "")
BASE_URL = os.environ.get("TRINET_BASE_URL", "")
EXTERNAL_KEY = os.environ.get("TRINET_API_KEY", "")
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"
CONTAINER_PORT = 8000
READY_TIMEOUT_S = 120.0

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (IMAGE or (BASE_URL and EXTERNAL_KEY)),
        reason="set TRINET_IMAGE (needs Docker) or TRINET_BASE_URL and TRINET_API_KEY",
    ),
]


@dataclass(frozen=True)
class Service:
    base_url: str
    api_key: str

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}


def _wait_until_ready(base_url: str, logs: Any = None) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    last = "no answer yet"
    while time.monotonic() < deadline:
        try:
            response = httpx2.get(f"{base_url}/health/ready", timeout=3.0)
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        except httpx2.HTTPError as exc:
            last = type(exc).__name__
        time.sleep(1.0)
    detail = f"\n--- container logs ---\n{logs()}" if logs is not None else ""
    pytest.fail(f"service not ready after {READY_TIMEOUT_S:.0f}s ({last}){detail}")


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Service]:
    if BASE_URL and not IMAGE:
        _wait_until_ready(BASE_URL.rstrip("/"))
        yield Service(BASE_URL.rstrip("/"), EXTERNAL_KEY)
        return

    from testcontainers.core.container import DockerContainer

    key = secrets.token_urlsafe(32)  # throwaway: lives in a temp file and in this process only
    secret_dir = tmp_path_factory.mktemp("secrets")
    secret_dir.chmod(0o755)
    key_file = secret_dir / "api_key"
    key_file.write_text(key, encoding="utf-8")
    key_file.chmod(0o444)  # the container runs as a non-root uid that is not ours

    container = (
        DockerContainer(IMAGE)
        .with_exposed_ports(CONTAINER_PORT)
        .with_volume_mapping(str(key_file), "/run/secrets/api_key", "ro")
        # testcontainers >= 4.15 owns the docker-py tmpfs kwarg (not usable via with_kwargs)
        .with_tmpfs_mount("/tmp", "rw,noexec,nosuid,nodev,size=64m,mode=1777")  # noqa: S108
        .with_kwargs(
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="1g",
            pids_limit=256,
        )
    )
    with container:
        base_url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(CONTAINER_PORT)}"

        def logs() -> str:
            stdout, stderr = container.get_logs()
            return (stdout + stderr).decode("utf-8", errors="replace")[-4000:]

        _wait_until_ready(base_url, logs)
        yield Service(base_url, key)


@pytest.fixture(scope="module")
def http() -> Iterator[httpx2.Client]:
    with httpx2.Client(timeout=15.0) as client:
        yield client


def _golden_request(model: str) -> dict[str, Any]:
    """A valid request for ``model``, taken from the golden file of the committed artifact."""
    directory = latest_versions(ARTIFACTS)[model]
    manifest = Manifest.model_validate_json((directory / "manifest.json").read_bytes())
    assert manifest.golden is not None, f"{model} ships no golden file"
    golden = load_golden(directory / manifest.golden.file)
    if isinstance(golden, TabularGolden):
        return {"json": {"instances": [golden.cases[0].record]}}
    assert isinstance(golden, ImageGolden)
    case = golden.cases[0]
    data = (directory / golden.directory / case.file).read_bytes()
    return {"files": {"file": (case.file, data, "application/octet-stream")}}


COMMITTED_MODELS = sorted(latest_versions(ARTIFACTS))


# --------------------------------------------------------------------------- tests


def test_the_service_is_ready_and_serves_every_committed_model(
    service: Service, http: httpx2.Client
) -> None:
    assert http.get(f"{service.base_url}/health/live").status_code == 200
    ready = http.get(f"{service.base_url}/health/ready")
    assert ready.status_code == 200, ready.text
    listing = http.get(f"{service.base_url}/v1/models", headers=service.headers)
    assert listing.status_code == 200, listing.text
    models = {m["name"]: m for m in listing.json()["models"]}
    assert set(COMMITTED_MODELS) <= set(models), f"served: {sorted(models)}"
    unavailable = {n: m.get("reason") for n, m in models.items() if not m["available"]}
    assert unavailable == {}


@pytest.mark.parametrize("model", COMMITTED_MODELS)
def test_one_prediction_per_model(service: Service, http: httpx2.Client, model: str) -> None:
    response = http.post(
        f"{service.base_url}/v1/models/{model}/predictions",
        headers=service.headers,
        **_golden_request(model),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_name"] == model
    assert response.headers["x-model-version"] == body["model_version"]
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["kind"] in {
        "classification",
        "regression",
        "image_classification",
    }


def test_requests_without_a_valid_key_are_rejected(service: Service, http: httpx2.Client) -> None:
    model = COMMITTED_MODELS[0]
    attempts = [
        http.get(f"{service.base_url}/v1/models"),
        http.get(f"{service.base_url}/v1/models", headers={"X-API-Key": "wrong-key-0123456789"}),
        http.post(f"{service.base_url}/v1/models/{model}/predictions", json={"instances": [{}]}),
    ]
    for response in attempts:
        assert response.status_code == 401
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.headers.get("www-authenticate")
        assert service.api_key not in response.text


def test_the_health_probes_need_no_key_and_the_schema_is_published(
    service: Service, http: httpx2.Client
) -> None:
    assert http.get(f"{service.base_url}/health/live").status_code == 200
    schema = http.get(f"{service.base_url}/openapi.json")
    assert schema.status_code == 200
    assert "/v1/models/{name}/predictions" in schema.json()["paths"]
