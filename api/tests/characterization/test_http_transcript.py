"""A transcript of the REAL server, replayed over a socket and compared with a golden file.

``python -m trinet_api`` is started in a subprocess (uvicorn, real sockets, real start-up
and artifact loading) on free ports, with a private copy of the artifacts and a private API key
file. One request per route and status class is replayed with ``http.client``; the answer
(status, a header whitelist and the JSON body, volatile fields masked) is compared with
``transcript.golden.json``. ``UPDATE_SNAPSHOTS=1`` regenerates the golden file.

The module imports nothing from ``trinet_api``: only the entry point string ``-m trinet_api``
is used, so the transcript is immune to every module move of the refactor. Masking: request
ids (checked by pattern, and against the body's ``request_id``), the port in ``Location``,
and floats rounded to five decimals (the same model on another CPU may differ in the last ulps).
The metrics port is checked for Prometheus text, for the families the replay must have fed and
for their samples, but NOT for the full list of families (new metrics are allowed to appear).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import pytest
from tests.characterization._http_helpers import (
    CANDLESTICK,
    CREDIT,
    GENERATED_REQUEST_ID,
    SECURITY_HEADERS,
    VOLATILITY,
    dump_json,
    read_json,
    snapshot_update_requested,
)

GOLDEN = Path(__file__).with_name("transcript.golden.json")
HOST = "127.0.0.1"
API_KEY = "transcript-api-key-0123456789abcdef-0123456789"
MAX_JSON_BYTES = 4096
STARTUP_TIMEOUT_S = 40.0
HEADER_WHITELIST = (
    *SECURITY_HEADERS,
    "content-type",
    "x-model-version",
    "retry-after",
    "www-authenticate",
    "allow",
    "location",
)
REQUEST_ID_MASK = "<request-id>"
FAMILIES = (
    "trinet_http_requests_total",
    "trinet_http_request_duration_seconds",
    "trinet_inference_duration_seconds",
    "trinet_predicted_class_total",
    "trinet_http_inflight_requests",
    "trinet_inflight_inferences",
    "trinet_model_info",
)
PROBE_ROUTES = ("/health/live", "/health/ready")
SAMPLE = re.compile(r"^(?P<name>[a-z_]+)\{(?P<labels>[^}]*)\} (?P<value>\S+)$")


class ServerError(Exception):
    pass


# --------------------------------------------------------------------------- the subprocess


def free_ports(count: int) -> list[int]:
    """Distinct ports that were free a moment ago (bind-then-release: callers retry once)."""
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind((HOST, 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


@dataclass
class Server:
    artifacts: Path
    workdir: Path
    port: int
    metrics_port: int
    process: subprocess.Popen[bytes] | None = None
    log: TextIO | None = None
    log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.log_path = self.workdir / f"server-{self.port}.log"

    def environment(self) -> dict[str, str]:
        key_file = self.workdir / "api_key"
        key_file.write_text(API_KEY + "\n", encoding="utf-8")
        key_file.chmod(0o600)
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRINET_")}
        env.update(
            {
                "TRINET_ARTIFACTS_DIR": str(self.artifacts),
                "TRINET_API_KEY_FILE": str(key_file),
                "TRINET_HOST": HOST,
                "TRINET_PORT": str(self.port),
                "TRINET_METRICS_ENABLED": "true",
                "TRINET_METRICS_HOST": HOST,
                "TRINET_METRICS_PORT": str(self.metrics_port),
                "TRINET_LOG_LEVEL": "WARNING",
                "TRINET_RATE_LIMIT": "1000/minute",
                "TRINET_MAX_JSON_BYTES": str(MAX_JSON_BYTES),
                "PYTHONUNBUFFERED": "1",
            }
        )
        return env

    def start(self) -> None:
        self.log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv, our own interpreter
            [sys.executable, "-m", "trinet_api"],
            env=self.environment(),
            cwd=self.workdir,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def tail(self, lines: int = 25) -> str:
        if self.log is not None:
            self.log.flush()
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def wait_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        for path in ("/health/live", "/health/ready"):
            while True:
                if self.process.poll() is not None:
                    raise ServerError(f"exited with {self.process.returncode}:\n{self.tail()}")
                if time.monotonic() > deadline:
                    raise ServerError(f"{path} not 200 in time:\n{self.tail()}")
                try:
                    if fetch(self.port, "GET", path).status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.05)
        text = scrape(self.metrics_port)
        if "trinet_http_inflight_requests" not in text:  # someone else owns the metrics port
            raise ServerError("the metrics port does not serve this server's metrics")

    def stop(self) -> None:
        process, self.process = self.process, None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            if self.log is not None:
                self.log.close()
                self.log = None


def start_server(artifacts: Path, workdir: Path) -> Server:
    """A ready server; one retry with new ports if the first attempt cannot start."""
    failures: list[str] = []
    for _attempt in range(2):
        port, metrics_port = free_ports(2)
        server = Server(artifacts, workdir, port, metrics_port)
        server.start()
        try:
            server.wait_ready()
        except ServerError as exc:
            failures.append(str(exc))
            server.stop()
        else:
            return server
    raise AssertionError("the server did not start:\n" + "\n----\n".join(failures))


@pytest.fixture
def server(artifacts_copy: Path, tmp_path: Path) -> Iterator[Server]:
    workdir = tmp_path / "run"
    workdir.mkdir()
    started = start_server(artifacts_copy, workdir)
    try:
        yield started
    finally:
        started.stop()


# --------------------------------------------------------------------------- the client


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict[str, str]
    body: bytes


def fetch(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> Reply:
    connection = http.client.HTTPConnection(HOST, port, timeout=15)
    try:
        sent = {"Connection": "close", **(headers or {})}
        connection.request(method, path, body=body, headers=sent)
        response = connection.getresponse()
        return Reply(
            response.status,
            {name.lower(): value for name, value in response.getheaders()},
            response.read(),
        )
    finally:
        connection.close()


def scrape(port: int) -> str:
    reply = fetch(port, "GET", "/metrics")
    assert reply.status == 200
    return reply.body.decode("utf-8")


def multipart(data: bytes, *, field_name: str = "file") -> tuple[bytes, str]:
    boundary = "----transcript-boundary-7d3f"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="scene.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- the script


@dataclass(frozen=True)
class Step:
    name: str
    method: str
    path: str
    authed: bool = True
    content_type: str | None = None
    body: bytes | None = None
    request_id: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "authed": self.authed,
            "content_type": self.content_type,
            "body_bytes": None if self.body is None else len(self.body),
            "request_id": self.request_id,
        }


def script(records: dict[str, Any], png: bytes) -> list[Step]:
    json_type = "application/json"
    predict = "/v1/models/{}/predictions"
    form, form_type = multipart(png)
    bad_form, _ = multipart(png, field_name="image")

    def payload(name: str) -> bytes:
        return json.dumps({"instances": [records[name]]}).encode()

    return [
        Step("health_live", "GET", "/health/live", authed=False),
        Step("health_live_with_a_client_request_id", "GET", "/health/live", False, None, None, "transcript-echo-0001"),
        Step("health_ready", "GET", "/health/ready", authed=False),
        Step("list_models", "GET", "/v1/models"),
        Step("get_model_tabular", "GET", f"/v1/models/{CREDIT}"),
        Step("get_model_image", "GET", f"/v1/models/{CANDLESTICK}"),
        Step("create_prediction_classification", "POST", predict.format(CREDIT), True, json_type, payload(CREDIT)),
        Step("create_prediction_regression", "POST", predict.format(VOLATILITY), True, json_type, payload(VOLATILITY)),
        Step("create_prediction_image", "POST", predict.format(CANDLESTICK), True, form_type, form),
        Step("unauthorized_no_key", "GET", "/v1/models", authed=False),
        Step("unauthorized_before_not_found", "GET", "/v1/models/nope", authed=False),
        Step("model_not_found", "GET", "/v1/models/nope"),
        Step("not_found_unknown_path", "GET", "/nope", authed=False),
        Step("not_found_metrics_on_the_public_port", "GET", "/metrics"),
        Step("method_not_allowed", "DELETE", "/v1/models"),
        Step("unsupported_media_type_tabular", "POST", predict.format(CREDIT), True, "text/plain", b"{}"),
        Step("unsupported_media_type_image_gets_json", "POST", predict.format(CANDLESTICK), True, json_type, b"{}"),
        Step("invalid_input_missing_fields", "POST", predict.format(CREDIT), True, json_type, b'{"instances": [{}]}'),
        Step("invalid_input_not_json", "POST", predict.format(CREDIT), True, json_type, b"{nope"),
        Step("invalid_input_image_field_missing", "POST", predict.format(CANDLESTICK), True, form_type, bad_form),
        Step("validation_error_path", "GET", "/v1/models/NOPE"),
        Step("payload_too_large", "POST", predict.format(CREDIT), True, json_type, b" " * (MAX_JSON_BYTES + 1)),
        Step("redirect_trailing_slash", "GET", "/health/live/", authed=False),
    ]


def mask(value: Any, request_id: str) -> Any:
    """Normalise volatile parts of a JSON body: ids, and floats to five decimals."""
    if isinstance(value, dict):
        return {
            key: REQUEST_ID_MASK if key == "request_id" and inner == request_id else mask(inner, request_id)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [mask(item, request_id) for item in value]
    if isinstance(value, float):
        return round(value, 5)
    return value


def record(step: Step, reply: Reply, port: int) -> dict[str, Any]:
    headers = {name: reply.headers[name] for name in HEADER_WHITELIST if name in reply.headers}
    request_id = reply.headers["x-request-id"]
    if step.request_id is None:
        assert GENERATED_REQUEST_ID.fullmatch(request_id), request_id
        headers["x-request-id"] = REQUEST_ID_MASK
    else:
        assert request_id == step.request_id
        headers["x-request-id"] = request_id
    if "location" in headers:
        headers["location"] = headers["location"].replace(f"{HOST}:{port}", "<host>")
    assert "server" not in reply.headers
    assert "date" in reply.headers  # uvicorn's default; not recorded
    media = reply.headers.get("content-type", "")
    body: Any
    if media in ("application/json", "application/problem+json"):
        parsed = json.loads(reply.body)
        if "request_id" in parsed:
            assert parsed["request_id"] == request_id
        body = mask(parsed, request_id)
    else:
        body = reply.body.decode("utf-8")
    return {
        "name": step.name,
        "request": step.describe(),
        "response": {"status": reply.status, "headers": headers, "body": body},
    }


# --------------------------------------------------------------------------- metrics


def parse_samples(text: str) -> tuple[dict[str, str], list[tuple[str, dict[str, str], float]]]:
    types: dict[str, str] = {}
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(maxsplit=3)
            types[name] = kind
        elif match := SAMPLE.match(line):
            labels = dict(re.findall(r'(\w+)="([^"]*)"', match["labels"]))
            samples.append((match["name"], labels, float(match["value"])))
    return types, samples


def metrics_transcript(port: int, expected_requests: int) -> dict[str, Any]:
    """What the metrics port says once the replay is over (probes excluded: they are polled)."""
    deadline = time.monotonic() + 10
    while True:
        text = scrape(port)
        types, samples = parse_samples(text)
        requests = [
            (labels, value)
            for name, labels, value in samples
            if name == "trinet_http_requests_total" and labels["route"] not in PROBE_ROUTES
        ]
        if sum(value for _, value in requests) >= expected_requests or time.monotonic() > deadline:
            break
        time.sleep(0.05)

    def lines(sample_name: str, *keys: str) -> list[str]:
        return sorted(
            " ".join([*(labels[k] for k in keys), f"{value:g}"])
            for name, labels, value in samples
            if name == sample_name and labels.get("route") not in PROBE_ROUTES
        )

    return {
        "family_types": {name: types.get(name) for name in FAMILIES},
        "requests_total": lines("trinet_http_requests_total", "method", "route", "model", "status"),
        "request_duration_counts": lines(
            "trinet_http_request_duration_seconds_count", "method", "route", "model"
        ),
        "inference_duration_counts": lines(
            "trinet_inference_duration_seconds_count", "model", "runtime"
        ),
        "predicted_class_total": lines("trinet_predicted_class_total", "model", "label"),
        "model_info": lines("trinet_model_info", "model", "version", "status", "runtime"),
    }


# --------------------------------------------------------------------------- the test


def test_the_real_server_answers_the_recorded_transcript(
    server: Server, golden_records: dict[str, Any], png_bytes: bytes
) -> None:
    steps = script(golden_records, png_bytes)
    entries = []
    for step in steps:
        headers: dict[str, str] = {}
        if step.authed:
            headers["X-API-Key"] = API_KEY
        if step.content_type is not None:
            headers["Content-Type"] = step.content_type
        if step.request_id is not None:
            headers["X-Request-ID"] = step.request_id
        reply = fetch(server.port, step.method, step.path, headers=headers, body=step.body)
        entries.append(record(step, reply, server.port))

    public_metrics = fetch(server.port, "GET", "/metrics")  # also a request: it is a 404 here
    assert public_metrics.status == 404

    scraped = fetch(server.metrics_port, "GET", "/metrics")
    media = scraped.headers["content-type"].split(";")[0]
    served = sum(1 for step in steps if step.path not in PROBE_ROUTES) + 1  # + the one above
    current = {
        "format": 1,
        "requests": entries,
        "metrics_port": {
            "status": scraped.status,
            "media_type": media,
            "body_starts_with_help": scraped.body.decode().startswith("# HELP "),
            **metrics_transcript(server.metrics_port, served),
        },
    }
    if snapshot_update_requested():
        GOLDEN.write_text(dump_json(current), encoding="utf-8")
    expected = read_json(GOLDEN)
    mismatched = [
        c["name"] for c, e in zip(current["requests"], expected["requests"], strict=False) if c != e
    ]
    assert current == expected, (
        f"the HTTP transcript changed (UPDATE_SNAPSHOTS=1 to accept). Differing steps: {mismatched}"
    )
