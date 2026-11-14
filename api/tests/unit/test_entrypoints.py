"""``python -m trinet_api`` and ``python -m trinet_api.healthcheck`` (the Docker contract)."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
import uvicorn

from trinet_api import __main__ as entrypoint
from trinet_api import healthcheck


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Handler(BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(type(self).status if self.path == "/health/live" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def server() -> Iterator[tuple[int, type[_Handler]]]:
    handler = type("Handler", (_Handler,), {"status": 200})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_port, handler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_healthy_service_exits_zero(
    server: tuple[int, type[_Handler]], monkeypatch: pytest.MonkeyPatch
) -> None:
    port, _ = server
    assert healthcheck.check(port) is True
    monkeypatch.setenv("TRINET_PORT", str(port))
    assert healthcheck.main() == 0


@pytest.mark.parametrize("status", [204, 500, 503])
def test_any_non_200_answer_is_unhealthy(
    server: tuple[int, type[_Handler]], monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    port, handler = server
    handler.status = status
    monkeypatch.setenv("TRINET_PORT", str(port))
    assert healthcheck.main() == 1


def test_nothing_listening_is_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRINET_PORT", str(free_port()))
    assert healthcheck.main() == 1


def test_a_garbage_port_is_unhealthy_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRINET_PORT", "eighty")
    assert healthcheck.main() == 1


def test_the_port_defaults_to_8000(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    monkeypatch.delenv("TRINET_PORT", raising=False)

    def fake_check(port: int, timeout: float = 0) -> bool:
        seen.append(port)
        return True

    monkeypatch.setattr(healthcheck, "check", fake_check)
    assert healthcheck.main() == 0
    assert seen == [8000]


def test_main_starts_one_uvicorn_process_with_the_app_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setenv("TRINET_PORT", "8123")
    monkeypatch.setenv("TRINET_HOST", "127.0.0.1")
    monkeypatch.setenv("TRINET_LOG_LEVEL", "warning")
    entrypoint.main()
    ((args, kwargs),) = calls
    assert args == ("trinet_api.interface.http.app:create_app",)
    assert kwargs["factory"] is True
    assert kwargs["workers"] == 1
    assert (kwargs["host"], kwargs["port"]) == ("127.0.0.1", 8123)
    assert kwargs["log_config"] is None
    assert kwargs["access_log"] is False
    assert kwargs["server_header"] is False


def test_the_factory_path_resolves() -> None:
    import importlib

    module, _, attribute = entrypoint.FACTORY.partition(":")
    assert callable(getattr(importlib.import_module(module), attribute))
