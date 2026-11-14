"""Docker HEALTHCHECK: exit 0 iff ``GET /health/live`` answers 200 (stdlib only)."""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 2.0


def check(port: int, timeout: float = TIMEOUT_S) -> bool:
    url = f"http://127.0.0.1:{port}/health/live"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def main() -> int:
    try:
        port = int(os.environ.get("TRINET_PORT", "8000"))
    except ValueError:
        return 1
    return 0 if check(port) else 1


if __name__ == "__main__":
    sys.exit(main())
