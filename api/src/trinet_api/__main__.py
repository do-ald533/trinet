"""``python -m trinet_api``: one uvicorn process serving the app factory."""

from __future__ import annotations

import uvicorn

from trinet_api.infrastructure.observability import configure_logging
from trinet_api.settings import Settings

FACTORY = "trinet_api.interface.http.app:create_app"


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        FACTORY,
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,  # scale with container replicas, not workers (see PLANO 5.4)
        log_config=None,  # structlog owns the logging configuration
        access_log=False,  # one access line per request comes from our middleware
        server_header=False,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()
