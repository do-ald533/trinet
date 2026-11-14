"""12-factor configuration (``TRINET_*`` environment variables)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from limits import parse_many
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_API_KEY_LENGTH = 16


class ConfigurationError(RuntimeError):
    """The service cannot start safely with the given configuration."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRINET_", extra="ignore", frozen=True)

    host: str = "0.0.0.0"  # noqa: S104 - container entrypoint, published by the orchestrator
    port: Annotated[int, Field(ge=1, le=65535)] = 8000
    metrics_enabled: bool = True
    metrics_host: str = "0.0.0.0"  # noqa: S104 - internal network only, port is never published
    metrics_port: Annotated[int, Field(ge=1, le=65535)] = 9100

    artifacts_dir: Path = Path("/app/artifacts")
    # The secret file wins over the environment variable (Docker/compose secrets).
    api_key_file: Path | None = Path("/run/secrets/api_key")
    api_key: SecretStr | None = None

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    max_concurrency: Annotated[int, Field(ge=1, le=64)] = 4
    queue_timeout_s: Annotated[float, Field(gt=0, le=60)] = 2.0
    ort_intra_op_threads: Annotated[int, Field(ge=1, le=64)] = 1
    rate_limit: str = "60/minute"
    max_batch: Annotated[int, Field(ge=1, le=10_000)] = 256
    max_json_bytes: Annotated[int, Field(ge=1_024)] = 1024 * 1024
    max_upload_bytes: Annotated[int, Field(ge=1_024)] = 6 * 1024 * 1024
    max_image_pixels: Annotated[int, Field(ge=1)] = 12_000_000

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("rate_limit")
    @classmethod
    def _valid_rate_limit(cls, value: str) -> str:
        try:
            parse_many(value)
        except ValueError as exc:
            raise ValueError(f"invalid rate limit {value!r} (expected e.g. '60/minute')") from exc
        return value

    def resolve_api_key(self) -> SecretStr:
        """The API key (file first, then ``TRINET_API_KEY``); fail closed if there is none."""
        secret: str | None = None
        if self.api_key_file is not None and self.api_key_file.is_file():
            secret = self.api_key_file.read_text(encoding="utf-8").strip()
        elif self.api_key is not None:
            secret = self.api_key.get_secret_value().strip()
        if not secret:
            raise ConfigurationError(
                "no API key configured: mount a secret at TRINET_API_KEY_FILE "
                "(default /run/secrets/api_key) or set TRINET_API_KEY"
            )
        if len(secret) < MIN_API_KEY_LENGTH:
            raise ConfigurationError(
                f"the API key must have at least {MIN_API_KEY_LENGTH} characters"
            )
        return SecretStr(secret)
