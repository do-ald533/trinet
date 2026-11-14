from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trinet_api.settings import MIN_API_KEY_LENGTH, ConfigurationError, Settings

KEY = "k" * MIN_API_KEY_LENGTH


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.startswith("TRINET_"):
            monkeypatch.delenv(name)


def test_defaults_match_the_deployment_contract() -> None:
    settings = Settings()
    assert settings.host == "0.0.0.0"  # noqa: S104
    assert settings.port == 8000
    assert settings.metrics_port == 9100
    assert settings.artifacts_dir == Path("/app/artifacts")
    assert settings.api_key_file == Path("/run/secrets/api_key")
    assert settings.max_concurrency == 4
    assert settings.ort_intra_op_threads == 1
    assert settings.rate_limit == "60/minute"
    assert settings.max_batch == 256
    assert settings.log_level == "INFO"


def test_environment_variables_use_the_trinet_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRINET_PORT", "9000")
    monkeypatch.setenv("TRINET_METRICS_PORT", "9200")
    monkeypatch.setenv("TRINET_ARTIFACTS_DIR", "/data/models")
    monkeypatch.setenv("TRINET_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("TRINET_LOG_LEVEL", "debug")
    monkeypatch.setenv("TRINET_RATE_LIMIT", "5/second; 100/hour")
    monkeypatch.setenv("PORT", "1")  # unprefixed names are ignored
    settings = Settings()
    assert (settings.port, settings.metrics_port) == (9000, 9200)
    assert settings.artifacts_dir == Path("/data/models")
    assert settings.max_concurrency == 2
    assert settings.log_level == "DEBUG"
    assert settings.rate_limit == "5/second; 100/hour"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TRINET_PORT", "0"),
        ("TRINET_PORT", "70000"),
        ("TRINET_PORT", "http"),
        ("TRINET_MAX_CONCURRENCY", "0"),
        ("TRINET_MAX_BATCH", "0"),
        ("TRINET_QUEUE_TIMEOUT_S", "-1"),
        ("TRINET_LOG_LEVEL", "chatty"),
        ("TRINET_RATE_LIMIT", "lots"),
        ("TRINET_RATE_LIMIT", "60/fortnight"),
    ],
)
def test_invalid_values_fail_at_start_up(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


def test_the_secret_file_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = tmp_path / "api_key"
    secret.write_text("  from-the-file-0123456789\n")
    monkeypatch.setenv("TRINET_API_KEY", KEY)
    monkeypatch.setenv("TRINET_API_KEY_FILE", str(secret))
    assert Settings().resolve_api_key().get_secret_value() == "from-the-file-0123456789"


def test_the_environment_is_used_when_the_file_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TRINET_API_KEY", f" {KEY} ")
    monkeypatch.setenv("TRINET_API_KEY_FILE", str(tmp_path / "missing"))
    assert Settings().resolve_api_key().get_secret_value() == KEY


def test_no_key_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRINET_API_KEY_FILE", str(tmp_path / "missing"))
    with pytest.raises(ConfigurationError, match="no API key"):
        Settings().resolve_api_key()


def test_an_empty_secret_file_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    secret = tmp_path / "api_key"
    secret.write_text("\n")
    monkeypatch.setenv("TRINET_API_KEY_FILE", str(secret))
    monkeypatch.setenv("TRINET_API_KEY", KEY)  # the file wins even when it is empty
    with pytest.raises(ConfigurationError, match="no API key"):
        Settings().resolve_api_key()


def test_short_keys_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TRINET_API_KEY_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("TRINET_API_KEY", "short")
    with pytest.raises(ConfigurationError, match="at least"):
        Settings().resolve_api_key()


def test_the_key_never_appears_in_representations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRINET_API_KEY", KEY)
    settings = Settings()
    assert KEY not in repr(settings)
    assert KEY not in str(settings)
    assert KEY not in settings.model_dump_json()
    assert KEY not in repr(settings.resolve_api_key())


def test_settings_are_immutable() -> None:
    with pytest.raises(ValidationError):
        Settings().port = 1  # type: ignore[misc]
