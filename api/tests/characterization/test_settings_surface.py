"""Characterization (R0): the ``TRINET_*`` environment surface of ``Settings``.

``settings.surface.json`` pins, for every field, the environment variable, the type, the
default and the constraints (ge/gt/le/lt/min_length/...). The pin is a SUPERSET contract:
renamed or removed variables, changed defaults and tightened or loosened constraints are
detected, new variables are allowed. The behaviour that operators rely on is pinned next to
it: how each variable is parsed, the bounds, and ``resolve_api_key`` (fail closed, minimum
length, secret file versus environment, whitespace stripping).

Regenerate the JSON with ``UPDATE_SNAPSHOTS=1 pytest`` (same convention as the OpenAPI snapshot).
"""

from __future__ import annotations

import json
import os
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError
from pydantic.fields import FieldInfo

# Update these imports if packages move (refactor R1/R2): the only place this file names them.
from trinet_api.domain.entities import ModelRegistry
from trinet_api.interface.http.app import create_app
from trinet_api.settings import MIN_API_KEY_LENGTH, ConfigurationError, Settings

PIN = Path(__file__).with_name("settings.surface.json")
UPDATE = os.environ.get("UPDATE_SNAPSHOTS") == "1"
KEY = "k" * 16
CONFIG_KEYS = (
    "env_prefix",
    "extra",
    "frozen",
    "case_sensitive",
    "env_file",
    "env_ignore_empty",
    "env_nested_delimiter",
    "secrets_dir",
)
CONSTRAINT_ATTRIBUTES = (
    "ge",
    "gt",
    "le",
    "lt",
    "min_length",
    "max_length",
    "multiple_of",
    "pattern",
)

#: Text to put in the variable -> the Python value that ``Settings`` must expose.
SAMPLES: dict[str, tuple[str, Any]] = {
    "host": ("10.1.2.3", "10.1.2.3"),
    "port": ("1234", 1234),
    "metrics_enabled": ("false", False),
    "metrics_host": ("127.0.0.9", "127.0.0.9"),
    "metrics_port": ("4321", 4321),
    "artifacts_dir": ("/srv/models", Path("/srv/models")),
    "api_key_file": ("/srv/key", Path("/srv/key")),
    "api_key": ("sample-key-0123456789", SecretStr("sample-key-0123456789")),
    "log_level": ("debug", "DEBUG"),
    "max_concurrency": ("3", 3),
    "queue_timeout_s": ("1.5", 1.5),
    "ort_intra_op_threads": ("2", 2),
    "rate_limit": ("5/second", "5/second"),
    "max_batch": ("10", 10),
    "max_json_bytes": ("2048", 2048),
    "max_upload_bytes": ("4096", 4096),
    "max_image_pixels": ("5", 5),
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.upper().startswith("TRINET_"):
            monkeypatch.delenv(name)


# --------------------------------------------------------------------------- the pin


def type_name(annotation: Any) -> str:
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in typing.get_args(annotation)) + "]"
    if origin in (typing.Union, types.UnionType):
        return " | ".join(type_name(a) for a in typing.get_args(annotation))
    if annotation is type(None):
        return "None"
    return str(getattr(annotation, "__name__", annotation))


def plain(value: Any) -> Any:
    """A JSON-friendly rendering of a default."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SecretStr):
        return "<secret>"
    return value


def constraints(field: FieldInfo) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for item in field.metadata:
        for attribute in CONSTRAINT_ATTRIBUTES:
            if hasattr(item, attribute):
                found[attribute] = getattr(item, attribute)
    return found


def current_surface() -> dict[str, Any]:
    prefix = Settings.model_config["env_prefix"]
    return {
        "model_config": {key: Settings.model_config.get(key) for key in CONFIG_KEYS},
        "min_api_key_length": MIN_API_KEY_LENGTH,
        "fields": {
            name: {
                "env": f"{prefix}{name}".upper(),
                "type": type_name(field.annotation),
                "required": field.is_required(),
                "default": plain(field.default),
                "constraints": constraints(field),
            }
            for name, field in Settings.model_fields.items()
        },
    }


def pinned_surface() -> dict[str, Any]:
    if UPDATE:
        PIN.write_text(json.dumps(current_surface(), indent=2, sort_keys=True) + "\n")
    pinned: dict[str, Any] = json.loads(PIN.read_text())
    return pinned


def test_the_environment_surface_is_a_superset_of_the_pinned_one() -> None:
    pinned, current = pinned_surface(), current_surface()
    problems = []
    for name, expected in pinned["fields"].items():
        found = current["fields"].get(name)
        if found is None:
            problems.append(f"{expected['env']} ({name}) was removed or renamed")
        elif found != expected:
            problems.append(f"{expected['env']} changed: {expected} -> {found}")
    assert not problems, "\n".join(problems) + "\n(UPDATE_SNAPSHOTS=1 accepts the change)"
    assert current["model_config"] == pinned["model_config"]
    assert current["min_api_key_length"] == pinned["min_api_key_length"] == 16


def test_the_pin_lists_every_variable_and_its_expected_shape() -> None:
    """Hand-written copy of the essentials, so the JSON cannot silently rot."""
    fields = pinned_surface()["fields"]
    assert {n: f["env"] for n, f in fields.items() if n in SAMPLES} == {
        n: f"TRINET_{n.upper()}" for n in SAMPLES
    }
    assert fields["port"]["default"] == 8000
    assert fields["port"]["constraints"] == {"ge": 1, "le": 65535}
    assert fields["metrics_port"]["default"] == 9100
    assert fields["api_key"]["type"] == "SecretStr | None"
    assert fields["api_key"]["default"] is None
    assert fields["api_key_file"]["default"] == "/run/secrets/api_key"
    assert fields["log_level"]["type"] == "Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']"
    assert fields["queue_timeout_s"]["constraints"] == {"gt": 0, "le": 60}
    assert fields["max_json_bytes"]["default"] == 1024 * 1024
    assert fields["max_upload_bytes"]["default"] == 6 * 1024 * 1024
    assert all(not f["required"] for f in fields.values()), "every variable has a default"


def test_the_samples_cover_every_pinned_variable() -> None:
    missing = set(pinned_surface()["fields"]) - set(SAMPLES)
    assert not missing, f"add SAMPLES entries for the new variables: {sorted(missing)}"


# --------------------------------------------------------------------------- how it is read


@pytest.mark.parametrize("field", sorted(SAMPLES))
def test_each_variable_feeds_exactly_its_own_field(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    baseline = Settings()
    text, expected = SAMPLES[field]
    monkeypatch.setenv(f"TRINET_{field.upper()}", text)
    settings = Settings()
    assert getattr(settings, field) == expected
    for other in SAMPLES:
        if other != field:
            assert getattr(settings, other) == getattr(baseline, other), other


def test_defaults_without_any_environment() -> None:
    settings = Settings()
    expected = {
        name: field["default"]
        for name, field in pinned_surface()["fields"].items()
        if name != "api_key"
    }
    assert settings.model_dump(mode="json", exclude={"api_key"}) == expected
    assert settings.api_key is None


def test_bounds_follow_the_pinned_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every pinned numeric bound is enforced at its boundary, through the environment."""
    problems: list[str] = []
    for name, field in pinned_surface()["fields"].items():
        limits = field["constraints"]
        unsupported = set(limits) - {"ge", "gt", "le"}
        assert not unsupported, f"teach this test about {unsupported} ({name})"
        cast = float if field["type"] == "float" else int
        accepted: list[float] = []
        rejected: list[float] = []
        if "ge" in limits:
            accepted.append(limits["ge"])
            rejected.append(limits["ge"] - 1)
        if "gt" in limits:
            rejected.append(limits["gt"])
            accepted.append(limits["gt"] + (0.5 if cast is float else 1))
        if "le" in limits:
            accepted.append(limits["le"])
            rejected.append(limits["le"] + (0.5 if cast is float else 1))
        for value in accepted:
            monkeypatch.setenv(field["env"], str(cast(value)))
            if getattr(Settings(), name) != cast(value):
                problems.append(f"{field['env']}={value} should be accepted")
        for value in rejected:
            monkeypatch.setenv(field["env"], str(cast(value)))
            try:
                Settings()
            except ValidationError:
                continue
            problems.append(f"{field['env']}={value} should be rejected")
        monkeypatch.delenv(field["env"], raising=False)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", ["TRINET_PORT", "TRINET_MAX_BATCH", "TRINET_METRICS_PORT"])
@pytest.mark.parametrize("value", ["", "http", "12x"])
def test_unparsable_numbers_fail_at_start_up(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("text", ["0", "false", "False", "FALSE", "no", "off"])
def test_metrics_can_be_switched_off_with_the_usual_spellings(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("TRINET_METRICS_ENABLED", text)
    assert Settings().metrics_enabled is False


@pytest.mark.parametrize("text", ["1", "true", "True", "yes", "on"])
def test_metrics_can_be_switched_on_with_the_usual_spellings(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("TRINET_METRICS_ENABLED", text)
    assert Settings().metrics_enabled is True


@pytest.mark.parametrize("text", ["maybe", ""])
def test_an_unreadable_boolean_is_refused_not_guessed(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("TRINET_METRICS_ENABLED", text)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("text", "expected"), [("debug", "DEBUG"), ("Info", "INFO"), ("WARNING", "WARNING")]
)
def test_log_level_is_case_insensitive_and_normalised_to_upper_case(
    monkeypatch: pytest.MonkeyPatch, text: str, expected: str
) -> None:
    monkeypatch.setenv("TRINET_LOG_LEVEL", text)
    assert Settings().log_level == expected


@pytest.mark.parametrize("text", ["warn", "trace", "", "10"])
def test_unknown_log_levels_are_refused(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setenv("TRINET_LOG_LEVEL", text)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("text", ["60/minute", "5/second; 100/hour", "10/second;5/minute", "1/day"])
def test_valid_rate_limit_expressions_are_kept_verbatim(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("TRINET_RATE_LIMIT", text)
    assert Settings().rate_limit == text


@pytest.mark.parametrize("text", ["lots", "60/fortnight", "", "-1/second", "60"])
def test_invalid_rate_limit_expressions_are_refused(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv("TRINET_RATE_LIMIT", text)
    with pytest.raises(ValidationError):
        Settings()


def test_only_the_prefixed_names_are_read_and_unknown_ones_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORT", "1")
    monkeypatch.setenv("HOST", "elsewhere")
    monkeypatch.setenv("API_KEY", "not-this-one-0123456789")
    monkeypatch.setenv("TRINET_THIS_IS_NOT_A_SETTING", "1")  # extra="ignore": no startup failure
    settings = Settings()
    assert (settings.port, settings.host, settings.api_key) == (8000, "0.0.0.0", None)  # noqa: S104


def test_variable_names_are_matched_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("trinet_port", "9001")
    assert Settings().port == 9001


def test_settings_are_immutable_and_never_print_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRINET_API_KEY", KEY)
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.port = 1  # type: ignore[misc]
    for rendering in (repr(settings), str(settings), settings.model_dump_json()):
        assert KEY not in rendering


# --------------------------------------------------------------------------- resolve_api_key


def key_file(tmp_path: Path, content: str | bytes) -> Path:
    path = tmp_path / "api_key"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_the_minimum_length_is_16_characters() -> None:
    assert MIN_API_KEY_LENGTH == 16
    assert issubclass(ConfigurationError, RuntimeError)


def test_no_key_anywhere_fails_closed(tmp_path: Path) -> None:
    for settings in (
        Settings(api_key_file=None),
        Settings(api_key_file=tmp_path / "missing"),
        Settings(api_key_file=tmp_path),  # a directory is not a secret file
    ):
        with pytest.raises(ConfigurationError, match="no API key configured"):
            settings.resolve_api_key()


def test_a_blank_environment_key_fails_closed() -> None:
    for blank in ("", " ", "\n", " \t\r\n "):
        with pytest.raises(ConfigurationError, match="no API key configured"):
            Settings(api_key_file=None, api_key=SecretStr(blank)).resolve_api_key()


@pytest.mark.parametrize("source", ["env", "file"])
def test_the_length_limit_is_16_after_stripping(tmp_path: Path, source: str) -> None:
    def resolve(secret: str) -> SecretStr:
        if source == "env":
            return Settings(api_key_file=None, api_key=SecretStr(secret)).resolve_api_key()
        return Settings(api_key_file=key_file(tmp_path, secret)).resolve_api_key()

    assert resolve("a" * 16).get_secret_value() == "a" * 16
    assert resolve(f"  {'a' * 16} \n").get_secret_value() == "a" * 16
    for short in ("a" * 15, f"  {'a' * 15}  \n", "x"):
        with pytest.raises(ConfigurationError, match="at least 16 characters"):
            resolve(short)


def test_the_secret_file_is_stripped_of_line_endings_and_blanks_but_not_inside(
    tmp_path: Path,
) -> None:
    expected = {
        "k" * 20 + "\n": "k" * 20,
        "k" * 20 + "\r\n": "k" * 20,
        "k" * 20 + "\n\n\n": "k" * 20,
        "\t " + "k" * 20 + " \n": "k" * 20,
        "k" * 8 + " " + "k" * 8 + "\n": "k" * 8 + " " + "k" * 8,
        "k" * 8 + "\n" + "k" * 8 + "\n": "k" * 8 + "\n" + "k" * 8,
    }
    for content, secret in expected.items():
        settings = Settings(api_key_file=key_file(tmp_path, content))
        assert settings.resolve_api_key().get_secret_value() == secret, repr(content)


def test_the_secret_file_wins_over_the_environment_even_when_it_is_unusable(tmp_path: Path) -> None:
    good_env = SecretStr("e" * 20)
    winner = Settings(api_key_file=key_file(tmp_path, "f" * 20 + "\n"), api_key=good_env)
    assert winner.resolve_api_key().get_secret_value() == "f" * 20
    for unusable, message in (("\n", "no API key"), ("short\n", "at least")):
        settings = Settings(api_key_file=key_file(tmp_path, unusable), api_key=good_env)
        with pytest.raises(ConfigurationError, match=message):  # no silent fallback to the env
            settings.resolve_api_key()


def test_the_environment_is_used_when_there_is_no_usable_file(tmp_path: Path) -> None:
    for file in (None, tmp_path / "missing", tmp_path):
        settings = Settings(api_key_file=file, api_key=SecretStr(f" {KEY}\n"))
        assert settings.resolve_api_key().get_secret_value() == KEY


def test_an_empty_secret_file_variable_falls_back_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose files often define ``TRINET_API_KEY_FILE=`` (empty): it becomes ``Path('.')``."""
    monkeypatch.setenv("TRINET_API_KEY_FILE", "")
    monkeypatch.setenv("TRINET_API_KEY", KEY)
    assert Settings().resolve_api_key().get_secret_value() == KEY


def test_the_resolved_key_is_a_secret_that_hides_itself(tmp_path: Path) -> None:
    resolved = Settings(api_key_file=key_file(tmp_path, KEY)).resolve_api_key()
    assert isinstance(resolved, SecretStr)
    assert KEY not in repr(resolved)
    assert KEY not in str(resolved)


def test_the_default_secret_file_is_the_docker_secret_location() -> None:
    assert Settings().api_key_file == Path("/run/secrets/api_key")


def test_create_app_fails_closed_before_serving_without_a_key(tmp_path: Path) -> None:
    """The factory raises at construction time, not at the first request."""
    with pytest.raises(ConfigurationError, match="no API key configured"):
        create_app(Settings(api_key_file=None), registry=ModelRegistry({}))
    with pytest.raises(ConfigurationError, match="at least 16 characters"):
        create_app(
            Settings(api_key_file=None, api_key=SecretStr("short")), registry=ModelRegistry({})
        )
