"""Domain errors. The HTTP layer maps each one to an RFC 9457 problem (see ``problems.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Issue:
    """One validation problem. Never carries the offending value (it may be sensitive)."""

    loc: tuple[str | int, ...]
    code: str
    message: str


class TriNetError(Exception):
    """Base class of every error the API raises on purpose."""

    code: ClassVar[str] = "internal_error"


class ModelNotFound(TriNetError):
    code = "model_not_found"

    def __init__(self, name: str) -> None:
        super().__init__(f"model {name!r} does not exist")
        self.name = name


class ModelNotAvailable(TriNetError):
    """The model exists in ``artifacts/`` but could not be loaded (the others keep serving)."""

    code = "model_not_available"

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"model {name!r} is not available ({reason})")
        self.name = name
        self.reason = reason


class InvalidInput(TriNetError):
    code = "invalid_input"

    def __init__(self, issues: tuple[Issue, ...] | list[Issue]) -> None:
        super().__init__(f"{len(issues)} validation error(s)")
        self.issues = tuple(issues)


class UnsupportedInput(TriNetError):
    """The payload kind or file format is not one the model accepts (HTTP 415)."""

    code = "unsupported_media_type"


class PayloadTooLarge(TriNetError):
    code = "payload_too_large"


class Overloaded(TriNetError):
    """No inference slot became free within the queue timeout (HTTP 503 + Retry-After)."""

    code = "overloaded"

    def __init__(self, retry_after_s: int) -> None:
        super().__init__("the service is busy, retry later")
        self.retry_after_s = retry_after_s


class InferenceFailed(TriNetError):
    """The runtime raised, or produced an output that violates the manifest."""

    code = "inference_failed"


class NotReady(TriNetError):
    """No model can serve yet (readiness probe)."""

    code = "not_ready"
