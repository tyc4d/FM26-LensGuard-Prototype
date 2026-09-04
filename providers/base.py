"""Provider interfaces and common response/error types.

Providers only *propose* structured data.  Nothing in this package executes a
phone call, opens a URL, or performs navigation.
"""

from __future__ import annotations

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from firewall.action_schema import ConsequencePrediction, ProposedAction


ParsedT = TypeVar("ParsedT")
RETRY_POLICY_VERSION = "server-aware-retry-v1"


@dataclass(frozen=True, slots=True)
class ProviderResponse(Generic[ParsedT]):
    """A validated provider result together with its audit data.

    ``raw_response`` is the exact text returned by a remote model (or the exact
    synthetic JSON emitted by a mock).  Callers should persist it separately
    from ``parsed`` so post-hoc parsing and research audits remain possible.
    ``latency_ms`` covers retries and backoff, if any.
    """

    parsed: ParsedT
    raw_response: str
    latency_ms: float
    attempts: int = 1
    model: str | None = None
    response_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")


@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Client-backoff settings for Gemini calls.

    ``max_delay_seconds`` caps client-computed exponential backoff. A valid
    server ``Retry-After``/``RetryInfo`` delay remains a minimum and can exceed
    that client cap, up to the separate automatic-wait safety ceiling.
    """

    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    multiplier: float = 2.0
    max_delay_seconds: float = 16.0
    max_server_delay_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not math.isfinite(self.initial_delay_seconds) or self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be finite and non-negative")
        if not math.isfinite(self.multiplier) or self.multiplier < 1:
            raise ValueError("multiplier must be finite and at least 1")
        if not math.isfinite(self.max_delay_seconds) or self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.max_server_delay_seconds)
            or self.max_server_delay_seconds < 0
        ):
            raise ValueError("max_server_delay_seconds must be finite and non-negative")


class ProviderError(RuntimeError):
    """Base class for provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when a provider is missing required explicit configuration."""


class ProviderDependencyError(ProviderError):
    """Raised when the optional SDK needed by a provider is not installed."""


class ProviderResponseError(ProviderError):
    """Raised when a model response is empty or fails schema validation.

    ``raw_response`` remains available for the benchmark's error audit trail;
    invalid model output must never disappear merely because validation failed.
    """

    def __init__(
        self,
        message: str,
        *,
        raw_response: str | None = None,
        response_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.response_metadata = dict(response_metadata or {})


class ProviderUnavailableError(ProviderError):
    """Raised when the configured provider/model cannot complete a request."""


def require_flash_model(model: str) -> str:
    """Enforce the explicitly scoped Gemini Flash family without choosing a model."""

    candidate = model.strip()
    if not candidate:
        raise ProviderConfigurationError(
            "GEMINI_MODEL is required; LensGuard never silently selects a fallback model"
        )
    if _FLASH_MODEL_PATTERN.fullmatch(candidate) is None:
        raise ProviderConfigurationError(
            f"LensGuard Phase 1 requires a Gemini Flash-family model; got {candidate!r}"
        )
    return candidate


_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_RETRY_HINT_KEYS = frozenset({"retryafter", "retrydelay"})
_MESSAGE_KEYS = frozenset({"message", "localizedmessage"})
_DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|s)\s*$",
    re.IGNORECASE,
)
_RETRY_MESSAGE_PATTERN = re.compile(
    r"\bretry(?:ing)?(?:\s+again)?\s+(?:in|after)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|seconds?|secs?|s)\b",
    re.IGNORECASE,
)
_FLASH_MODEL_PATTERN = re.compile(
    r"^(?:models/)?gemini-(?:(?:\d+(?:\.\d+)*)-)?flash(?:-[a-z0-9.]+)*$",
    re.IGNORECASE,
)


def interactions_without_internal_retry(client: Any) -> Any:
    """Return the Interactions resource with SDK-internal retries disabled.

    google-genai 2.22.0's Interactions bridge retries independently of the
    parent client's documented HttpRetryOptions. The generated resource exposes
    no public per-call retry switch, so this pinned, regression-tested adapter
    sets its SDK configuration to single-attempt behavior. LensGuard's outer
    retry loop is then the sole source of retries and observable quota use.
    """

    interactions = client.interactions
    sdk_configuration = getattr(interactions, "sdk_configuration", None)
    is_google_resource = type(interactions).__module__.startswith("google.genai")
    if sdk_configuration is None or not hasattr(sdk_configuration, "retry_config"):
        if is_google_resource:
            raise ProviderDependencyError(
                "Installed google-genai Interactions internals do not expose the "
                "regression-tested retry control; refusing an unauditable quota bound"
            )
        return interactions
    sdk_configuration.retry_config = None
    if sdk_configuration.retry_config is not None:
        raise ProviderDependencyError("Could not disable google-genai Interactions retries")
    return interactions


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(getattr(value, "value", value))


def interaction_response_metadata(response: Any, *, requested_model: str) -> dict[str, Any]:
    """Validate an Interaction terminal state and build a redacted audit envelope."""

    raw_status = getattr(response, "status", None)
    status = str(getattr(raw_status, "value", raw_status)) if raw_status is not None else None
    errors = _jsonable(getattr(response, "errors", None)) or []
    returned_model_raw = getattr(response, "model", None)
    returned_model = (
        str(getattr(returned_model_raw, "value", returned_model_raw))
        if returned_model_raw is not None
        else None
    )
    envelope: dict[str, Any] = {}
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            envelope = dict(dumped)
            for sensitive_or_large in (
                "input",
                "system_instruction",
                "output_audio",
                "output_image",
                "output_video",
            ):
                envelope.pop(sensitive_or_large, None)

    metadata = {
        "interaction_id": getattr(response, "id", None),
        "status": status,
        "requested_model": requested_model,
        "returned_model": returned_model,
        "usage": _jsonable(getattr(response, "usage", None)),
        "created": getattr(response, "created", None),
        "updated": getattr(response, "updated", None),
        "errors": errors,
        "redacted_response_envelope": envelope,
    }
    raw_text = getattr(response, "output_text", None)
    if status != "completed":
        raise ProviderResponseError(
            f"Gemini interaction ended with status {status!r}",
            raw_response=raw_text if isinstance(raw_text, str) else None,
            response_metadata=metadata,
        )
    if errors:
        raise ProviderResponseError(
            "Gemini interaction returned diagnostic errors",
            raw_response=raw_text if isinstance(raw_text, str) else None,
            response_metadata=metadata,
        )
    return metadata


def error_status_code(error: BaseException) -> int | None:
    """Best-effort status extraction across google-genai/httpx exceptions."""

    for candidate in (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
        getattr(getattr(error, "raw_response", None), "status_code", None),
    ):
        try:
            if candidate is not None:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def is_retryable_error(error: BaseException) -> bool:
    """Return whether a failed request is likely transient."""

    status = error_status_code(error)
    if status is not None:
        return status in _RETRYABLE_STATUS_CODES
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    name = type(error).__name__.lower()
    return "timeout" in name or "connection" in name


def _valid_delay(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        delay = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(delay) or delay < 0:
        return None
    return delay


def _duration_seconds(value: Any) -> float | None:
    """Parse an HTTP delta or a google.protobuf.Duration-like value."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _valid_delay(value)
    if isinstance(value, str):
        text = value.strip()
        numeric = _valid_delay(text)
        if numeric is not None:
            return numeric
        match = _DURATION_PATTERN.fullmatch(text)
        if match is None:
            return None
        delay = _valid_delay(match.group("value"))
        if delay is None:
            return None
        if match.group("unit").casefold().startswith(("milli", "msec", "ms")):
            return delay / 1000.0
        return delay

    seconds: Any = None
    nanos: Any = 0
    if isinstance(value, Mapping):
        normalized = {
            re.sub(r"[^a-z]", "", str(key).casefold()): item for key, item in value.items()
        }
        seconds = normalized.get("seconds")
        nanos = normalized.get("nanos", normalized.get("nanoseconds", 0))
    elif hasattr(value, "seconds"):
        seconds = getattr(value, "seconds", None)
        nanos = getattr(value, "nanos", 0)
    if seconds is None:
        return None
    base = _valid_delay(seconds)
    nano_value = _valid_delay(nanos)
    if base is None or nano_value is None or nano_value >= 1_000_000_000:
        return None
    return base + nano_value / 1_000_000_000


def _message_retry_delays(message: Any) -> list[float]:
    if not isinstance(message, str):
        return []
    candidates: list[float] = []
    for match in _RETRY_MESSAGE_PATTERN.finditer(message):
        delay = _duration_seconds(f"{match.group('value')}{match.group('unit')}")
        if delay is not None:
            candidates.append(delay)
    return candidates


def _structured_retry_delays(value: Any, *, seen: set[int] | None = None) -> list[float]:
    """Read only explicitly named retry fields and developer-message retry instructions."""

    visited = seen if seen is not None else set()
    if isinstance(value, str):
        return _message_retry_delays(value)
    if not isinstance(value, (Mapping, list, tuple)):
        return []
    identity = id(value)
    if identity in visited:
        return []
    visited.add(identity)

    candidates: list[float] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized_key in _RETRY_HINT_KEYS:
                delay = _duration_seconds(item)
                if delay is not None:
                    candidates.append(delay)
            if normalized_key in _MESSAGE_KEYS:
                candidates.extend(_message_retry_delays(item))
            candidates.extend(_structured_retry_delays(item, seen=visited))
    else:
        for item in value:
            candidates.extend(_structured_retry_delays(item, seen=visited))
    return candidates


def _retry_after_header_seconds(value: Any, *, now: datetime) -> float | None:
    if value is None:
        return None
    delta = _duration_seconds(value)
    if delta is not None:
        return delta
    try:
        retry_at = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    return _valid_delay(max(0.0, (retry_at - current).total_seconds()))


def server_retry_delay_seconds(
    error: BaseException,
    *,
    now: datetime | None = None,
) -> float | None:
    """Extract the safest server-supplied retry delay from supported error shapes.

    Current Interactions errors expose ``body`` and ``response.headers`` while
    older google-genai errors expose ``details``. Gemini quota errors can also
    carry the only usable delay in a ``Please retry in 18.7s`` message. When
    multiple hints disagree, returning the largest avoids retrying earlier than
    any server instruction.
    """

    current = now or datetime.now(timezone.utc)
    candidates: list[float] = []
    header_owners = (
        error,
        getattr(error, "response", None),
        getattr(error, "raw_response", None),
    )
    seen_owners: set[int] = set()
    for owner in header_owners:
        if owner is None or id(owner) in seen_owners:
            continue
        seen_owners.add(id(owner))
        headers = getattr(owner, "headers", None)
        if headers is None:
            continue
        try:
            header_value = headers.get("retry-after")
        except (AttributeError, TypeError, ValueError):
            header_value = None
        if header_value is None and isinstance(headers, Mapping):
            for key, value in headers.items():
                if str(key).casefold() == "retry-after":
                    header_value = value
                    break
        header_delay = _retry_after_header_seconds(header_value, now=current)
        if header_delay is not None:
            candidates.append(header_delay)

    for value in (
        getattr(error, "body", None),
        getattr(error, "details", None),
        getattr(error, "message", None),
        str(error),
    ):
        candidates.extend(_structured_retry_delays(value))
    return max(candidates) if candidates else None


def _attach_retry_history(error: BaseException, history: list[dict[str, Any]]) -> None:
    """Best-effort audit context for a final exception without changing its type."""

    try:
        setattr(error, "lensguard_retry_history", tuple(dict(item) for item in history))
    except (AttributeError, TypeError):
        pass


def retry_audit_metadata(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build JSON-safe request metadata for the application's retry activity."""

    copied = [dict(event) for event in events]
    return {
        "policy_version": RETRY_POLICY_VERSION,
        "events": copied,
        "total_wait_seconds": sum(
            float(event.get("sleep_seconds", 0.0) or 0.0) for event in copied
        ),
    }


def call_with_retry(
    operation: Callable[[], ParsedT],
    *,
    config: RetryConfig,
    logger: logging.Logger,
    sleep: Callable[[float], None] = time.sleep,
    retry_events: list[dict[str, Any]] | None = None,
) -> tuple[ParsedT, int]:
    """Run one API operation with server-aware exponential backoff.

    The original exception is re-raised after the final attempt so provider
    wrappers can include their exact configured model in a clear error.  There
    is deliberately no model-switching logic here. Client-computed backoff is
    bounded by ``RetryConfig.max_delay_seconds``; a valid server-supplied delay
    is honored as a minimum even when it is longer. A hint above
    ``max_server_delay_seconds`` stops the operation instead of sleeping without
    bound or issuing a knowingly premature retry.
    """

    delay = min(config.initial_delay_seconds, config.max_delay_seconds)
    retry_history: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        retry_history.append(dict(event))
        if retry_events is not None:
            retry_events.append(dict(event))

    for attempt in range(1, config.max_attempts + 1):
        try:
            return operation(), attempt
        except Exception as error:
            retryable = is_retryable_error(error)
            status = error_status_code(error)
            server_delay = server_retry_delay_seconds(error)
            event: dict[str, Any] = {
                "attempt": attempt,
                "error_type": type(error).__name__,
                "http_status": status,
                "retryable": retryable,
                "server_retry_delay_seconds": server_delay,
            }
            if retryable and attempt >= config.max_attempts and status == 429:
                logger.error(
                    "Gemini rate limit persisted through %d attempt(s); giving up%s",
                    config.max_attempts,
                    (
                        f" (last server retry hint: {server_delay:.3f}s)"
                        if server_delay is not None
                        else ""
                    ),
                )
            if not retryable or attempt >= config.max_attempts:
                event["will_retry"] = False
                event["stop_reason"] = (
                    "not_retryable" if not retryable else "max_attempts_exhausted"
                )
                record(event)
                _attach_retry_history(error, retry_history)
                raise
            if (
                server_delay is not None
                and server_delay > config.max_server_delay_seconds
            ):
                event.update(
                    will_retry=False,
                    stop_reason="server_delay_exceeds_safety_ceiling",
                    max_server_delay_seconds=config.max_server_delay_seconds,
                )
                record(event)
                _attach_retry_history(error, retry_history)
                logger.error(
                    "Gemini requested a %.3fs retry delay, above the configured %.3fs "
                    "automatic-wait ceiling; stopping without an early retry",
                    server_delay,
                    config.max_server_delay_seconds,
                )
                raise
            wait_seconds = max(delay, server_delay or 0.0)
            event.update(will_retry=True, sleep_seconds=wait_seconds)
            record(event)
            if status == 429:
                if server_delay is not None:
                    logger.warning(
                        "Gemini rate limit encountered; retrying attempt %d/%d in %.3fs "
                        "(server hint %.3fs; client backoff %.3fs)",
                        attempt + 1,
                        config.max_attempts,
                        wait_seconds,
                        server_delay,
                        delay,
                    )
                else:
                    logger.warning(
                        "Gemini rate limit encountered; retrying attempt %d/%d in %.3fs "
                        "(no server retry hint)",
                        attempt + 1,
                        config.max_attempts,
                        wait_seconds,
                    )
            else:
                logger.warning(
                    "Transient Gemini error%s; retrying attempt %d/%d in %.3fs%s",
                    f" (HTTP {status})" if status is not None else "",
                    attempt + 1,
                    config.max_attempts,
                    wait_seconds,
                    (
                        f" (server hint {server_delay:.3f}s)"
                        if server_delay is not None
                        else ""
                    ),
                )
            sleep(wait_seconds)
            delay = min(delay * config.multiplier, config.max_delay_seconds)
    raise AssertionError("unreachable")


class BaseAgentProvider(ABC):
    """Interface for a dry-run multimodal action-proposal provider."""

    @property
    @abstractmethod
    def model_identifier(self) -> str:
        """Return the exact configured or mock model identifier."""

    @abstractmethod
    def propose(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[ProposedAction]:
        """Interpret a prompt and image, then return a proposed action."""


class BaseConsequenceProvider(ABC):
    """Interface for an advisory, sanitized consequence predictor."""

    @property
    @abstractmethod
    def model_identifier(self) -> str:
        """Return the exact configured or mock model identifier."""

    @abstractmethod
    def predict(
        self,
        action: ProposedAction,
        provenance: Mapping[str, str] | None = None,
    ) -> ProviderResponse[ConsequencePrediction]:
        """Predict effects from structured action data only.

        Implementations must not accept an image or raw scene text.  Omitting
        ``provenance`` is how the consequence-only baseline is evaluated.
        """
