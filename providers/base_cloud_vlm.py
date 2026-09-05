"""Auditable cloud transport, independent of LensGuard scientific scoring.

Adapters make one semantic attempt. Only transport failures can resend the
serialized request, and SDK retries are disabled by each concrete adapter.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import os
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from providers.base import ProviderConfigurationError, error_status_code
from providers.gemini_agent import _read_image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECRET_ENV_NAMES = ("OPENAI_API_KEY", "GEMINI_API_KEY")
TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def require_secret_safety(repository_root: Path = REPOSITORY_ROOT) -> None:
    """Refuse live construction unless root .env is ignored and untracked."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], cwd=repository_root,
        capture_output=True, check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--", ".env"], cwd=repository_root,
        capture_output=True, check=False,
    )
    if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout.strip():
        raise ProviderConfigurationError(
            "SECRET_SAFETY_FAILED: .env must be gitignored and untracked before API use"
        )


def _secret_values() -> tuple[str, ...]:
    values = [os.getenv(name, "") for name in SECRET_ENV_NAMES]
    # Reading here is only for full-value redaction; it never loads credentials.
    from dotenv import dotenv_values

    local = dotenv_values(REPOSITORY_ROOT / ".env")
    values.extend(local.get(name) or "" for name in SECRET_ENV_NAMES)
    return tuple(sorted({value for value in values if value}, key=len, reverse=True))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    # Never stringify arbitrary SDK objects: their repr may contain credentials.
    return {"unserialized_type": type(value).__name__}


def redact_secrets(value: Any) -> Any:
    secrets = _secret_values()

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
                # Authentication services sometimes echo the prefix with a masked tail.
                # Derive patterns privately; neither prefixes nor full values are logged.
                prefix = secret[:8]
                if len(prefix) == 8:
                    item = re.sub(
                        re.escape(prefix) + r"[A-Za-z0-9_.*…-]*", "[REDACTED]", item,
                    )
            item = re.sub(
                r"\b(?:sk-[A-Za-z0-9_.*…-]{4,}|AIza[A-Za-z0-9_.*…-]{4,})",
                "[REDACTED]", item,
            )
            return item
        if isinstance(item, Mapping):
            return {visit(str(key)): visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        return item

    return visit(jsonable(value))


def disable_sdk_debug_logging() -> None:
    """Prevent SDK HTTP diagnostics from writing authorization headers."""
    for name in ("openai", "httpx", "httpcore", "google.genai", "google_genai"):
        logger = logging.getLogger(name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
    # Parent logger levels do not filter a child's already-created debug record.
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.startswith(
            ("openai.", "httpx.", "httpcore.", "google.genai.", "google_genai.")
        ):
            logger.disabled = True
            logger.setLevel(logging.CRITICAL)


@dataclass(frozen=True, slots=True)
class CloudRequest:
    image_path: Path
    prompt: str
    response_schema: dict[str, Any]
    case_id: str = ""
    arm: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Cloud request prompt must be nonempty")
        if not isinstance(self.response_schema, dict):
            raise ValueError("Cloud request schema must be a JSON object")
        object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "response_schema", copy.deepcopy(self.response_schema))

    def image_base64(self) -> tuple[str, str]:
        image_bytes, mime_type = _read_image(self.image_path)
        return base64.b64encode(image_bytes).decode("ascii"), mime_type


@dataclass(frozen=True, slots=True)
class CloudResponse:
    provider: str
    model: str
    model_id: str
    api_interface: str
    timestamp_utc: str
    raw_response: Any
    output_text: str
    completed: bool
    error_type: str | None
    error_detail: str | None
    http_status: int | None
    api_status: str | None
    request_id: str | None
    latency_ms: float
    usage: dict[str, Any]
    provider_config: dict[str, Any]
    transport_attempts: int
    rate_limit_events: int = 0
    total_backoff_seconds: float = 0.0
    stop_provider: bool = False
    hard_quota: bool = False
    returned_model: str | None = None
    transport_events: list[dict[str, Any]] = field(default_factory=list)
    raw_response_redacted: bool = False
    estimated_cost_usd: float | None = None
    cost_basis: str = "Unavailable: no request-reported cost or configured pricing"
    scientific_attempt: int = 1

    @property
    def transport_attempt_count(self) -> int:
        return self.transport_attempts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _error_body(error: BaseException) -> Any:
    body = getattr(error, "body", None)
    if body is not None:
        return jsonable(body)
    response = getattr(error, "response", None) or getattr(error, "raw_response", None)
    if response is not None:
        try:
            return jsonable(response.json())
        except Exception:
            text = getattr(response, "text", None)
            if isinstance(text, str):
                return {"body_text": text}
    return {"exception_type": type(error).__name__, "message": str(error)}


def classify_error(error: BaseException) -> tuple[str, bool, bool, bool]:
    """Return category, retryable, stop-provider, hard-quota without logging."""
    status = error_status_code(error)
    body = _error_body(error)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            pass
    message = (json.dumps(body, sort_keys=True) + " " + str(error)).casefold()
    # Daily limits and billing quotas cannot be fixed by the RPM retry loop.
    hard_quota = status == 429 and (any(part in message for part in (
        "per day", "per_day", "perday", "daily", "insufficient_quota",
        "billing_hard_limit", "hard quota", "hard_quota", "limit: 0",
        '"quota_value": "0"', '"quotavalue": "0"',
    )) or bool(re.search(r'"(?:quota_?value|limit)"\s*:\s*"?0(?:"|\s*[,}])', message)))
    if hard_quota:
        return "RATE_LIMIT_EXHAUSTED", False, True, True
    if status == 404 or any(part in message for part in (
        "model_not_found", "model not found", "model is not available",
        "model does not exist", "model is unavailable", "unsupported model",
    )):
        return "MODEL_UNAVAILABLE", False, True, False
    if status == 401 or any(part in message for part in (
        "api_key_invalid", "invalid_api_key", "invalid api key", "api key not valid",
    )):
        return "AUTHENTICATION_FAILED", False, True, False
    if status in {402, 403}:
        return "ACCESS_OR_BILLING_RESTRICTION", False, True, False
    if status == 429:
        return "RATE_LIMIT_EXHAUSTED", True, False, False
    timeout = isinstance(error, (TimeoutError, ConnectionError)) or any(
        word in type(error).__name__.casefold() for word in ("timeout", "connection")
    )
    if status in TRANSIENT_HTTP_STATUSES or timeout:
        return "TRANSPORT_ERROR", True, False, False
    if status in {400, 405, 415, 422} or isinstance(error, (TypeError, ValueError)):
        return "API_COMPATIBILITY_ERROR", False, True, False
    return "API_ERROR", False, True, False


class CloudVLMProvider(ABC):
    provider: str
    api_interface: str
    default_model: str
    model_environment: str
    transport_delays: tuple[float, ...]

    def __init__(
        self, *, model: str | None = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.model = model if model is not None else os.getenv(self.model_environment, self.default_model)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ProviderConfigurationError("Requested cloud model ID must be nonempty")
        self._sleep = sleep
        self._clock = clock
        self._utcnow = utcnow
        self._lock = threading.Lock()
        self._stopped = False
        self._initialization_error: str | None = None

    @property
    @abstractmethod
    def provider_config(self) -> dict[str, Any]: ...

    @abstractmethod
    def build_payload(self, request: CloudRequest) -> dict[str, Any]: ...

    @abstractmethod
    def _send(self, payload: dict[str, Any]) -> Any: ...

    @abstractmethod
    def _usage(self, response: Any) -> dict[str, Any]: ...

    def _before_request(self) -> None:
        pass

    def _after_success(self) -> None:
        pass

    def _bounded_sleep(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0:
            interval = min(remaining, 60.0)
            self._sleep(interval)
            remaining -= interval

    def infer(self, request: CloudRequest) -> CloudResponse:
        # Holding the lock across pacing and retries enforces Gemini concurrency=1.
        with self._lock:
            return self._infer_locked(request)

    def _infer_locked(self, request: CloudRequest) -> CloudResponse:
        if self._stopped:
            raise ProviderConfigurationError("Provider has stopped; no further requests are permitted")
        payload = self.build_payload(request)
        # Freeze once before first send. Every transport retry receives equal bytes/data.
        payload = json.loads(json.dumps(payload, sort_keys=True))
        self._before_request()
        started = self._clock()
        timestamp = self._utcnow().astimezone(timezone.utc).isoformat()
        events: list[dict[str, Any]] = []
        rate_limits = 0
        backoff = 0.0
        common: dict[str, Any] = {
            "provider": self.provider, "model": self.model, "model_id": self.model,
            "api_interface": self.api_interface, "timestamp_utc": timestamp,
            "provider_config": self.provider_config,
        }
        if self._initialization_error:
            self._stopped = True
            return CloudResponse(
                **common, raw_response={"error_type": self._initialization_error},
                output_text="", completed=False, error_type=self._initialization_error,
                error_detail=self._initialization_error, http_status=None, api_status=None,
                request_id=None, latency_ms=0.0, usage={}, transport_attempts=0,
                stop_provider=True,
            )
        for attempt in range(1, len(self.transport_delays) + 2):
            try:
                response = self._send(copy.deepcopy(payload))
            except Exception as error:
                category, retryable, stop, hard_quota = classify_error(error)
                status = error_status_code(error)
                rate_limits += int(status == 429)
                original_body = _error_body(error)
                body = redact_secrets(original_body)
                event = {
                    "transport_attempt": attempt, "http_status": status,
                    "error_type": category, "raw_error_response": body,
                }
                events.append(event)
                if retryable and attempt <= len(self.transport_delays):
                    delay = self.transport_delays[attempt - 1]
                    event["backoff_seconds"] = delay
                    self._bounded_sleep(delay)
                    backoff += delay
                    continue
                self._stopped = stop
                return CloudResponse(
                    **common, raw_response=body, output_text="", completed=False,
                    error_type=category, error_detail=category, http_status=status,
                    api_status=category, request_id=getattr(error, "request_id", None),
                    latency_ms=max(0.0, (self._clock() - started) * 1000), usage={},
                    transport_attempts=attempt, rate_limit_events=rate_limits,
                    total_backoff_seconds=backoff, stop_provider=stop, hard_quota=hard_quota,
                    transport_events=events, raw_response_redacted=body != jsonable(original_body),
                )
            # A returned model result is never retried, even if empty or malformed.
            self._after_success()
            original_envelope = jsonable(response)
            envelope = redact_secrets(original_envelope)
            output_text = _field(response, "output_text", "")
            output_text = output_text if isinstance(output_text, str) else ""
            status = str(_field(response, "status", "completed"))
            errors = _field(response, "errors") or _field(response, "error")
            completed = status == "completed" and not errors
            return CloudResponse(
                **common, raw_response=envelope, output_text=redact_secrets(output_text),
                completed=completed, error_type=None if completed else "MODEL_RESPONSE_INCOMPLETE",
                error_detail=None if completed else "MODEL_RESPONSE_INCOMPLETE",
                http_status=200, api_status=status,
                request_id=_field(response, "_request_id") or _field(response, "id"),
                latency_ms=max(0.0, (self._clock() - started) * 1000),
                usage=redact_secrets(self._usage(response)), transport_attempts=attempt,
                rate_limit_events=rate_limits, total_backoff_seconds=backoff,
                returned_model=_field(response, "model"), transport_events=events,
                raw_response_redacted=envelope != original_envelope,
            )
        raise AssertionError("Bounded transport loop did not terminate")

    def close(self) -> None:
        if getattr(self, "_owns_client", False):
            close = getattr(self.client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> CloudVLMProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
