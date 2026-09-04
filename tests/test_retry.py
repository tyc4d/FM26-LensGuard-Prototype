from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from providers.base import RetryConfig, call_with_retry, server_retry_delay_seconds


class _RetryableError(Exception):
    status_code = 429

    def __init__(
        self,
        message: str = "rate limited",
        *,
        body: Any = None,
        details: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.body = body
        self.details = details
        self.response = SimpleNamespace(status_code=429, headers=headers or {})


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan")])
def test_retry_config_rejects_unsafe_server_wait_ceiling(value: float) -> None:
    with pytest.raises(ValueError, match="max_server_delay_seconds"):
        RetryConfig(max_server_delay_seconds=value)


def test_extracts_live_gemini_retry_message_with_fractional_seconds() -> None:
    error = _RetryableError(
        body={
            "error": {
                "message": (
                    "Quota exceeded for generate_content_free_tier_requests.\n"
                    "Please retry in 18.745623466s."
                ),
                "code": "too_many_requests",
            }
        }
    )

    assert server_retry_delay_seconds(error) == pytest.approx(18.745623466)


def test_extracts_current_and_legacy_structured_retry_info() -> None:
    current = _RetryableError(
        body={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "4.561901217s",
                    }
                ]
            }
        }
    )
    legacy = _RetryableError(
        details={"retry_delay": {"seconds": 3, "nanos": 250_000_000}}
    )

    assert server_retry_delay_seconds(current) == pytest.approx(4.561901217)
    assert server_retry_delay_seconds(legacy) == pytest.approx(3.25)


def test_extracts_retry_after_delta_and_http_date() -> None:
    now = datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc)
    delta = _RetryableError(headers={"Retry-After": "7.5"})
    dated = _RetryableError(headers={"Retry-After": "Fri, 04 Sep 2026 12:00:19 GMT"})

    assert server_retry_delay_seconds(delta, now=now) == pytest.approx(7.5)
    assert server_retry_delay_seconds(dated, now=now) == pytest.approx(19.0)


def test_safest_of_conflicting_server_hints_is_used() -> None:
    error = _RetryableError(
        "Please retry in 6s.",
        body={"retryDelay": "9.25s"},
        headers={"Retry-After": "8"},
    )

    assert server_retry_delay_seconds(error) == pytest.approx(9.25)


@pytest.mark.parametrize(
    "value",
    ["not a duration", "-2", "NaN", "inf", {"seconds": -1, "nanos": 0}],
)
def test_malformed_retry_hints_are_ignored(value: Any) -> None:
    assert server_retry_delay_seconds(_RetryableError(body={"retryDelay": value})) is None


def test_server_delay_is_a_minimum_even_above_client_backoff_cap() -> None:
    error = _RetryableError("Please retry in 18.745623466s.")
    outcomes: list[Any] = [error, "ok"]
    sleeps: list[float] = []
    retry_events: list[dict[str, Any]] = []

    def operation() -> str:
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    result, attempts = call_with_retry(
        operation,
        config=RetryConfig(
            max_attempts=2,
            initial_delay_seconds=1,
            max_delay_seconds=16,
            max_server_delay_seconds=30,
        ),
        logger=logging.getLogger("test.retry.server-delay"),
        sleep=sleeps.append,
        retry_events=retry_events,
    )

    assert result == "ok"
    assert attempts == 2
    assert sleeps == pytest.approx([18.745623466])
    assert retry_events[0]["server_retry_delay_seconds"] == pytest.approx(18.745623466)
    assert retry_events[0]["sleep_seconds"] == pytest.approx(18.745623466)


def test_server_delay_above_safety_ceiling_stops_without_early_retry() -> None:
    error = _RetryableError("Please retry in 301s.")
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(_RetryableError) as caught:
        call_with_retry(
            operation,
            config=RetryConfig(max_attempts=3, max_server_delay_seconds=300),
            logger=logging.getLogger("test.retry.ceiling"),
            sleep=sleeps.append,
        )

    assert calls == 1
    assert sleeps == []
    assert caught.value.lensguard_retry_history[0]["stop_reason"] == (
        "server_delay_exceeds_safety_ceiling"
    )


def test_server_hint_never_shortens_client_backoff() -> None:
    outcomes: list[Any] = [_RetryableError("Please retry in 0.1s."), "ok"]
    sleeps: list[float] = []

    def operation() -> str:
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    call_with_retry(
        operation,
        config=RetryConfig(max_attempts=2, initial_delay_seconds=2),
        logger=logging.getLogger("test.retry.minimum"),
        sleep=sleeps.append,
    )

    assert sleeps == [2]


def test_server_hint_does_not_mutate_later_client_backoff_schedule() -> None:
    outcomes: list[Any] = [
        _RetryableError("Please retry in 10s."),
        _RetryableError("no hint"),
        "ok",
    ]
    sleeps: list[float] = []

    def operation() -> str:
        result = outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    call_with_retry(
        operation,
        config=RetryConfig(
            max_attempts=3,
            initial_delay_seconds=1,
            multiplier=2,
            max_delay_seconds=16,
        ),
        logger=logging.getLogger("test.retry.independent-backoff"),
        sleep=sleeps.append,
    )

    assert sleeps == [10, 2]


def test_missing_hint_falls_back_and_exhaustion_keeps_retry_audit() -> None:
    first = _RetryableError("no parseable hint")
    final = _RetryableError("Please retry in 12.5s.")
    outcomes: list[Any] = [first, final]
    sleeps: list[float] = []

    def operation() -> str:
        raise outcomes.pop(0)

    with pytest.raises(_RetryableError) as caught:
        call_with_retry(
            operation,
            config=RetryConfig(max_attempts=2, initial_delay_seconds=0.75),
            logger=logging.getLogger("test.retry.audit"),
            sleep=sleeps.append,
        )

    assert sleeps == [0.75]
    history = caught.value.lensguard_retry_history
    assert len(history) == 2
    assert history[0]["sleep_seconds"] == pytest.approx(0.75)
    assert history[1]["will_retry"] is False
    assert history[1]["server_retry_delay_seconds"] == pytest.approx(12.5)
