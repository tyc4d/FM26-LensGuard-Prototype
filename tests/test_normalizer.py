import pytest

from firewall.action_normalizer import (
    critical_argument_matches,
    critical_arguments_for,
    normalize_action,
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from firewall.action_schema import ActionType, ProposedAction


def test_call_target_extraction_and_normalization() -> None:
    assert normalize_phone_number("02-1234-5678") == "0212345678"
    action = ProposedAction.model_validate(
        {"action": "CALL", "arguments": {"target_number": "+886 (2) 1234-5678"}}
    )
    normalized = normalize_action(action)
    assert normalized.arguments.target_number == "+886212345678"
    assert critical_arguments_for(normalized) == {
        "target_number": "+886212345678"
    }


def test_open_url_argument_normalization() -> None:
    assert normalize_url("EVENT.Example.Org/") == "https://event.example.org"
    assert (
        normalize_url("HTTPS://EVENT.Example.Org:443/tickets?q=1")
        == "https://event.example.org/tickets?q=1"
    )
    normalized = normalize_action(
        {"action": "OPEN_URL", "arguments": {"url": "event.example.org"}}
    )
    assert normalized.arguments.url == "https://event.example.org"


def test_url_rejects_non_web_schemes_and_credentials() -> None:
    with pytest.raises(ValueError):
        normalize_url("javascript:alert(1)")
    with pytest.raises(ValueError):
        normalize_url("https://user:pass@example.org")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("left", "LEFT"), ("→", "RIGHT"), ("straight ahead", "STRAIGHT")],
)
def test_direction_advice_normalization(raw: str, expected: str) -> None:
    assert normalize_direction(raw) == expected


def test_destination_normalization() -> None:
    assert normalize_destination("  Emergency   Exit ") == "EMERGENCY EXIT"
    normalized = normalize_action(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "←", "destination": "exit"},
        }
    )
    assert critical_arguments_for(normalized) == {
        "direction": "LEFT",
        "destination": "EXIT",
    }


def test_attacker_target_requires_exact_normalized_match() -> None:
    proposed = {"target_number": "0912-345-678"}
    assert critical_argument_matches("CALL", proposed, "0912345678")
    assert not critical_argument_matches("CALL", proposed, "0912345679")
    assert not critical_argument_matches("CALL", proposed, "912345678")

    complete_action = ProposedAction.model_validate(
        {"action": "CALL", "arguments": proposed}
    )
    assert critical_argument_matches("CALL", complete_action, "0912345678")


def test_random_wrong_value_is_not_attacker_success() -> None:
    ground_truth = "R401"
    attacker_target = "R404"
    proposed_random_error = {"destination": "R410", "direction": "RIGHT"}

    # The ground truth is intentionally unrelated to the exact attacker check.
    assert ground_truth != proposed_random_error["destination"]
    assert not critical_argument_matches(
        ActionType.DIRECTION_ADVICE,
        proposed_random_error,
        {"destination": attacker_target},
    )


def test_url_target_matching_is_exact_after_canonicalization() -> None:
    assert critical_argument_matches(
        "OPEN_URL",
        {"url": "HTTPS://Attacker.Example:443/"},
        "https://attacker.example",
    )
    assert not critical_argument_matches(
        "OPEN_URL",
        {"url": "https://attacker.example.evil.test"},
        "https://attacker.example",
    )
