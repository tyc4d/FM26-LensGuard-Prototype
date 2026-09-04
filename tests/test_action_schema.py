import pytest
from pydantic import ValidationError

from firewall.action_schema import (
    ActionType,
    CallArguments,
    ConsequencePrediction,
    DirectionAdviceArguments,
    FirewallDecision,
    ProposedAction,
)


def test_call_action_uses_typed_arguments() -> None:
    action = ProposedAction.model_validate(
        {
            "action": "CALL",
            "arguments": {"target_number": "02-1234-5678"},
            "reason_summary": "Call the front desk",
            "confidence": 0.91,
        }
    )

    assert action.action is ActionType.CALL
    assert isinstance(action.arguments, CallArguments)
    assert action.arguments.target_number == "02-1234-5678"


def test_action_specific_arguments_reject_wrong_or_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ProposedAction.model_validate(
            {"action": "CALL", "arguments": {"url": "https://example.org"}}
        )
    with pytest.raises(ValidationError):
        ProposedAction.model_validate(
            {
                "action": "OPEN_URL",
                "arguments": {"url": "https://example.org", "surprise": True},
            }
        )


def test_direction_destination_may_be_unknown_but_direction_is_required() -> None:
    action = ProposedAction.model_validate(
        {"action": "DIRECTION_ADVICE", "arguments": {"direction": "LEFT"}}
    )
    assert isinstance(action.arguments, DirectionAdviceArguments)
    assert action.arguments.destination is None

    with pytest.raises(ValidationError):
        ProposedAction.model_validate(
            {"action": "DIRECTION_ADVICE", "arguments": {"destination": "EXIT"}}
        )


def test_none_forbids_tool_arguments() -> None:
    action = ProposedAction.model_validate({"action": "NONE", "arguments": {}})
    assert action.action is ActionType.NONE
    with pytest.raises(ValidationError):
        ProposedAction.model_validate(
            {"action": "NONE", "arguments": {"target_number": "0912345678"}}
        )


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        ProposedAction.model_validate(
            {"action": "NONE", "arguments": {}, "confidence": 1.01}
        )


def test_consequence_schema_validation() -> None:
    prediction = ConsequencePrediction.model_validate(
        {
            "effects": ["external_communication"],
            "severity": "medium",
            "reversibility": "low",
            "uncertainties": [],
        }
    )
    assert prediction.severity.value == "medium"

    with pytest.raises(ValidationError):
        ConsequencePrediction.model_validate(
            {
                "effects": ["external_communication"],
                "severity": "catastrophic",
                "reversibility": "low",
                "uncertainties": [],
            }
        )


def test_firewall_decision_serializes_to_requested_shape() -> None:
    decision = FirewallDecision.model_validate(
        {
            "decision": "CONFIRM",
            "action": "CALL",
            "critical_arguments": {"target_number": "0912345678"},
            "provenance": {"target_number": "camera_unverified"},
            "predicted_consequences": ["external_communication"],
            "policy_rules_triggered": ["CALL_UNVERIFIED_TARGET"],
            "user_message": "Confirm before calling.",
        }
    )
    assert decision.model_dump(mode="json")["decision"] == "CONFIRM"

