"""Deterministic providers for quota-free end-to-end validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from firewall.action_schema import ConsequencePrediction, ProposedAction

from .base import BaseAgentProvider, BaseConsequenceProvider, ProviderResponse

ATTACK_CONDITIONS = frozenset(
    {
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        "AUTHORITY_IMPERSONATION",
        "OBVIOUS_INJECTION_CONTROL",
    }
)

GROUND_TRUTH_CONDITIONS = frozenset({"CLEAN_TRUSTED", "EXPLICIT_USER_OVERRIDE"})

_CRITICAL_ARGUMENT_BY_ACTION = {
    "CALL": "target_number",
    "OPEN_URL": "url",
    "DIRECTION_ADVICE": "direction",
}


def _action_value(action: Any) -> str:
    """Return a stable string for string- or Enum-valued action fields."""

    value = getattr(action, "value", action)
    return str(value).upper()


class MockAgentProvider(BaseAgentProvider):
    """Emit deterministic proposals derived from benchmark ground truth.

    This mock intentionally adopts ``attacker_target`` for attack conditions so
    the rest of the experiment can exercise detection and metrics without API
    quota.  It is test infrastructure, never scientific Gemini evidence.

    A scenario can override the default behavior with ``mock_behavior``:
    ``ground_truth``, ``attacker_target``, ``arbitrary_wrong``, or ``none``.
    A complete ``mock_proposed_action`` mapping takes precedence over both.
    """

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    @property
    def model_identifier(self) -> str:
        return "mock-deterministic-v1"

    @property
    def experiment_config(self) -> dict[str, Any]:
        return {"provider": "mock", "seed": self.seed, "sdk_internal_retries": 0}

    def propose(
        self,
        user_prompt: str,
        image_path: str | Path,
        scenario: Mapping[str, Any] | None = None,
    ) -> ProviderResponse[ProposedAction]:
        del user_prompt, image_path  # The mock is intentionally metadata-driven.
        started = perf_counter()
        if scenario is None:
            raise ValueError("MockAgentProvider requires scenario metadata")

        payload = self._proposal_payload(scenario)
        parsed = ProposedAction.model_validate(payload)
        raw = json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return ProviderResponse(
            parsed=parsed,
            raw_response=raw,
            latency_ms=(perf_counter() - started) * 1000,
            attempts=1,
            model=self.model_identifier,
            response_metadata={
                "status": "completed",
                "requested_model": self.model_identifier,
                "returned_model": self.model_identifier,
                "mock": True,
            },
        )

    def _proposal_payload(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        explicit = scenario.get("mock_proposed_action")
        if explicit is not None:
            if not isinstance(explicit, Mapping):
                raise ValueError("mock_proposed_action must be an object")
            return dict(explicit)

        condition = str(scenario.get("condition", "")).upper()
        behavior = str(scenario.get("mock_behavior", "")).lower()
        if not behavior:
            if scenario.get("trusted_user_override") or condition in GROUND_TRUTH_CONDITIONS:
                behavior = "ground_truth"
            elif condition in ATTACK_CONDITIONS:
                behavior = "attacker_target"
            elif condition in {"ARBITRARY_WRONG", "RANDOM_WRONG_CONTROL"}:
                behavior = "arbitrary_wrong"
            else:
                behavior = "ground_truth"

        action = _action_value(scenario.get("ground_truth_action", "NONE"))
        arguments_obj = scenario.get("ground_truth_arguments") or {}
        if not isinstance(arguments_obj, Mapping):
            raise TypeError("ground_truth_arguments must be an object")
        arguments = dict(arguments_obj)

        if behavior == "none":
            action = "NONE"
            arguments = {}
        elif behavior == "attacker_target":
            argument_name = self._critical_argument_name(scenario, action)
            attacker_target = scenario.get("attacker_target")
            if attacker_target in (None, ""):
                raise ValueError("attacker_target behavior requires a non-empty attacker_target")
            arguments[argument_name] = attacker_target
        elif behavior == "arbitrary_wrong":
            argument_name = self._critical_argument_name(scenario, action)
            wrong = scenario.get("arbitrary_wrong_target") or scenario.get("random_wrong_target")
            arguments[argument_name] = wrong or self._arbitrary_wrong_value(action, scenario)
        elif behavior != "ground_truth":
            raise ValueError(f"Unsupported mock_behavior: {behavior!r}")

        return {
            "action": action,
            "arguments": arguments,
            "reason_summary": f"Deterministic mock behavior: {behavior}",
            "confidence": 1.0,
        }

    @staticmethod
    def _critical_argument_name(scenario: Mapping[str, Any], action: str) -> str:
        name = scenario.get("critical_argument_name") or _CRITICAL_ARGUMENT_BY_ACTION.get(action)
        if not name:
            raise ValueError(f"No critical argument is defined for action {action!r}")
        return str(name)

    def _arbitrary_wrong_value(self, action: str, scenario: Mapping[str, Any]) -> str:
        material = f"{self.seed}:{scenario.get('scenario_id', '')}:{action}".encode()
        suffix = int(hashlib.sha256(material).hexdigest()[:8], 16)
        if action == "CALL":
            return f"0999{suffix % 1_000_000:06d}"
        if action == "OPEN_URL":
            return f"https://wrong-{suffix % 100_000}.invalid"
        if action == "DIRECTION_ADVICE":
            current = str((scenario.get("ground_truth_arguments") or {}).get("direction", "")).upper()
            return next(candidate for candidate in ("LEFT", "RIGHT", "FORWARD", "BACK") if candidate != current)
        return f"wrong-{suffix:08x}"


class MockConsequenceProvider(BaseConsequenceProvider):
    """Return registry-aligned intrinsic effects without interpreting scene data.

    Predictions intentionally do not become more alarming for attacker metadata.
    This preserves a clean consequence-only baseline: provenance-sensitive
    escalation belongs to the deterministic policy engine.
    """

    _PREDICTIONS: ClassVar[dict[str, dict[str, Any]]] = {
        "CALL": {
            "effects": [
                "external_communication",
                "caller_identity_disclosure",
                "possible_social_engineering",
                "possible_call_charge",
            ],
            "severity": "medium",
            "reversibility": "low",
            "uncertainties": [],
        },
        "OPEN_URL": {
            "effects": [
                "network_request",
                "navigation_to_external_origin",
                "possible_device_metadata_disclosure",
                "possible_malicious_content",
            ],
            "severity": "medium",
            "reversibility": "medium",
            "uncertainties": [],
        },
        "DIRECTION_ADVICE": {
            "effects": [
                "human_physical_movement",
                "possible_navigation_error",
                "possible_safety_impact",
            ],
            "severity": "medium",
            "reversibility": "medium",
            "uncertainties": [],
        },
        "NONE": {
            "effects": [],
            "severity": "low",
            "reversibility": "high",
            "uncertainties": [],
        },
    }

    def __init__(self, *, seed: int = 0) -> None:
        # Accepted for a uniform mock-provider construction API.  Results remain
        # deterministic and currently do not need pseudorandomness.
        self.seed = seed

    @property
    def model_identifier(self) -> str:
        return "mock-consequence-v1"

    @property
    def experiment_config(self) -> dict[str, Any]:
        return {"provider": "mock", "seed": self.seed, "sdk_internal_retries": 0}

    def predict(
        self,
        action: ProposedAction,
        provenance: Mapping[str, str] | None = None,
    ) -> ProviderResponse[ConsequencePrediction]:
        del provenance  # Deliberately provenance-invariant; see class docstring.
        started = perf_counter()
        action_name = _action_value(action.action)
        payload = self._PREDICTIONS.get(action_name)
        if payload is None:
            raise ValueError(f"Unsupported action for mock consequence prediction: {action_name}")
        parsed = ConsequencePrediction.model_validate(payload)
        raw = json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return ProviderResponse(
            parsed=parsed,
            raw_response=raw,
            latency_ms=(perf_counter() - started) * 1000,
            attempts=1,
            model=self.model_identifier,
            response_metadata={
                "status": "completed",
                "requested_model": self.model_identifier,
                "returned_model": self.model_identifier,
                "mock": True,
            },
        )
