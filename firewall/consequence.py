"""Provenance-blind consequence baseline.

The consequence predictor is advisory.  This baseline intentionally cannot use
the action argument's source, so it is unable to distinguish an ordinary medium-
risk call from the same call pointed at an attacker-selected number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .action_schema import ConsequencePrediction, Decision, Severity


def consequence_only_decision(
    consequence: ConsequencePrediction | Mapping[str, Any],
    policy_config: Mapping[str, Any] | None = None,
) -> Decision:
    """Return the deterministic provenance-blind baseline decision.

    Semantics are deliberately clear and stable: ``high`` advisory severity
    requires confirmation; ``low`` and ``medium`` are allowed.  No effect name,
    action argument, scene field, or provenance label is inspected.
    """

    prediction = (
        consequence
        if isinstance(consequence, ConsequencePrediction)
        else ConsequencePrediction.model_validate(consequence)
    )
    baseline = (policy_config or {}).get("consequence_only", {})
    high_decision = baseline.get("high_severity_decision", Decision.CONFIRM.value)
    otherwise_decision = baseline.get("otherwise_decision", Decision.ALLOW.value)
    return Decision(high_decision if prediction.severity is Severity.HIGH else otherwise_decision)
