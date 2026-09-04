"""Public API for the Phase 1 LensGuard action firewall."""

from .action_normalizer import (
    attacker_target_matches,
    critical_argument_matches,
    critical_arguments_for,
    normalize_action,
    normalize_destination,
    normalize_direction,
    normalize_phone_number,
    normalize_url,
)
from .action_schema import (
    ActionType,
    CallArguments,
    ConsequencePrediction,
    Decision,
    DirectionAdviceArguments,
    EmptyArguments,
    FirewallDecision,
    OpenURLArguments,
    ProposedAction,
    ProvenanceSource,
    Reversibility,
    Severity,
)
from .consequence import consequence_only_decision
from .policy_engine import evaluate_policy, load_policy, validate_registry_policy
from .provenance import OracleProvenance, load_oracle_provenance

__all__ = [
    "ActionType",
    "CallArguments",
    "ConsequencePrediction",
    "Decision",
    "DirectionAdviceArguments",
    "EmptyArguments",
    "FirewallDecision",
    "OpenURLArguments",
    "OracleProvenance",
    "ProposedAction",
    "ProvenanceSource",
    "Reversibility",
    "Severity",
    "attacker_target_matches",
    "consequence_only_decision",
    "critical_argument_matches",
    "critical_arguments_for",
    "evaluate_policy",
    "load_oracle_provenance",
    "load_policy",
    "validate_registry_policy",
    "normalize_action",
    "normalize_destination",
    "normalize_direction",
    "normalize_phone_number",
    "normalize_url",
]
