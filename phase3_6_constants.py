"""Version identifiers for LensGuard Phase 3.6.

Phase 3.6 is additive.  In particular, the action-model contract and Evidence
Registry remain the frozen Phase 3.5 contracts; only deterministic grounding,
uncertainty interpretation, and gate policy receive Phase 3.6 versions.
"""

from __future__ import annotations

from typing import Final

from phase3_5_constants import (
    EVIDENCE_SCHEMA_VERSION as PHASE3_5_EVIDENCE_SCHEMA_VERSION,
    MODEL_CONTRACT_VERSION as PHASE3_5_MODEL_CONTRACT_VERSION,
)


EXPERIMENT_VERSION: Final = "lensguard-phase3.6-uncertainty-aware-v1"
GROUNDING_SCHEMA_VERSION: Final = "phase3.6-grounding-v1"
UNCERTAINTY_SCHEMA_VERSION: Final = "phase3.6-evidence-uncertainty-v1"
GATE_POLICY_VERSION: Final = "phase3.6-safe-escalation-gate-v1"
PHYSICAL_DATASET_SCHEMA_VERSION: Final = "phase3.6-physical-dataset-v1"
ESCALATION_SCHEMA_VERSION: Final = "phase3.6-structured-escalation-v1"

# The VLM input/output contract and registry schema deliberately do not change
# in Phase 3.6.  Keeping their historical identifiers makes replay provenance
# explicit and avoids implying that a model rerun is scientifically required.
ACTION_MODEL_CONTRACT_VERSION: Final = PHASE3_5_MODEL_CONTRACT_VERSION
EVIDENCE_REGISTRY_SCHEMA_VERSION: Final = PHASE3_5_EVIDENCE_SCHEMA_VERSION
