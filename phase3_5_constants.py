"""Version and vocabulary constants for LensGuard Phase 3.5.

This module is intentionally independent of every Phase 2 module.  Phase 3.5
is an additive experiment and must never change the identifiers or contracts
used to produce the frozen Phase 2 and Phase 2.5 baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


EXPERIMENT_VERSION: Final = "lensguard-phase3.5-grounded-provenance-v1"
EVIDENCE_SCHEMA_VERSION: Final = "phase3.5-evidence-registry-v1"
MODEL_CONTRACT_VERSION: Final = "phase3.5-grounded-action-v1"
POLICY_VERSION: Final = "phase3.5-grounded-gate-v1"
ACTION_REGISTRY_VERSION: Final = "phase3.5-action-registry-v1"
PHYSICAL_DATASET_SCHEMA_VERSION: Final = "phase3.5-physical-dataset-v1"
ACTION_ONLY_PROMPT_VERSION: Final = "phase3.5-action-only-v1"
GROUNDED_ACTION_PROMPT_VERSION: Final = "phase3.5-grounded-action-prompt-v2"
LOCAL_SCHEMA_TRANSPORT_VERSION_PHASE3_5: Final = (
    "phase3.5-local-json-schema-transport-v1"
)
METRICS_VERSION: Final = "phase3.5-metrics-v1"
RUNNER_VERSION: Final = "phase3.5-runner-v1"
REPORT_VERSION: Final = "phase3.5-report-v1"


class Phase35ActionType(StrEnum):
    CALL = "CALL"
    OPEN_URL = "OPEN_URL"
    DIRECTION_ADVICE = "DIRECTION_ADVICE"
    SAFETY_ADVICE = "SAFETY_ADVICE"
    RESTAURANT_RESERVATION = "RESTAURANT_RESERVATION"
    NONE = "NONE"


CRITICAL_ARGUMENTS: Final = MappingProxyType(
    {
        Phase35ActionType.CALL: ("target_number",),
        Phase35ActionType.OPEN_URL: ("url",),
        Phase35ActionType.DIRECTION_ADVICE: ("direction", "destination"),
        Phase35ActionType.SAFETY_ADVICE: ("safe_to_proceed", "hazard"),
        Phase35ActionType.RESTAURANT_RESERVATION: (
            "restaurant",
            "target_number",
            "time",
            "party_size",
        ),
        Phase35ActionType.NONE: (),
    }
)


class ScenarioFamily(StrEnum):
    CALL = "CALL"
    NAVIGATION = "NAVIGATION"
    SAFETY = "SAFETY"
    RESTAURANT_RESERVATION = "RESTAURANT_RESERVATION"


class PhysicalConditionId(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    C5 = "C5"
    C6 = "C6"


class LightingClass(StrEnum):
    BRIGHT = "bright"
    DIM = "dim"


class AttackPosition(StrEnum):
    TOP_RIGHT = "TR"
    BOTTOM_LEFT = "BL"


@dataclass(frozen=True, slots=True)
class PhysicalConditionSpec:
    condition_id: PhysicalConditionId
    distance_m: float
    camera_angle_deg: int
    lighting_class: LightingClass
    attack_position: AttackPosition


_CONDITION_SPECS = {
    PhysicalConditionId.C0: PhysicalConditionSpec(
        PhysicalConditionId.C0, 1.5, 0, LightingClass.BRIGHT, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C1: PhysicalConditionSpec(
        PhysicalConditionId.C1, 1.0, 0, LightingClass.BRIGHT, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C2: PhysicalConditionSpec(
        PhysicalConditionId.C2, 2.0, 0, LightingClass.BRIGHT, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C3: PhysicalConditionSpec(
        PhysicalConditionId.C3, 2.5, 0, LightingClass.BRIGHT, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C4: PhysicalConditionSpec(
        PhysicalConditionId.C4, 1.5, 45, LightingClass.BRIGHT, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C5: PhysicalConditionSpec(
        PhysicalConditionId.C5, 1.5, 0, LightingClass.DIM, AttackPosition.TOP_RIGHT
    ),
    PhysicalConditionId.C6: PhysicalConditionSpec(
        PhysicalConditionId.C6, 1.5, 0, LightingClass.BRIGHT, AttackPosition.BOTTOM_LEFT
    ),
}
PHYSICAL_CONDITION_SPECS: Final = MappingProxyType(_CONDITION_SPECS)


PHYSICAL_BASE_SCENES: Final = (
    "CALL-01",
    "CALL-02",
    "CALL-03",
    "CALL-04",
    "NAV-01",
    "NAV-02",
    "NAV-03",
    "NAV-04",
    "SAFE-01",
    "SAFE-02",
    "SAFE-03",
    "SAFE-04",
    "RESTAURANT-01",
    "RESTAURANT-02",
    "RESTAURANT-03",
    "RESTAURANT-04",
)

SCENE_FAMILY_PREFIX: Final = MappingProxyType(
    {
        "CALL": ScenarioFamily.CALL,
        "NAV": ScenarioFamily.NAVIGATION,
        "SAFE": ScenarioFamily.SAFETY,
        "RESTAURANT": ScenarioFamily.RESTAURANT_RESERVATION,
    }
)


class EvidenceContentType(StrEnum):
    TEXT = "text"
    OBJECT = "object"
    SPATIAL = "spatial"
    SYMBOL = "symbol"
    OTHER = "other"
    USER_INPUT = "user_input"


class RegistryOrigin(StrEnum):
    BENCHMARK_ANNOTATION = "benchmark_annotation"
    USER_PROMPT = "user_prompt"
    AUTOMATIC_PERCEPTION = "automatic_perception"
    PHYSICAL_ANNOTATION = "physical_annotation"


class EvidencePhysicalSource(StrEnum):
    """Physical/source vocabulary available to new Phase 3.5 records.

    ``explicit_user`` is restricted to non-camera user evidence.  The
    physical-dataset region vocabulary itself uses the eight values specified
    by :class:`PhysicalSource` in ``phase3_5_dataset_schema``.
    """

    ORIGINAL_PACKAGING = "original_packaging"
    OFFICIAL_SIGN = "official_sign"
    ENVIRONMENT_OBJECT = "environment_object"
    ATTACKER_STICKER = "attacker_sticker"
    ATTACKER_PAPER = "attacker_paper"
    RESTAURANT_MATERIAL = "restaurant_material"
    USER_INPUT = "user_input"
    EXPLICIT_USER = "explicit_user"
    OTHER = "other"


class ControlClass(StrEnum):
    LEGITIMATE = "legitimate"
    ATTACKER_CONTROLLED = "attacker_controlled"
    NEUTRAL = "neutral"


EXPECTED_PHYSICAL_IMAGE_COUNT: Final = (
    len(PHYSICAL_BASE_SCENES) * len(PHYSICAL_CONDITION_SPECS)
)
