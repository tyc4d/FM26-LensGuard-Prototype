"""Frozen scientific inputs shared by both cloud providers and future pilot ingestion.

The cloud JSON-schema wrapper only closes known argument-reference map shapes
for provider compatibility. Prompts, action definitions, registries and the
downstream scientific validators remain the frozen Phase 3.5/3.6 contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_phase2 import load_phase2_dataset
from benchmark_phase3_5 import _build_registry
from phase2_benchmark_lock import sha256_file, verify_phase2_benchmark_lock
from phase3_5_constants import CRITICAL_ARGUMENTS
from phase3_5_schema import GroundedActionOutput, Phase35ActionOutput
from provenance.evidence_registry_phase3_5 import EvidenceRegistry
from providers.base_cloud_vlm import CloudRequest
from providers.local.phase3_5_adapter import (
    build_phase3_5_action_only_prompt,
    build_phase3_5_grounded_prompt,
)

ROOT = Path(__file__).resolve().parent
FROZEN_HEAD = "667d68f7b046ea159cde0187557f530c819f086a"
ARMS = ("ACTION_ONLY", "GROUNDED")
SMOKE_CASE_IDS = (
    "p2_call_hotel__clean_trusted",
    "p2_url_summit__clean_trusted",
    "p2_direction_exit__clean_trusted",
)
DATASET_PATH = ROOT / "dataset_phase2/metadata.json"


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def load_cases(smoke: bool = False) -> list[dict[str, Any]]:
    """Verify the locked corpus before returning the same 81 cases or fixed 3."""
    verify_phase2_benchmark_lock()
    _, scenarios = load_phase2_dataset(DATASET_PATH)
    ids = [scenario["scenario_id"] for scenario in scenarios]
    if len(ids) != 81 or len(set(ids)) != 81:
        raise ValueError("Cloud benchmark requires exactly the frozen 81 distinct cases")
    for scenario in scenarios:
        expected_image = (ROOT / str(scenario["image_path"])).resolve()
        if Path(scenario["_resolved_image_path"]).resolve() != expected_image:
            raise ValueError("Cloud corpus image resolution differs from the locked repository")
    if smoke:
        indexed = {scenario["scenario_id"]: scenario for scenario in scenarios}
        if not set(SMOKE_CASE_IDS) <= set(indexed):
            raise ValueError("Fixed smoke case is missing from the frozen corpus")
        return [indexed[case_id] for case_id in SMOKE_CASE_IDS]
    return scenarios


def _validate_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError("Cloud arm must be ACTION_ONLY or GROUNDED")


def shared_response_schema(arm: str) -> dict[str, Any]:
    """One schema for both APIs; evidence IDs remain unconstrained strings.

    OpenAI strict structured output requires every object to have known keys,
    additionalProperties=false and all properties required. Six closed reference
    map alternatives preserve every supported action's existing argument keys.
    Cross-field action/argument/reference validation stays in the shared frozen
    evaluator, and invented IDs remain observable scientific output.
    """
    _validate_arm(arm)
    output_model = Phase35ActionOutput if arm == "ACTION_ONLY" else GroundedActionOutput
    schema = copy.deepcopy(output_model.model_json_schema())
    if arm == "GROUNDED":
        reference_value = schema["properties"]["argument_evidence_refs"]["additionalProperties"]
        schema["properties"]["argument_evidence_refs"] = {
            "title": "Argument Evidence Refs",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {name: copy.deepcopy(reference_value) for name in arguments},
                    "required": list(arguments),
                    "additionalProperties": False,
                }
                for arguments in CRITICAL_ARGUMENTS.values()
            ],
        }

    def close_objects(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for child in node.values():
                close_objects(child)
        elif isinstance(node, list):
            for child in node:
                close_objects(child)

    close_objects(schema)
    return schema


def build_cloud_request(
    image_path: str | Path,
    user_prompt: str,
    arm: str,
    registry: EvidenceRegistry | None = None,
    case_id: str = "",
) -> CloudRequest:
    """Accept an existing image and independently built registry, including pilot inputs.

    Physical annotation ingestion uses Phase36PhysicalAnnotationRegistryAdapter
    before calling this function. No capture, perception inference, authenticity
    verification, or scientific physical result is produced here.
    """
    _validate_arm(arm)
    if arm == "ACTION_ONLY":
        if registry is not None:
            raise ValueError("ACTION_ONLY must not receive an evidence registry")
        prompt = build_phase3_5_action_only_prompt(user_prompt)
    else:
        if not isinstance(registry, EvidenceRegistry):
            raise ValueError("GROUNDED requires an independently constructed EvidenceRegistry")
        prompt = build_phase3_5_grounded_prompt(user_prompt, registry.as_model_input())
    return CloudRequest(
        image_path=Path(image_path), prompt=prompt, response_schema=shared_response_schema(arm),
        case_id=case_id, arm=arm,
    )


def prepare_case(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a case's provider-independent image, prompts and evidence before inference."""
    case_id = str(scenario["scenario_id"])
    image_path = Path(str(scenario.get("_resolved_image_path") or ROOT / scenario["image_path"]))
    relative_image = image_path.resolve().relative_to(ROOT).as_posix()
    registry = _build_registry(scenario)
    snapshot = registry.model_dump(include_dataset_labels=True)
    user_prompt = str(scenario["user_prompt"])
    requests = {
        arm: build_cloud_request(
            image_path, user_prompt, arm, registry if arm == "GROUNDED" else None, case_id,
        )
        for arm in ARMS
    }
    prompts = {arm: request.prompt for arm, request in requests.items()}
    return {
        "case_id": case_id,
        "image_path": relative_image,
        "image_sha256": sha256_file(image_path),
        "user_prompt": user_prompt,
        "registry_snapshot": snapshot,
        "model_registry": registry.as_model_input(),
        "registry_sha256": sha256_json(snapshot),
        "prompts": prompts,
        "prompt_sha256": {
            arm: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for arm, prompt in prompts.items()
        },
        "schemas": {arm: request.response_schema for arm, request in requests.items()},
    }
