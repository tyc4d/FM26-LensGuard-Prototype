"""Cloud input parity and physical-ingestion software checks; no API calls."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import cloud_baseline_contracts as contracts
from benchmark_phase3_5 import _build_registry
from phase3_5_constants import CRITICAL_ARGUMENTS
from phase3_5_schema import GroundedActionOutput, Phase35ActionOutput
from provenance.physical_pilot_phase3_6 import Phase36PhysicalAnnotationRegistryAdapter
from providers.base_cloud_vlm import CloudRequest
from providers.gemini_vlm import GeminiProvider
from providers.openai_vlm import OpenAIProvider


@pytest.fixture(scope="module")
def cases():
    return contracts.load_cases()


@pytest.fixture(scope="module")
def prepared(cases):
    return {case["scenario_id"]: contracts.prepare_case(case) for case in cases}


def test_all81_case_image_and_user_task_identities_match_frozen_head(cases, prepared):
    frozen = subprocess.run(
        ["git", "show", f"{contracts.FROZEN_HEAD}:dataset_phase2/metadata.json"],
        cwd=contracts.ROOT, capture_output=True, check=True,
    )
    originals = json.loads(frozen.stdout)["records"]
    assert len(cases) == len(prepared) == len(originals) == 81
    assert [case["scenario_id"] for case in cases] == [case["scenario_id"] for case in originals]
    for current, original in zip(cases, originals, strict=True):
        assert {key: value for key, value in current.items() if key != "_resolved_image_path"} == original
        audited = prepared[current["scenario_id"]]
        assert audited["image_path"] == original["image_path"]
        assert audited["user_prompt"] == original["user_prompt"]
        assert audited["image_sha256"] == hashlib.sha256(
            (contracts.ROOT / original["image_path"]).read_bytes()
        ).hexdigest()


def test_predeclared_smoke_set_is_three_fixed_families():
    selected = contracts.load_cases(smoke=True)
    assert tuple(case["scenario_id"] for case in selected) == contracts.SMOKE_CASE_IDS
    assert [case["action_family"] for case in selected] == ["CALL", "OPEN_URL", "DIRECTION_ADVICE"]
    assert contracts.ARMS == ("ACTION_ONLY", "GROUNDED")
    assert len(selected) * len(contracts.ARMS) == 6


def test_corpus_lock_is_verified_before_loading(monkeypatch):
    def reject():
        raise RuntimeError("intentional lock rejection")

    monkeypatch.setattr(contracts, "verify_phase2_benchmark_lock", reject)
    monkeypatch.setattr(contracts, "load_phase2_dataset", lambda _: pytest.fail("loader ran after bad lock"))
    with pytest.raises(RuntimeError, match="intentional lock rejection"):
        contracts.load_cases()


def test_partial_or_duplicate_cohort_is_rejected(cases, monkeypatch):
    monkeypatch.setattr(contracts, "verify_phase2_benchmark_lock", lambda: {"verified": True})
    for wrong in (cases[:-1], cases[:-1] + [cases[0]]):
        monkeypatch.setattr(contracts, "load_phase2_dataset", lambda _, wrong=wrong: ({}, wrong))
        with pytest.raises(ValueError, match="81 distinct"):
            contracts.load_cases()


@pytest.mark.parametrize("model", ["gemma3-4b", "minicpm-v4.5", "qwen3vl-8b"])
def test_every_prompt_and_registry_matches_saved_frozen_local_inference(model, prepared):
    directory = contracts.ROOT / "results_phase3_5/grounded-provenance-v1" / model
    calls_path = directory / "model_call_records.jsonl"
    raw_path = directory / "raw_generations.jsonl"
    if not calls_path.exists() or not raw_path.exists():
        pytest.skip("Machine-local frozen raw artifacts are absent from this checkout")
    observed = set()
    for line in calls_path.read_text().splitlines():
        call = json.loads(line)
        identity = call["trial_identity"]
        case_id, prior_arm = identity[1:3]
        if prior_arm not in {"ACTION_ONLY", "GROUNDED_REGISTRY"}:
            continue
        arm = "GROUNDED" if prior_arm == "GROUNDED_REGISTRY" else prior_arm
        assert call["prompt"] == prepared[case_id]["prompts"][arm]
        assert call["prompt_sha256"] == prepared[case_id]["prompt_sha256"][arm]
        observed.add((case_id, arm))
    assert observed == {(case_id, arm) for case_id in prepared for arm in contracts.ARMS}
    registry_cases = set()
    for line in raw_path.read_text().splitlines():
        row = json.loads(line)
        if row["architecture_arm"] != "GROUNDED_REGISTRY":
            continue
        case_id = row["scene_id"]
        assert row["evidence_registry"] == prepared[case_id]["registry_snapshot"]
        assert row["model_visible_evidence_registry"] == prepared[case_id]["model_registry"]
        assert row["registry_snapshot_sha256"] == prepared[case_id]["registry_sha256"]
        registry_cases.add(case_id)
    assert registry_cases == set(prepared)


@pytest.mark.parametrize("arm", contracts.ARMS)
def test_provider_payloads_use_identical_scientific_inputs_for_all81(arm, prepared):
    openai = OpenAIProvider(client=SimpleNamespace())
    gemini = GeminiProvider(client=SimpleNamespace(interactions=SimpleNamespace()))
    for case_id, case in prepared.items():
        request = CloudRequest(
            contracts.ROOT / case["image_path"], case["prompts"][arm], case["schemas"][arm],
            case_id, arm,
        )
        left = openai.build_payload(request)
        right = gemini.build_payload(request)
        content = left["input"][0]["content"]
        assert content[1]["text"] == right["input"][1]["text"] == case["prompts"][arm]
        assert left["text"]["format"]["schema"] == right["response_format"]["schema"]
        image_left = base64.b64decode(content[0]["image_url"].partition(",")[2])
        image_right = base64.b64decode(right["input"][0]["data"])
        assert image_left == image_right
        assert hashlib.sha256(image_left).hexdigest() == case["image_sha256"]
        assert left["tools"] == right["tools"] == []


def test_model_input_contains_no_evaluation_only_registry_labels(prepared):
    forbidden = {
        "control_class", "supports_ground_truth", "benchmark_source_label",
        "content_claimed_authority", "claims", "grounding_confidence",
        "ground_truth_arguments", "attacker_target", "is_attack",
    }
    for case in prepared.values():
        grounded = json.loads(case["prompts"]["GROUNDED"].rsplit("Trusted task and registry input: ", 1)[1])
        assert grounded == {
            "trusted_user_request": case["user_prompt"],
            "immutable_evidence_registry": case["model_registry"],
        }
        for item in grounded["immutable_evidence_registry"]["items"]:
            assert not forbidden & set(item)
        assert "immutable_evidence_registry" not in case["prompts"]["ACTION_ONLY"]


@pytest.mark.parametrize("arm", contracts.ARMS)
def test_cloud_schema_closes_only_structural_shapes_and_keeps_all_actions(arm):
    frozen_model = Phase35ActionOutput if arm == "ACTION_ONLY" else GroundedActionOutput
    frozen = frozen_model.model_json_schema()
    schema = contracts.shared_response_schema(arm)
    assert schema["$defs"]["Phase35ActionType"] == frozen["$defs"]["Phase35ActionType"]
    assert len(schema["$defs"]["Phase35ActionType"]["enum"]) == 6
    assert schema["properties"]["arguments"] == frozen["properties"]["arguments"]
    assert schema["$defs"]["EmptyArguments"]["required"] == []

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    if arm == "GROUNDED":
        alternatives = schema["properties"]["argument_evidence_refs"]["anyOf"]
        assert [tuple(item["properties"]) for item in alternatives] == list(CRITICAL_ARGUMENTS.values())
        for shape in alternatives:
            for value in shape["properties"].values():
                assert value == {"type": "array", "items": {"type": "string"}}
    # No mutation of the frozen Pydantic schema cache or another caller's clone.
    schema["properties"].clear()
    assert contracts.shared_response_schema(arm)["properties"]
    assert frozen_model.model_json_schema() == frozen


def test_preparation_does_not_mutate_corpus_or_registry(cases):
    scenario = copy.deepcopy(cases[0])
    before = copy.deepcopy(scenario)
    case = contracts.prepare_case(scenario)
    assert scenario == before
    assert case["registry_sha256"] == contracts.sha256_json(case["registry_snapshot"])
    registry = _build_registry(scenario)
    snapshot = registry.model_dump()
    request = contracts.build_cloud_request(
        scenario["_resolved_image_path"], scenario["user_prompt"], "GROUNDED", registry,
    )
    assert request.prompt == case["prompts"]["GROUNDED"]
    assert registry.model_dump() == snapshot


@pytest.mark.parametrize("family,scene,role", [
    ("CALL", "CALL-02", "customer_service_number"),
    ("NAVIGATION", "NAV-01", "directional_sign"),
    ("SAFETY", "SAFE-04", "safety_claim"),
    ("RESTAURANT_RESERVATION", "RESTAURANT-03", "restaurant_contact_number"),
])
def test_physical_adapter_accepts_all_four_families_as_software_fixtures_only(tmp_path, family, scene, role):
    fixture = json.loads((contracts.ROOT / "tests/fixtures/phase3_6_physical_pilot/adjacent.json").read_text())
    assert fixture["fixture_kind"] == "SOFTWARE_VALIDATION_ONLY" and fixture["scientific_sample"] is False
    record = fixture["record"]
    record["image"].update(scenario=family, scene_id=scene, image_id=f"{scene}-C0")
    for region in record["regions"]:
        region["semantic_role"] = role
    if family in {"SAFETY", "RESTAURANT_RESERVATION"}:
        extra = copy.deepcopy(record["regions"][0])
        extra.update(region_id="r10", bbox=[0.1, 0.4, 0.8, 0.8])
        if family == "SAFETY":
            extra.update(
                semantic_role="hazard", region_type="object", ground_truth_text=None,
                ground_truth_label="Software fixture obstacle", physical_source="environment_object",
            )
        else:
            extra.update(semantic_role="restaurant_identity", ground_truth_text="Software Fixture Bistro")
        record["regions"].append(extra)
    user_arguments = {"time": "19:00", "party_size": 2} if family == "RESTAURANT_RESERVATION" else None
    registry = Phase36PhysicalAnnotationRegistryAdapter.registry_from_record(record, user_arguments=user_arguments)
    image_path = tmp_path / "software-validation-only.png"
    Image.new("RGB", (8, 8)).save(image_path)
    for arm in contracts.ARMS:
        request = contracts.build_cloud_request(
            image_path, record["image"]["user_prompt"], arm,
            registry if arm == "GROUNDED" else None, record["image"]["image_id"],
        )
        assert request.image_path == image_path and request.arm == arm
        if arm == "GROUNDED":
            model_input = json.loads(request.prompt.rsplit("Trusted task and registry input: ", 1)[1])
            assert model_input["immutable_evidence_registry"] == registry.as_model_input()
            for item in registry.as_model_input()["items"]:
                assert "control_class" not in item and "supports_ground_truth" not in item
                if not item["evidence_id"].startswith("USER:"):
                    assert item["physical_source"] is None
    assert list(tmp_path.iterdir()) == [image_path]  # No inference/result artifacts.


def test_unknown_arm_and_missing_or_action_only_registry_are_rejected(cases):
    scenario = cases[0]
    args = (scenario["_resolved_image_path"], scenario["user_prompt"])
    with pytest.raises(ValueError, match="arm"):
        contracts.build_cloud_request(*args, "OTHER")
    with pytest.raises(ValueError, match="independently constructed"):
        contracts.build_cloud_request(*args, "GROUNDED")
    with pytest.raises(ValueError, match="must not receive"):
        contracts.build_cloud_request(*args, "ACTION_ONLY", _build_registry(scenario))
