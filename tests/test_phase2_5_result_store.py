import csv
from copy import deepcopy

import pytest

from result_store import read_jsonl
from result_store_phase2_5 import (
    PHASE2_5_IDENTITY_FIELDS,
    assert_phase2_5_resume_compatible,
    completed_phase2_5_identities,
    final_phase2_5_trials,
    next_phase2_5_attempt_index,
    normalize_latency_ms,
    normalize_phase2_5_telemetry,
    normalize_vram_bytes,
    persist_phase2_5_attempt,
    phase2_5_attempt_accounting,
    phase2_5_trial_identity,
    read_phase2_5_csv,
    validate_phase2_5_attempts,
    validate_phase2_5_telemetry,
)


def _telemetry():
    return {
        "model_load_time_ms": 2_000.0,
        "preprocessing_latency_ms": 20.0,
        "inference_latency_ms": 120.0,
        "generation_latency_ms": 100.0,
        "thin_gate_latency_ms": 0.2,
        "evidence_mapper_latency_ms": 0.4,
        "input_token_count": 40,
        "output_token_count": 10,
        "generated_tokens": 10,
        "tokens_per_second": 100.0,
        "gpu_memory_allocated_before_inference_bytes": 8_000,
        "gpu_peak_memory_allocated_bytes": 12_000,
        "gpu_peak_memory_reserved_bytes": 14_000,
        "model_dtype": "bf16",
        "quantization": "none",
        "attention_backend": "sdpa",
        "image_width": 1024,
        "image_height": 768,
    }


def _attempt(status="completed", **updates):
    row = {
        "scene_id": "scene-1",
        "condition": "CLEAN_TRUSTED",
        "architecture_arm": "INLINE_PROVENANCE",
        "provider": "local",
        "model_id": "gemma3-4b",
        "model_revision": "revision-a",
        "run": 1,
        "prompt_version": "ZERO_SHOT_V1",
        "dataset_version": "phase2-v1",
        "policy_version": "phase2-policy-v1",
        "status": status,
        "zero_shot_prompt_version": "ZERO_SHOT_V1",
        "benchmark_lock_id": "lensguard-phase2-frozen-v1",
        "benchmark_lock_sha256": "a" * 64,
        "selected_case_count": 1,
        "benchmark_case_count": 81,
        "structured_output_valid": status == "completed",
        **_telemetry(),
    }
    row.update(updates)
    return row


def _v2_attempt(status="completed", **updates):
    usable = status == "completed"
    row = _attempt(
        status,
        zero_shot_prompt_version="ZERO_SHOT_V2",
        schema_transport_version="phase2.5-local-json-schema-transport-v2",
        parse_success=usable,
        schema_valid=usable,
        normalization_applied=False,
        normalization_method=None,
        normalized_schema_valid=usable,
        contract_semantically_valid=usable,
        structured_output_valid=usable,
        action_candidate=(
            {"action": "CALL", "arguments": {"target_number": "+12025550106"}}
            if usable
            else None
        ),
        action_correct=True if usable else None,
        critical_argument_correct=True if usable else None,
        provenance_semantically_valid=True if usable else None,
        unsafe_execution=False if usable else None,
        gate_decision="ALLOW" if usable else None,
        unsafe_automatic_execution=False if usable else None,
        failure_category=None if usable else "inference_runtime",
        failure_categories=[] if usable else ["inference_runtime"],
    )
    row.update(updates)
    return row


def _accept_core(rows):
    assert all(row["model"] == row["model_id"] for row in rows)


def test_phase2_5_identity_is_exact_required_tuple():
    row = _attempt()
    assert phase2_5_trial_identity(row) == tuple(
        str(row[field]) for field in PHASE2_5_IDENTITY_FIELDS
    )


def test_retry_dedup_prefers_last_success_without_counting_retry_as_trial():
    failed = _attempt("error", marker="failed")
    succeeded = _attempt("completed", marker="success")
    later_error = _attempt("error", marker="later-error")

    assert final_phase2_5_trials([failed, succeeded, later_error]) == [succeeded]
    assert completed_phase2_5_identities([failed, succeeded]) == {
        phase2_5_trial_identity(succeeded)
    }
    assert next_phase2_5_attempt_index([failed, succeeded], failed) == 3
    accounting = phase2_5_attempt_accounting([failed, succeeded])
    assert accounting["raw_attempts"] == 2
    assert accounting["unique_scientific_trials"] == 1
    assert accounting["superseded_error_attempts"] == 1


def test_model_revision_and_provider_are_part_of_trial_identity():
    base = _attempt()
    new_revision = _attempt(model_revision="revision-b")
    cloud = _attempt(provider="gemini")
    assert len(final_phase2_5_trials([base, new_revision, cloud])) == 3


def test_validation_delegates_to_phase2_core_with_non_mutating_model_alias():
    observed = []

    def core_validator(rows):
        observed.extend(rows)
        assert rows[0]["model"] == "gemma3-4b"

    row = _attempt()
    validate_phase2_5_attempts([row], core_validator=core_validator)
    assert "model" not in row
    assert len(observed) == 1


@pytest.mark.parametrize(
    ("item_statuses", "expected_legacy_status"),
    [
        (("matched", "hallucinated"), "hallucinated"),
        (("hallucinated", "matched"), "hallucinated"),
        (("matched", "missing"), "missing"),
    ],
)
def test_v2_validation_bridges_frozen_multi_item_ambiguous_text_summary(
    item_statuses, expected_legacy_status
):
    """Keep V2 semantics while adapting only the frozen validator's view."""

    observed = []

    def core_validator(rows):
        observed.extend(rows)
        projected = rows[0]["provenance_evaluations"][0]
        assert projected["evidence_status"] == expected_legacy_status
        assert projected["text_match_correct"] is False

    evaluation = {
        "argument_name": "direction",
        "evidence_status": "ambiguous",
        "text_match_correct": False,
        "reported_evidence_items": [
            {
                "evidence_status": item_statuses[0],
                "supports_argument": item_statuses[0] == "matched",
            },
            {
                "evidence_status": item_statuses[1],
                "supports_argument": item_statuses[1] == "matched",
            },
        ],
    }
    row = _v2_attempt(
        provenance_evaluations=[evaluation],
        provenance_semantically_valid=False,
        failure_category="provenance_semantic_failure",
        failure_categories=["provenance_semantic_failure"],
    )

    validate_phase2_5_attempts([row], core_validator=core_validator)

    assert observed
    assert row["provenance_evaluations"][0]["evidence_status"] == "ambiguous"
    assert row["provenance_evaluations"][0]["text_match_correct"] is False
    assert tuple(
        item["evidence_status"]
        for item in row["provenance_evaluations"][0]["reported_evidence_items"]
    ) == item_statuses


@pytest.mark.parametrize(
    ("profile", "item_statuses"),
    [
        ("ZERO_SHOT_V1", ("matched", "hallucinated")),
        ("ZERO_SHOT_V2", ("matched", "unsupported")),
    ],
)
def test_multi_item_bridge_does_not_broaden_other_profiles_or_shapes(
    profile, item_statuses
):
    observed = []
    evaluation = {
        "argument_name": "direction",
        "evidence_status": "ambiguous",
        "text_match_correct": False,
        "reported_evidence_items": [
            {
                "evidence_status": status,
                "supports_argument": status == "matched",
            }
            for status in item_statuses
        ],
    }
    row = _v2_attempt(
        zero_shot_prompt_version=profile,
        provenance_evaluations=[evaluation],
    )

    validate_phase2_5_attempts(
        [row], core_validator=lambda rows: observed.extend(rows)
    )

    assert observed[0]["provenance_evaluations"][0]["evidence_status"] == "ambiguous"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run", 0, "invalid run"),
        ("provider", "", "provider"),
        ("model_revision", None, "model_revision"),
        ("policy_version", "", "policy_version"),
    ],
)
def test_validation_rejects_incomplete_extended_identity(field, value, message):
    row = _attempt()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


def test_validation_rejects_conflicting_legacy_model_field():
    row = _attempt(model="some-other-model")
    with pytest.raises(ValueError, match="inconsistent with model_id"):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark_lock_sha256", "not-a-digest", "benchmark_lock_sha256"),
        ("benchmark_lock_id", "", "benchmark_lock_id"),
        ("selected_case_count", 0, "case-scope"),
        ("benchmark_case_count", 0, "case-scope"),
        ("structured_output_valid", None, "structured_output_valid"),
    ],
)
def test_validation_rejects_missing_or_invalid_freeze_metadata(
    field, value, message
):
    row = _attempt()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


def test_completed_trial_cannot_claim_invalid_structured_output():
    row = _attempt(structured_output_valid=False)
    with pytest.raises(ValueError, match="completed.*invalid structured output"):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


def test_default_validation_reaches_frozen_phase2_validator():
    with pytest.raises(ValueError, match="Phase 2 attempt"):
        validate_phase2_5_attempts([_attempt()])


def test_telemetry_normalizes_scalars_aliases_units_and_rate():
    raw = _telemetry()
    raw.pop("input_token_count")
    raw.pop("output_token_count")
    raw.pop("tokens_per_second")
    raw.update(input_tokens=40, output_tokens=10, model_dtype="torch.bfloat16")
    normalized = normalize_phase2_5_telemetry(raw)

    assert normalized["input_token_count"] == 40
    assert normalized["output_token_count"] == 10
    assert normalized["model_dtype"] == "bf16"
    assert normalized["tokens_per_second"] == 100.0
    assert normalize_vram_bytes(1.5, unit="GiB") == round(1.5 * 1024**3)
    assert normalize_latency_ms(1.5, unit="seconds") == 1_500.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("preprocessing_latency_ms", -1, "preprocessing_latency_ms"),
        ("gpu_peak_memory_allocated_bytes", 7_999, "before inference"),
        ("gpu_peak_memory_reserved_bytes", 11_999, "exceeds peak reserved"),
        ("image_width", 0, "image_width"),
        ("output_token_count", 9, "generated_tokens"),
        ("model_dtype", None, "model_dtype"),
    ],
)
def test_telemetry_validation_rejects_invalid_measurements(field, value, message):
    telemetry = _telemetry()
    telemetry[field] = value
    with pytest.raises(ValueError, match=message):
        validate_phase2_5_telemetry(telemetry)


def test_nested_telemetry_is_supported_but_conflicts_are_rejected():
    row = _attempt()
    nested = {key: row.pop(key) for key in _telemetry()}
    row["local_performance"] = nested
    validate_phase2_5_attempts([row], core_validator=_accept_core)

    row["image_width"] = 99
    with pytest.raises(ValueError, match="conflicting nested/top-level image_width"):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


def test_resume_compatibility_checks_science_and_runtime_profile():
    rows = [_attempt(), _attempt(architecture_arm="ACTION_ONLY", prompt_version="ACTION_V1")]
    assert_phase2_5_resume_compatible(
        rows,
        provider="local",
        model_id="gemma3-4b",
        model_revision="revision-a",
        dataset_version="phase2-v1",
        policy_version="phase2-policy-v1",
        prompt_versions={"ZERO_SHOT_V1", "ACTION_V1"},
        model_dtype="bfloat16",
        quantization="none",
        attention_backend="sdpa",
    )

    with pytest.raises(ValueError, match="model_revision"):
        assert_phase2_5_resume_compatible(
            rows,
            provider="local",
            model_id="gemma3-4b",
            model_revision="revision-b",
            dataset_version="phase2-v1",
            policy_version="phase2-policy-v1",
        )
    with pytest.raises(ValueError, match="attention_backend"):
        assert_phase2_5_resume_compatible(
            rows,
            provider="local",
            model_id="gemma3-4b",
            model_revision="revision-a",
            dataset_version="phase2-v1",
            policy_version="phase2-policy-v1",
            attention_backend="flash_attention_2",
        )


def test_persistence_keeps_raw_attempts_and_atomically_refreshes_final_csv(tmp_path):
    raw_path = tmp_path / "raw_generations.jsonl"
    final_path = tmp_path / "final_trials.csv"
    failed = _attempt("error", marker="failed")
    succeeded = _attempt("completed", marker="success")

    persist_phase2_5_attempt(
        raw_path, final_path, failed, core_validator=_accept_core
    )
    persist_phase2_5_attempt(
        raw_path, final_path, succeeded, core_validator=_accept_core
    )

    assert read_jsonl(raw_path) == [failed, succeeded]
    with final_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["marker"] == "success"
    assert not final_path.with_suffix(".csv.tmp").exists()


def test_persistence_refuses_non_json_telemetry_before_append(tmp_path):
    row = deepcopy(_attempt())
    row["inference_latency_ms"] = float("nan")
    raw_path = tmp_path / "raw.jsonl"
    with pytest.raises(ValueError, match="strict-JSON"):
        persist_phase2_5_attempt(
            raw_path,
            tmp_path / "final.csv",
            row,
            core_validator=_accept_core,
        )
    assert not raw_path.exists()


def test_v2_validation_accepts_explicit_list_normalization_diagnostics():
    row = _v2_attempt(
        schema_valid=False,
        normalization_applied=True,
        normalization_method="single_argument_list",
    )

    validate_phase2_5_attempts([row], core_validator=_accept_core)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_transport_version": None}, "schema_transport_version"),
        ({"parse_success": None}, "parse_success"),
        (
            {"normalization_applied": True, "normalization_method": None},
            "normalization_method",
        ),
        (
            {"normalized_schema_valid": False, "structured_output_valid": True},
            "raw schema validity requires",
        ),
        ({"unsafe_execution": True, "gate_decision": None}, "gate did not run"),
        ({"failure_categories": ["not-a-category"]}, "failure_categories"),
        (
            {
                "failure_category": "action_prediction_failure",
                "failure_categories": [
                    "action_prediction_failure",
                    "schema_mismatch",
                ],
            },
            "unstable failure_categories order",
        ),
    ],
)
def test_v2_validation_rejects_inconsistent_contract_diagnostics(updates, message):
    row = _v2_attempt(**updates)
    with pytest.raises(ValueError, match=message):
        validate_phase2_5_attempts([row], core_validator=_accept_core)


def test_resume_refuses_v1_v2_profile_or_schema_transport_mixing():
    legacy = [_attempt()]
    with pytest.raises(ValueError, match="zero_shot_prompt_version"):
        assert_phase2_5_resume_compatible(
            legacy,
            provider="local",
            model_id="gemma3-4b",
            model_revision="revision-a",
            dataset_version="phase2-v1",
            policy_version="phase2-policy-v1",
            zero_shot_prompt_version="ZERO_SHOT_V2",
            schema_transport_version="phase2.5-local-json-schema-transport-v2",
        )

    current = [_v2_attempt()]
    with pytest.raises(ValueError, match="schema_transport_version"):
        assert_phase2_5_resume_compatible(
            current,
            provider="local",
            model_id="gemma3-4b",
            model_revision="revision-a",
            dataset_version="phase2-v1",
            policy_version="phase2-policy-v1",
            zero_shot_prompt_version="ZERO_SHOT_V2",
            schema_transport_version="some-other-transport",
        )


def test_v2_csv_round_trip_preserves_tri_state_and_failure_list(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    final_path = tmp_path / "final.csv"
    row = _v2_attempt(
        "error",
        parse_success=True,
        schema_valid=False,
        normalized_schema_valid=False,
        contract_semantically_valid=False,
        action_candidate={
            "action": "CALL",
            "arguments": {"target_number": "+12025550106"},
        },
        action_correct=True,
        critical_argument_correct=True,
        failure_category="schema_mismatch",
        failure_categories=["schema_mismatch"],
    )
    persist_phase2_5_attempt(
        raw_path, final_path, row, core_validator=_accept_core
    )

    [decoded] = read_phase2_5_csv(final_path)
    assert decoded["parse_success"] is True
    assert decoded["schema_valid"] is False
    assert decoded["action_correct"] is True
    assert decoded["unsafe_execution"] is None
    assert decoded["provenance_semantically_valid"] is None
    assert decoded["normalization_method"] is None
    assert decoded["failure_categories"] == ["schema_mismatch"]
