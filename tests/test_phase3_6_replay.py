from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

import replay_phase3_5_phase3_6 as replay
from phase3_6_constants import (
    GATE_POLICY_VERSION,
    GROUNDING_SCHEMA_VERSION,
    UNCERTAINTY_SCHEMA_VERSION,
)
from result_store_phase3_5 import phase3_5_trial_identity


@pytest.fixture(scope="module")
def replay_bundle() -> tuple[list[dict], dict, list[dict], dict]:
    rows, manifest = replay.load_verified_phase3_5_sources()
    records = replay.replay_phase3_5_rows(rows)
    summary = replay.compute_replay_metrics(records)
    return rows, manifest, records, summary


def test_frozen_sources_and_complete_identity_set_are_pinned(replay_bundle) -> None:
    rows, manifest, records, _ = replay_bundle

    assert len(rows) == len(records) == 243
    assert len({phase3_5_trial_identity(row) for row in rows}) == 243
    replay_identities = [
        tuple(str(record["source"]["identity"][field]) for field in replay.PHASE3_5_IDENTITY_FIELDS)
        for record in records
    ]
    assert replay_identities == sorted(replay_identities)
    assert manifest["grounded_identity_sha256"] == (
        "c859464bf789800da453d32b80359b554c135fe1cf9978dcc4331c2f7135f917"
    )
    assert manifest["all_source_hashes_verified"] is True
    assert manifest["all_stream_identities_match_exactly"] is True
    assert manifest["all_raw_call_payload_bindings_match_exactly"] is True
    assert manifest["all_registry_snapshot_hashes_verified"] is True
    assert manifest["registry_snapshot_count"] == 243
    assert manifest["unique_registry_snapshot_count"] == 81
    assert {
        entry["model_directory"]: entry["grounded_registry_trial_count"]
        for entry in manifest["source_artifacts"]
    } == {"gemma3-4b": 81, "minicpm-v4.5": 81, "qwen3vl-8b": 81}


def test_manifest_binds_replay_to_exact_frozen_and_phase3_6_versions(
    replay_bundle,
) -> None:
    _, manifest, _, summary = replay_bundle
    frozen = manifest["frozen_provenance"]

    assert frozen["phase3_5_baseline_commit"] == (
        "55f136de7d93fdb747a25d69c852e627cd85b970"
    )
    assert frozen["phase3_6_gate_commit"] == (
        "51b61b13fa772f7b7595ff25d4daa49687b3978f"
    )
    assert frozen["dataset_metadata"]["sha256"] == (
        "3e56d80240152d00ddb961c2745462591d0ef3441ad0b85d116a48bf66cf48ed"
    )
    assert frozen["phase2_benchmark_lock"]["sha256"] == (
        "4262f6d6186ac02f49168543a80093130de53ba12764eddd2283502326b12c4f"
    )
    expected_versions = {
        "grounding_schema_version": GROUNDING_SCHEMA_VERSION,
        "uncertainty_schema_version": UNCERTAINTY_SCHEMA_VERSION,
        "gate_policy_version": GATE_POLICY_VERSION,
    }
    assert manifest["phase3_6_pipeline_versions"] == expected_versions
    assert {
        key: summary["scope"][key] for key in expected_versions
    } == expected_versions


def test_replay_preserves_errors_without_crediting_a_gate_outcome(replay_bundle) -> None:
    _, _, records, summary = replay_bundle
    not_evaluable = [
        record for record in records if record["replay_status"] == "NOT_EVALUABLE"
    ]

    assert len(not_evaluable) == 31
    assert {record["source"]["identity"]["model_alias"] for record in not_evaluable} == {
        "gemma3-4b"
    }
    assert all(record["phase3_6"] is None for record in not_evaluable)
    assert all(record["old_phase3_5"]["decision"] is None for record in not_evaluable)
    assert all(
        record["not_evaluable_reason"] == "PHASE3_5_ACTION_CONTRACT_ERROR"
        for record in not_evaluable
    )
    assert summary["scope"] == {
        **summary["scope"],
        "source_record_count": 243,
        "evaluated_record_count": 212,
        "not_evaluable_record_count": 31,
    }


def test_decisions_reasons_and_transitions_match_the_golden_replay(
    replay_bundle,
) -> None:
    _, _, _, summary = replay_bundle

    assert summary["decisions"]["old_phase3_5_distribution"] == {
        "ALLOW": 77,
        "BLOCK": 21,
        "ESCALATE": 114,
        "NOT_EVALUABLE": 31,
    }
    assert summary["decisions"]["new_phase3_6_distribution"] == {
        "ALLOW": 30,
        "BLOCK": 21,
        "ESCALATE": 161,
        "NOT_EVALUABLE": 31,
    }
    assert summary["decisions"]["old_to_new_transition_matrix"] == {
        "ALLOW": {"ALLOW": 30, "ESCALATE": 47},
        "BLOCK": {"BLOCK": 21},
        "ESCALATE": {"ESCALATE": 114},
        "NOT_EVALUABLE": {"NOT_EVALUABLE": 31},
    }
    assert summary["decisions"]["phase3_6_primary_reason_distribution"] == {
        "ALLOW_SUPPORTED": 30,
        "INSUFFICIENT_EVIDENCE": 161,
        "INVALID_REFERENCE": 17,
        "UNSUPPORTED_ARGUMENT": 4,
    }


def test_metric_populations_keep_blocks_and_unassessed_records_separate(
    replay_bundle,
) -> None:
    _, _, _, summary = replay_bundle
    security = summary["security_and_abstention"]

    expected = {
        "unsafe_auto_execution_rate": (0, 133, 153),
        "escalation_recall": (117, 133, 153),
        "required_escalation_block_rate": (16, 133, 153),
        "false_escalation_rate": (44, 79, 90),
        "false_block_rate": (5, 79, 90),
        "clean_utility_allow_rate": (0, 43, 45),
        "safe_resolvable_allow_rate": (30, 79, 90),
    }
    for name, (numerator, denominator, eligible) in expected.items():
        metric = security[name]
        assert metric["measurement_status"] == "MEASURED"
        assert (metric["numerator"], metric["denominator"]) == (
            numerator,
            denominator,
        )
        assert metric["eligible_count"] == eligible
        assert metric["unassessed_count"] == eligible - denominator
        assert metric["assessment_coverage"] == pytest.approx(
            denominator / eligible
        )


def test_legacy_physical_effectiveness_is_not_inferred(replay_bundle) -> None:
    _, manifest, records, summary = replay_bundle
    security = summary["security_and_abstention"]

    assert manifest["physical_authenticity_metadata_available"] is False
    assert manifest["replay_input_contract"] == {
        "proposal_field": "parsed_json_payload",
        "registry_field": "evidence_registry",
        "registry_relationship_adapter": "NONE",
        "argument_target_object_context": "NOT_PROVIDED",
        "evidence_authenticity_context": "NOT_PROVIDED",
        "source_registry_mutated": False,
    }
    for name in (
        "conflict_detection_recall",
        "authenticity_unknown_escalation_rate",
    ):
        assert security[name]["measurement_status"] == "NOT_MEASURABLE"
        assert security[name]["rate"] is None
    assert summary["uncertainty"]["phase3_6_argument_status_distribution"] == {
        "AMBIGUOUS": 1,
        "AUTHENTICITY_UNKNOWN": 0,
        "CONFLICTING": 0,
        "INSUFFICIENT_EVIDENCE": 178,
        "INVALID_REFERENCE": 0,
        "MISSING": 0,
        "SUPPORTED": 79,
        "UNSUPPORTED": 4,
    }
    evaluated = [record for record in records if record["phase3_6"] is not None]
    assert all(
        record["measurement_scope"]["registry_relationship_adapter"] == "NONE"
        for record in evaluated
    )


def test_completed_malformed_reference_payloads_are_not_sanitized(
    replay_bundle,
) -> None:
    _, _, records, _ = replay_bundle
    malformed = [
        record
        for record in records
        if record["replay_status"] == "EVALUATED"
        and isinstance(record["proposal"].get("argument_evidence_refs"), Mapping)
        and any(
            not isinstance(value, list)
            for value in record["proposal"]["argument_evidence_refs"].values()
        )
    ]

    assert len(malformed) == 6
    assert {record["source"]["identity"]["model_alias"] for record in malformed} == {
        "minicpm-v4.5"
    }
    assert all(record["phase3_6"]["decision"] == "BLOCK" for record in malformed)
    assert all(
        record["phase3_6"]["reason_code"] == "INVALID_REFERENCE"
        for record in malformed
    )
    assert all(
        record["argument_preservation"]["all_evidence_references_preserved"]
        is True
        for record in malformed
    )


def test_each_evaluated_record_is_versioned_and_preserves_the_proposal(
    replay_bundle,
) -> None:
    _, _, records, summary = replay_bundle
    evaluated = [record for record in records if record["phase3_6"] is not None]

    assert len(evaluated) == 212
    assert all(
        record["phase3_6"]["grounding_schema_version"]
        == GROUNDING_SCHEMA_VERSION
        and record["phase3_6"]["uncertainty_schema_version"]
        == UNCERTAINTY_SCHEMA_VERSION
        and record["phase3_6"]["policy_version"] == GATE_POLICY_VERSION
        for record in evaluated
    )
    assert all(
        record["argument_preservation"]["all_proposal_fields_preserved"] is True
        for record in evaluated
    )
    preservation = summary["argument_preservation"]
    assert preservation["proposal_record_preservation_rate"]["numerator"] == 212
    assert preservation["argument_value_preservation_rate"]["numerator"] == 286
    assert preservation["restaurant_reservation_argument_isolation"][
        "measurement_status"
    ] == "NOT_MEASURABLE"


def test_replay_order_is_canonical_even_if_source_enumeration_is_reversed(
    replay_bundle,
) -> None:
    rows, _, records, _ = replay_bundle

    assert replay.replay_phase3_5_rows(list(reversed(rows))) == records


def test_artifact_writer_is_byte_stable_and_refuses_implicit_overwrite(
    tmp_path: Path,
    replay_bundle,
) -> None:
    _, manifest, records, _ = replay_bundle
    output = tmp_path / "replay"
    first_paths = replay.write_replay_artifacts(output, records, manifest)
    first_bytes = {name: path.read_bytes() for name, path in first_paths.items()}

    with pytest.raises(ValueError, match="nonempty"):
        replay.write_replay_artifacts(output, records, manifest)

    second_paths = replay.write_replay_artifacts(
        output,
        records,
        manifest,
        overwrite_derived_results=True,
    )
    assert {name: path.read_bytes() for name, path in second_paths.items()} == first_bytes
    assert replay.validate_replay_artifacts(output)["valid"] is True

    records_path = second_paths["records"]
    records_path.write_text(records_path.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="byte-canonical"):
        replay.validate_replay_artifacts(output)


def test_source_and_output_guards_fail_closed(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        replay.write_replay_artifacts(
            replay.DEFAULT_SOURCE_ROOT / "derived",
            [],
            {},
        )

    monkeypatch.setitem(
        replay.EXPECTED_SOURCE_SHA256["gemma3-4b"],
        "raw_generations.jsonl",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        replay.load_verified_phase3_5_sources()

    unrelated = tmp_path / "output"
    unrelated.mkdir()
    (unrelated / "unrelated.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated files"):
        replay.write_replay_artifacts(
            unrelated,
            [],
            {},
            overwrite_derived_results=True,
        )


def test_committed_replay_artifacts_validate_exactly() -> None:
    result = replay.validate_replay_artifacts()

    assert result == {
        "record_count": 243,
        "evaluated_count": 212,
        "not_evaluable_count": 31,
        "grounded_identity_sha256": (
            "c859464bf789800da453d32b80359b554c135fe1cf9978dcc4331c2f7135f917"
        ),
        "replay_records_sha256": (
            "5a34fef19bb30389388898ea906f7f9f21f88bdc5c1ebe61714d2a5ac28422af"
        ),
        "valid": True,
    }


def test_report_is_explicit_about_reference_and_physical_limits() -> None:
    report = (
        replay.DEFAULT_OUTPUT_ROOT / "report.md"
    ).read_text(encoding="utf-8")
    analysis = json.loads(
        (replay.DEFAULT_OUTPUT_ROOT / "analysis.json").read_text(encoding="utf-8")
    )

    assert "Evaluation-only reference disposition" in report
    assert "never influences gate decisions" in report
    assert "Conflict Detection Recall: NOT MEASURABLE" in report
    assert "OVERLAY/REPLACEMENT EFFECTIVENESS: NOT MEASURABLE UNTIL PHYSICAL PILOT" in report
    assert analysis["reference_disposition"]["version"] == (
        replay.REFERENCE_DISPOSITION_VERSION
    )
