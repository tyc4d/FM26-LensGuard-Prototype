from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import audit_phase3_5_corpus as corpus_audit
from phase2_benchmark_lock import Phase2BenchmarkLockError, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def generated_audit(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    output_dir = (
        tmp_path_factory.mktemp("phase3_5_audit")
        / "results_phase3_5"
        / "grounded-provenance-v1"
    )
    return corpus_audit.generate_corpus_audit(
        output_dir,
        project_root=PROJECT_ROOT,
    )


def test_audit_verifies_lock_and_keeps_lossless_source_payload(
    generated_audit: dict[str, Any],
) -> None:
    source_path = PROJECT_ROOT / "dataset_phase2/metadata.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    audit = generated_audit["audit"]

    assert audit["frozen_lock_verification"]["verified"] is True
    assert audit["phase2_corpus"] == source
    assert audit["sources"]["metadata_sha256"] == sha256_file(source_path)
    assert audit["sources"]["image_tree_sha256"] == (
        "a7dcd80f480088de5192f564b3892307d51b4509b62ae2e11aded201511aa5f5"
    )
    assert len(audit["sources"]["image_sha256"]) == 81


def test_audit_reports_exact_counts_actions_conditions_and_annotations(
    generated_audit: dict[str, Any],
) -> None:
    audit = generated_audit["audit"]

    assert audit["counts"]["records"] == 81
    assert audit["counts"]["images"] == 81
    assert audit["counts"]["semantic_scenarios"] == 15
    assert audit["counts"]["region_occurrences"] == 162
    assert audit["counts"]["actions"] == {
        "CALL": 27,
        "DIRECTION_ADVICE": 27,
        "OPEN_URL": 27,
    }
    assert audit["counts"]["conditions"] == {
        "AUTHORITY_IMPERSONATION": 15,
        "BENIGN_UNTRUSTED_SUBSTITUTION": 15,
        "CLEAN_TRUSTED": 15,
        "EXPLICIT_USER_OVERRIDE": 15,
        "NO_VERIFIED_GROUND_TRUTH": 3,
        "OBVIOUS_INJECTION_CONTROL": 15,
        "TRUSTED_BUT_CONFLICTING_UPDATE": 3,
    }
    assert audit["observed_annotation_capabilities"]["critical_arguments_by_action"] == {
        "CALL": ["target_number"],
        "DIRECTION_ADVICE": ["direction", "destination"],
        "OPEN_URL": ["url"],
    }
    assert "source_type" in audit["observed_annotation_capabilities"]["region_fields"]
    assert "argument_provenance" in audit["observed_annotation_capabilities"][
        "record_fields"
    ]


def test_region_csv_retains_full_regions_and_labels_derived_ids(
    generated_audit: dict[str, Any],
) -> None:
    csv_path = generated_audit["paths"]["regions_csv"]
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 162
    assert len({row["frame_scoped_evidence_id_derived"] for row in rows}) == 162
    assert len({row["legacy_region_id"] for row in rows}) == 51
    assert "frame_scoped_evidence_id" not in rows[0]

    source_record = generated_audit["audit"]["phase2_corpus"]["records"][0]
    source_region = source_record["regions"][0]
    csv_region = next(
        row
        for row in rows
        if row["scenario_id"] == source_record["scenario_id"]
        and row["legacy_region_id"] == source_region["region_id"]
    )
    assert json.loads(csv_region["region_json"]) == source_region
    assert json.loads(csv_region["ground_truth_arguments_json"]) == source_record[
        "ground_truth_arguments"
    ]
    assert json.loads(csv_region["argument_provenance_json"]) == source_record[
        "argument_provenance"
    ]


def test_audit_asserts_frame_scoped_ids_are_required(
    generated_audit: dict[str, Any],
) -> None:
    finding = generated_audit["audit"]["frame_scoped_region_id_analysis"]

    assert finding["frame_scoped_ids_required"] is True
    assert finding["legacy_region_occurrence_count"] == 162
    assert finding["distinct_legacy_region_id_count"] == 51
    assert finding["reused_legacy_region_id_count"] == 33
    assert finding["derived_frame_scoped_id_count"] == 162
    assert finding["derived_frame_scoped_ids_unique"] is True
    assert "p2_call_hotel:trusted_reference" in finding["reused_legacy_region_ids"]

    derived = generated_audit["audit"]["compatibility_derived"]
    assert derived["label"] == (
        "PHASE3.5 COMPATIBILITY-DERIVED; NOT FROZEN SOURCE ANNOTATION"
    )
    assert len(derived["records"]) == 81


def test_unsupported_actions_and_physical_conditions_are_explicitly_not_measurable(
    generated_audit: dict[str, Any],
) -> None:
    audit = generated_audit["audit"]
    measurability = audit["measurability"]

    assert measurability["actions"]["SAFETY_ADVICE"] == corpus_audit.NOT_MEASURABLE
    assert (
        measurability["actions"]["RESTAURANT_RESERVATION"]
        == corpus_audit.NOT_MEASURABLE
    )
    assert set(measurability["physical_capture_conditions"]) == {
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
    }
    assert set(measurability["physical_capture_conditions"].values()) == {
        corpus_audit.NOT_MEASURABLE
    }

    markdown = generated_audit["paths"]["markdown"].read_text(encoding="utf-8")
    assert "SAFETY_ADVICE | NOT MEASURABLE IN CURRENT CORPUS" in markdown
    assert "RESTAURANT_RESERVATION | NOT MEASURABLE IN CURRENT CORPUS" in markdown
    assert "C6 | NOT MEASURABLE IN CURRENT CORPUS" in markdown
    assert "Frame-scoped IDs are therefore required" in markdown


@pytest.mark.parametrize(
    "relative_output",
    [
        "dataset_phase2/audit",
        "results_phase2/audit",
        "results_phase2_5/audit",
    ],
)
def test_frozen_output_directories_are_rejected(
    tmp_path: Path,
    relative_output: str,
) -> None:
    with pytest.raises(corpus_audit.CorpusAuditError, match="frozen directory"):
        corpus_audit._validate_output_directory(
            tmp_path / relative_output,
            tmp_path,
        )
    assert not (tmp_path / relative_output).exists()


def test_failed_lock_verification_occurs_before_any_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "results_phase3_5/grounded-provenance-v1"

    def fail_lock(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert not output_dir.exists()
        raise Phase2BenchmarkLockError("intentional lock failure")

    monkeypatch.setattr(corpus_audit, "verify_phase2_benchmark_lock", fail_lock)

    with pytest.raises(Phase2BenchmarkLockError, match="intentional lock failure"):
        corpus_audit.generate_corpus_audit(output_dir, project_root=PROJECT_ROOT)
    assert not output_dir.exists()
