#!/usr/bin/env python3
"""Audit the frozen Phase 2 corpus for Phase 3.5 compatibility.

This module is deliberately read-only with respect to Phase 2 and Phase 2.5.
It verifies the Phase 2 benchmark lock before loading the 81 frozen records and
writes only a lossless inventory and derived compatibility observations to a
caller-selected Phase 3.5 output directory.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from phase2_benchmark_lock import (
    Phase2BenchmarkLockError,
    load_phase2_benchmark_lock,
    sha256_file,
    verify_phase2_benchmark_lock,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results_phase3_5/grounded-provenance-v1"
AUDIT_VERSION = "lensguard-phase3.5-corpus-audit-v1"
EXPECTED_RECORD_COUNT = 81
NOT_MEASURABLE = "NOT MEASURABLE IN CURRENT CORPUS"

_FROZEN_OUTPUT_ROOTS = (
    "dataset_phase2",
    "results_phase2",
    "results_phase2_5",
)
_CURRENT_ACTIONS = ("CALL", "OPEN_URL", "DIRECTION_ADVICE")
_FUTURE_ACTIONS = (
    "CALL",
    "OPEN_URL",
    "DIRECTION_ADVICE",
    "SAFETY_ADVICE",
    "RESTAURANT_RESERVATION",
)
_PHYSICAL_CONDITION_IDS = tuple(f"C{index}" for index in range(7))


class CorpusAuditError(ValueError):
    """The frozen corpus could not be represented by a valid Phase 3.5 audit."""


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_output_directory(output_dir: Path, project_root: Path) -> None:
    """Reject every output target inside a frozen Phase 2/2.5 directory."""

    resolved = output_dir.resolve()
    for relative_root in _FROZEN_OUTPUT_ROOTS:
        frozen_root = (project_root / relative_root).resolve()
        if _is_within(resolved, frozen_root):
            raise CorpusAuditError(
                f"Refusing to write Phase 3.5 audit inside frozen directory: "
                f"{frozen_root}"
            )


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CorpusAuditError(
            f"Could not read frozen Phase 2 metadata: {metadata_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise CorpusAuditError(
            f"Invalid frozen Phase 2 metadata JSON at {metadata_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise CorpusAuditError("Frozen Phase 2 metadata must be a JSON object")
    return payload


def _validate_records(metadata: Mapping[str, Any], project_root: Path) -> list[dict[str, Any]]:
    records = metadata.get("records")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise CorpusAuditError("Frozen Phase 2 metadata records must be a list of objects")
    if metadata.get("record_count") != EXPECTED_RECORD_COUNT:
        raise CorpusAuditError(
            "Frozen Phase 2 declared record_count must be exactly "
            f"{EXPECTED_RECORD_COUNT}, got {metadata.get('record_count')!r}"
        )
    if len(records) != EXPECTED_RECORD_COUNT:
        raise CorpusAuditError(
            f"Expected exactly {EXPECTED_RECORD_COUNT} frozen records, got {len(records)}"
        )

    scenario_ids: list[str] = []
    image_paths: list[str] = []
    for index, record in enumerate(records):
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise CorpusAuditError(f"Record {index} has no valid scenario_id")
        scenario_ids.append(scenario_id)

        image_path = record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise CorpusAuditError(f"Record {scenario_id!r} has no valid image_path")
        resolved_image = _resolve_path(project_root, image_path)
        if not _is_within(resolved_image, project_root):
            raise CorpusAuditError(
                f"Record {scenario_id!r} image escapes the project root: {image_path!r}"
            )
        if not resolved_image.is_file():
            raise CorpusAuditError(
                f"Record {scenario_id!r} image does not exist: {image_path!r}"
            )
        image_paths.append(image_path)

        regions = record.get("regions")
        if not isinstance(regions, list) or any(
            not isinstance(region, dict) for region in regions
        ):
            raise CorpusAuditError(f"Record {scenario_id!r} regions must be a list of objects")
        region_ids = [region.get("region_id") for region in regions]
        if any(not isinstance(region_id, str) or not region_id for region_id in region_ids):
            raise CorpusAuditError(f"Record {scenario_id!r} has an invalid region_id")
        if len(region_ids) != len(set(region_ids)):
            raise CorpusAuditError(
                f"Record {scenario_id!r} repeats a region_id within one frame"
            )

    if len(set(scenario_ids)) != EXPECTED_RECORD_COUNT:
        raise CorpusAuditError("Frozen Phase 2 scenario_id values must be unique")
    if len(set(image_paths)) != EXPECTED_RECORD_COUNT:
        raise CorpusAuditError("Frozen Phase 2 image_path values must be unique")
    if metadata.get("image_count") != EXPECTED_RECORD_COUNT:
        raise CorpusAuditError(
            "Frozen Phase 2 declared image_count must be exactly "
            f"{EXPECTED_RECORD_COUNT}, got {metadata.get('image_count')!r}"
        )
    return records


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _critical_arguments_by_action(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for record in records:
        action = record.get("action_family")
        names = record.get("critical_argument_names")
        if not isinstance(action, str) or not isinstance(names, list) or any(
            not isinstance(name, str) for name in names
        ):
            raise CorpusAuditError("Every record must declare action and critical arguments")
        existing = result.setdefault(action, [])
        result[action] = _ordered_unique([*existing, *names])
    return dict(sorted(result.items()))


def _frame_scoped_id_analysis(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    legacy_frames: dict[str, list[str]] = defaultdict(list)
    scoped_ids: list[str] = []
    projections: list[dict[str, Any]] = []

    for record in records:
        frame_id = str(record["scenario_id"])
        critical_names = list(record["critical_argument_names"])
        ground_truth_arguments = record.get("ground_truth_arguments")
        if not isinstance(ground_truth_arguments, Mapping):
            raise CorpusAuditError(
                f"Record {frame_id!r} ground_truth_arguments must be an object"
            )
        critical_arguments = {
            name: ground_truth_arguments[name]
            for name in critical_names
            if name in ground_truth_arguments
        }
        if len(critical_arguments) != len(critical_names):
            raise CorpusAuditError(
                f"Record {frame_id!r} lacks a declared critical argument value"
            )

        derived_regions: list[dict[str, str]] = []
        for region in record["regions"]:
            legacy_id = str(region["region_id"])
            legacy_frames[legacy_id].append(frame_id)
            scoped_id = f"{frame_id}:{legacy_id}"
            scoped_ids.append(scoped_id)
            derived_regions.append(
                {
                    "legacy_region_id": legacy_id,
                    "frame_scoped_evidence_id": scoped_id,
                }
            )

        attacker_target = record.get("attacker_target")
        primary_critical_name = record.get("critical_argument_name")
        derived_attack_target: dict[str, Any] | None = None
        if attacker_target is not None and isinstance(primary_critical_name, str):
            derived_attack_target = {primary_critical_name: attacker_target}

        projections.append(
            {
                "source_scenario_id": frame_id,
                "frame_id": frame_id,
                "action": record.get("action_family"),
                "critical_arguments": critical_arguments,
                "attack_target": derived_attack_target,
                "regions": derived_regions,
            }
        )

    reused = {
        legacy_id: _ordered_unique(frames)
        for legacy_id, frames in sorted(legacy_frames.items())
        if len(set(frames)) > 1
    }
    total = sum(len(record["regions"]) for record in records)
    if not reused:
        raise CorpusAuditError(
            "Expected legacy region IDs to be reused across Phase 2 frames; "
            "the frame-scoping premise can no longer be asserted"
        )
    if len(scoped_ids) != total or len(set(scoped_ids)) != total:
        raise CorpusAuditError("Derived frame-scoped evidence IDs are not unique")

    analysis = {
        "frame_scoped_ids_required": True,
        "assertion": (
            "Legacy region IDs are reused across condition-specific image frames; "
            "a globally addressed Evidence Registry must include frame identity."
        ),
        "derivation": "<scenario_id>:<legacy_region_id>",
        "legacy_region_occurrence_count": total,
        "distinct_legacy_region_id_count": len(legacy_frames),
        "reused_legacy_region_id_count": len(reused),
        "reused_legacy_region_ids": reused,
        "derived_frame_scoped_id_count": len(scoped_ids),
        "derived_frame_scoped_ids_unique": True,
    }
    return analysis, projections


def _build_measurability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {str(record.get("action_family")) for record in records}
    action_status = {
        action: (
            "MEASURABLE IN CURRENT CORPUS" if action in observed else NOT_MEASURABLE
        )
        for action in _FUTURE_ACTIONS
    }
    return {
        "actions": action_status,
        "future_physical_scenario_families": {
            "CALL": "COMPATIBLE ACTION PRESENT; PHYSICAL CAPTURE NOT EVALUATED",
            "NAVIGATION": (
                "COMPATIBLE VIA DIRECTION_ADVICE; PHYSICAL CAPTURE NOT EVALUATED"
            ),
            "SAFETY": NOT_MEASURABLE,
            "RESTAURANT_RESERVATION": NOT_MEASURABLE,
        },
        "physical_capture_conditions": {
            condition_id: NOT_MEASURABLE for condition_id in _PHYSICAL_CONDITION_IDS
        },
        "physical_capture_dimensions": {
            "distance_m": NOT_MEASURABLE,
            "camera_angle_deg": NOT_MEASURABLE,
            "lighting_class": NOT_MEASURABLE,
            "measured_lux": NOT_MEASURABLE,
            "attack_position": NOT_MEASURABLE,
        },
        "warning": (
            "The current condition labels are synthetic semantic/security conditions, "
            "not the future C0-C6 physical capture conditions."
        ),
    }


def build_corpus_audit(
    metadata: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    metadata_path: str | Path | None = None,
    lock_path: str | Path | None = None,
    lock_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an in-memory audit after the caller has verified the Phase 2 lock.

    ``phase2_corpus`` is a lossless JSON-value copy of the source metadata.
    Phase 3.5 interpretations are confined to ``compatibility_derived``.
    """

    root = Path(project_root).resolve()
    source_metadata_path = (
        _resolve_path(root, metadata_path)
        if metadata_path is not None
        else root / "dataset_phase2/metadata.json"
    )
    source_lock_path = (
        _resolve_path(root, lock_path)
        if lock_path is not None
        else root / "config/phase2_benchmark_lock.json"
    )
    records = _validate_records(metadata, root)
    actions = Counter(str(record["action_family"]) for record in records)
    conditions = Counter(str(record["condition"]) for record in records)
    partitions = Counter(str(record["dataset_partition"]) for record in records)
    source_types = Counter(
        str(region.get("source_type"))
        for record in records
        for region in record["regions"]
    )
    claimed_authorities = Counter(
        str(region.get("content_claimed_authority"))
        for record in records
        for region in record["regions"]
    )
    frame_analysis, compatibility_records = _frame_scoped_id_analysis(records)
    lock_manifest = load_phase2_benchmark_lock(source_lock_path)

    image_hashes: dict[str, str] = {}
    for record in records:
        image_path = str(record["image_path"])
        image_hashes[image_path] = sha256_file(_resolve_path(root, image_path))

    region_count = sum(len(record["regions"]) for record in records)
    audit = {
        "audit_version": AUDIT_VERSION,
        "audit_scope": "FROZEN_PHASE2_CORPUS_FOR_PHASE3_5_COMPATIBILITY",
        "not_measurable_marker": NOT_MEASURABLE,
        "frozen_lock_verification": dict(lock_verification),
        "sources": {
            "metadata_path": _display_path(source_metadata_path, root),
            "metadata_sha256": sha256_file(source_metadata_path),
            "benchmark_lock_path": _display_path(source_lock_path, root),
            "benchmark_lock_sha256": sha256_file(source_lock_path),
            "locked_artifact_sha256": dict(
                lock_manifest.get("artifacts", {}).get("files", {})
            ),
            "image_tree_sha256": lock_verification.get("dataset_images", {}).get(
                "tree_sha256"
            ),
            "image_sha256": image_hashes,
        },
        "counts": {
            "records": len(records),
            "images": len(image_hashes),
            "semantic_scenarios": len(
                {str(record["base_scenario_id"]) for record in records}
            ),
            "region_occurrences": region_count,
            "actions": dict(sorted(actions.items())),
            "conditions": dict(sorted(conditions.items())),
            "dataset_partitions": dict(sorted(partitions.items())),
            "region_source_types": dict(sorted(source_types.items())),
            "region_claimed_authorities": dict(sorted(claimed_authorities.items())),
        },
        "observed_annotation_capabilities": {
            "actions": list(_CURRENT_ACTIONS),
            "critical_arguments_by_action": _critical_arguments_by_action(records),
            "record_fields": sorted({key for record in records for key in record}),
            "region_fields": sorted(
                {key for record in records for region in record["regions"] for key in region}
            ),
            "note": (
                "Phase 2 source_type and content_claimed_authority are retained exactly. "
                "They are not relabelled as Phase 3.5 physical_source, semantic_role, "
                "control_class, or confidence."
            ),
            "not_present_and_not_inferred": [
                "physical_source",
                "semantic_role",
                "control_class",
                "detection_confidence",
                "ocr_confidence",
                "grounding_confidence",
            ],
        },
        "measurability": _build_measurability(records),
        "frame_scoped_region_id_analysis": frame_analysis,
        "compatibility_derived": {
            "label": "PHASE3.5 COMPATIBILITY-DERIVED; NOT FROZEN SOURCE ANNOTATION",
            "records": compatibility_records,
        },
        "phase2_corpus": metadata,
    }
    return audit


_REGION_COLUMNS = (
    "scenario_id",
    "base_scenario_id",
    "condition",
    "action_family",
    "user_prompt",
    "image_path",
    "critical_argument_names_json",
    "ground_truth_arguments_json",
    "attacker_target_json",
    "attacker_target_source",
    "attack_source",
    "argument_provenance_json",
    "visual_argument_provenance_json",
    "verified_reference_json",
    "reference_region_id",
    "alternate_region_id",
    "expected_selected_region_id",
    "region_index",
    "legacy_region_id",
    "frame_scoped_evidence_id_derived",
    "bbox_json",
    "bbox_coordinate_space",
    "region_text",
    "source_type",
    "content_claimed_authority",
    "trusted_source",
    "claims_json",
    "visual_features_json",
    "region_json",
)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_regions_csv(audit: Mapping[str, Any]) -> str:
    """Render one CSV row per source region without mutating source values."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_REGION_COLUMNS, lineterminator="\n")
    writer.writeheader()
    records = audit["phase2_corpus"]["records"]
    for record in records:
        for region_index, region in enumerate(record["regions"]):
            writer.writerow(
                {
                    "scenario_id": record["scenario_id"],
                    "base_scenario_id": record["base_scenario_id"],
                    "condition": record["condition"],
                    "action_family": record["action_family"],
                    "user_prompt": record["user_prompt"],
                    "image_path": record["image_path"],
                    "critical_argument_names_json": _json_cell(
                        record["critical_argument_names"]
                    ),
                    "ground_truth_arguments_json": _json_cell(
                        record["ground_truth_arguments"]
                    ),
                    "attacker_target_json": _json_cell(record.get("attacker_target")),
                    "attacker_target_source": record.get("attacker_target_source"),
                    "attack_source": record.get("attack_source"),
                    "argument_provenance_json": _json_cell(
                        record.get("argument_provenance")
                    ),
                    "visual_argument_provenance_json": _json_cell(
                        record.get("visual_argument_provenance")
                    ),
                    "verified_reference_json": _json_cell(record.get("verified_reference")),
                    "reference_region_id": record.get("reference_region_id"),
                    "alternate_region_id": record.get("alternate_region_id"),
                    "expected_selected_region_id": record.get(
                        "expected_selected_region_id"
                    ),
                    "region_index": region_index,
                    "legacy_region_id": region["region_id"],
                    "frame_scoped_evidence_id_derived": (
                        f"{record['scenario_id']}:{region['region_id']}"
                    ),
                    "bbox_json": _json_cell(region.get("bbox")),
                    "bbox_coordinate_space": region.get("bbox_coordinate_space"),
                    "region_text": region.get("text"),
                    "source_type": region.get("source_type"),
                    "content_claimed_authority": region.get(
                        "content_claimed_authority"
                    ),
                    "trusted_source": region.get("trusted_source"),
                    "claims_json": _json_cell(region.get("claims")),
                    "visual_features_json": _json_cell(region.get("visual_features")),
                    "region_json": _json_cell(region),
                }
            )
    return output.getvalue()


def _markdown_table(rows: Sequence[Sequence[Any]]) -> list[str]:
    if not rows:
        return []
    rendered = [[str(value).replace("|", "\\|") for value in row] for row in rows]
    return [
        "| " + " | ".join(rendered[0]) + " |",
        "| " + " | ".join("---" for _ in rendered[0]) + " |",
        *("| " + " | ".join(row) + " |" for row in rendered[1:]),
    ]


def render_markdown(audit: Mapping[str, Any]) -> str:
    """Render a concise human-readable summary of the complete JSON audit."""

    counts = audit["counts"]
    verification = audit["frozen_lock_verification"]
    frame_ids = audit["frame_scoped_region_id_analysis"]
    measurability = audit["measurability"]
    action_rows: list[list[Any]] = [["Action", "Records", "Critical arguments"]]
    critical = audit["observed_annotation_capabilities"]["critical_arguments_by_action"]
    for action, count in counts["actions"].items():
        action_rows.append([action, count, ", ".join(critical[action])])

    condition_rows: list[list[Any]] = [["Current condition", "Records"]]
    condition_rows.extend(
        [condition, count] for condition, count in counts["conditions"].items()
    )
    support_rows: list[list[Any]] = [["Phase 3.5 action", "Status"]]
    support_rows.extend(measurability["actions"].items())
    physical_rows: list[list[Any]] = [["Future physical condition", "Status"]]
    physical_rows.extend(measurability["physical_capture_conditions"].items())

    lines = [
        "# Phase 3.5 frozen-corpus compatibility audit",
        "",
        "The Phase 2 benchmark lock passed before this audit read or wrote any corpus artifact.",
        "",
        f"- Benchmark: `{verification['benchmark_id']}`",
        f"- Lock SHA-256: `{audit['sources']['benchmark_lock_sha256']}`",
        f"- Metadata SHA-256: `{audit['sources']['metadata_sha256']}`",
        f"- Image-tree SHA-256: `{audit['sources']['image_tree_sha256']}`",
        f"- Records/images: {counts['records']}/{counts['images']}",
        f"- Semantic base scenes: {counts['semantic_scenarios']}",
        f"- Annotated region occurrences: {counts['region_occurrences']}",
        "",
        "## Existing compatible actions",
        "",
        *_markdown_table(action_rows),
        "",
        "## Existing synthetic conditions",
        "",
        *_markdown_table(condition_rows),
        "",
        "## Phase 3.5 measurability",
        "",
        *_markdown_table(support_rows),
        "",
        *_markdown_table(physical_rows),
        "",
        measurability["warning"],
        "",
        "## Region identity finding",
        "",
        (
            f"The {frame_ids['legacy_region_occurrence_count']} region occurrences use "
            f"{frame_ids['distinct_legacy_region_id_count']} distinct legacy IDs; "
            f"{frame_ids['reused_legacy_region_id_count']} legacy IDs occur in more than "
            "one image frame. Frame-scoped IDs are therefore required."
        ),
        "",
        "The JSON keeps the complete frozen metadata payload under `phase2_corpus`. "
        "Frame IDs and evidence IDs appear only under `compatibility_derived` and in "
        "the CSV column explicitly suffixed `_derived`.",
        "",
        "## Annotation limits",
        "",
        "Legacy `source_type` and `content_claimed_authority` labels are preserved. "
        "No `physical_source`, `semantic_role`, `control_class`, perception confidence, "
        "or grounding confidence is inferred. SAFETY and RESTAURANT_RESERVATION are not "
        "represented by experimental cases in this corpus.",
        "",
    ]
    return "\n".join(lines)


def generate_corpus_audit(
    output_dir: str | Path | None = None,
    *,
    project_root: str | Path = PROJECT_ROOT,
    lock_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify, audit, and write the three Phase 3.5 corpus-audit artifacts."""

    root = Path(project_root).resolve()
    source_lock_path = (
        _resolve_path(root, lock_path)
        if lock_path is not None
        else root / "config/phase2_benchmark_lock.json"
    )
    source_metadata_path = (
        _resolve_path(root, metadata_path)
        if metadata_path is not None
        else root / "dataset_phase2/metadata.json"
    )

    # Scientific precondition: this is intentionally the first repository read
    # that can fail and occurs before output validation, mkdir, or file writes.
    lock_verification = verify_phase2_benchmark_lock(
        source_lock_path,
        project_root=root,
    )
    metadata = _read_metadata(source_metadata_path)
    audit = build_corpus_audit(
        metadata,
        project_root=root,
        metadata_path=source_metadata_path,
        lock_path=source_lock_path,
        lock_verification=lock_verification,
    )

    target = (
        _resolve_path(root, output_dir)
        if output_dir is not None
        else root / "results_phase3_5/grounded-provenance-v1"
    )
    _validate_output_directory(target, root)
    if target.exists() and not target.is_dir():
        raise CorpusAuditError(f"Audit output path is not a directory: {target}")
    target.mkdir(parents=True, exist_ok=True)

    json_path = target / "corpus_audit.json"
    csv_path = target / "corpus_audit_regions.csv"
    markdown_path = target / "corpus_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path.write_text(render_regions_csv(audit), encoding="utf-8", newline="")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")

    return {
        "audit": audit,
        "paths": {
            "json": json_path,
            "regions_csv": csv_path,
            "markdown": markdown_path,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--lock", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate_corpus_audit(
            args.output_dir,
            project_root=args.project_root,
            lock_path=args.lock,
            metadata_path=args.metadata,
        )
    except (CorpusAuditError, Phase2BenchmarkLockError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    printable = {
        "verified": result["audit"]["frozen_lock_verification"]["verified"],
        "record_count": result["audit"]["counts"]["records"],
        "region_count": result["audit"]["counts"]["region_occurrences"],
        "paths": {name: str(path) for name, path in result["paths"].items()},
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
