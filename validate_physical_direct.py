"""Replay DIRECT structural parsing and verify input/response identity; no ground truth scoring."""

from __future__ import annotations

import argparse
import subprocess

from benchmark_physical_direct import MODEL_IDS, SMOKE_ROOT, smoke_root, validate_provider
from cloud_baseline_store import read_json, write_index
from physical_direct_inputs import (
    BASELINE_HEAD, INPUT_MANIFEST, RESULT_ROOT, ROOT, inspect_archive, load_manifest,
)


def validate_current() -> dict:
    """Validate preserved trials, including incomplete runs, without writing indexes."""
    manifest = load_manifest()
    originals = inspect_archive(ROOT / "TestData.zip")
    for actual, expected in zip(originals, manifest["records"], strict=True):
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("Archive metadata differs from frozen input manifest")
    results = {}
    for path in sorted(RESULT_ROOT.glob("**/plans/*.json")):
        plan = read_json(path)
        key = str(path.relative_to(RESULT_ROOT))
        results[key] = validate_provider(path.parent.parent, plan["provider"])
    full_recorded = sum(value["recorded_trials"] for key, value in results.items()
                        if key.startswith("plans/"))
    return {"validation_version": "physical-direct-current-v1", "valid": True,
            "full_planned_trials": 54 * len(MODEL_IDS),
            "full_recorded_trials": full_recorded,
            "full_missing_trials": 54 * len(MODEL_IDS) - full_recorded,
            "full_models_without_plan": [alias for alias in MODEL_IDS
                                         if f"plans/{alias}.json" not in results],
            "image_count": len(originals), "scope": "existing artifacts only",
            "completeness_required": False, "scientific_scoring": "NOT PERFORMED",
            "results": results}


def validate_all() -> dict:
    load_manifest()
    inspect_archive(ROOT / "TestData.zip")
    diagnostic = validate_provider(SMOKE_ROOT, "openai")
    if diagnostic["recorded_trials"] != 1 or diagnostic["completed_trials"] != 0:
        raise ValueError("Unexpected compatibility diagnostic responses")
    identities = set()
    result = {}
    for run_type, root, expected in (("full", RESULT_ROOT, 270), ("smoke", SMOKE_ROOT, 20)):
        roots = {a: smoke_root(a) if run_type == "smoke" else root for a in MODEL_IDS}
        models = {a: validate_provider(roots[a], a, require_complete=True) for a in MODEL_IDS}
        records = [read_json(p) for a in MODEL_IDS
                   for p in (roots[a] / "records" / a).glob("*/*.json")]
        if len(records) != expected:
            raise ValueError("Incorrect full/smoke trial count")
        for row in records:
            identity = (row["image_id"], row["model_alias"], row["run_type"])
            if identity in identities or row["run_type"] != run_type or row["arm"] != "DIRECT":
                raise ValueError("Duplicate identity or unexpected experiment arm")
            identities.add(identity)
        result[run_type] = {"expected_trials": expected, "recorded_trials": len(records),
                            "completed_trials": sum(r["completed"] for r in records), "models": models}
    historical = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE_HEAD], cwd=ROOT, text=True).splitlines())
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", BASELINE_HEAD], cwd=ROOT, text=True).splitlines())
    for name in historical & changed:
        if name in {"README.md", ".gitignore"}:
            original = subprocess.check_output(["git", "show", f"{BASELINE_HEAD}:{name}"], cwd=ROOT)
            if (ROOT / name).read_bytes().startswith(original):
                continue
        raise ValueError(f"Historical tracked file changed: {name}")
    return {"valid": True, "input_manifest": str(INPUT_MANIFEST.relative_to(ROOT)),
            "unique_identities": len(identities), "historical_files_unchanged": True,
            "raw_response_and_normalization_replay": "PASS", "results": result,
            "additional_api_compatibility_diagnostic": {
                "recorded_requests": 1, "model_responses": 0,
                "path": str((SMOKE_ROOT / "raw/openai/direct/IMG_3483.jpeg.json").relative_to(ROOT)),
                "included_in_full_or_active_smoke": False},
            "ground_truth_frozen": False, "scientific_correctness_scoring": "NOT PERFORMED",
            "oracle_started": False, "automatic_started": False, "gate_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-validation", action="store_true")
    parser.add_argument("--current", action="store_true",
                        help="Read-only validation of existing, possibly incomplete runs")
    args = parser.parse_args()
    import json
    if args.current and args.write_validation:
        parser.error("--current is read-only; redirect stdout to a separate snapshot file")
    report = validate_current() if args.current else validate_all()
    if args.write_validation:
        write_index(RESULT_ROOT / "validation.json", report)
    print(json.dumps(report, indent=2))
