"""Replay DIRECT structural parsing and verify input/response identity; no ground truth scoring."""

from __future__ import annotations

import argparse
import subprocess

from benchmark_physical_direct import MODEL_IDS, SMOKE_ROOT, validate_provider
from cloud_baseline_store import read_json, write_index
from physical_direct_inputs import (
    BASELINE_HEAD, INPUT_MANIFEST, RESULT_ROOT, ROOT, inspect_archive, load_manifest,
)


def validate_all() -> dict:
    load_manifest()
    inspect_archive(ROOT / "TestData.zip")
    identities = set()
    result = {}
    for run_type, root, expected in (("full", RESULT_ROOT, 270), ("smoke", SMOKE_ROOT, 20)):
        models = {alias: validate_provider(root, alias, require_complete=True) for alias in MODEL_IDS}
        records = [read_json(p) for p in (root / "records").glob("*/*/*.json")]
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
    if historical & changed:
        raise ValueError("Historical tracked files changed")
    return {"valid": True, "input_manifest": str(INPUT_MANIFEST.relative_to(ROOT)),
            "unique_identities": len(identities), "historical_files_unchanged": True,
            "raw_response_and_normalization_replay": "PASS", "results": result,
            "ground_truth_frozen": False, "scientific_correctness_scoring": "NOT PERFORMED",
            "oracle_started": False, "automatic_started": False, "gate_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-validation", action="store_true")
    args = parser.parse_args()
    import json
    report = validate_all()
    if args.write_validation:
        write_index(RESULT_ROOT / "validation.json", report)
    print(json.dumps(report, indent=2))
