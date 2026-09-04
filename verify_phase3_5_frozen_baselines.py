#!/usr/bin/env python3
"""Read-only integrity check for the baselines consumed by Phase 3.5."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2_benchmark_lock import PROJECT_ROOT, verify_phase2_benchmark_lock


PHASE2_5_ROOT = PROJECT_ROOT / "results_phase2_5/contract-v2-full"
EXPECTED_PHASE2_5_TREE_SHA256 = (
    "378eb1c17b1bb536ca3b19d65c1b9c33376b8fa959894de1f49b99f599a9ef5e"
)
EXPECTED_PHASE2_5_FILE_COUNT = 763
EXPECTED_PHASE2_5_REPORT_SHA256 = (
    "29578fc61133b0dace955e51dc1b0fe5a5f8ee786230e9e253bffee7abf7358a"
)
EXPECTED_ANALYSIS_SHA256 = {
    "gemma3-4b": "c3259677642656230c2f8c3299c788ac42162a21afa9e48da2be48ac3a8a5294",
    "minicpm-v4.5": "513816a02e7e00856d4a76b4539949f2790aa0d5fa0d50ac6d208b768a4b9975",
    "qwen3vl-8b": "d88030c2e9e6d954974cf8a4c3671b0f0bf7f061906b9b6a510d0386590d5485",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase2_5_tree_sha256(root: Path = PHASE2_5_ROOT) -> tuple[str, int]:
    """Match `find | sort | sha256sum | sha256sum` from the project root."""

    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        project_relative = path.relative_to(PROJECT_ROOT)
        digest.update(
            f"{sha256_file(path)}  {project_relative.as_posix()}\n".encode("utf-8")
        )
    return digest.hexdigest(), len(files)


def verify_frozen_baselines() -> dict[str, Any]:
    phase2 = verify_phase2_benchmark_lock()
    tree_hash, file_count = phase2_5_tree_sha256()
    report_hash = sha256_file(PHASE2_5_ROOT / "report_local_models.md")
    models: dict[str, Any] = {}
    errors: list[str] = []
    if tree_hash != EXPECTED_PHASE2_5_TREE_SHA256:
        errors.append("Phase 2.5 canonical result tree SHA-256 changed")
    if file_count != EXPECTED_PHASE2_5_FILE_COUNT:
        errors.append("Phase 2.5 canonical result file count changed")
    if report_hash != EXPECTED_PHASE2_5_REPORT_SHA256:
        errors.append("Phase 2.5 aggregate report SHA-256 changed")
    for model, expected_analysis in EXPECTED_ANALYSIS_SHA256.items():
        directory = PHASE2_5_ROOT / model
        analysis_hash = sha256_file(directory / "analysis.json")
        raw_count = sum(
            1 for line in (directory / "raw_generations.jsonl").open(encoding="utf-8")
            if line.strip()
        )
        models[model] = {
            "analysis_sha256": analysis_hash,
            "expected_analysis_sha256": expected_analysis,
            "raw_generation_rows": raw_count,
        }
        if analysis_hash != expected_analysis:
            errors.append(f"Phase 2.5 {model} analysis SHA-256 changed")
        if raw_count != 243:
            errors.append(f"Phase 2.5 {model} raw-generation row count changed")
    result = {
        "schema_version": "phase3.5-frozen-baseline-integrity-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "phase2": phase2,
        "phase2_5": {
            "root": str(PHASE2_5_ROOT),
            "file_count": file_count,
            "expected_file_count": EXPECTED_PHASE2_5_FILE_COUNT,
            "tree_sha256": tree_hash,
            "expected_tree_sha256": EXPECTED_PHASE2_5_TREE_SHA256,
            "aggregate_report_sha256": report_hash,
            "expected_aggregate_report_sha256": EXPECTED_PHASE2_5_REPORT_SHA256,
            "models": models,
        },
        "verified": not errors,
        "errors": errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_frozen_baselines()
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
