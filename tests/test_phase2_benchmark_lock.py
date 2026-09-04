from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from phase2_benchmark_lock import (
    Phase2BenchmarkLockError,
    compute_image_tree_sha256,
    sha256_file,
    verify_phase2_benchmark_lock,
)


def _version_provenance() -> dict[str, dict[str, str]]:
    declared = {
        "dataset_version",
        "generator_version",
        "prompt_version",
        "policy_version",
        "action_registry_version",
    }
    implicit = {
        "action_schema_version",
        "evidence_schema_version",
        "evaluator_version",
    }
    return {
        **{name: {"kind": "declared", "source": "test"} for name in declared},
        **{
            name: {"kind": "implicit_frozen_by_sha256", "source": "test"}
            for name in implicit
        },
    }


def _temporary_lock(
    root: Path,
    *,
    actual_dataset_version: str = "dataset-v1",
    expected_dataset_version: str = "dataset-v1",
) -> tuple[Path, dict[str, Any]]:
    (root / "dataset/images").mkdir(parents=True)
    (root / "providers").mkdir()
    metadata = root / "dataset/metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "dataset_version": actual_dataset_version,
                "generator_version": "generator-v1",
            }
        ),
        encoding="utf-8",
    )
    prompt = root / "providers/prompts.py"
    prompt.write_text(
        'PHASE2_ACTION_PROMPT_VERSION = "action-v1"\n',
        encoding="utf-8",
    )
    frozen = root / "frozen.txt"
    frozen.write_text("immutable semantics\n", encoding="utf-8")
    (root / "dataset/images/a.png").write_bytes(b"image-a")
    (root / "dataset/images/b.png").write_bytes(b"image-b")
    image_hash, image_paths = compute_image_tree_sha256(root, "dataset/images")

    manifest: dict[str, Any] = {
        "manifest_version": "lensguard-phase2-benchmark-lock-v1",
        "benchmark_id": "temporary-phase2-lock",
        "baseline_git_commit": "test-commit",
        "hash_algorithm": "sha256",
        "dataset_version": expected_dataset_version,
        "generator_version": "generator-v1",
        "prompt_version": {"action_only": "action-v1"},
        "policy_version": "policy-v1",
        "action_registry_version": "registry-v1",
        "action_schema_version": "action-schema-v1",
        "evidence_schema_version": "evidence-schema-v1",
        "evaluator_version": "evaluator-v1",
        "version_provenance": _version_provenance(),
        "declared_values": {
            "dataset_version": {
                "path": "dataset/metadata.json",
                "format": "json",
                "key_path": ["dataset_version"],
                "expected": expected_dataset_version,
            }
        },
        "prompt_constants": {
            "PHASE2_ACTION_PROMPT_VERSION": {
                "path": "providers/prompts.py",
                "expected": "action-v1",
            }
        },
        "artifacts": {
            "files": {
                "dataset/metadata.json": sha256_file(metadata),
                "providers/prompts.py": sha256_file(prompt),
                "frozen.txt": sha256_file(frozen),
            },
            "dataset_images": {
                "root": "dataset/images",
                "glob": "*.png",
                "count": len(image_paths),
                "tree_sha256": image_hash,
            },
        },
    }
    manifest_path = root / "lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_phase2_benchmark_lock_verifies_temporary_frozen_root(tmp_path: Path) -> None:
    manifest_path, _ = _temporary_lock(tmp_path)

    result = verify_phase2_benchmark_lock(manifest_path, project_root=tmp_path)

    assert result["verified"] is True
    assert result["verified_file_count"] == 3
    assert result["declared_values"] == {"dataset_version": "dataset-v1"}
    assert result["prompt_constants"] == {"PHASE2_ACTION_PROMPT_VERSION": "action-v1"}
    assert result["dataset_images"]["count"] == 2


def test_phase2_benchmark_lock_rejects_tampered_file(tmp_path: Path) -> None:
    manifest_path, _ = _temporary_lock(tmp_path)
    (tmp_path / "frozen.txt").write_text("changed semantics\n", encoding="utf-8")

    with pytest.raises(Phase2BenchmarkLockError, match="artifact hash mismatch for frozen.txt"):
        verify_phase2_benchmark_lock(manifest_path, project_root=tmp_path)


def test_phase2_benchmark_lock_rejects_declared_version_mismatch(tmp_path: Path) -> None:
    manifest_path, _ = _temporary_lock(
        tmp_path,
        actual_dataset_version="dataset-v2",
        expected_dataset_version="dataset-v1",
    )

    with pytest.raises(
        Phase2BenchmarkLockError,
        match="declared value mismatch for dataset_version",
    ):
        verify_phase2_benchmark_lock(manifest_path, project_root=tmp_path)


def test_phase2_benchmark_lock_detects_image_tree_drift(tmp_path: Path) -> None:
    manifest_path, manifest = _temporary_lock(tmp_path)
    original = manifest["artifacts"]["dataset_images"]["tree_sha256"]
    (tmp_path / "dataset/images/c.png").write_bytes(b"image-c")

    changed, paths = compute_image_tree_sha256(tmp_path, "dataset/images")
    assert changed != original
    assert paths == [
        "dataset/images/a.png",
        "dataset/images/b.png",
        "dataset/images/c.png",
    ]
    with pytest.raises(Phase2BenchmarkLockError) as caught:
        verify_phase2_benchmark_lock(manifest_path, project_root=tmp_path)
    assert "dataset image count mismatch" in str(caught.value)
    assert "dataset image tree hash mismatch" in str(caught.value)
