#!/usr/bin/env python3
"""Load and verify the immutable LensGuard Phase 2 scientific benchmark lock.

The verifier is intentionally provider-independent and read-only.  Phase 2.5
providers may add runtime code and metadata, but a benchmark run must fail
before inference if any frozen dataset, prompt, schema, mapper, policy, gate,
attack evaluator, or metric implementation differs from the locked baseline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK_PATH = PROJECT_ROOT / "config/phase2_benchmark_lock.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_VERSION_PROVENANCE = {
    "dataset_version": "declared",
    "generator_version": "declared",
    "prompt_version": "declared",
    "policy_version": "declared",
    "action_registry_version": "declared",
    "action_schema_version": "implicit_frozen_by_sha256",
    "evidence_schema_version": "implicit_frozen_by_sha256",
    "evaluator_version": "implicit_frozen_by_sha256",
}
_PROMPT_VERSION_KEYS = {
    "PHASE2_ACTION_PROMPT_VERSION": "action_only",
    "PHASE2_INLINE_PROMPT_VERSION": "inline_provenance",
    "PHASE2_TWO_PASS_PROMPT_VERSION": "two_pass_evidence",
}


class Phase2BenchmarkLockError(ValueError):
    """The Phase 2 benchmark no longer matches its scientific lock."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_within_root(project_root: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise Phase2BenchmarkLockError(f"{label} must be a non-empty relative path")
    declared = Path(relative_path)
    if declared.is_absolute():
        raise Phase2BenchmarkLockError(f"{label} must be relative: {relative_path!r}")
    root = project_root.resolve()
    resolved = (root / declared).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Phase2BenchmarkLockError(
            f"{label} escapes the project root: {relative_path!r}"
        ) from error
    return resolved


def load_phase2_benchmark_lock(path: str | Path = DEFAULT_LOCK_PATH) -> dict[str, Any]:
    """Load a benchmark-lock JSON object without verifying repository contents."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise Phase2BenchmarkLockError(
            f"Could not read Phase 2 benchmark lock: {source}"
        ) from error
    except json.JSONDecodeError as error:
        raise Phase2BenchmarkLockError(
            f"Invalid Phase 2 benchmark lock JSON at {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise Phase2BenchmarkLockError("Phase 2 benchmark lock must be a JSON object")
    return payload


def compute_image_tree_sha256(
    project_root: str | Path,
    image_root: str | Path,
    pattern: str = "*.png",
) -> tuple[str, list[str]]:
    """Hash sorted per-image SHA-256 lines using the manifest's documented recipe."""

    root = Path(project_root).resolve()
    images = _resolve_within_root(root, str(image_root), label="dataset image root")
    if not isinstance(pattern, str) or not pattern:
        raise Phase2BenchmarkLockError("dataset image glob must be a non-empty string")
    if not images.is_dir():
        raise Phase2BenchmarkLockError(f"Dataset image root is not a directory: {images}")

    entries: list[tuple[str, Path]] = []
    for candidate in images.glob(pattern):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise Phase2BenchmarkLockError(
                f"Dataset image escapes the project root: {candidate}"
            ) from error
        entries.append((relative, resolved))
    entries.sort(key=lambda item: item[0])
    lines = "".join(f"{sha256_file(path)}  {relative}\n" for relative, path in entries)
    return _sha256_bytes(lines.encode("utf-8")), [relative for relative, _ in entries]


def _read_data_file(path: Path, data_format: str) -> Any:
    if data_format == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if data_format == "yaml":
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - project dependency guard
            raise Phase2BenchmarkLockError(
                "PyYAML is required to verify Phase 2 YAML declarations"
            ) from error
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise Phase2BenchmarkLockError(f"Unsupported declaration format: {data_format!r}")


def _nested_value(payload: Any, key_path: Sequence[Any]) -> Any:
    current = payload
    for key in key_path:
        if not isinstance(key, str) or not isinstance(current, Mapping) or key not in current:
            rendered = "/".join(str(item) for item in key_path)
            raise Phase2BenchmarkLockError(f"Declared key path was not found: {rendered}")
        current = current[key]
    return current


def _python_literal_constant(path: Path, constant_name: str) -> Any:
    """Read a top-level Python literal without importing provider dependencies."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        value_node: ast.expr | None = None
        names: list[str] = []
        if isinstance(statement, ast.Assign):
            value_node = statement.value
            names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            value_node = statement.value
            names = [statement.target.id]
        if constant_name not in names or value_node is None:
            continue
        try:
            return ast.literal_eval(value_node)
        except (TypeError, ValueError) as error:
            raise Phase2BenchmarkLockError(
                f"Prompt constant {constant_name} in {path} is no longer a literal"
            ) from error
    raise Phase2BenchmarkLockError(f"Prompt constant {constant_name} was not found in {path}")


def _validate_manifest_shape(manifest: Mapping[str, Any], errors: list[str]) -> None:
    if manifest.get("manifest_version") != "lensguard-phase2-benchmark-lock-v1":
        errors.append(
            "unsupported manifest_version: "
            f"{manifest.get('manifest_version')!r}"
        )
    if manifest.get("hash_algorithm") != "sha256":
        errors.append(f"unsupported hash_algorithm: {manifest.get('hash_algorithm')!r}")
    provenance = manifest.get("version_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("version_provenance must be an object")
        return
    for name, expected_kind in _REQUIRED_VERSION_PROVENANCE.items():
        entry = provenance.get(name)
        if not isinstance(entry, Mapping):
            errors.append(f"version_provenance is missing {name}")
        elif entry.get("kind") != expected_kind:
            errors.append(
                f"version provenance mismatch for {name}: expected {expected_kind!r}, "
                f"got {entry.get('kind')!r}"
            )


def verify_phase2_benchmark_lock(
    manifest_path: str | Path = DEFAULT_LOCK_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen Phase 2 benchmark and return an auditable summary.

    Every discrepancy is collected so a failed pre-run check identifies all
    drift at once.  The function performs no writes and does not inspect model
    weights, provider caches, or Phase 2.5-only runtime files.
    """

    source = Path(manifest_path)
    manifest = load_phase2_benchmark_lock(source)
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    errors: list[str] = []
    _validate_manifest_shape(manifest, errors)

    verified_files: dict[str, str] = {}
    artifacts = manifest.get("artifacts")
    locked_files = artifacts.get("files") if isinstance(artifacts, Mapping) else None
    if not isinstance(locked_files, Mapping) or not locked_files:
        errors.append("artifacts.files must be a non-empty object")
    else:
        for relative_path, expected_hash in locked_files.items():
            if (
                not isinstance(expected_hash, str)
                or _SHA256_PATTERN.fullmatch(expected_hash) is None
            ):
                errors.append(f"invalid expected SHA-256 for {relative_path!r}: {expected_hash!r}")
                continue
            try:
                path = _resolve_within_root(root, relative_path, label="locked artifact path")
                actual_hash = sha256_file(path)
            except (OSError, Phase2BenchmarkLockError) as error:
                errors.append(f"could not hash locked artifact {relative_path!r}: {error}")
                continue
            verified_files[str(relative_path)] = actual_hash
            if actual_hash != expected_hash:
                errors.append(
                    f"artifact hash mismatch for {relative_path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )

    verified_declarations: dict[str, Any] = {}
    declarations = manifest.get("declared_values")
    if not isinstance(declarations, Mapping) or not declarations:
        errors.append("declared_values must be a non-empty object")
    else:
        data_cache: dict[tuple[Path, str], Any] = {}
        for name, declaration in declarations.items():
            if not isinstance(declaration, Mapping):
                errors.append(f"declared value {name!r} must be an object")
                continue
            try:
                path = _resolve_within_root(
                    root, declaration.get("path"), label=f"declared value {name} path"
                )
                data_format = declaration.get("format")
                if not isinstance(data_format, str):
                    raise Phase2BenchmarkLockError(
                        f"declared value {name} format must be a string"
                    )
                cache_key = (path, data_format)
                if cache_key not in data_cache:
                    data_cache[cache_key] = _read_data_file(path, data_format)
                key_path = declaration.get("key_path")
                if not isinstance(key_path, list) or not key_path:
                    raise Phase2BenchmarkLockError(
                        f"declared value {name} key_path must be a non-empty list"
                    )
                actual = _nested_value(data_cache[cache_key], key_path)
            except (OSError, ValueError, Phase2BenchmarkLockError) as error:
                errors.append(f"could not verify declared value {name}: {error}")
                continue
            expected = declaration.get("expected")
            verified_declarations[str(name)] = actual
            if actual != expected:
                errors.append(
                    f"declared value mismatch for {name}: expected {expected!r}, got {actual!r}"
                )
            manifest_value = manifest.get(str(name))
            if manifest_value != expected:
                errors.append(
                    f"manifest version mismatch for {name}: top-level value "
                    f"{manifest_value!r} != declared expectation {expected!r}"
                )

    verified_prompts: dict[str, Any] = {}
    prompt_constants = manifest.get("prompt_constants")
    if not isinstance(prompt_constants, Mapping) or not prompt_constants:
        errors.append("prompt_constants must be a non-empty object")
    else:
        for name, declaration in prompt_constants.items():
            if not isinstance(declaration, Mapping):
                errors.append(f"prompt constant {name!r} must be an object")
                continue
            try:
                path = _resolve_within_root(
                    root, declaration.get("path"), label=f"prompt constant {name} path"
                )
                actual = _python_literal_constant(path, str(name))
            except (OSError, SyntaxError, Phase2BenchmarkLockError) as error:
                errors.append(f"could not verify prompt constant {name}: {error}")
                continue
            expected = declaration.get("expected")
            verified_prompts[str(name)] = actual
            if actual != expected:
                errors.append(
                    f"prompt constant mismatch for {name}: expected {expected!r}, got {actual!r}"
                )
            prompt_key = _PROMPT_VERSION_KEYS.get(str(name))
            prompt_versions = manifest.get("prompt_version")
            if prompt_key is not None and (
                not isinstance(prompt_versions, Mapping)
                or prompt_versions.get(prompt_key) != expected
            ):
                errors.append(
                    f"manifest prompt_version mismatch for {prompt_key}: expected {expected!r}"
                )

    verified_images: dict[str, Any] = {}
    image_spec = artifacts.get("dataset_images") if isinstance(artifacts, Mapping) else None
    if not isinstance(image_spec, Mapping):
        errors.append("artifacts.dataset_images must be an object")
    else:
        expected_tree_hash = image_spec.get("tree_sha256")
        expected_count = image_spec.get("count")
        if (
            not isinstance(expected_tree_hash, str)
            or _SHA256_PATTERN.fullmatch(expected_tree_hash) is None
        ):
            errors.append(f"invalid dataset image tree SHA-256: {expected_tree_hash!r}")
        else:
            try:
                tree_hash, image_paths = compute_image_tree_sha256(
                    root,
                    image_spec.get("root"),
                    str(image_spec.get("glob", "*.png")),
                )
            except (OSError, Phase2BenchmarkLockError) as error:
                errors.append(f"could not verify dataset image tree: {error}")
            else:
                verified_images = {
                    "count": len(image_paths),
                    "tree_sha256": tree_hash,
                }
                if not isinstance(expected_count, int) or isinstance(expected_count, bool):
                    errors.append(f"invalid expected dataset image count: {expected_count!r}")
                elif len(image_paths) != expected_count:
                    errors.append(
                        f"dataset image count mismatch: expected {expected_count}, "
                        f"got {len(image_paths)}"
                    )
                if tree_hash != expected_tree_hash:
                    errors.append(
                        "dataset image tree hash mismatch: "
                        f"expected {expected_tree_hash}, got {tree_hash}"
                    )

    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise Phase2BenchmarkLockError(
            "Phase 2 benchmark lock verification failed:\n" + details
        )

    return {
        "verified": True,
        "benchmark_id": manifest.get("benchmark_id"),
        "manifest_version": manifest.get("manifest_version"),
        "manifest_path": str(source.resolve()),
        "manifest_sha256": sha256_file(source),
        "baseline_git_commit": manifest.get("baseline_git_commit"),
        "verified_file_count": len(verified_files),
        "declared_values": verified_declarations,
        "prompt_constants": verified_prompts,
        "dataset_images": verified_images,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_phase2_benchmark_lock(
            args.manifest,
            project_root=args.project_root,
        )
    except Phase2BenchmarkLockError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
