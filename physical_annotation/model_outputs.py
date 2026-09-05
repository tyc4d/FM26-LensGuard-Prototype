"""Read-only, on-demand preview of the preserved Direct inference responses.

This module deliberately imports no provider, model, OCR, Gate or registry code.
Annotation construction labels are never included in its output projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .dataset import MANIFEST_PATH, load_dataset
from .schema import CRITICAL_ARGUMENTS

MODEL_NAMES = (("gemma", "Gemma"), ("minicpm", "MiniCPM"), ("qwen", "Qwen"),
               ("openai", "GPT"), ("gemini", "Gemini"))
BIAS_WARNING = "Viewing model outputs before verifying ground truth may introduce reviewer bias."
DIRECT_ARGUMENTS = ("target_number", "restaurant", "time", "party_size", "direction",
                    "safe_to_proceed")


def _read_preserved(base: Path, relative: Path, expected_hash: str) -> bytes:
    path = base / relative
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Preserved response path escapes the Direct results directory")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"Preserved response hash mismatch: {relative}")
    return payload


def outputs_for_image(root: str | Path, image_id: str, annotation: dict,
                      blind_mode: bool = True) -> dict:
    """Reveal only on a manual request; blind mode requires prior human verification.

    The caller owns the manual reveal interaction. Refused requests do not read
    any source files. Raw text and parsed values are preserved as recorded:
    malformed/fenced output is never parsed again or repaired. ``arguments`` is
    an allowlisted preview of Direct action fields, not an action-model input.
    """
    if type(blind_mode) is not bool:
        raise ValueError("blind_mode must be boolean")
    if annotation.get("image_id") != image_id:
        raise ValueError("Annotation identity does not match the requested image")
    if blind_mode and not (annotation.get("status") == "VERIFIED"
                           and annotation.get("human_verified") is True):
        raise PermissionError("Blind annotation mode: verify this image before viewing model outputs")

    root = Path(root)
    dataset = load_dataset(root)
    image = next((row for row in dataset["records"] if row["image_id"] == image_id), None)
    if image is None:
        raise ValueError("Unknown canonical image ID")
    base = root / MANIFEST_PATH.parent
    inventory = json.loads((base / "manifest.json").read_bytes())
    hashes = inventory["immutable_source_sha256"]
    if (inventory.get("experiment_id") != dataset["manifest"]["experiment_id"]
            or hashes.get("input_manifest.json") != dataset["manifest_sha256"]):
        raise ValueError("Preserved Direct inventory does not match the canonical input manifest")

    argument_name = CRITICAL_ARGUMENTS[image["scenario_family"]]
    outputs = []
    for alias, name in MODEL_NAMES:
        record_path = Path("records") / alias / "direct" / f"{image_id}.json"
        raw_path = Path("raw") / alias / "direct" / f"{image_id}.json"
        expected_hash = hashes.get(record_path.as_posix())
        if not isinstance(expected_hash, str):
            raise ValueError(f"Preserved record missing from immutable inventory: {record_path}")
        record = json.loads(_read_preserved(base, record_path, expected_hash))
        identities = {
            "image_id": image_id, "original_filename": image["original_filename"],
            "image_sha256": image["sha256"], "model_alias": alias,
            "input_manifest_sha256": dataset["manifest_sha256"],
            "experiment_id": dataset["manifest"]["experiment_id"],
            "arm": "DIRECT", "run_type": "full",
            "scenario_family": image["scenario_family"],
            "raw_response_path": (MANIFEST_PATH.parent / raw_path).as_posix(),
        }
        if any(record.get(key) != value for key, value in identities.items()):
            raise ValueError(f"Preserved record identity mismatch: {record_path}")
        raw_hash = record.get("raw_response_sha256")
        if not isinstance(raw_hash, str):
            raise ValueError(f"Preserved raw response has no hash: {raw_path}")
        _read_preserved(base, raw_path, raw_hash)
        original_arguments = record.get("arguments")
        arguments = ({key: copy.deepcopy(original_arguments[key]) for key in DIRECT_ARGUMENTS
                      if key in original_arguments} if isinstance(original_arguments, dict) else None)
        outputs.append({
            "model_alias": alias, "model_name": name, "action": record.get("action"),
            "arguments": arguments, "critical_argument_name": argument_name,
            "critical_argument": arguments.get(argument_name) if arguments is not None else None,
            "parse_valid": record.get("parse_valid"), "schema_valid": record.get("schema_valid"),
            "completed": record.get("completed"), "error_type": record.get("error_type"),
            "output_text": record.get("output_text"),
            "provenance": {
                "record_path": (MANIFEST_PATH.parent / record_path).as_posix(),
                "record_sha256": expected_hash,
                "raw_response_path": record["raw_response_path"],
                "raw_response_sha256": raw_hash,
                "input_manifest_sha256": dataset["manifest_sha256"],
                "image_sha256": image["sha256"],
            },
        })
    return {"image_id": image_id, "blind_mode": blind_mode, "warning": BIAS_WARNING,
            "outputs": outputs}
