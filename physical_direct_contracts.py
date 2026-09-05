"""Separate direct physical-pilot format and syntax validation.

This module supplies the same prompt and schema to every selected model. It
does not assess whether a proposed action is correct or safe, and makes no
execution decision. The frozen Phase 3.5/3.6 contracts are not imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config/physical_direct_prompts_v1.yaml"
SMOKE_CONFIG_PATH = ROOT / "config/physical_direct_smoke_v1.json"
_CONFIG_BYTES = CONFIG_PATH.read_bytes()
_CONFIG = yaml.safe_load(_CONFIG_BYTES)
CONFIG_SHA256 = hashlib.sha256(_CONFIG_BYTES).hexdigest()
EXPERIMENT_ID = _CONFIG["experiment_id"]
SCHEMA_VERSION = _CONFIG["schema_version"]
PROMPT_VERSION = _CONFIG["prompt_version"]
DIRECT_SCHEMA: dict[str, Any] = copy.deepcopy(_CONFIG["response_schema"])
TASK_PROMPTS: dict[str, str] = dict(_CONFIG["task_prompts"])
GLOBAL_WRAPPER: str = _CONFIG["global_wrapper"]
SMOKE_CASES: tuple[dict[str, str], ...] = tuple(
    json.loads(SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))["cases"]
)
_ACTION_ARGUMENT_SHAPES = {
    action: tuple(frozenset(shape) for shape in shapes)
    for action, shapes in _CONFIG["policy"]["action_argument_shapes"].items()
}


def config_sha256() -> str:
    """Hash the exact config bytes used to construct this module's contract."""
    if hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Physical direct config changed after it was loaded")
    return CONFIG_SHA256


def build_prompt(scenario: str) -> str:
    """Build the common wrapper/schema plus one verbatim predeclared task."""
    if not isinstance(scenario, str) or scenario not in TASK_PROMPTS:
        raise ValueError("Unknown physical direct scenario")
    config_sha256()
    schema = json.dumps(DIRECT_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (
        f"{GLOBAL_WRAPPER}\n\nJSON schema:\n{schema}\n\nUser request:\n"
        f"{json.dumps(TASK_PROMPTS[scenario], ensure_ascii=False)}"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(_: str) -> Any:
    raise ValueError("Nonfinite JSON number")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite JSON number")
    return number


def _matches_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Check the small frozen schema vocabulary without coercion or repair."""
    if "anyOf" in schema:
        return any(_matches_schema(value, choice) for choice in schema["anyOf"])
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= set(value):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(
            _matches_schema(item, properties[key])
            for key, item in value.items() if key in properties
        )
    if kind == "string":
        return isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
    if kind == "integer":
        return type(value) is int and ("minimum" not in schema or value >= schema["minimum"])
    if kind == "boolean":
        return type(value) is bool
    if kind == "null":
        return value is None
    raise ValueError("Unsupported physical direct schema vocabulary")


def parse_output(raw: str) -> dict[str, Any]:
    """Parse a complete JSON object, preserving decoded schema-invalid output.

    A valid JSON scalar or array is preserved but fails the object parse
    contract. Duplicate keys and nonfinite numbers fail parsing entirely.
    No markdown removal, substring extraction, value normalization, or model
    retry is performed. Invalid fields remain visible in ``parsed_response``.
    """
    result: dict[str, Any] = {
        "parsed_response": None,
        "parse_valid": False,
        "schema_valid": False,
        "action": None,
        "arguments": None,
        "decision_text": None,
        "error_type": "MALFORMED_JSON",
    }
    if not isinstance(raw, str):
        return result
    try:
        decoded = json.loads(
            raw, object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite, parse_float=_finite_float,
        )
    except (ValueError, TypeError, RecursionError):
        return result
    result["parsed_response"] = decoded
    result["error_type"] = "INVALID_SCHEMA"
    if not isinstance(decoded, dict):
        return result
    result.update({
        "parse_valid": True,
        "action": decoded.get("action"),
        "arguments": decoded.get("arguments"),
        "decision_text": decoded.get("decision_text"),
    })
    if not _matches_schema(decoded, DIRECT_SCHEMA):
        return result
    allowed_shapes = _ACTION_ARGUMENT_SHAPES[decoded["action"]]
    if frozenset(decoded["arguments"]) not in allowed_shapes:
        return result
    result["schema_valid"] = True
    result["error_type"] = None
    return result
