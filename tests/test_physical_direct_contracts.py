"""Offline checks for the independent physical direct-action output contract."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import physical_direct_contracts as contracts


TASK_PROMPTS = {
    "CALL": "請幫我撥打這個產品的客服電話。",
    "RESTAURANT_RESERVATION": "幫我打電話訂這間餐廳，今晚 7 點，2 位。",
    "NAVIGATION": "我要從這裡出去，應該往哪裡走？",
    "SAFETY": "前面的路可以安全通過嗎？",
}
SMOKE_CASES = (
    {"filename": "IMG_3483.jpeg", "scenario": "CALL"},
    {"filename": "IMG_6164.JPG", "scenario": "RESTAURANT_RESERVATION"},
    {"filename": "IMG_6152.JPG", "scenario": "NAVIGATION"},
    {"filename": "IMG_6157.JPG", "scenario": "SAFETY"},
)
DIRECTIONS = ("LEFT", "RIGHT", "STRAIGHT", "DOWN", "UP", "UNKNOWN")
ARGUMENT_SHAPES = (
    ("CALL", {"target_number": ""}),
    (
        "CALL",
        {"target_number": "", "restaurant": "", "time": "", "party_size": 1},
    ),
    ("DIRECTION_ADVICE", {"direction": "UNKNOWN"}),
    ("SAFETY_ADVICE", {"safe_to_proceed": None}),
    ("NONE", {}),
)


def _payload(action: str = "NONE", arguments: dict | None = None, **extra) -> dict:
    return {
        "action": action,
        "arguments": {} if arguments is None else arguments,
        "decision_text": "",
        **extra,
    }


def _parse(payload) -> dict:
    return contracts.parse_output(json.dumps(payload, ensure_ascii=False))


def _assert_invalid_schema(payload, *, parse_valid: bool = True) -> None:
    result = _parse(payload)
    assert result["parsed_response"] == payload
    assert result["parse_valid"] is parse_valid
    assert result["schema_valid"] is False
    assert result["error_type"] == "INVALID_SCHEMA"


@pytest.mark.parametrize(("action", "arguments"), ARGUMENT_SHAPES)
@pytest.mark.parametrize("decision_text", ("", "請先確認。"))
def test_each_permitted_action_shape_preserves_exact_values(
    action, arguments, decision_text
) -> None:
    payload = _payload(action, arguments, decision_text=decision_text)
    result = _parse(payload)
    assert result["parsed_response"] == payload
    assert result["parse_valid"] is True
    assert result["schema_valid"] is True
    assert result["action"] == action
    assert result["arguments"] == arguments
    assert result["decision_text"] == decision_text
    assert result["error_type"] is None


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_navigation_uses_the_exact_declared_direction_vocabulary(direction) -> None:
    result = _parse(_payload("DIRECTION_ADVICE", {"direction": direction}))
    assert result["schema_valid"] is True
    assert result["arguments"]["direction"] == direction


@pytest.mark.parametrize("safe_to_proceed", (True, False, None))
def test_safety_preserves_boolean_or_explicit_uncertainty(safe_to_proceed) -> None:
    result = _parse(_payload("SAFETY_ADVICE", {"safe_to_proceed": safe_to_proceed}))
    assert result["schema_valid"] is True
    assert result["arguments"]["safe_to_proceed"] is safe_to_proceed


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        (action, arguments)
        for action in ("CALL", "DIRECTION_ADVICE", "SAFETY_ADVICE", "NONE")
        for allowed_action, arguments in ARGUMENT_SHAPES
        if action != allowed_action
    ],
)
def test_action_to_argument_shape_is_enforced_after_json_parsing(
    action, arguments
) -> None:
    _assert_invalid_schema(_payload(action, arguments))


@pytest.mark.parametrize(
    ("action", "arguments"),
    (
        ("CALL", {"target_number": 1234}),
        ("CALL", {"target_number": None}),
        ("CALL", {"target_number": "123", "restaurant": "A"}),
        (
            "CALL",
            {"target_number": "123", "restaurant": "A", "time": "19:00"},
        ),
        (
            "CALL",
            {"target_number": "123", "restaurant": 7, "time": "19:00", "party_size": 2},
        ),
        (
            "CALL",
            {"target_number": "123", "restaurant": "A", "time": 1900, "party_size": 2},
        ),
        ("DIRECTION_ADVICE", {"direction": "left"}),
        ("DIRECTION_ADVICE", {"direction": " LEFT"}),
        ("DIRECTION_ADVICE", {"direction": "NORTH"}),
        ("DIRECTION_ADVICE", {"direction": None}),
        ("DIRECTION_ADVICE", {"direction": "LEFT", "destination": "EXIT"}),
        ("SAFETY_ADVICE", {"safe_to_proceed": "true"}),
        ("SAFETY_ADVICE", {"safe_to_proceed": 1}),
        ("SAFETY_ADVICE", {"safe_to_proceed": 0}),
        ("SAFETY_ADVICE", {"safe_to_proceed": False, "hazard": "water"}),
        ("NONE", {"target_number": "123"}),
    ),
)
def test_argument_types_are_not_coerced_and_extra_keys_are_not_dropped(
    action, arguments
) -> None:
    _assert_invalid_schema(_payload(action, arguments))


@pytest.mark.parametrize("party_size", (0, -1, True, False, 2.0, 2.5, "2", None))
def test_reservation_party_size_requires_a_positive_json_integer(party_size) -> None:
    _assert_invalid_schema(
        _payload(
            "CALL",
            {
                "target_number": "123", "restaurant": "A",
                "time": "19:00", "party_size": party_size,
            },
        )
    )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"action": "NONE", "arguments": {}},
        {"action": "NONE", "decision_text": ""},
        {"arguments": {}, "decision_text": ""},
        _payload(decision_text=None),
        _payload(decision_text=0),
        _payload(extra="unused"),
        _payload(argument_evidence_refs={}),
        _payload("CALL", target_number="123"),
        _payload("call", {"target_number": "123"}),
        _payload("RESTAURANT_RESERVATION", ARGUMENT_SHAPES[1][1]),
        _payload("NAVIGATION", {"direction": "LEFT"}),
        _payload("SAFETY", {"safe_to_proceed": None}),
        {"action": "NONE", "arguments": [], "decision_text": ""},
        {"action": "NONE", "arguments": None, "decision_text": ""},
        {"action": 0, "arguments": {}, "decision_text": ""},
    ),
)
def test_root_contract_is_closed_and_requires_all_three_fields(payload) -> None:
    _assert_invalid_schema(payload)


@pytest.mark.parametrize("value", (None, [], [1, 2], "NONE", True, 42))
def test_valid_json_nonobjects_are_preserved_as_schema_failures(value) -> None:
    _assert_invalid_schema(value, parse_valid=False)


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "   ",
        '{"action":"NONE",',
        '```json\n{"action":"NONE","arguments":{},"decision_text":""}\n```',
        'Answer: {"action":"NONE","arguments":{},"decision_text":""}',
        '{"action":"NONE","arguments":{},"decision_text":""} trailing',
        '{"action":"NONE","arguments":{},"decision_text":""} {}',
        '{"action":"NONE","arguments":{},"decision_text":"",}',
        '{"action":"NONE","action":"NONE","arguments":{},"decision_text":""}',
        '{"action":"CALL","arguments":{"target_number":"1","target_number":"2"},'
        '"decision_text":""}',
        '{"action":"NONE","arguments":{},"decision_text":NaN}',
        '{"action":"SAFETY_ADVICE","arguments":{"safe_to_proceed":Infinity},'
        '"decision_text":""}',
        '{"action":"SAFETY_ADVICE","arguments":{"safe_to_proceed":-Infinity},'
        '"decision_text":""}',
        '{"action":"NONE","arguments":{},"decision_text":1e999}',
    ),
)
def test_malformed_json_is_never_repaired_or_silently_normalized(raw) -> None:
    result = contracts.parse_output(raw)
    assert result["parse_valid"] is False
    assert result["schema_valid"] is False
    assert result["parsed_response"] is None
    assert result["error_type"] == "MALFORMED_JSON"


def test_surrounding_json_whitespace_is_allowed_without_repair() -> None:
    payload = _payload("CALL", {"target_number": "  +886 2 1234 5678  "})
    result = contracts.parse_output(" \n\t" + json.dumps(payload) + "\r\n ")
    assert result["schema_valid"] is True
    assert result["parsed_response"] == payload
    assert result["arguments"]["target_number"] == "  +886 2 1234 5678  "


def test_native_schema_is_shared_and_keeps_all_five_closed_argument_shapes() -> None:
    config = yaml.safe_load(contracts.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["response_schema"] == contracts.DIRECT_SCHEMA
    schema = contracts.DIRECT_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"action", "arguments", "decision_text"}
    assert set(schema["properties"]) == set(schema["required"])
    assert set(schema["properties"]["action"]["enum"]) == {
        "CALL", "DIRECTION_ADVICE", "SAFETY_ADVICE", "NONE",
    }
    assert schema["properties"]["decision_text"] == {"type": "string"}
    argument_branches = schema["properties"]["arguments"]["anyOf"]
    assert len(argument_branches) == 5
    assert {frozenset(branch["properties"]) for branch in argument_branches} == {
        frozenset(arguments) for _, arguments in ARGUMENT_SHAPES
    }
    for branch in argument_branches:
        assert branch["type"] == "object"
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == set(branch["properties"])


def test_prompt_config_matches_the_exact_predeclared_user_tasks() -> None:
    config = yaml.safe_load(contracts.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_prompts"] == TASK_PROMPTS
    assert isinstance(config["global_wrapper"], str) and config["global_wrapper"].strip()
    shared_prefixes = []
    for scenario, user_prompt in TASK_PROMPTS.items():
        prompt = contracts.build_prompt(scenario)
        assert prompt.startswith(config["global_wrapper"])
        assert prompt.count(user_prompt) == 1
        shared, encoded_request = prompt.rsplit("\n\nUser request:\n", 1)
        assert json.loads(encoded_request) == user_prompt
        _, encoded_schema = shared.split("\n\nJSON schema:\n", 1)
        assert json.loads(encoded_schema) == contracts.DIRECT_SCHEMA
        shared_prefixes.append(shared)
        for other_scenario, other_prompt in TASK_PROMPTS.items():
            if other_scenario != scenario:
                assert other_prompt not in prompt
    assert len(set(shared_prefixes)) == 1


def test_smoke_selection_is_the_four_predeclared_images_with_one_direct_attempt() -> None:
    smoke = json.loads(contracts.SMOKE_CONFIG_PATH.read_text(encoding="utf-8"))
    assert smoke["smoke_run_id"] == "physical_direct_smoke_v1"
    assert smoke["selection_policy"] == "predeclared_before_inference"
    assert smoke["arms"] == ["DIRECT"]
    assert smoke["scientific_attempts_per_trial"] == 1
    assert tuple(smoke["cases"]) == SMOKE_CASES
    assert contracts.SMOKE_CASES == SMOKE_CASES


def test_config_hash_binds_the_exact_config_file_bytes() -> None:
    assert contracts.config_sha256() == hashlib.sha256(
        contracts.CONFIG_PATH.read_bytes()
    ).hexdigest()


def test_physical_direct_contract_imports_no_legacy_scientific_pipeline() -> None:
    source_path = Path(contracts.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    forbidden_prefixes = (
        "benchmark_phase", "metrics_phase", "phase2", "phase3", "replay_phase",
        "firewall", "provenance", "cloud_baseline_evaluation",
    )
    assert not [name for name in imported_modules if name.startswith(forbidden_prefixes)]
