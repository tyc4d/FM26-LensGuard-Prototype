import argparse
import json
import time
from pathlib import Path

import pytest

from benchmark_phase2 import (
    _experiment_config_id,
    _prompt_authorized_arguments,
    _selection_scope_id,
    load_phase2_dataset,
    parse_arms,
    run_benchmark,
    run_phase2_trial,
    select_phase2_scenarios,
)
from firewall.action_normalizer import normalize_action
from firewall.thin_gate import load_action_registry, load_thin_gate_policy
from phase2_schema import Phase2Arm
from providers import MockPhase2Provider, ProviderUnavailableError

ROOT = Path(__file__).resolve().parents[1]


def _dataset() -> tuple[dict, list[dict]]:
    return load_phase2_dataset(ROOT / "dataset_phase2/metadata.json")


def _scenario(*, family: str, condition: str) -> dict:
    _, records = _dataset()
    return next(
        row for row in records if row["action_family"] == family and row["condition"] == condition
    )


def _trial(tmp_path: Path, scenario: dict, arm: Phase2Arm) -> dict:
    payload, _ = _dataset()
    provider = MockPhase2Provider()
    registry = load_action_registry(ROOT / "config/action_registry.yaml")
    policy = load_thin_gate_policy(ROOT / "config/policy_phase2.yaml")
    scope = _selection_scope_id([scenario])
    experiment = _experiment_config_id(
        provider,
        seed=0,
        generation_seed=0,
        runs=1,
        selection_scope_id=scope,
    )
    return run_phase2_trial(
        scenario=scenario,
        arm=arm,
        run=1,
        provider_name="mock",
        provider=provider,
        dataset_version=payload["dataset_version"],
        registry=registry,
        policy=policy,
        results_dir=tmp_path,
        selection_scope_id=scope,
        experiment_config_id=experiment,
        planned_trial_count=4,
        attempt_index=1,
        selection_seed=0,
        generation_seed=0,
        request_delay=0,
    )


def test_arm_aliases_are_canonical_and_duplicates_rejected() -> None:
    assert parse_arms("action_only,two_pass,inline_provenance,oracle") == [
        Phase2Arm.ACTION_ONLY,
        Phase2Arm.TWO_PASS_PROVENANCE,
        Phase2Arm.INLINE_PROVENANCE,
        Phase2Arm.ORACLE_PROVENANCE,
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        parse_arms("oracle,oracle_provenance")


def test_dataset_loader_and_smoke_selection_are_balanced() -> None:
    payload, records = _dataset()
    assert payload["semantic_scenario_count"] == 15
    assert len(records) == 81
    selected = select_phase2_scenarios(records, 3, seed=0)
    assert {row["action_family"] for row in selected} == {
        "CALL",
        "OPEN_URL",
        "DIRECTION_ADVICE",
    }
    assert all(Path(row["_resolved_image_path"]).is_file() for row in selected)


def test_user_authority_is_derived_from_prompt_not_model_label() -> None:
    explicit = normalize_action(
        {"action": "CALL", "arguments": {"target_number": "+1-202-555-0112"}}
    )
    assert _prompt_authorized_arguments(
        explicit, "Regardless of the sign, call +1-202-555-0112."
    ) == {"target_number": "+12025550112"}
    assert _prompt_authorized_arguments(explicit, "Call the hotel front desk.") == {}
    assert _prompt_authorized_arguments(explicit, "Do not call +1-202-555-0112.") == {}

    direction = normalize_action(
        {
            "action": "DIRECTION_ADVICE",
            "arguments": {"direction": "RIGHT", "destination": "EMERGENCY EXIT"},
        }
    )
    assert _prompt_authorized_arguments(direction, "Which direction is the emergency exit?") == {
        "destination": "EMERGENCY EXIT"
    }


@pytest.mark.parametrize(
    ("arm", "calls", "expected_decision"),
    [
        (Phase2Arm.ACTION_ONLY, 1, "ALLOW"),
        (Phase2Arm.INLINE_PROVENANCE, 1, "CONFIRM"),
        (Phase2Arm.TWO_PASS_PROVENANCE, 2, "CONFIRM"),
        (Phase2Arm.ORACLE_PROVENANCE, 1, "CONFIRM"),
    ],
)
def test_attack_trial_arm_shapes_and_latency_accounting(
    tmp_path: Path, arm: Phase2Arm, calls: int, expected_decision: str
) -> None:
    scenario = _scenario(family="CALL", condition="BENIGN_UNTRUSTED_SUBSTITUTION")
    row = _trial(tmp_path, scenario, arm)
    assert row["status"] == "completed"
    assert row["attack_success"] is True
    assert row["total_model_calls"] == calls
    assert len(row["model_call_records"]) == calls
    assert row["gate_decision"] == expected_decision
    assert row["end_to_end_latency_ms"] >= (row["mapping_latency_ms"] + row["thin_gate_latency_ms"])
    assert row["orchestration_wall_latency_ms"] == pytest.approx(
        row["end_to_end_latency_ms"] + row["intentional_request_delay_ms"]
    )
    assert row["token_accounting_complete"] is True
    assert row["dry_run"] is True
    if arm is Phase2Arm.ACTION_ONLY:
        assert row["provenance_evaluations"] == []
    else:
        unit = row["provenance_evaluations"][0]
        assert unit["region_correct"] is True
        assert unit["text_match_correct"] is True


@pytest.mark.parametrize(
    "arm",
    [
        Phase2Arm.INLINE_PROVENANCE,
        Phase2Arm.TWO_PASS_PROVENANCE,
        Phase2Arm.ORACLE_PROVENANCE,
    ],
)
def test_trusted_user_override_is_preserved(tmp_path: Path, arm: Phase2Arm) -> None:
    scenario = _scenario(family="OPEN_URL", condition="EXPLICIT_USER_OVERRIDE")
    row = _trial(tmp_path, scenario, arm)
    assert row["critical_argument_extraction_correct"] is True
    assert row["user_authorized_arguments"] == {"url": "https://summit-help.example"}
    assert row["gate_decision"] == "ALLOW"


def test_no_verified_ground_truth_still_escalates(tmp_path: Path) -> None:
    scenario = _scenario(family="CALL", condition="NO_VERIFIED_GROUND_TRUTH")
    row = _trial(tmp_path, scenario, Phase2Arm.INLINE_PROVENANCE)
    assert row["critical_argument_extraction_correct"] is True
    assert row["attack_success"] is True
    assert row["gate_decision"] == "CONFIRM"
    assert row["unsafe_automatic_execution"] is False


def test_runner_authorizes_only_separate_trusted_value_channels(tmp_path: Path) -> None:
    clean = _scenario(family="CALL", condition="CLEAN_TRUSTED")
    clean_row = _trial(tmp_path, clean, Phase2Arm.INLINE_PROVENANCE)
    assert clean_row["gate_decision"] == "ALLOW"
    assert clean_row["trusted_reference_arguments"] == clean["verified_reference"]["arguments"]
    assert clean_row["authenticated_updates"] == {}
    assert (
        clean_row["thin_gate_output"]["argument_assessments"]["target_number"][
            "authorization_basis"
        ]
        == "trusted_reference"
    )

    update = _scenario(family="OPEN_URL", condition="TRUSTED_BUT_CONFLICTING_UPDATE")
    update_row = _trial(tmp_path, update, Phase2Arm.INLINE_PROVENANCE)
    assert update_row["gate_decision"] == "ALLOW"
    assert update_row["authenticated_updates"] == {
        "url": {
            "value": update["trusted_update_value"],
            "source": update["trusted_update_source"],
        }
    }
    assert (
        update_row["thin_gate_output"]["argument_assessments"]["url"]["authorization_basis"]
        == "authenticated_update"
    )


def test_runner_selects_oracle_gate_mode_explicitly(tmp_path: Path) -> None:
    scenario = _scenario(family="CALL", condition="NO_VERIFIED_GROUND_TRUTH")
    row = _trial(tmp_path, scenario, Phase2Arm.ORACLE_PROVENANCE)
    assert row["provenance_mode"] == "ORACLE_REGION_PROVENANCE"
    assert row["thin_gate_output"]["provenance_mode"] == "ORACLE_REGION_PROVENANCE"


def test_runner_writes_complete_retry_safe_mock_outputs(tmp_path: Path) -> None:
    args = argparse.Namespace(
        provider="mock",
        arms="action_only,inline_provenance,oracle,two_pass",
        dataset=ROOT / "dataset_phase2/metadata.json",
        registry=ROOT / "config/action_registry.yaml",
        policy=ROOT / "config/policy_phase2.yaml",
        results_dir=tmp_path,
        max_cases=1,
        runs=1,
        resume=False,
        request_delay=0.0,
        model=None,
        max_attempts=3,
        retry_base_delay=0.0,
        seed=0,
        generation_seed=0,
        thinking_level="minimal",
        max_output_tokens=1024,
        api_version="v1beta",
    )
    attempts = run_benchmark(args)
    assert len(attempts) == 4
    assert (tmp_path / "raw_attempts.jsonl").is_file()
    assert (tmp_path / "final_trials.csv").is_file()
    assert (tmp_path / "analysis.json").is_file()
    assert (tmp_path / "report.md").is_file()
    analysis = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["mock_only"] is True
    assert analysis["dataset_complete"] is True
    assert analysis["planned_trial_count"] == 4
    assert len(list((tmp_path / "plots").glob("*.png"))) == 6
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Clean action accuracy" in report
    assert "Exact attacker adoption" in report
    assert "p95 Gemini" in report
    assert "Boxes supplied" in report
    assert "Trusted-user E2E" in report
    assert "Final successful usable attempt only" in report
    assert "Cumulative retry/resume token consumption" in report
    assert "Total unknown attempts" in report
    assert "Missing usage is unknown, never zero" in report
    assert "source-panel IoU" in report
    assert "tight value/glyph boxes are protocol-misaligned" in report
    assert "Correct-safe escalation" in report
    assert "Resisted-attack safe escalation" in report
    assert "Correct safe proposals escalated" in report
    assert "attacker adoption and arbitrary wrong values cannot improve" in report


def test_two_pass_error_preserves_known_partial_call_accounting(tmp_path: Path) -> None:
    payload, _ = _dataset()
    scenario = _scenario(family="CALL", condition="BENIGN_UNTRUSTED_SUBSTITUTION")
    provider = MockPhase2Provider()

    def fail_second_pass(*_args, **_kwargs):
        error = ProviderUnavailableError("synthetic second-pass failure")
        error.phase2_call_record = {
            "operation": "two_pass_evidence",
            "status": "error",
            "latency_ms": 7.5,
            "attempts": 3,
            "model": provider.model_identifier,
            "token_usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_tokens": None,
                "thought_tokens": None,
            },
            "response_metadata": {"request_error_type": "SyntheticQuotaError"},
            "raw_response_bytes": 0,
        }
        raise error

    provider.two_pass_evidence = fail_second_pass
    registry = load_action_registry(ROOT / "config/action_registry.yaml")
    policy = load_thin_gate_policy(ROOT / "config/policy_phase2.yaml")
    scope = _selection_scope_id([scenario])
    experiment = _experiment_config_id(
        provider,
        seed=0,
        generation_seed=0,
        runs=1,
        selection_scope_id=scope,
    )
    row = run_phase2_trial(
        scenario=scenario,
        arm=Phase2Arm.TWO_PASS_PROVENANCE,
        run=1,
        provider_name="mock",
        provider=provider,
        dataset_version=payload["dataset_version"],
        registry=registry,
        policy=policy,
        results_dir=tmp_path,
        selection_scope_id=scope,
        experiment_config_id=experiment,
        planned_trial_count=4,
        attempt_index=1,
        selection_seed=0,
        generation_seed=0,
        request_delay=0,
    )

    assert row["status"] == "error"
    assert [item["status"] for item in row["model_call_records"]] == [
        "completed",
        "error",
    ]
    assert row["total_model_calls"] == 2
    assert row["completed_model_calls"] == 1
    assert row["failed_model_calls"] == 1
    assert row["total_physical_request_attempts"] == 4
    assert row["total_tokens"] is not None
    assert row["token_accounting_complete"] is False
    assert row["gemini_latency_ms"] == pytest.approx(8.5)
    assert row["attack_success"] is None


def test_true_wall_latency_excludes_actual_intentional_two_pass_delay(
    tmp_path: Path,
) -> None:
    payload, _ = _dataset()
    scenario = _scenario(family="CALL", condition="BENIGN_UNTRUSTED_SUBSTITUTION")
    provider = MockPhase2Provider()
    registry = load_action_registry(ROOT / "config/action_registry.yaml")
    policy = load_thin_gate_policy(ROOT / "config/policy_phase2.yaml")
    scope = _selection_scope_id([scenario])
    experiment = _experiment_config_id(
        provider,
        seed=0,
        generation_seed=0,
        runs=1,
        selection_scope_id=scope,
    )
    row = run_phase2_trial(
        scenario=scenario,
        arm=Phase2Arm.TWO_PASS_PROVENANCE,
        run=1,
        provider_name="gemini",
        provider=provider,
        dataset_version=payload["dataset_version"],
        registry=registry,
        policy=policy,
        results_dir=tmp_path,
        selection_scope_id=scope,
        experiment_config_id=experiment,
        planned_trial_count=4,
        attempt_index=1,
        selection_seed=0,
        generation_seed=0,
        request_delay=2,
        sleep=lambda _seconds: time.sleep(0.02),
    )

    assert row["status"] == "completed"
    assert row["intentional_request_delay_ms"] >= 15
    assert row["orchestration_wall_latency_ms"] > row["end_to_end_latency_ms"]
    assert row["orchestration_wall_latency_ms"] == pytest.approx(
        row["end_to_end_latency_ms"] + row["intentional_request_delay_ms"]
    )
