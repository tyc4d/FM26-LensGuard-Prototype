import json
from copy import deepcopy
from pathlib import Path

import yaml
import pytest

import benchmark as benchmark_module
from analyze_phase1 import analyze
from benchmark import (
    load_dataset,
    parse_args,
    run_benchmark,
    run_trial,
    select_scenarios,
    validate_dataset_registry,
)
from firewall import load_policy
from generate_report import build_report
from providers import MockAgentProvider, MockConsequenceProvider
from result_store import append_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mock_trial_runs_the_complete_dry_run_pipeline(tmp_path):
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    scenario = next(
        row
        for row in records
        if row["action_family"] == "CALL"
        and row["condition"] == "BENIGN_UNTRUSTED_SUBSTITUTION"
    )
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())

    result = run_trial(
        scenario=scenario,
        run=1,
        provider_name="mock",
        agent=MockAgentProvider(),
        predictor=MockConsequenceProvider(),
        dataset_version=version,
        policy=policy,
        registry=registry,
        results_dir=tmp_path,
        request_delay=0,
    )

    assert result["status"] == "completed"
    assert result["dry_run"] is True
    assert result["attack_success"] is True
    assert result["critical_argument_source"] == scenario["attack_source"]
    assert result["no_firewall_decision"] == "ALLOW"
    assert result["consequence_only_decision"] == "ALLOW"
    assert result["full_firewall_decision"] in {"WARN", "CONFIRM", "BLOCK"}
    assert result["source_provenance_only_decision"] in {"WARN", "CONFIRM", "BLOCK"}
    assert result["verified_conflict_only_decision"] in {"WARN", "CONFIRM", "BLOCK"}
    assert result["attacker_controlled_influence"] is True
    assert result["security_relevant_influence_no_firewall"] is True
    assert result["security_relevant_influence_consequence_only"] is True
    assert result["security_relevant_influence_full_firewall"] is False
    assert result["security_relevant_influence"] is False
    for key in (
        "raw_agent_response_path",
        "raw_consequence_only_response_path",
        "raw_consequence_response_path",
    ):
        assert Path(result[key]).is_file()


def test_clean_and_explicit_user_cases_are_not_labeled_attack_success(tmp_path):
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())
    selected = [
        next(row for row in records if row["condition"] == condition)
        for condition in ("CLEAN_TRUSTED", "EXPLICIT_USER_OVERRIDE")
    ]
    for index, scenario in enumerate(selected, 1):
        result = run_trial(
            scenario=scenario,
            run=index,
            provider_name="mock",
            agent=MockAgentProvider(),
            predictor=MockConsequenceProvider(),
            dataset_version=version,
            policy=policy,
            registry=registry,
            results_dir=tmp_path,
            request_delay=0,
        )
        assert result["attack_success"] is False
        assert result["security_relevant_influence"] is False
        assert result["full_firewall_decision"] == "ALLOW"


def test_malformed_or_non_oracle_metadata_is_an_error_not_a_policy_catch(tmp_path):
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    scenario = dict(records[1])
    scenario["provenance_mode"] = "MODEL_ESTIMATED_PROVENANCE"
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())

    result = run_trial(
        scenario=scenario,
        run=1,
        provider_name="mock",
        agent=MockAgentProvider(),
        predictor=MockConsequenceProvider(),
        dataset_version=version,
        policy=policy,
        registry=registry,
        results_dir=tmp_path,
        request_delay=0,
    )

    assert result["status"] == "error"
    assert result["error_type"] == "ValueError"
    assert "non-oracle" in result["error_message"]
    assert result.get("full_firewall_decision") is None


def test_three_case_selector_covers_every_action_family() -> None:
    _, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    selected = select_scenarios(records, 3, seed=0)
    assert {row["dataset_partition"] for row in selected} == {"CORE"}
    assert {row["action_family"] for row in selected} == {
        "CALL",
        "OPEN_URL",
        "DIRECTION_ADVICE",
    }
    assert {row["condition"] for row in selected} == {
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        "AUTHORITY_IMPERSONATION",
        "OBVIOUS_INJECTION_CONTROL",
    }
    assert [row["scenario_id"] for row in selected] == [
        row["scenario_id"] for row in select_scenarios(records, 3, seed=0)
    ]


def test_dataset_oracle_sources_are_classified_by_action_registry() -> None:
    _, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())
    validate_dataset_registry(records, registry)

    invalid = dict(records[0])
    invalid["official_source"] = "undeclared_source"
    with pytest.raises(ValueError, match="undeclared_source"):
        validate_dataset_registry([invalid], registry)


def test_matched_conditions_and_source_variants_use_paired_request_seeds(tmp_path) -> None:
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())
    agent = MockAgentProvider(seed=71)
    predictor = MockConsequenceProvider(seed=71)

    base_id = "call_hotel_front_desk"
    core_scenarios = [
        row
        for row in records
        if row["dataset_partition"] == "CORE"
        and row["base_scenario_id"] == base_id
    ]
    assert len(core_scenarios) == 5

    core_results = [
        run_trial(
            scenario=scenario,
            run=1,
            provider_name="mock",
            agent=agent,
            predictor=predictor,
            dataset_version=version,
            policy=policy,
            registry=registry,
            results_dir=tmp_path / "core",
            request_delay=0,
            selection_seed=19,
        )
        for scenario in core_scenarios
    ]
    assert {row["status"] for row in core_results} == {"completed"}
    assert {row["condition"] for row in core_results} == {
        "CLEAN_TRUSTED",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        "AUTHORITY_IMPERSONATION",
        "OBVIOUS_INJECTION_CONTROL",
        "EXPLICIT_USER_OVERRIDE",
    }
    assert {row["seed_pairing_key"] for row in core_results} == {base_id}
    assert len({row["agent_request_seed"] for row in core_results}) == 1
    assert len({row["predictor_request_seed"] for row in core_results}) == 1

    source_group = "source_authority_call_hotel_front_desk"
    source_scenarios = [
        row
        for row in records
        if row.get("source_authority_group_id") == source_group
    ]
    assert len(source_scenarios) == 5
    source_results = [
        run_trial(
            scenario=scenario,
            run=1,
            provider_name="mock",
            agent=agent,
            predictor=predictor,
            dataset_version=version,
            policy=policy,
            registry=registry,
            results_dir=tmp_path / "source",
            request_delay=0,
            selection_seed=19,
        )
        for scenario in source_scenarios
    ]
    assert {row["status"] for row in source_results} == {"completed"}
    assert {row["seed_pairing_key"] for row in source_results} == {source_group}
    assert len({row["agent_request_seed"] for row in source_results}) == 1
    assert len({row["predictor_request_seed"] for row in source_results}) == 1

    second_run = run_trial(
        scenario=core_scenarios[0],
        run=2,
        provider_name="mock",
        agent=agent,
        predictor=predictor,
        dataset_version=version,
        policy=policy,
        registry=registry,
        results_dir=tmp_path / "second-run",
        request_delay=0,
        selection_seed=19,
    )
    assert second_run["status"] == "completed"
    assert second_run["seed_pairing_key"] == core_results[0]["seed_pairing_key"]
    assert second_run["agent_request_seed"] != core_results[0]["agent_request_seed"]
    assert second_run["predictor_request_seed"] != core_results[0][
        "predictor_request_seed"
    ]


def test_analysis_marks_mock_results_as_non_gemini_evidence(tmp_path) -> None:
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    scenario = next(row for row in records if row["condition"] == "CLEAN_TRUSTED")
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())
    result = run_trial(
        scenario=scenario,
        run=1,
        provider_name="mock",
        agent=MockAgentProvider(),
        predictor=MockConsequenceProvider(),
        dataset_version=version,
        policy=policy,
        registry=registry,
        results_dir=tmp_path / "evidence",
        request_delay=0,
    )
    input_path = tmp_path / "raw.jsonl"
    append_jsonl(input_path, result)

    output = analyze(input_path, tmp_path / "analysis.json", None)

    assert output["mock_only"] is True
    assert output["result_kind"] == "mock_validation"
    assert output["eligible_as_gemini_evidence"] is False


def test_report_rejects_registry_version_drift(tmp_path) -> None:
    version, records = load_dataset(PROJECT_ROOT / "dataset/metadata.json")
    scenario = next(row for row in records if row["condition"] == "CLEAN_TRUSTED")
    policy = load_policy(PROJECT_ROOT / "config/policy.yaml")
    registry = yaml.safe_load((PROJECT_ROOT / "config/action_registry.yaml").read_text())
    result = run_trial(
        scenario=scenario,
        run=1,
        provider_name="mock",
        agent=MockAgentProvider(),
        predictor=MockConsequenceProvider(),
        dataset_version=version,
        policy=policy,
        registry=registry,
        results_dir=tmp_path,
        request_delay=0,
    )
    mismatched_registry = deepcopy(registry)
    mismatched_registry["registry_version"] = "future-registry-version"

    with pytest.raises(ValueError, match="does not match"):
        build_report([result], mismatched_registry, tmp_path / "raw.jsonl")


def test_mock_runner_exits_nonzero_when_any_trial_errors(tmp_path, monkeypatch) -> None:
    args = parse_args(
        [
            "--provider",
            "mock",
            "--dataset",
            str(PROJECT_ROOT / "dataset/metadata.json"),
            "--policy",
            str(PROJECT_ROOT / "config/policy.yaml"),
            "--registry",
            str(PROJECT_ROOT / "config/action_registry.yaml"),
            "--results-dir",
            str(tmp_path / "failed-mock"),
            "--max-cases",
            "1",
        ]
    )

    monkeypatch.setattr(
        benchmark_module,
        "run_trial",
        lambda **_: {
            "status": "error",
            "error_type": "SyntheticFailure",
            "error_message": "test failure",
        },
    )
    with pytest.raises(ValueError, match="validation did not pass"):
        run_benchmark(args)
