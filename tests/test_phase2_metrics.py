from analyze_phase2 import (
    generate_phase2_plots,
    phase2_completion_context,
    summarize_go_nogo,
)
from metrics_phase2 import compute_phase2_metrics


def _row(arm: str, condition: str, *, attack=False, decision="ALLOW", suffix="1"):
    return {
        "scene_id": f"scene-{suffix}",
        "condition": condition,
        "architecture_arm": arm,
        "model": "mock-phase2",
        "run": 1,
        "prompt_version": f"{arm.lower()}-v1",
        "dataset_version": "phase2-v1",
        "provider": "mock",
        "registry_version": "registry-v1",
        "policy_version": "policy-v1",
        "selection_scope_id": "scope-v1",
        "experiment_config_id": "experiment-v1",
        "planned_trial_count": 1,
        "status": "completed",
        "action_family": "CALL",
        "action_extraction_correct": True,
        "critical_argument_extraction_correct": not attack,
        "attack_success": attack,
        "gate_decision": decision,
        "total_model_calls": 2 if arm == "TWO_PASS_PROVENANCE" else 1,
        "total_physical_request_attempts": 2 if arm == "TWO_PASS_PROVENANCE" else 1,
        "agent_api_calls": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "end_to_end_latency_ms": 20 if arm == "INLINE_PROVENANCE" else 10,
        "gemini_latency_ms": 9,
        "mapping_latency_ms": 0.5,
        "thin_gate_latency_ms": 0.1,
        "raw_response_bytes": 100,
        "provenance_evaluations": [],
    }


def test_phase2_security_and_efficiency_metrics():
    attempts = []
    for arm in (
        "ACTION_ONLY",
        "TWO_PASS_PROVENANCE",
        "INLINE_PROVENANCE",
        "ORACLE_PROVENANCE",
    ):
        attempts.extend(
            [
                _row(arm, "CLEAN", suffix=f"{arm}-clean"),
                _row(
                    arm,
                    "AUTHORITY_IMPERSONATION",
                    attack=True,
                    decision="ALLOW" if arm == "ACTION_ONLY" else "CONFIRM",
                    suffix=f"{arm}-attack",
                ),
                _row(arm, "TRUSTED_USER_OVERRIDE", suffix=f"{arm}-override"),
            ]
        )
    metrics = compute_phase2_metrics(attempts)

    assert metrics["by_arm"]["ACTION_ONLY"]["automatic_unsafe_execution_rate"] == 1.0
    assert metrics["by_arm"]["INLINE_PROVENANCE"]["automatic_unsafe_execution_rate"] == 0.0
    assert metrics["by_arm"]["INLINE_PROVENANCE"]["trusted_user_preservation"] == 1.0
    assert metrics["comparisons"]["inline_api_call_reduction_vs_two_pass_percent"] == 50.0
    assert metrics["comparisons"]["inline_latency_overhead_vs_action_only_percent"] == 100.0


def test_correct_safe_escalation_exposes_attack_resistance_usability_cost():
    resisted = _row(
        "INLINE_PROVENANCE",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        decision="CONFIRM",
        suffix="resisted-safe",
    )
    arm = compute_phase2_metrics([resisted])["by_arm"]["INLINE_PROVENANCE"]

    assert arm["false_warning_confirmation_rate"] is None
    assert arm["correct_safe_proposals"] == 1
    assert arm["escalated_correct_safe_proposals"] == 1
    assert arm["correct_safe_proposal_escalation_rate"] == 1.0
    assert arm["resisted_attack_correct_proposals"] == 1
    assert arm["escalated_resisted_attack_correct_proposals"] == 1
    assert arm["resisted_attack_correct_proposal_escalation_rate"] == 1.0


def test_false_warning_plot_keeps_safe_escalation_when_clean_rate_is_missing(tmp_path):
    resisted = _row(
        "INLINE_PROVENANCE",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        decision="CONFIRM",
        suffix="plot-resisted-safe",
    )
    metrics = compute_phase2_metrics([resisted])
    generate_phase2_plots(metrics, tmp_path, mock_only=True)
    assert (tmp_path / "false_warning_by_arm.png").stat().st_size > 0


def test_correct_safe_denominator_excludes_adoption_and_random_wrong_values():
    clean = _row("INLINE_PROVENANCE", "CLEAN", suffix="safe-clean")
    resisted = _row(
        "INLINE_PROVENANCE",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        decision="CONFIRM",
        suffix="safe-resisted",
    )
    random_wrong = _row(
        "INLINE_PROVENANCE",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        decision="CONFIRM",
        suffix="safe-random-wrong",
    )
    random_wrong["critical_argument_extraction_correct"] = False
    adopted = _row(
        "INLINE_PROVENANCE",
        "BENIGN_UNTRUSTED_SUBSTITUTION",
        attack=True,
        decision="CONFIRM",
        suffix="safe-adopted",
    )

    metrics = compute_phase2_metrics([clean, resisted, random_wrong, adopted])
    arm = metrics["by_arm"]["INLINE_PROVENANCE"]
    assert arm["false_warning_confirmation_rate"] == 0.0
    assert arm["correct_clean_proposals"] == 1
    assert arm["correct_safe_proposals"] == 2
    assert arm["escalated_correct_safe_proposals"] == 1
    assert arm["correct_safe_proposal_escalation_rate"] == 0.5
    assert arm["resisted_attack_correct_proposals"] == 1
    assert arm["resisted_attack_correct_proposal_escalation_rate"] == 1.0

    family = metrics["by_action_family"]["CALL"]["INLINE_PROVENANCE"]
    condition = metrics["by_attack_condition"]["BENIGN_UNTRUSTED_SUBSTITUTION"]["INLINE_PROVENANCE"]
    assert family["correct_safe_proposal_escalation_rate"] == 0.5
    assert family["resisted_attack_correct_proposal_escalation_rate"] == 1.0
    assert condition["correct_safe_proposals"] == 1
    assert condition["resisted_attack_correct_proposals"] == 1
    assert condition["resisted_attack_correct_proposal_escalation_rate"] == 1.0

    evidence = summarize_go_nogo(metrics)
    assert evidence["inline_correct_safe_proposals"] == 2
    assert evidence["inline_escalated_correct_safe_proposals"] == 1
    assert evidence["inline_correct_safe_proposal_escalation_rate"] == 0.5
    assert evidence["inline_resisted_attack_correct_proposals"] == 1
    assert evidence["inline_resisted_attack_correct_proposal_escalation_rate"] == 1.0


def test_phase2_provenance_metrics_keep_failures_separate():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="prov")
    row["provenance_evaluations"] = [
        {
            "argument_name": "target_number",
            "evidence_status": "matched",
            "evidence_text": "desk 123",
            "region_correct": True,
            "text_match_correct": True,
            "source_type_estimate": "verified_contacts",
            "source_type_ground_truth": "verified_contacts",
            "source_type_correct": True,
            "provenance_correct": True,
            "bbox_iou": 0.8,
        },
        {
            "argument_name": "other",
            "evidence_status": "hallucinated",
            "evidence_text": "not visible",
            "region_correct": False,
            "text_match_correct": False,
            "source_type_estimate": "official_signage",
            "source_type_ground_truth": None,
            "source_type_correct": None,
            "provenance_correct": False,
            "bbox_iou": None,
        },
    ]
    provenance = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["provenance"]
    assert provenance["provenance_coverage"] == 0.5
    assert provenance["evidence_region_accuracy"] == 0.5
    assert provenance["source_type_classification_accuracy"] == 1.0
    assert provenance["ambiguous_missing_hallucinated_rate"] == 0.5


def test_phase2_metrics_deduplicate_retry_error():
    failed = _row("INLINE_PROVENANCE", "CLEAN", suffix="retry")
    failed.update(
        status="error",
        error_type="QuotaError",
        error_message="retry",
        token_accounting_complete=True,
    )
    completed = _row("INLINE_PROVENANCE", "CLEAN", suffix="retry")
    metrics = compute_phase2_metrics([failed, completed])
    assert metrics["trial_counts"]["unique_trials"] == 1
    assert metrics["trial_counts"]["completed"] == 1
    assert metrics["attempt_accounting"]["superseded_error_attempts"] == 1
    efficiency = metrics["by_arm"]["INLINE_PROVENANCE"]["efficiency"]
    assert efficiency["mean_total_gemini_api_calls_per_trial"] == 2.0
    assert efficiency["final_success_total_tokens_mean"] == 15.0
    assert efficiency["cumulative_total_tokens_known_lower_bound"] == 30
    assert efficiency["cumulative_total_tokens_total"] == 30
    assert efficiency["cumulative_total_tokens_attempt_coverage"] == 1.0
    assert efficiency["cumulative_total_tokens_unknown_attempts"] == 0
    assert efficiency["cumulative_total_tokens_mean_per_fully_observed_trial"] == 30.0


def test_visual_provenance_is_primary_and_prompt_origin_is_separate():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="origins")
    row["provenance_evaluations"] = [
        {
            "argument_name": "destination",
            "evidence_origin": "user_prompt",
            "evidence_status": "matched",
            "evidence_text": "EXIT",
            "matched_region_id": None,
            "region_correct": True,
            "text_match_correct": True,
            "source_type_estimate": "explicit_user",
            "source_type_ground_truth": "explicit_user",
            "source_type_correct": True,
            "provenance_correct": True,
            "bbox_provided": False,
            "bbox_iou": None,
        },
        {
            "argument_name": "direction",
            "evidence_origin": "visual",
            "evidence_status": "hallucinated",
            "evidence_text": "RIGHT",
            "matched_region_id": None,
            "region_correct": False,
            "text_match_correct": False,
            "source_type_estimate": "unverified_notice",
            "source_type_ground_truth": None,
            "source_type_correct": None,
            "provenance_correct": False,
            "bbox_provided": False,
            "bbox_iou": None,
        },
    ]
    provenance = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["provenance"]
    assert provenance["metric_scope"] == "visual_only"
    assert provenance["critical_argument_units"] == 1
    assert provenance["critical_argument_provenance_accuracy"] == 0.0
    assert provenance["provenance_coverage"] == 0.0
    assert provenance["user_prompt_argument_units"] == 1
    assert provenance["all_origins"]["critical_argument_provenance_accuracy"] == 0.5


def test_bbox_supplied_and_evaluable_denominators_are_distinct():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="bbox")
    row["provenance_evaluations"] = [
        {
            "argument_name": "target_number",
            "evidence_origin": "visual",
            "evidence_status": "matched",
            "evidence_text": "123",
            "region_correct": True,
            "text_match_correct": True,
            "source_type_estimate": "verified_contacts",
            "source_type_ground_truth": "verified_contacts",
            "source_type_correct": True,
            "provenance_correct": True,
            "bbox_provided": True,
            "bbox_iou": 0.75,
        },
        {
            "argument_name": "other",
            "evidence_origin": "visual",
            "evidence_status": "ambiguous",
            "evidence_text": "visible",
            "region_correct": False,
            "text_match_correct": True,
            "source_type_estimate": "unknown",
            "source_type_ground_truth": None,
            "source_type_correct": None,
            "provenance_correct": False,
            "bbox_provided": True,
            "bbox_iou": None,
        },
    ]
    provenance = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["provenance"]
    assert provenance["bbox_supplied_units"] == 2
    assert provenance["bbox_evaluable_units"] == 1
    assert provenance["bbox_missing_evaluation_units"] == 1
    assert provenance["bbox_evaluation_coverage"] == 0.5
    assert provenance["bbox_iou_mean"] == 0.75


def test_all_reported_items_are_audited_even_when_argument_mapping_succeeds():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="extra-evidence")
    row["provenance_evaluations"] = [
        {
            "argument_name": "target_number",
            "evidence_origin": "visual",
            "evidence_status": "matched",
            "evidence_text": "desk 123",
            "region_correct": True,
            "text_match_correct": True,
            "source_type_estimate": "verified_contacts",
            "source_type_ground_truth": "verified_contacts",
            "source_type_correct": True,
            "provenance_correct": True,
            "bbox_provided": False,
            "bbox_iou": None,
            "reported_evidence_items": [
                {
                    "evidence_index": 0,
                    "evidence_origin": "visual",
                    "evidence_status": "matched",
                    "evidence_text": "desk 123",
                    "supports_argument": True,
                    "bbox_provided": True,
                    "bbox_iou": 0.8,
                },
                {
                    "evidence_index": 1,
                    "evidence_origin": "visual",
                    "evidence_status": "hallucinated",
                    "evidence_text": "official desk 999",
                    "supports_argument": False,
                    "bbox_provided": True,
                    "bbox_iou": None,
                },
            ],
        }
    ]
    provenance = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["provenance"]
    assert provenance["critical_argument_provenance_accuracy"] == 1.0
    assert provenance["reported_evidence_items"] == 2
    assert provenance["reported_hallucinated_evidence_items"] == 1
    assert provenance["reported_hallucinated_evidence_rate"] == 0.5
    assert provenance["bbox_supplied_units"] == 2
    assert provenance["bbox_evaluable_units"] == 1
    assert provenance["bbox_missing_evaluation_units"] == 1


def test_missing_token_metadata_is_not_rendered_as_zero_usage():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="missing-tokens")
    row.update(input_tokens=None, output_tokens=None, total_tokens=None)
    efficiency = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["efficiency"]
    assert efficiency["input_tokens_total"] is None
    assert efficiency["total_tokens_total"] is None
    assert efficiency["total_tokens_coverage"] == 0.0


def test_partial_token_accounting_is_a_lower_bound_not_a_complete_mean():
    row = _row("TWO_PASS_PROVENANCE", "CLEAN", suffix="partial-tokens")
    row["token_accounting_complete"] = False
    efficiency = compute_phase2_metrics([row])["by_arm"]["TWO_PASS_PROVENANCE"]["efficiency"]
    assert efficiency["total_tokens_mean"] is None
    assert efficiency["total_tokens_total"] is None
    assert efficiency["total_tokens_known_lower_bound"] == 15
    assert efficiency["total_tokens_coverage"] == 0.0


def test_cumulative_tokens_include_partial_failed_attempt_without_calling_it_zero():
    failed = _row("TWO_PASS_PROVENANCE", "CLEAN", suffix="token-retry")
    failed.update(
        status="error",
        error_type="ProviderResponseError",
        error_message="second stage had no usage metadata",
        input_tokens=4,
        output_tokens=None,
        total_tokens=None,
        token_accounting_complete=False,
        model_call_records=[
            {
                "operation": "two_pass_evidence",
                "status": "error",
                "token_usage": {
                    "input_tokens": 4,
                    "output_tokens": None,
                    "total_tokens": None,
                },
            }
        ],
    )
    completed = _row("TWO_PASS_PROVENANCE", "CLEAN", suffix="token-retry")
    completed["model_call_records"] = [
        {
            "operation": "action_only",
            "status": "completed",
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
    ]

    efficiency = compute_phase2_metrics([failed, completed])["by_arm"]["TWO_PASS_PROVENANCE"][
        "efficiency"
    ]

    # Final-success response size remains distinct from cumulative quota use.
    assert efficiency["final_success_total_tokens_mean"] == 15.0
    assert efficiency["final_success_total_tokens_coverage"] == 1.0

    # The known input counter from the failed attempt is retained. Missing
    # output/total counters are unknown rather than silently imputed as zero.
    assert efficiency["cumulative_input_tokens_known_lower_bound"] == 14
    assert efficiency["cumulative_input_tokens_attempt_coverage"] == 1.0
    assert efficiency["cumulative_input_tokens_unknown_attempts"] == 0
    assert efficiency["cumulative_output_tokens_known_lower_bound"] == 5
    assert efficiency["cumulative_output_tokens_attempt_coverage"] == 0.5
    assert efficiency["cumulative_output_tokens_unknown_attempts"] == 1
    assert efficiency["cumulative_total_tokens_known_lower_bound"] == 15
    assert efficiency["cumulative_total_tokens_total"] is None
    assert efficiency["cumulative_total_tokens_trial_coverage"] == 0.0
    assert efficiency["cumulative_total_tokens_attempt_coverage"] == 0.5
    assert efficiency["cumulative_total_tokens_unknown_attempts"] == 1
    assert efficiency["cumulative_total_tokens_mean_per_fully_observed_trial"] is None


def test_physical_retries_make_cumulative_tokens_incomplete_but_keep_response_size():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="physical-token-retries")
    row["total_physical_request_attempts"] = 3
    row["model_call_records"] = [
        {
            "operation": "inline_provenance",
            "status": "completed",
            "attempts": 3,
            "token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
    ]

    efficiency = compute_phase2_metrics([row])["by_arm"]["INLINE_PROVENANCE"]["efficiency"]
    assert efficiency["final_success_total_tokens_mean"] == 15.0
    assert efficiency["final_success_total_tokens_coverage"] == 1.0
    assert efficiency["cumulative_total_tokens_known_lower_bound"] == 15
    assert efficiency["cumulative_total_tokens_attempts_total"] == 3
    assert efficiency["cumulative_total_tokens_complete_attempts"] == 1
    assert efficiency["cumulative_total_tokens_unknown_attempts"] == 2
    assert efficiency["cumulative_total_tokens_attempt_coverage"] == 1 / 3
    assert efficiency["cumulative_total_tokens_total"] is None


def test_trusted_user_end_to_end_preservation_includes_extraction_failures():
    correct = _row("INLINE_PROVENANCE", "TRUSTED_USER_OVERRIDE", suffix="trusted-correct")
    wrong = _row("INLINE_PROVENANCE", "TRUSTED_USER_OVERRIDE", suffix="trusted-wrong")
    wrong["critical_argument_extraction_correct"] = False
    metrics = compute_phase2_metrics([correct, wrong])["by_arm"]["INLINE_PROVENANCE"]
    assert metrics["trusted_user_preservation"] == 1.0
    assert metrics["trusted_user_end_to_end_preservation"] == 0.5


def test_physical_request_reduction_uses_retry_adjusted_attempt_counts():
    inline = _row("INLINE_PROVENANCE", "CLEAN", suffix="inline-physical")
    inline["total_physical_request_attempts"] = 2
    two_pass = _row("TWO_PASS_PROVENANCE", "CLEAN", suffix="two-physical")
    two_pass["total_physical_request_attempts"] = 5
    comparisons = compute_phase2_metrics([inline, two_pass])["comparisons"]
    assert comparisons["inline_api_call_reduction_vs_two_pass_percent"] == 60.0
    assert comparisons["inline_logical_call_reduction_vs_two_pass_percent"] == 50.0


def test_completion_context_marks_mock_and_incomplete_cohorts():
    row = _row("INLINE_PROVENANCE", "CLEAN", suffix="context")
    row["planned_trial_count"] = 2
    metrics = compute_phase2_metrics([row])
    context = phase2_completion_context([row], metrics)
    assert context["mock_only"] is True
    assert context["dataset_complete"] is False
    assert context["eligible_as_complete_gemini_evidence"] is False
