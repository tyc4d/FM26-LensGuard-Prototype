#!/usr/bin/env python3
"""Show one LensGuard Phase 2 dry-run decision in a human-readable CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark_phase2 import (
    _experiment_config_id,
    _selection_scope_id,
    load_phase2_dataset,
    run_phase2_trial,
)
from firewall.thin_gate import load_action_registry, load_thin_gate_policy
from phase2_schema import Phase2Arm, canonical_phase2_arm
from providers import GeminiPhase2Provider, MockPhase2Provider, ProviderError, RetryConfig

DEFAULT_SCENE = "p2_call_hotel__benign_untrusted_substitution"


def _provider(args: argparse.Namespace) -> Any:
    if args.provider == "mock":
        return MockPhase2Provider(seed=args.seed)
    return GeminiPhase2Provider(
        model=args.model or os.getenv("GEMINI_MODEL"),
        retry_config=RetryConfig(
            max_attempts=args.max_attempts,
            initial_delay_seconds=args.retry_base_delay,
            max_server_delay_seconds=args.retry_max_server_delay,
        ),
        seed=args.seed,
        thinking_level=args.thinking_level,
        max_output_tokens=args.max_output_tokens,
        api_version=args.api_version,
    )


def _print_demo(row: dict[str, Any]) -> None:
    print("USER")
    print(row["user_prompt"])
    print("\nIMAGE")
    print(row["image_path"])
    print("\nAI PROPOSED ACTION")
    arguments = ", ".join(f"{name}={value}" for name, value in row["proposed_arguments"].items())
    print(f"{row['proposed_action']}({arguments})")

    evidence = row.get("self_reported_argument_evidence", {})
    print("\nSELF-REPORTED SUPPORTING EVIDENCE")
    if evidence:
        for argument, items in evidence.items():
            if not items:
                print(f"- {argument}: <missing>")
            for item in items:
                print(
                    f"- {argument}: {item['evidence_text']!r} "
                    f"(estimate={item['source_type_estimate']}, "
                    f"confidence={item['confidence']:.2f})"
                )
    elif row["architecture_arm"] == Phase2Arm.ORACLE_PROVENANCE.value:
        print("- Oracle benchmark evidence was used (not deployable).")
    else:
        print("- None")

    print("\nMAPPED VISUAL SOURCE")
    evaluations = row.get("provenance_evaluations", [])
    if not evaluations:
        print("- No provenance mapping in this arm")
    for item in evaluations:
        print(
            f"- {item['argument_name']}: region={item['matched_region_id'] or 'none'}, "
            f"status={item['evidence_status']}, method={item['match_method'] or 'none'}, "
            f"estimated_type={item['source_type_estimate'] or 'unknown'}"
        )

    print("\nACTION REGISTRY EFFECTS")
    for effect in row.get("static_effects", []):
        print(f"- {effect.replace('_', ' ')}")
    print("\nTHIN GATE")
    print(row["gate_decision"])
    print("\nREASON")
    print(row["gate_user_message"])
    print("\nTIMING")
    print(f"Model: {row['gemini_latency_ms'] / 1000:.3f} s")
    print(f"Provenance mapping: {row['mapping_latency_ms']:.3f} ms")
    print(f"Policy gate: {row['thin_gate_latency_ms']:.3f} ms")
    print("\nDRY RUN")
    print("No call, URL navigation, or physical navigation is executed.")


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    payload, records = load_phase2_dataset(args.dataset)
    try:
        scenario = next(row for row in records if row["scenario_id"] == args.scenario_id)
    except StopIteration as error:
        raise ValueError(f"Unknown Phase 2 scenario ID: {args.scenario_id}") from error
    arm = canonical_phase2_arm(args.arm)
    provider = _provider(args)
    registry = load_action_registry(args.registry)
    policy = load_thin_gate_policy(args.policy)
    scope = _selection_scope_id([scenario])
    experiment = _experiment_config_id(
        provider,
        seed=args.seed,
        generation_seed=args.seed,
        runs=1,
        selection_scope_id=scope,
    )
    row = run_phase2_trial(
        scenario=scenario,
        arm=arm,
        run=1,
        provider_name=args.provider,
        provider=provider,
        dataset_version=str(payload["dataset_version"]),
        registry=registry,
        policy=policy,
        results_dir=args.artifacts_dir,
        selection_scope_id=scope,
        experiment_config_id=experiment,
        planned_trial_count=1,
        attempt_index=1,
        selection_seed=args.seed,
        generation_seed=args.seed,
        request_delay=args.request_delay,
    )
    close = getattr(provider, "close", None)
    if callable(close):
        close()
    if row["status"] != "completed":
        raise ValueError(
            f"Demo provider failed with {row.get('error_type')}: {row.get('error_message')}"
        )
    _print_demo(row)
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--arm", default="inline_provenance")
    parser.add_argument("--scenario-id", default=DEFAULT_SCENE)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_phase2/metadata.json"))
    parser.add_argument("--registry", type=Path, default=Path("config/action_registry.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("config/policy_phase2.yaml"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("results_phase2/demo"))
    parser.add_argument("--model")
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-server-delay", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--thinking-level",
        choices=("minimal", "low", "medium", "high"),
        default="minimal",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--api-version", default="v1beta")
    return parser.parse_args(argv)


def main() -> None:
    load_dotenv(override=False)
    try:
        run_demo(parse_args())
    except (ProviderError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
