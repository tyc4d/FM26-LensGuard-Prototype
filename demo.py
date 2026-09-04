#!/usr/bin/env python3
"""Human-readable, side-effect-free LensGuard CLI demonstration."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from benchmark import _oracle_provenance, load_dataset
from firewall import evaluate_policy, load_policy, normalize_action
from providers import (
    GeminiAgentProvider,
    GeminiConsequenceProvider,
    MockAgentProvider,
    MockConsequenceProvider,
    ProviderError,
    RetryConfig,
)


def _title(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _pick_scenario(records: list[dict], scenario_id: str | None) -> dict:
    if scenario_id:
        for record in records:
            if record["scenario_id"] == scenario_id:
                return record
        raise ValueError(f"Unknown scenario id: {scenario_id}")
    for record in records:
        if (
            record.get("action_family") == "CALL"
            and record.get("condition") == "BENIGN_UNTRUSTED_SUBSTITUTION"
        ):
            return record
    return records[0]


def run_demo(args: argparse.Namespace) -> None:
    _, records = load_dataset(args.dataset)
    scenario = _pick_scenario(records, args.scenario_id)
    if args.provider == "mock":
        agent = MockAgentProvider(seed=0)
        predictor = MockConsequenceProvider(seed=0)
    else:
        retry = RetryConfig(
            max_attempts=args.max_attempts,
            max_server_delay_seconds=args.retry_max_server_delay,
        )
        model = args.model or os.getenv("GEMINI_MODEL")
        agent = GeminiAgentProvider(model=model, retry_config=retry)
        predictor = GeminiConsequenceProvider(model=model, retry_config=retry)

    proposal_response = agent.propose(
        scenario["user_prompt"], scenario["_resolved_image_path"], scenario
    )
    proposal = normalize_action(proposal_response.parsed)
    provenance = _oracle_provenance(scenario, proposal)
    if args.provider == "gemini" and args.request_delay:
        time.sleep(args.request_delay)
    consequence = predictor.predict(proposal, provenance).parsed
    firewall = evaluate_policy(
        proposal,
        provenance,
        scenario=scenario,
        consequence=consequence,
        policy_config=load_policy(args.policy),
    )

    print("LensGuard Phase 1 — DRY RUN / ORACLE PROVENANCE MODE")
    print(f"\nImage\n{scenario['image_path']}")
    print(f"\nUser\n{scenario['user_prompt']}")
    print(f"\nAI PROPOSED ACTION\n{proposal.action.value}")
    for name, value in proposal.arguments.model_dump(exclude_none=True).items():
        print(f"{_title(name)}: {value}")
    print("\nSOURCE")
    for name, source in provenance.items():
        print(f"{_title(name)}: {source}")
    print("\nPREDICTED CONSEQUENCES")
    for effect in consequence.effects:
        print(f"- {_title(effect)}")
    print(f"\nFIREWALL\n{firewall.decision.value}")
    print("\nREASON")
    print(", ".join(firewall.policy_rules_triggered))
    print(f"\nUSER WARNING\n\"{firewall.user_message}\"")
    print("\nNo action was executed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "gemini"), default="mock")
    parser.add_argument("--scenario-id")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/metadata.json"))
    parser.add_argument("--policy", type=Path, default=Path("config/policy.yaml"))
    parser.add_argument("--model", help="Exact model; defaults to GEMINI_MODEL")
    parser.add_argument("--request-delay", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-max-server-delay", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=False)
    try:
        run_demo(parse_args())
    except (ProviderError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
