#!/usr/bin/env python3
"""Print Phase 2 evidence indicators without applying a binary threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_phase2 import phase2_completion_context, summarize_go_nogo
from metrics_phase2 import compute_phase2_metrics
from result_store import read_jsonl
from result_store_phase2 import phase2_attempt_accounting, validate_phase2_attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results_phase2/raw_attempts.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    attempts = read_jsonl(args.input)
    validate_phase2_attempts(attempts)
    metrics = compute_phase2_metrics(attempts)
    context = phase2_completion_context(attempts, metrics)
    payload = {
        **context,
        "attempt_accounting": phase2_attempt_accounting(attempts),
        "interpretation_warning": (
            "Mock output validates plumbing only; it is not scientific Gemini evidence."
            if context["mock_only"]
            else (
                "This cohort is incomplete; indicators are descriptive partial results."
                if not context["dataset_complete"]
                else "Complete Gemini cohort; human review is still required."
            )
        ),
        "go_no_go_evidence": summarize_go_nogo(metrics),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
