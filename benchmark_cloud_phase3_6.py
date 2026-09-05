#!/usr/bin/env python3
"""Separate cloud benchmark over the frozen 81-image LensGuard corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from benchmark_phase3_5 import _build_registry
from cloud_baseline_contracts import ARMS, FROZEN_HEAD, ROOT, load_cases, prepare_case
from cloud_baseline_evaluation import normalize_response, summarize_records
from cloud_baseline_store import CloudResultStore, digest, file_digest, read_json, trial_id, write_index
from providers.base_cloud_vlm import CloudRequest, CloudResponse, require_secret_safety
from providers.gemini_vlm import GeminiProvider
from providers.openai_vlm import OpenAIProvider

PROVIDERS = {"openai": OpenAIProvider, "gemini": GeminiProvider}
NAMESPACE = ROOT / "results_cloud_baseline"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_plan(provider: str, model: str, config: dict, cases: list[dict], experiment_id: str,
               *, smoke: bool) -> dict:
    return {
        "experiment_id": experiment_id, "provider": provider, "model": model,
        "frozen_head": FROZEN_HEAD, "mode": "smoke" if smoke else "full",
        "provider_config": config, "arms": list(ARMS),
        "scientific_attempts_per_trial": 1,
        "external_tools": [], "registry_relationship_adapter": "NONE",
        "physical_measurements": "NOT_MEASURABLE",
        "cases": [prepare_case(case) for case in cases],
    }


def _rate(value: dict | None) -> str:
    if not value or value.get("rate") is None:
        return "NOT MEASURABLE"
    return f"{value['numerator']}/{value['denominator']} ({value['rate']:.2%})"


def render_provider_report(summary: dict, manifest: dict) -> str:
    lines = [f"# {manifest['provider']} — {manifest['model']}", "",
             f"Experiment: `{manifest['experiment_id']}`. INCOMPLETE: **{manifest['incomplete']}**.",
             f"Scientific requests planned: {manifest['planned_trials']}; "
             f"completed: {manifest['completed_trials']}; recorded: {manifest['recorded_trials']}.",
             f"Transport retries: {manifest['transport_retry_count']}; "
             f"429 / RESOURCE_EXHAUSTED events: {manifest['rate_limit_events']}; "
             f"backoff: {manifest['total_backoff_seconds']} seconds.",
             f"Incomplete due to quota: {'YES' if manifest['incomplete_due_to_quota'] else 'NO'}.",
             f"Stop reasons: {', '.join(manifest['stop_reasons']) or 'none'}.", "",
             "All rates below show their assessed denominator. Partial cohorts must not be "
             "treated as complete-model aggregate comparisons.", "",
             "| Arm | Critical E2E | Schema validity | Exact evidence | Camera evidence | Unsafe attacker-target execution |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, metrics in summary.get("metrics", {}).get("by_arm", {}).items():
        if arm == "ORACLE":
            continue
        lines.append(f"| {arm} | {_rate(metrics['utility']['critical_argument_accuracy_end_to_end'])} "
                     f"| {_rate(metrics['structural']['schema_validity'])} "
                     f"| {_rate(metrics['evidence_selection']['correct_evidence_selection'])} "
                     f"| {_rate(metrics['evidence_selection']['correct_visual_region_selection'])} "
                     f"| {_rate(metrics['security']['automatic_unsafe_execution'])} |")
    lines += ["", "Phase 3.6 abstention, latency, usage and cost details:", "", "```json",
              json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, sort_keys=True),
              "```", "", "Latency measures network/API transport, including retries and backoff; "
              "it excludes artificial request pacing and is not equivalent to local GPU runtime.",
              "Native schema enforcement differs from the local prompt-only schema transport.",
              "", "Physical overlay/replacement effectiveness, authenticity uncertainty, physical Safety, "
              "physical Restaurant Reservation and C0–C6 robustness: **NOT MEASURABLE**.",
              "CLOUD PHYSICAL EVALUATION: READY FOR INPUT, NOT YET MEASURED.", ""]
    return "\n".join(lines)


def finalize(store: CloudResultStore) -> dict:
    store.validate()
    manifest = store.refresh_indexes()
    summary = summarize_records(store.records(), planned_trials=len(store.expected))
    write_index(store.root / f"{store.provider}_summary.json", summary)
    write_index(store.root / f"{store.provider}_report.md",
                render_provider_report(summary, manifest), text=True)
    return manifest


def run_provider(provider, cases: list[dict], root: Path, experiment_id: str,
                 *, smoke: bool, resume: bool = False) -> dict:
    plan = build_plan(provider.provider, provider.model, provider.provider_config, cases,
                      experiment_id, smoke=smoke)
    store = CloudResultStore(root, provider.provider, plan, resume=resume)
    lock = root / f".{provider.provider}.running"
    # Exclusive process lock; stale locks require inspection, never blind deletion.
    with lock.open("x") as handle:
        handle.write(f"pid={os.getpid()}\n")
    try:
        existing = {trial_id(r["case_id"], r["arm"]): r for r in store.records()}
        stopped = [r for r in existing.values() if r.get("stop_provider") and not r.get("hard_quota")]
        if stopped:
            return finalize(store)
        resumed_gemini_needs_spacing = resume and provider.provider == "gemini" and any(
            row["completed"] for row in existing.values()
        )
        for case, prepared in zip(cases, plan["cases"], strict=True):
            for arm in ARMS:
                identity = trial_id(prepared["case_id"], arm)
                if identity in existing:
                    continue
                request_hash = digest({"case": prepared, "arm": arm, "model": provider.model,
                                       "provider": provider.provider, "config": provider.provider_config})
                raw_path = store.path("raw", prepared["case_id"], arm)
                if raw_path.exists():
                    # Recover only deterministic evaluation after a preserved response.
                    envelope = read_json(raw_path)
                    if envelope["request_sha256"] != request_hash:
                        raise ValueError("Preserved response request differs on resume")
                    response = CloudResponse(**envelope["response"])
                    if provider.provider == "gemini" and response.completed:
                        resumed_gemini_needs_spacing = True
                else:
                    if file_digest(Path(case["_resolved_image_path"])) != prepared["image_sha256"]:
                        raise ValueError("Image changed after the pre-inference plan was frozen")
                    if resumed_gemini_needs_spacing:
                        provider._bounded_sleep(provider.request_delay_seconds + 2.0)
                        resumed_gemini_needs_spacing = False
                    store.begin(prepared["case_id"], arm, request_hash, utcnow())
                    request = CloudRequest(
                        image_path=Path(case["_resolved_image_path"]),
                        prompt=prepared["prompts"][arm], response_schema=prepared["schemas"][arm],
                        case_id=prepared["case_id"], arm=arm,
                    )
                    sequence = len(store.records()) + 1
                    print(f"[{provider.provider} {sequence}/{len(store.expected)}] "
                          f"case={request.case_id} arm={arm} model={provider.model} "
                          f"start={utcnow()} status=START", flush=True)
                    response = provider.infer(request)
                    raw_path = store.save_raw(request.case_id, arm, response.to_dict(), request_hash)
                registry = _build_registry(case)
                if registry.model_dump(include_dataset_labels=True) != prepared["registry_snapshot"]:
                    raise ValueError("Registry changed after the pre-inference plan was frozen")
                record = normalize_response(
                    response=response, scenario=case, arm=arm, registry=registry,
                    experiment_id=experiment_id, raw_response_path=raw_path.relative_to(ROOT).as_posix(),
                )
                record.update({"raw_response_sha256": file_digest(raw_path),
                               "request_sha256": request_hash,
                               "image_sha256": prepared["image_sha256"],
                               "registry_sha256": prepared["registry_sha256"],
                               "prompt_sha256": prepared["prompt_sha256"][arm]})
                store.save_record(record)
                store.refresh_indexes()
                print(f"[{provider.provider} {len(store.records())}/{len(store.expected)}] "
                      f"case={prepared['case_id']} arm={arm} model={provider.model} "
                      f"status={response.error_type or 'OK'} latency={response.latency_ms:.0f}ms "
                      f"transport_attempts={response.transport_attempts} "
                      f"rate_limit_events={response.rate_limit_events} completed={response.completed} "
                      f"next_request_in={getattr(provider, 'next_request_delay_seconds', 0):.1f}s", flush=True)
                if response.stop_provider:
                    return finalize(store)
        return finalize(store)
    finally:
        store.refresh_indexes()
        lock.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openai", "gemini", "all"], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    require_secret_safety()
    load_dotenv(ROOT / ".env", override=False)
    run_id = args.run_id or ("phase3_6_cloud_smoke_v1" if args.smoke else "phase3_6_cloud_v1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run ID must be a safe identifier")
    output = NAMESPACE / "phase3_6_cloud_v1" / "smoke" / run_id if args.smoke else NAMESPACE / run_id
    if not output.resolve().is_relative_to(NAMESPACE.resolve()):
        raise ValueError("Output must remain in the cloud namespace")
    names = list(PROVIDERS) if args.provider == "all" else [args.provider]
    cases = load_cases(smoke=args.smoke)
    total = 0
    for name in names:
        model = os.getenv(PROVIDERS[name].model_environment, PROVIDERS[name].default_model)
        count = len(cases) * len(ARMS)
        print(f"provider={name} model={model} case_count={len(cases)} "
              f"arm_count={len(ARMS)} planned_request_count={count}", flush=True)
        total += count
    print(f"total_planned_request_count={total}", flush=True)
    if args.dry_run:
        return 0
    if args.validate_only:
        for name in names:
            plan = read_json(output / "plans" / f"{name}.json")
            store = CloudResultStore(output, name, plan, resume=True)
            print(json.dumps(store.validate(require_complete=args.require_complete), sort_keys=True))
        return 0
    # All target collisions checked before any provider makes a request.
    if not args.resume and any((output / "plans" / f"{name}.json").exists() for name in names):
        raise FileExistsError("Target provider run exists; refusing scientific response replacement")
    for key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        print(f"{key} present: {'YES' if os.getenv(key) else 'NO'}", flush=True)
    exit_code = 0
    for name in names:
        with PROVIDERS[name]() as provider:
            manifest = run_provider(provider, cases, output, run_id, smoke=args.smoke, resume=args.resume)
        if manifest["incomplete"]:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Exception values can include SDK or user-provided content. Diagnostics
        # remain in preserved envelopes; the console prints the category only.
        print(f"BENCHMARK_STOPPED: {type(error).__name__}", flush=True)
        raise SystemExit(2) from None
