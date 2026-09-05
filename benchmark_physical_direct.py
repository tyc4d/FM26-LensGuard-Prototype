"""Physical DIRECT inference only: immutable requests/responses, no scientific scoring."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cloud_baseline_store import CloudResultStore, digest, file_digest, read_json, write_index
from physical_direct_contracts import (
    CONFIG_PATH, DIRECT_SCHEMA, SMOKE_CONFIG_PATH, build_prompt, parse_output,
)
from physical_direct_inputs import (
    EXPERIMENT_ID, INPUT_MANIFEST, RESULT_ROOT, ROOT, extract_originals, load_manifest,
)
from physical_direct_local import LOCAL_MODELS, LocalDirectProvider
from providers.base_cloud_vlm import CloudRequest, redact_secrets, require_secret_safety

MODEL_IDS = {**{k: v["model_id"] for k, v in LOCAL_MODELS.items()},
             "openai": "gpt-5.6-sol", "gemini": "gemini-3.1-flash-lite"}
SMOKE_ROOT = RESULT_ROOT / "smoke" / "physical_direct_smoke_v1"


def smoke_root(alias: str) -> Path:
    # The original OpenAI smoke contains one preserved pre-generation HTTP 400.
    # Other models' semantic smoke attempts are retained, never repeated.
    return (SMOKE_ROOT.parent / "physical_direct_smoke_v1_openai_high_detail"
            if alias == "openai" else SMOKE_ROOT)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_cases(run_type: str) -> list[dict]:
    records = load_manifest()["records"]
    if run_type == "full":
        return records
    if run_type != "smoke":
        raise ValueError("Unknown run type")
    selected = read_json(SMOKE_CONFIG_PATH)["cases"]
    by_name = {r["original_filename"]: r for r in records}
    cases = [by_name[c["filename"]] for c in selected]
    if len(cases) != 4 or any(r["scenario_family"] != c["scenario"]
                               for r, c in zip(cases, selected)):
        raise ValueError("Smoke identity/scenario mismatch")
    return cases


def make_plan(alias: str, run_type: str, provider_config: dict) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID, "run_type": run_type,
        "model_alias": alias, "provider": alias, "model": MODEL_IDS[alias],
        "model_revision": LOCAL_MODELS.get(alias, {}).get("revision"),
        "arms": ["DIRECT"], "scientific_attempts_per_trial": 1,
        "input_manifest_sha256": file_digest(INPUT_MANIFEST),
        "prompt_config_sha256": file_digest(CONFIG_PATH),
        "smoke_config_sha256": file_digest(SMOKE_CONFIG_PATH),
        "response_schema": copy.deepcopy(DIRECT_SCHEMA),
        "provider_config": copy.deepcopy(provider_config),
        "ground_truth_frozen": False, "scientific_scoring": "NOT PERFORMED",
        "cases": [{"case_id": c["original_filename"], "image_sha256": c["sha256"],
                   "scenario_family": c["scenario_family"],
                   "prompt": build_prompt(c["scenario_family"])}
                  for c in selected_cases(run_type)],
    }


def request_hash(plan: dict, case: dict) -> str:
    return digest({"plan_sha256": digest(plan), "case": case, "arm": "DIRECT"})


def normalized_record(store: CloudResultStore, case: dict, response: dict, raw: Path) -> dict:
    """Structural parsing only; review labels never enter inference or evaluation."""
    from physical_direct_reporting import estimate_cost

    if (response.get("provider") != store.provider
            or response.get("model") != store.plan["model"]
            or response.get("model_id") != store.plan["model"]):
        raise ValueError("Preserved response provider/model differs from the frozen request")
    info = next(r for r in load_manifest()["records"] if r["image_id"] == case["case_id"])
    parsed = parse_output(response.get("output_text", ""))
    if not response["completed"]:
        parsed["schema_valid"] = False
    record = {k: v for k, v in response.items() if k != "raw_response"}
    record.update(parsed)
    record.update({
        "experiment_id": EXPERIMENT_ID, "run_type": store.plan["run_type"],
        "model_alias": store.provider, "provider": store.provider,
        "model": store.plan["model"], "model_id": store.plan["model"],
        "model_version": store.plan.get("model_revision") or store.plan["model"],
        "case_id": case["case_id"], "image_id": case["case_id"],
        "original_filename": case["case_id"], "image_sha256": case["image_sha256"],
        "scenario_family": info["scenario_family"], "quality_class": info["quality_class"],
        "inference_contamination_risk": info["inference_contamination_risk"],
        "arm": "DIRECT", "scientific_attempt": 1,
        "raw_response_path": os.path.relpath(raw, ROOT),
        "raw_response_sha256": file_digest(raw),
        "request_sha256": request_hash(store.plan, case),
        "prompt_sha256": digest(case["prompt"]), "schema_sha256": digest(DIRECT_SCHEMA),
        "input_manifest_sha256": store.plan["input_manifest_sha256"],
        "prompt_config_sha256": store.plan["prompt_config_sha256"],
        "error_type": response.get("error_type") or parsed["error_type"],
        "api_error_type": response.get("error_type"),
        "scientific_scoring_status": "NEEDS_HUMAN_REVIEW",
    })
    record.update(estimate_cost(response))
    return record


def finalize(store: CloudResultStore) -> dict:
    from physical_direct_reporting import render_model_report, summarize_records

    store.refresh_indexes()
    summary = summarize_records(store.records(), len(store.expected))
    write_index(store.root / f"{store.provider}_summary.json", summary)
    write_index(store.root / f"{store.provider}_summary.md", render_model_report(summary), text=True)
    return summary


def validate_provider(root: Path, alias: str, *, require_complete: bool = False) -> dict:
    path = root / "plans" / f"{alias}.json"
    plan = read_json(path)
    expected = make_plan(alias, plan["run_type"], plan["provider_config"])
    if plan != expected:
        raise ValueError("Frozen plan differs from current input/prompt/model contract")
    store = CloudResultStore(root, alias, plan, resume=True)
    result = store.validate(require_complete=require_complete)
    cases = {c["case_id"]: c for c in plan["cases"]}
    for row in store.records():
        case = cases[row["case_id"]]
        raw = store.path("raw", row["case_id"], "DIRECT")
        envelope = read_json(raw)
        if envelope["request_sha256"] != request_hash(plan, case):
            raise ValueError("Request differs from predeclared plan")
        if normalized_record(store, case, envelope["response"], raw) != row:
            raise ValueError("Normalized record cannot be reproduced from preserved raw response")
        if row["completed"] and alias in LOCAL_MODELS:
            if row.get("model_revision") != LOCAL_MODELS[alias]["revision"]:
                raise ValueError("Local checkpoint revision differs")
    result["completed_trials"] = sum(r["completed"] for r in store.records())
    return result


def run_provider(alias: str, provider, run_type: str, *, root: Path | None = None,
                 resume: bool = False, image_paths: dict | None = None) -> dict:
    if provider.provider != alias or provider.model != MODEL_IDS[alias]:
        raise ValueError("Provider/model differs from the frozen model selection")
    root = Path(root or (smoke_root(alias) if run_type == "smoke" else RESULT_ROOT))
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f".{alias}.running"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    try:
        # Snapshot once before local model initialization adds observed runtime metadata.
        plan = make_plan(alias, run_type, copy.deepcopy(provider.provider_config))
        store = CloudResultStore(root, alias, plan, resume=resume)
        prior = store.records()
        if any(r.get("stop_provider") for r in prior):
            raise RuntimeError("Provider has a preserved stop condition; no automatic rerun")
        paths = image_paths if image_paths is not None else extract_originals()
        first_send = True
        for sequence, case in enumerate(plan["cases"], 1):
            name = case["case_id"]
            if store.path("records", name, "DIRECT").exists():
                continue
            raw = store.path("raw", name, "DIRECT")
            if raw.exists():
                envelope = read_json(raw)
                if envelope["request_sha256"] != request_hash(plan, case):
                    raise ValueError("Interrupted raw response has a mismatched request")
                response = envelope["response"]
            else:
                if file_digest(paths[name]) != case["image_sha256"]:
                    raise ValueError("Original image hash mismatch before inference")
                request = CloudRequest(paths[name], case["prompt"], DIRECT_SCHEMA, name, "DIRECT")
                if alias == "gemini" and first_send:
                    # A new process has no previous adapter pacing clock (including resume).
                    delay = float(plan["provider_config"]["request_delay_seconds"])
                    if not math.isfinite(delay) or delay < 8:
                        raise ValueError("Gemini requires a minimum 8-second pacing interval")
                    time.sleep(delay)
                first_send = False
                stamp = utcnow()
                store.begin(name, "DIRECT", request_hash(plan, case), stamp)
                print(f"[{alias} {sequence}/{len(plan['cases'])}] case={name} arm=DIRECT "
                      f"model={plan['model']} start={stamp}", flush=True)
                answer = provider.infer(request)
                response = answer.to_dict() if hasattr(answer, "to_dict") else answer
                response = redact_secrets(response)
                raw = store.save_raw(name, "DIRECT", response, request_hash(plan, case))
                if file_digest(paths[name]) != case["image_sha256"]:
                    raise ValueError("Original image changed during inference; raw preserved")
            store.save_record(normalized_record(store, case, response, raw))
            store.refresh_indexes()
            print(f"[{alias} {sequence}/{len(plan['cases'])}] case={name} arm=DIRECT "
                  f"status={response.get('api_status')} completed={response['completed']} "
                  f"latency={response['latency_ms']:.0f}ms "
                  f"transport_attempts={response['transport_attempts']} "
                  f"rate_limit_events={response.get('rate_limit_events', 0)} "
                  f"next_request_in={getattr(provider, 'next_request_delay_seconds', 0):.1f}s", flush=True)
            if response.get("stop_provider"):
                break
        return finalize(store)
    finally:
        lock.unlink()


def construct_provider(alias: str):
    if alias in LOCAL_MODELS:
        return LocalDirectProvider(alias)
    require_secret_safety()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    if alias == "openai":
        from physical_direct_openai import PhysicalDirectOpenAIProvider
        return PhysicalDirectOpenAIProvider(model=MODEL_IDS[alias])
    delay = float(os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "8"))
    if not math.isfinite(delay) or delay < 8:
        raise ValueError("Physical Gemini requests require at least 8 seconds pacing")
    from providers.gemini_vlm import GeminiProvider
    return GeminiProvider(model=MODEL_IDS[alias])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_IDS, "all"], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    run_type = "smoke" if args.smoke else "full"
    aliases = list(MODEL_IDS) if args.model == "all" else [args.model]
    cases = selected_cases(run_type)
    root = smoke_root(args.model) if args.smoke else RESULT_ROOT
    if args.dry_run:
        for alias in aliases:
            print(json.dumps({"model_alias": alias, "model": MODEL_IDS[alias],
                              "case_count": len(cases), "arm_count": 1,
                              "planned_requests": len(cases), "run_type": run_type}))
        print(f"Total planned requests: {len(cases) * len(aliases)}; network requests: 0")
        return 0
    if args.validate_only:
        print(json.dumps({a: validate_provider(smoke_root(a) if args.smoke else root, a,
                                              require_complete=args.require_complete)
                          for a in aliases}, indent=2))
        return 0
    if args.model == "all":
        for alias in aliases:
            python = LOCAL_MODELS.get(alias, {}).get("runtime_python", str(ROOT / ".venv/bin/python"))
            command = [python, str(Path(__file__).resolve()), "--model", alias, "--" + run_type]
            if args.resume:
                command.append("--resume")
            subprocess.run(command, cwd=ROOT, check=True)
        return 0
    provider = construct_provider(args.model)
    try:
        run_provider(args.model, provider, run_type, resume=args.resume)
        result = validate_provider(root, args.model, require_complete=True)
        return 0 if result["completed_trials"] == len(cases) else 2
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
