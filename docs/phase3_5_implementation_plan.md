# LensGuard Phase 3.5 implementation plan

Status: implementation start, 2026-09-04

Phase 3.5 is an additive experiment. The frozen Phase 2 corpus, benchmark lock,
action registry, prompts, evaluator, and results remain unchanged. The canonical
Phase 2.5 `ZERO_SHOT_V2` result tree is read only and is used only for historical
comparison.

1. Audit the 81 frozen records and export a lossless case/region inventory.
2. Add independently versioned Phase 3.5 action, evidence, model-contract,
   policy, runner, metrics, and report artifacts.
3. Build immutable, frame-scoped registries through an explicit perception
   interface. Adapt Phase 2 annotations only in the clearly labelled
   `ORACLE_REGISTRY` profile; keep automatic perception pluggable and separate
   from the action VLM.
4. Represent explicit user values as non-camera evidence and preserve provenance
   independently for every argument.
5. Parse the VLM's action, arguments, and pre-existing evidence IDs without
   semantic repair. Validate references, grounding, task policy, and the final
   `ALLOW` / `ESCALATE` / `BLOCK` decision deterministically.
6. Add future physical schemas for 16 scenes, seven conditions, text and
   non-text evidence, including restaurant multi-source binding and safety
   hazard veto representation.
7. Run the full non-GPU suite and reverify the frozen lock. Then run fixed
   nine-case smoke cohorts for Gemma 3 4B, MiniCPM-V 4.5, and Qwen3-VL 8B.
   Fix only implementation/contract defects; do not tune prompts from semantic
   failures.
8. After smoke validity, run fresh, single-attempt 81-case cohorts for
   `ACTION_ONLY`, `GROUNDED_REGISTRY`, and `ORACLE`, one BF16 model resident at a
   time. Retain raw output, call records, timing, and VRAM.
9. Report independent utility, structural, evidence-selection, grounding,
   security, and efficiency metrics. Mark absent safety and restaurant
   experiments exactly `NOT MEASURABLE IN CURRENT CORPUS`.
10. Reverify both frozen baselines by hashes/validators and assess readiness for
    the future 16 x 7 = 112-image physical collection.
