# Phase 1 implementation plan

LensGuard Phase 1 is an idea-validation experiment, not a production firewall or a wearable exploit. Work proceeds in five compact stages:

1. Define the threat model, protected actions, argument-level source labels, and deterministic policy rules.
2. Generate a controlled, versioned dataset of matched synthetic scenes for `CALL`, `OPEN_URL`, and `DIRECTION_ADVICE`.
3. Validate the complete dry-run pipeline with deterministic mock providers before spending Gemini quota.
4. Run a small, manually inspected Gemini smoke test, then a resumable quota-safe benchmark if the outputs are valid.
5. Calculate metrics and generate a report that separates measured observations from limitations and future work.

The primary experiment uses **ORACLE PROVENANCE MODE**. Source attribution comes from immutable scenario metadata and is resolved against the value actually proposed by the agent. Automatic provenance estimation is outside the primary experiment.
