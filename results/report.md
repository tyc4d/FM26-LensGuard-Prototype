# LensGuard Phase 1 report

> **NO GEMINI RESULTS YET.** No live benchmark was launched because this workspace has no
> `.env` credentials. This placeholder must not be interpreted as experimental evidence.

The complete quota-free pipeline validation is under [`results/mock/`](mock/). Those outputs
are clearly marked as mock-only and cannot support a research conclusion about Gemini.

Follow the README's separate smoke and main-cohort commands. After the main Gemini run, replace
this placeholder with:

```bash
uv run python analyze_phase1.py \
  --input results/raw_results.jsonl \
  --output results/analysis.json \
  --plots-dir results/plots
uv run python generate_report.py \
  --input results/raw_results.jsonl \
  --output results/report.md
```

The generated report will contain the threat model, methodology, model identifiers, primary
CORE metrics, separately labeled source-authority exploration, limitations, and GO/NO-GO
evidence without issuing an automatic verdict.
