# Scope and future work

## Current Phase 1 MVP

The current prototype is limited to:

- controlled synthetic camera-like images generated with Pillow;
- a configurable Gemini Flash model family, plus isolated mock providers;
- `CALL`, `OPEN_URL`, and `DIRECTION_ADVICE`;
- structured proposed actions and strict validation;
- **ORACLE PROVENANCE MODE** for the main experiment;
- sanitized structured consequence prediction;
- explicit deterministic action policy;
- dry-run decisions only: `ALLOW`, `WARN`, `CONFIRM`, or `BLOCK`;
- versioned result logging, metrics, plots, and evidence-oriented reporting.

Phase 1 asks whether known provenance is a useful action-gating primitive. It does not claim that provenance can yet be inferred reliably from pixels.

## Current Phase 2 MVP

Phase 2 reuses the same three protected actions and adds controlled, region-annotated Pillow
scenes. It compares Action Only, Two-Pass Provenance, Inline Provenance, and a non-deployable
Oracle upper bound. Inline Provenance asks Gemini Flash to emit an action plus self-reported
supporting evidence in one structured inference. A deterministic local mapper grounds that
evidence to annotated regions by conservative text matching and optional bbox IoU; the Thin Gate
then applies the static registry and source policy without another model call.

Phase 2 does not use an LLM consequence predictor in its main runtime path. It evaluates visual
evidence attribution and argument-level lineage, not latent causal provenance or physical-source
authenticity. Automatic-arm decisions treat the model estimate as untrusted descriptive evidence;
authorization requires separately corroborated user input or exact values from simulated trusted
reference/update channels. Benchmark region source types are evaluation-only. Oracle Provenance remains only
an upper bound. The current automatic-arm grounding adapter still matches returned evidence
against synthetic region text/boxes supplied by the benchmark generator. A deployable sensor path
must replace that scaffold with a trusted OCR/segmentation interface or validated model boxes.

## Not established by this experiment

The experiment does not establish a real physical-world attack, a vulnerability in Meta Ray-Ban or any other commercial wearable, equivalence between the Gemini API and a production wearable assistant, or readiness for production deployment. Pillow scenes are deliberately controlled and may not represent real camera noise, viewing geometry, attention, or product-level safety layers.

## Future work

- More reliable automatic provenance inference beyond self-reported evidence attribution.
- Physical printed-scene experiments and controlled capture under varied distance, angle, blur, glare, and occlusion.
- Webcam and wearable first-person-camera studies.
- Cross-model and cross-prompt evaluation.
- Real wearable integration with appropriate safety and consent review.
- Audio attacks and multimodal source conflicts.
- Additional protected actions: `SEND_MESSAGE`, `NAVIGATION`, `CALENDAR`, `LOCATION_SHARE`, and `PAYMENT`.
- IoT and robotics actions with stronger consequence and reversibility models.
- Spatial provenance and persistence across frames.
- Argument-level causal influence tracking rather than value matching alone.
- Calibration studies for warning fatigue, usable confirmation design, and false-warning costs.
- Robust normalization for international phone numbers, URL canonicalization, and richer direction semantics.
- Authenticated source mechanisms for contacts, application data, navigation, and official signage.
