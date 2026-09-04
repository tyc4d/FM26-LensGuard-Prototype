# Phase 3.6 replay of Phase 3.5 Grounded Registry outputs

Report version: `phase3.6-phase3.5-replay-report-v1`

## Replay scope

- Source records: 243
- Evaluated through both deterministic gates: 212
- Explicitly `NOT_EVALUABLE`: 31
- Model rerun performed: no
- Proposal input: preserved `parsed_json_payload`
- Registry input: preserved `evidence_registry`, unchanged
- Relationship adapter: none

All 31 non-evaluable records are preserved Phase 3.5 action-contract errors; no identity was dropped or repaired.

## Evaluation-only reference disposition

Reference version: `phase3.6-legacy-replay-reference-disposition-v1`

- `ALLOW`: `CLEAN_TRUSTED`, `EXPLICIT_USER_OVERRIDE`
- `ESCALATE`: all other frozen conditions, including `TRUSTED_BUT_CONFLICTING_UPDATE`
- Expected `BLOCK` cohort: none in the legacy corpus

This reference is fixed from documented Phase 3.6 policy semantics; it is used only for metrics and never influences gate decisions. The original Phase 2 expected posture remains present in every replay record.

## Old-to-new decisions

- `ALLOW -> ALLOW`: 30
- `ALLOW -> ESCALATE`: 47
- `BLOCK -> BLOCK`: 21
- `ESCALATE -> ESCALATE`: 114
- `NOT_EVALUABLE -> NOT_EVALUABLE`: 31

## Abstention and security metrics

- Unsafe Auto-Execution Rate: 0/133 (0.000000); assessed 133/153 eligible (coverage 0.869281)
- Escalation Recall: 117/133 (0.879699); assessed 133/153 eligible (coverage 0.869281)
- Required-Escalation Block Rate: 16/133 (0.120301); assessed 133/153 eligible (coverage 0.869281)
- False Escalation Rate: 44/79 (0.556962); assessed 79/90 eligible (coverage 0.877778)
- False Block Rate: 5/79 (0.063291); assessed 79/90 eligible (coverage 0.877778)
- Clean Utility / Allow Rate: 0/43 (0.000000); assessed 43/45 eligible (coverage 0.955556)
- Safe Resolvable Allow Rate: 30/79 (0.379747); assessed 79/90 eligible (coverage 0.877778)
- Conflict Detection Recall: NOT MEASURABLE
- Authenticity-Unknown Escalation Rate: NOT MEASURABLE

Blocking is counted separately from escalation. A block in a case whose expected Phase 3.6 outcome is user escalation does not count as successful escalation recall.

## Argument preservation

- Proposal records preserved: 212/212 assessed of 243 source records
- Argument values preserved: 286/286 assessed of 324 eligible
- Restaurant argument isolation: NOT MEASURABLE (no Restaurant Reservation actions in the legacy corpus)

## Interpretation boundary

The Phase 3.5 oracle registry deliberately does not encode Phase 3.6 semantic roles or target-object associations. The replay does not infer those fields from source labels, claim roles, bounding boxes, or model output. Consequently, new `INSUFFICIENT_EVIDENCE` outcomes are an expected compatibility limitation rather than evidence about physical detector effectiveness.

The legacy Phase 3.5 conflict proxy contains 118 argument labels, but it is not Phase 3.6 task-valid conflict truth.

OVERLAY/REPLACEMENT EFFECTIVENESS: NOT MEASURABLE UNTIL PHYSICAL PILOT

SAFETY PHYSICAL EFFECTIVENESS: NOT MEASURABLE UNTIL PHYSICAL PILOT

RESTAURANT PHYSICAL EFFECTIVENESS: NOT MEASURABLE UNTIL PHYSICAL PILOT
