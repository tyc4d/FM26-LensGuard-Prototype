# Phase 3.6 physical-pilot software fixtures

These JSON records test ingestion and validation only. They are not camera
captures, scientific benchmark samples, or evidence of physical effectiveness.
Their required `fixture_kind` and `scientific_sample: false` wrapper clearly
distinguishes them from dataset annotation records; it is an audit marker, not
an extraction-proof access-control boundary.

The fixtures exercise adjacent, partial-overlay, and full-replacement scene
construction. Construction metadata records how the scene was made; it does
not establish that visible information is authentic, legitimate, or malicious.
A replacement fixture therefore contains no invented bounding box for the
fully hidden original evidence.

`task_target_object_id` and per-region `associated_target_object_id` are stable
human relationship annotations for deterministic grounding analysis. They are
not shown to the action model and do not assert authenticity.

The current collection configuration selects Protocol A: one predefined attack
mode per base scene, carried across C0-C6. A balanced Latin-square assignment
keeps the existing 16 x 7 = 112 capture count and avoids tying every mode to the
same numeric scene suffix. Attack mode is still fully confounded with base scene
under Protocol A, despite being balanced across families and suffixes. Protocol
B would add attack-mode variants and change the planned image count, so it
requires an explicit user-approved protocol revision before use.

Each configuration entry records both its frozen Phase 3.5 scene basis and its
new Phase 3.6 construction description. These descriptions govern only the
future, uncollected Phase 3.6 pilot; they do not relabel, reinterpret, or modify
any Phase 3.5 plan, record, result, or scientific claim.
