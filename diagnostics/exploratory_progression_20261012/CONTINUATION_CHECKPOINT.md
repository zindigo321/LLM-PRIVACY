# Continuation checkpoint (derived analysis of UID-20261012-v2 r1)

Working file for resuming after an interruption. Not evidence. Directory: `diagnostics/exploratory_progression_20261012/`.

## Step 1 done: state verified (2026-10-12 ~21:15 +08:00)

- No earlier deliverables for this task existed.
- `FINAL_ARTIFACTS_SHA256.txt`: 42/42 entries match (hash and size). `manifest_r1.json` SHA-256 matches 9d3aba9a...8a8044.
- P5-A: `results/phase5/p5a_v2/artifact_manifest.json` run_root_artifacts, 39/39 non-checkpoint entries match the
  committed copies; `results/phase5/p5a_v2` is clean vs HEAD.
- R3 single record has AE + zero-predictor sufficient statistics, but no element sum, so its scalar-mean baseline is
  NOT recorded (bound only: scalar-mean MSE <= zero MSE).
- Production ResNet: HEAD `autoencoder.py` hardcodes stride 2; the working tree carries uncommitted SSBD changes
  (`StridedResNetAEConfig`). p5a.py, p5a_metrics.py, p5a_run.py, seeding.py are clean vs HEAD.
- Live hardware (CIM, read-only): i9-14900HX, 24 cores / 32 logical, 34,070,192,128 B RAM, RTX 4060 Laptop GPU.

## Plan of files

POLICY.md, ep_classify.py, test_ep_classify.py, assessment_uid_r1_ufit.json (classifier output),
compare_records.py, comparison.json, COMPARISON.md, NEXT_EXPERIMENT_SPEC.md.

## Status

- [x] POLICY.md  - [x] ep_classify.py (ruff clean; dry run on real records = PASS, 42/42 files verified)
- [x] test_ep_classify.py: 34 passed (pytest -p no:cacheprovider, from this directory)
- [x] assessment_uid_r1_ufit.json written by `ep_classify.py --out` (stdout in ep_classify.out.txt); read back.
  It records policy_sha256 fe14a225... and classifier_sha256 f7efd100...: DO NOT edit POLICY.md or ep_classify.py.
- [x] compare_records.py + comparison.json (both sides verified; stdout in compare_records.out.txt)
  U vs zero SSE ratio 0.2727411031905583, vs scalar mean 0.2728195664549294; R3 vs zero 0.9938081619334862,
  R3 scalar mean NOT recorded (ratio >= 0.9938 bound). R3 train MSE/zero (normalized, derived) at 3000 = 0.99373.
- [x] COMPARISON.md (conditions + metrics + non-matched explanation); comparison.json regenerated with param groups
  from the verified R3 checkpoint header (493,483 = 24,833 + 2x221,952 + 24,746; buffers 1,941).
- [x] NEXT_EXPERIMENT_SPEC.md (URC-v1 draft)  - [x] README.md
- [x] final read-back (2026-10-13 ~07:25): assessment policy/classifier hashes still match their files; UID final
  artifacts 42/42, manifest_r1 29/29 and frozen parent 7/7 still match; 34 tests pass; ruff check + format clean.
  Read-back fixes in the spec: submodule names (copied `groups()` needs encoder.sampling/core), the init-order claim,
  Uformer buffers, config loading. Fixes in COMPARISON.md: no direct EP application to R3; scope of the 45/45 check.
- [x] memory updated. TASK COMPLETE; nothing committed (human review boundary).

Dry-run numbers (eval): RSE 0.2727411031905583, cosine 0.8540472264633736, MSE 0.06054920855612749,
best baseline = scalar mean 0.22193865837012977, MSE/best 0.2728195664549294, M3 threshold 0.17755092669610384.
