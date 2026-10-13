# Uformer diagnostic archive (2026-10-12 to 2026-10-13)

This publication preserves the completed UID, URC and URP diagnostic packages and their local analysis records. It records reconstruction experiments on small inputs; it does not establish compression or reproduce the full CG-FedLLM workflow. Frozen configurations, original verdicts, failed attempts and historical evidence remain unchanged.

## Recorded outcomes

All values below are final eval-mode measurements, rounded for display. The linked reports and machine-readable records retain full precision. U-PATCH verdicts use the original input space.

| Arm | Tested model and input | Updates | RSE | Cosine | Original strict outcome | EP-20261012-v1 |
|---|---|---:|---:|---:|---|---|
| [U-CHECK](../diagnostics/uid_20261012_v2_r1_execution/REPORT.md) | 1x1 convolution pipeline control; synthetic input | 500 | ~7.2e-20 | ~1.000 | COMPLETED_PASS | Not assessed here |
| [U-FIT](../diagnostics/uid_20261012_v2_r1_execution/REPORT.md) | Uformer; one synthetic 32x32 input | 3,000 | 0.272741 | 0.854047 | COMPLETED_FIT_FAIL | PASS; post hoc |
| [R-FIT](../diagnostics/urc_20261013_v1_execution/REPORT.md) | Stride-1 ResNet-3; the same synthetic input | 3,000 | 0.224267 | 0.883533 | COMPLETED_FIT_FAIL | PASS; pre-specified |
| [U-PATCH](../diagnostics/urp_20261013_v1_execution/REPORT.md) | Uformer; one real LoRA-derived 32x32 patch | 3,000 | 0.193453 | 0.898105 | COMPLETED_FIT_FAIL | PASS; pre-specified |

The original strict criterion requires finite, defined final eval metrics with **RSE <= 0.01 and cosine >= 0.99**. [EP-20261012-v1](../diagnostics/exploratory_progression_20261012/POLICY.md) additionally defines evidence and execution validity, and requires **RSE <= 0.50, cosine >= 0.80, and MSE <= 0.80 times the better of the zero-prediction and scalar-mean baseline MSEs**. The scalar-mean predictor uses one scalar for the whole input. RSE is SSE divided by target energy, without a square root.

EP was selected after U-FIT's result was known; its U-FIT assessment is retrospective. It was specified before R-FIT and U-PATCH ran. A strict failure and an EP PASS answer different questions and both remain in the archive. U-CHECK is a pipeline control, not a Uformer result.

## What the evidence supports

The [matched synthetic comparison](../diagnostics/synthetic_comparison_summary_20261013/STAGE_SUMMARY.md) reports agreement on 35 shared workload conditions. R-FIT fitted that input better than U-FIT under the tested budget. The variants still differ in core blocks, parameter count, core normalization and initialization. There is one seed per variant, so this is not evidence of general architectural superiority or statistical significance.

U-PATCH reconstructed one registered real patch with MSE about **80.7% below the best constant baseline**. Its source is R3, time index 7, client 91, with slice `Phi[0,1024:1056,768:800]`. Its input and normalization differ from the synthetic runs, so comparisons with their numerical results are contextual, not matched experiments.

U-FIT and U-PATCH use 685,507-parameter Uformers; R-FIT uses 493,483 parameters. All three fit arms use an input of shape `[1,1,32,32]` and latent `[64,32,32]`: **64 times as many latent elements as input elements**. These single-input, single-seed fits establish neither compression nor generalization, whole-snapshot reconstruction, downstream utility, or successful FAF. Failure to reach near-identity within 3,000 updates does not establish general model incapacity.

## Evidence and archive limits

The linked reports describe the verification performed during those sessions, including metric recomputation and, for R-FIT and U-PATCH, forward-only saved-state replay. They are implementer reports, not independent acceptance. Publishing these files does not constitute a new experiment or a new execution of those checks.

The archive includes frozen source, configurations, manifests, selected input and fixture tensors, reports, learning curves, inventories, logs and ledger mirrors. The publication also copies 34 small original run files (child records, curves, jobs and primary ledgers) under [`diagnostics/uformer_archive_20261013/raw_runs/`](../diagnostics/uformer_archive_20261013/raw_runs/). [`archive_index.json`](../diagnostics/uformer_archive_20261013/archive_index.json) maps the original locations to 340 archived evidence files with byte counts and SHA-256 values; its external-artifact section identifies saved states and source-data bindings that are not bundled. Full external run roots, model checkpoints and source LoRA snapshots are not bundled. Some manifest bindings and analysis scripts reference historical paths under `D:/cgfed-runs`, the original repository path and its interpreter. Those paths are preserved to retain the original evidence bytes; a fresh clone alone cannot execute every historical verification or replay. Hash inventories identify artifacts but do not contain their payloads.

U-FIT's final weights were never saved; its recorded state hash cannot recover them. R-FIT and U-PATCH saved states externally. The [synthetic summary](../diagnostics/synthetic_comparison_summary_20261013/STAGE_SUMMARY.md#wording-clarification-no-record-changed) also documents the distinction between the URC analysis file's stored CRLF byte hash and the LF text hash quoted in its original report.

## Future primary gate

The requested future primary reconstruction thresholds are **RSE <= 0.50 and cosine >= 0.65**. **They are not implemented by this archive commit** and do not replace any archived configuration, classifier or recorded outcome. A subsequent change must define the complete future policy, wire it into the new runner and analyzer, and update its tests and documentation before further scientific starts. The [four-tile proposal](../diagnostics/urp_20261013_v1_execution/NEXT_EXPERIMENT_PROPOSAL.md) is a historical draft; its older EP-based decision rules are not an implementation of the requested new primary gate.
