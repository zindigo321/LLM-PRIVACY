# Stage summary: the closed synthetic Uformer / ResNet comparison (2026-10-13)

This is a derived summary, written after both runs had closed. It reopens no run and changes no protocol, record,
status, policy or report. Every number comes from `stage_summary.json`. That file was written by `extract_summary.py`
only after the files it reads matched the hash lists sealed at the end of each run (r1: 42/42 entries, URC run root and
mirror: 15/15, URC evidence inventory: 38/38). The original reports stay as they are:
`uid_20261012_v2_r1_execution/REPORT.md` and `urc_20261013_v1_execution/REPORT.md`.

## Conclusion

| | U-FIT (UID-20261012-v2 r1) | R-FIT (URC-20261013-v1) |
|---|---|---|
| Model | Uformer encoder-decoder (BasicUformer cores), 685,507 parameters | stride-1 ResNet-3 (residual cores), 493,483 parameters |
| Input | synthetic `1.6·U[0,1) − 0.8`, [1,1,32,32], SHA-256 `fc5e1283…d722` | the same bytes |
| Strict near-identity criterion (eval; RSE ≤ 0.01, cosine ≥ 0.99) | **COMPLETED_FIT_FAIL** | **COMPLETED_FIT_FAIL** |
| EP-20261012-v1 | **PASS** (post hoc: the policy was chosen after this result) | **PASS** (pre-specified) |
| Final eval RSE / cosine | 0.2727411031905583 / 0.8540472264633736 | 0.22426682990470648 / 0.8835333171508959 |

- **Both failed** their strict near-identity criterion, and **both passed** EP-20261012-v1. Neither of these results
  changes the other.
- **Matched workload.** The URC analysis found 0 violations in 35 shared conditions (MATCHED): input bytes and
  baselines, CPU fp32, 24/24 threads, determinism, torch build, the complete Adam settings (lr 2e-4), the 3,000-update
  budget, the curve schedule, and every guard and timing setting.
  - Under that matched workload, R-FIT had the lower reconstruction error and the higher cosine.
  - δRSE (R − U) = −0.0484742732858518.
  - δcosine (R − U) = +0.02948609068752228.
- **What was compared.** The result concerns the two variants as tested, including the four disclosed differences:
  - the core blocks;
  - the parameter count (0.72×);
  - the core normalization (BatchNorm vs. LayerNorm);
  - initialization: trunc-normal(0.02) in the Uformer cores versus PyTorch defaults, and therefore unequal
    decoder sampling-layer initial values.
- **What it does not establish:**
  - statistical significance: there was one seed per architecture, and seed-to-seed variation was never measured;
  - that either architecture is generally better;
  - compression: the latent [64,32,32] has 64 times as many elements as the input;
  - generalization, behaviour on real LoRA data, or downstream utility.

## Timing

The comparable figure is **end-to-end arm wall time**: R-FIT 87.531 s, U-FIT 506.344 s, a ratio of 5.78.

- **Scope.** The supervisor's monotonic clock runs from just before the arm's start record is written until the child
  process exits. That covers:
  - interpreter start-up, imports and CPU setup;
  - the model build, the input and the baselines;
  - the step-0 measurement;
  - the 3,000 updates with their curve-point measurements;
  - the final evaluation and the child record.

  For R-FIT it also covers writing and reading back two state files; U-FIT saved none. It does not cover package
  verification, which is charged to the invocation aggregate (U-FIT 510.516 s, which includes its 4.0 s U-CHECK arm;
  R-FIT 87.640 s).
- **Limitation.** Both runs used the same machine, interpreter, torch build and threads. They started about 13.7 h
  apart (2026-10-12T11:53:29Z and 2026-10-13T01:33:09Z). Background load was neither controlled nor recorded, so the
  ratio is an observation from two single runs, not a controlled speed measurement.

## Wording clarification (no record changed)

URC `REPORT.md` gives `analysis/analysis.json` the SHA-256 `728f7934…cf6`. That is the hash `urc_analyze.py` computed
over the LF text before writing it. The file was written in text mode on Windows, so its bytes have CRLF line endings
(12,735 bytes, SHA-256 `744ae6c4…0d45`, the value in `EVIDENCE_INVENTORY_SHA256.txt`). With line endings normalized to
LF, the bytes hash to `728f7934…cf6`. Both values describe the same content; the inventory value is the hash of the
file as stored. This was checked from the bytes (`clarifications` in `stage_summary.json`).

## Files

- `extract_summary.py`: read-only extraction (no model code; refuses on any hash mismatch).
- `stage_summary.json`: its output, 6,495 bytes, SHA-256
  `f07039a5ca8c120bd9cc10f188fac58890127061508b680bb879189ea43b7858`.
- `extract_summary.out.txt`: the console output.
- The first run of the script happened before a formatter-only edit. That run's output was byte-identical and was
  discarded; the saved output comes from the saved script.
