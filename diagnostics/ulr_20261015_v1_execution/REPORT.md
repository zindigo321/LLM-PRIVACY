# ULR-20261015-v1 result: Uformer Latent Reduction (256 -> 64 -> 16 transmitted fp32 elements)

The implementer wrote this report. It is not an independent acceptance. Every number comes from
`analysis/analysis.json` (`977c1a52b74da14d89b2dadce1d6d6cfad292327679adf043c4166e1474abef6`) or from
`report_tables.md`, which is rendered from it.

## 1. Outcome

- **A. The element reduction is real and verified.** For every arm state, the sender emitted exactly 16 c fp32 values
  per tile:
  - 256 (C16), 64 (C4) and 16 (C1) values, as 1,024, 256 and 64 little-endian bytes;
  - against a 1,024-element, 4,096-B fp32 input tile.

  A standalone receiver reconstructed every tile from those parsed bytes only. It was loaded from only the
  `expand.*` and `base.decoder.*` tensors (148 tensors). Payload lengths, wire shapes and the receiver path were
  checked in EVAL, by the torch-free independent verification and by a fresh-process replay. The C1 arm is a genuine
  1/64 element ratio and a 1/64 bare fp32 payload ratio.
- **B. UPP-20261014-v1: every arm passes.** C16-CONTROL, C4-REDUCED and C1-REDUCED are each PASS in EARLY and LATE
  (study PASS). The smallest tested adapted element count that passes is 16, but only just:
  - C1-REDUCED has held-out RSE 0.951 / 0.953 and cosine 0.223 / 0.218, about 5% below the all-zero output's RSE
    of 1.0;
  - one LATE tile fails.
- **C. Reduction damages reconstruction, and the fixed 6,000-update adaptation recovers little.**
  - **Damage at step zero.** Inserting the PCA bottleneck raised normalized MSE by 31-35% at 64 elements and 44-48%
    at 16 elements, against the unmodified source.
  - **Recovery.** Adaptation removed only 10-22% of that damage.
  - **Against C16.** The final reduced arms stay 1.32x (C4) and 1.44-1.48x (C1) worse than the equally adapted C16
    control, and they are worse on every tile.
  - **Input dependence.** It falls with the element count. The pooled SSE ratio of correct to wrong payloads is
    0.46 (C16), 0.75 (C4) and 0.89 (C1).
- **D. Every comparator beats the learned compressor.**
  - **Matching element counts.** The TRAIN-fitted input PCA beats the arm at each tested count: PCA-16 versus C1,
    PCA-64 versus C4 and PCA-256 versus C16. The pooled held-out SSE is 2%, 4% and 10% lower, and the arm wins on
    only 21-27%, 13-15% and 4% of held-out tiles.
  - **Matching bytes.** At 256 B, SCALAR-4 (1,024 2-bit indices) and the UOA source's 256-int8 latent both beat C4's
    64 fp32 values, so fewer elements did not buy more quality per byte. At 64 B, PCA-16 is better. At 1,024 B,
    whole-tile int8 and PCA-256 are better.
  - **Secondary target.** Normalized MSE stays 146x (C16), 195x (C4) and 217x (C1) above the 1e-4 secondary target
    on TRAIN.

**Inherited cost.** Every adapted model carries the source's 12,000 inherited updates plus 6,000 new ones. None was
trained from scratch, and this warm-start adaptation is not the residual-guided innovation.

## 2. What ran

- **Package.** `diagnostics/ulr_20261015_v1/manifest_ulr.json`, SHA-256
  `8db0e72f9f9315c105dd0e82b5bc189b15f078ff4b300b18222f666bd9b00bb6` (61 files, 12 external), frozen 2026-10-15
  11:08:58 +08:00.
  - Pre-launch verification: all_ok.
  - Authorization: the user's message of 2026-10-15 (`EXECUTION_AUTHORIZATION.md`). The digest itself was not
    human-approved.
- **Code base.** Worktree `D:/projects-cd/cg-fedllm-ulr`, branch `ulr-20261015-v1`, at the verified
  `origin/main` `0238e3b774e0f97f171bc5cbc53f1ec06aa49c7e`.
  - Literature main: `7877ef056f823d50c1f221ec99a092edfe43e9da`.
  - The UOA modules and their dependencies are vendored byte for byte (`SOURCE_INVENTORY.md`, 15 files, each equal
    to its UOA manifest entry).
- **Source.** UOA U-LEAKY-LONG `state_12000.safetensors`: file `ce8659b1...3646`, state `2f8e67bf...f3ea`. It was
  verified against the UOA training record and the UOA EVAL job.
- **Invocation.** One invocation, 2026-10-15 11:09:46 to 11:38:14 +08:00, exit 0. All five steps COMPLETED, every
  stderr is empty, there was no guard action, and the ledger is closed.

| Step | Wall (s) | Limit (s) | Sampled peak RSS (B) | Notes |
|---|---:|---:|---:|---|
| INIT | 103.6 | 600 | 760,819,712 | BN diagnostic, one channel PCA (0.008 s SVD), bridge inits, C16 identity check, step-zero TRAIN for c = 16 / 8 / 4 / 2 / 1 |
| C16-CONTROL | 449.4 | 1,800 | 716,201,984 | 6,000 updates, median 0.0644 s / update |
| C4-REDUCED | 450.2 | 1,800 | 724,107,264 | 6,000 updates, median 0.0648 s |
| C1-REDUCED | 445.8 | 1,800 | 737,935,360 | 6,000 updates, median 0.0640 s |
| EVAL | 257.8 | 1,500 | 878,379,008 | the only ULR opening of EARLY / LATE |

- **Counters.** 1 / 1 invocation, 3 / 3 neural starts (18,000 updates), 1 / 1 channel-PCA fit, 3 / 3 synthetic pilots
  (24.1 + 25.6 + 26.6 s, all GO), and 0 input-PCA or quantizer fits.
- **Time.**
  - Total scientific wall: about 2,030 s of 9,000. That is pilots 76.2 + invocation 1,707.2 + independent
    verification about 7 (from its output-file timestamps; it was not timed) + replay 238.8.
  - Evaluation plus verification: about 504 s of 2,000.
- **Memory and evidence.** Every child limit was at least 4.13 GB, and the EVAL OS peak working set was 1.11 GB. The
  retained new evidence is listed in `EVIDENCE_INVENTORY_SHA256.txt` (run root 167,668,070 B), against 512 MiB.
- **Background load** was not controlled: CPU load was 20% before and 15% after.

## 3. A. Did encode emit 256 / 64 / 16 values, and did the receiver use only them?

| Arm | Wire latent | Elements / tile | Payload / tile | Element ratio | Receiver state (decoder side only) |
|---|---|---:|---:|---:|---|
| C16-CONTROL | [16, 4, 4] | 256 | 1,024 B | 1/4 | 148 tensors, 154,048 B |
| C4-REDUCED | [4, 4, 4] | 64 | 256 B | 1/16 | 148 tensors, 153,280 B |
| C1-REDUCED | [1, 4, 4] | 16 | 64 B | 1/64 | 148 tensors, 153,088 B |

- **Wire format.** It is shape-aware: little-endian fp32 in [c, 4, 4] order. Malformed lengths raise. The old 256-value
  UCX / UOA codec refuses 64 values (unit test) and is not used by any arm.
- **Observed in EVAL.** The payload arrays have exactly 64 c bytes per tile for both arm states and all three cohorts.
  They equal the saved wire latents byte for byte; the independent verifier made 57 / 57 payload checks, wire
  latents included.
- **Receiver isolation.**
  - The receiver is a separate module holding `expand` and the decoder. It saw only parsed bytes plus the shared
    configuration (c, the TRAIN scale).
  - Unit tests show its output is bitwise the full model's decode, and that changing the input tensor after encoding
    does not change it.
  - The in-training full-TRAIN evaluations, which used the model's own decode, equal the EVAL receiver sums bit for
    bit at steps 0 and 6,000 for every arm.
- **Fresh-process replay.** The re-encoded payloads equal the saved bytes. New receivers decoding the SAVED bytes
  reproduce every prediction and control sum: 126 / 126 checks bit-identical.
- **What the receiver must hold.** The cores still run at [16, 4, 4] inside the decoder: expand turns 16 c received
  values into 256 internal features. No 256-value cache exists anywhere on the receiver side.

## 4. B. UPP-20261014-v1 (vendored hash-pinned classifier; per arm, final states)

| Unit | EARLY pooled RSE / cos | LATE pooled RSE / cos | EARLY | LATE | Study | Tiles PASS (E / L) |
|---|---|---|---|---|---|---|
| C16-CONTROL:FINAL | 0.6587 / 0.5885 | 0.6622 / 0.5851 | PASS | PASS | **PASS** | 168/168, 336/336 |
| C4-REDUCED:FINAL | 0.8682 / 0.3646 | 0.8742 / 0.3561 | PASS | PASS | **PASS** | 168/168, 336/336 |
| C1-REDUCED:FINAL | 0.9511 / 0.2226 | 0.9531 / 0.2183 | PASS | PASS | **PASS** | 168/168, 335/336 |
| C1-REDUCED:STEP0 (descriptive) | 0.9828 / 0.1426 | 0.9833 / 0.1407 | PASS | PASS | PASS | 142/168, 284/336 |

- **Comparators and references.** Every one passes: PCA-16 / 64 / 256, SCALAR-4, INT8-SHARED and the source FLOAT /
  INT8.
- **Report-only condition.** The constant-baseline MSE condition stays report-only, and no extra threshold was added.
- **Agreement.** The independent verifier re-derived the 6 pooled outcomes and 3 study decisions of the final arms.
- **What the PASS means.** "16 elements pass" is a permissive-screen statement. At 16 elements the reconstruction is
  only about 5% better than predicting zero. It is not evidence of useful compression.

## 5. C. Damage, recovery and input dependence

Normalized MSE (direct; final = 6,000 adaptation updates after the 12,000 inherited ones):

| Arm (elements) | Cohort | Source | Step 0 | Final | Step 0 / source | Final / step 0 | Damage recovered | Final / C16 final |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C16-CONTROL (256) | TRAIN / EARLY / LATE | 0.015294 / 0.014808 / 0.015488 | 0.015294 / 0.014808 / 0.015488 | 0.014632 / 0.014264 / 0.014990 | 1.000 (identity) | 0.957 / 0.963 / 0.968 | n/a (no damage) | 1 |
| C4-REDUCED (64) | TRAIN / EARLY / LATE | same | 0.020642 / 0.019461 / 0.020403 | 0.019484 / 0.018800 / 0.019788 | 1.350 / 1.314 / 1.317 | 0.944 / 0.966 / 0.970 | 21.6% / 14.2% / 12.5% | 1.332 / 1.318 / 1.320 |
| C1-REDUCED (16) | TRAIN / EARLY / LATE | same | 0.022647 / 0.021282 / 0.022258 | 0.021720 / 0.020596 / 0.021574 | 1.481 / 1.437 / 1.437 | 0.959 / 0.968 / 0.969 | 12.6% / 10.6% / 10.1% | 1.484 / 1.444 / 1.439 |

- **Step zero (observed).**
  - The C16 bridge reproduced the source within the preregistered tolerance: latent max error 9.5e-7, output max
    difference 1.06e-6, pooled normalized-MSE relative difference 1.1e-9. It is not bitwise identical.
  - Step-zero TRAIN RSE rises as the width falls. The forward-only widths 8 and 2 are never trained and are not
    adapted models.

    | c (elements) | 16 (256) | 8 (128) | 4 (64) | 2 (32) | 1 (16) |
    |---|---:|---:|---:|---:|---:|
    | TRAIN RSE | 0.661 | 0.799 | 0.892 | 0.949 | 0.979 |
    | TRAIN cosine | 0.582 | 0.449 | 0.330 | 0.229 | 0.153 |

  - The C16 "damage recovered" fraction is undefined, since step 0 equals the source to 1e-9.
- **The channel PCA spectrum is flat (observed).** The 16 TRAIN latent channel variances are spread out: cumulative
  explained variance is 11.5% at 1 component, 36.9% at 4, 62.9% at 8 and 100% at 16. Singular values run from 95.9
  to 52.3, with no near-degenerate pair. A rank-c truncation therefore discards most of the source latent's variance
  before adaptation.
- **Adaptation (observed).** TRAIN learning curves (eval mode, wire path) fall monotonically, with shrinking
  increments, in every arm (`report_tables.md` C3; `learning_curves_ulr.svg`).
  - The reduced arms improve on 94-99% of tiles relative to their step zero. They stay worse than the final C16 on
    every tile (0 / 1,440, 0 / 168, 0 / 336).
  - The C16 control also improves on the source: TRAIN 0.015294 -> 0.014632, RSE 0.661 -> 0.633.
  - The composed bridge map stayed full rank in C16, with singular values 0.81-1.15. In C1 the single wire channel's
    standard deviation grew from 0.63 to 1.86.
- **Input dependence.** On every held-out tile, every final arm beats both the wrong-snapshot payload (crossing
  clients) and the all-zero wire payload. The margin shrinks with the element count:

  | Pooled SSE ratio (EARLY / LATE) | C16 | C4 | C1 | Source FLOAT (256) |
  |---|---:|---:|---:|---:|
  | correct / wrong | 0.461 / 0.456 | 0.747 / 0.751 | 0.893 / 0.895 | 0.508 / 0.500 |
  | correct / zero wire | 0.627 / 0.633 | 0.858 / 0.863 | 0.901 / 0.905 | — |

  The C16 adaptation made its decoder slightly more input-dependent than the source (0.46 against 0.50).
- **Pre-training BN diagnostic (source U-LEAKY-LONG; copies; recorded only).**
  - On the four registered batches, eval-mode normalized MSE was 0.01477-0.01538 against 0.01483-0.01540 in train
    mode, at most 0.64% apart.
  - The maximum per-element output difference was 0.074-0.106 in normalized space.
  - This is the paired check that UOA had not run on its final models. It shows similar batch losses, not identical
    outputs. Nothing was changed because of it.

## 6. D. Comparators and the secondary target

**Matching element counts** (final arm against the input PCA with the same number of fp32 values, held-out EARLY /
LATE):

| Elements | Arm RSE | PCA RSE | Pooled SSE arm / PCA | Tiles where the arm is lower |
|---:|---|---|---|---|
| 16 (64 B) | C1 0.951 / 0.953 | PCA-16 0.931 / 0.938 | 1.022 / 1.017 | 36/168, 90/336 |
| 64 (256 B) | C4 0.868 / 0.874 | PCA-64 0.833 / 0.840 | 1.043 / 1.041 | 22/168, 50/336 |
| 256 (1,024 B) | C16 0.659 / 0.662 | PCA-256 0.597 / 0.601 | 1.104 / 1.103 | 7/168, 13/336 |

**Matching bytes** (element kinds differ; ranked by pooled held-out SSE, best first):

- **64 B.**
  - PCA-16: RSE 0.931 / 0.938.
  - C1: 0.951 / 0.953.
- **256 B.**
  - SCALAR-4 (1,024 2-bit indices): 0.120 / 0.118.
  - UOA source INT8 latent (256 int8 values): 0.684 / 0.684.
  - PCA-64 (64 fp32): 0.833 / 0.840.
  - C4 (64 fp32): 0.868 / 0.874.
- **1,024 B.**
  - INT8-SHARED (1,024 int8): 2.2e-4 / 2.1e-4.
  - PCA-256: 0.597 / 0.601.
  - C16: 0.659 / 0.662.
  - Source FLOAT: 0.684 / 0.684.

- **No byte saving at 256 B.** Spending 256 B as 64 fp32 learned values is worse than spending them as 256 int8 values
  of the same source latent, or as 1,024 2-bit scalar codes. Reducing the element count did not improve quality per
  byte here.
- **Reproduction against UOA.** The comparators and the source references reproduce the UOA records bit for bit on
  all three cohorts: PCA-256, PCA-64, SCALAR-4, INT8-SHARED, and the source FLOAT and INT8. So do the source TRAIN /
  EARLY / LATE latents.

**Secondary target (normalized MSE <= 1e-4; UOA LOSS_CONTRACT reading R2):**

| Method | TRAIN | EARLY | LATE | TRAIN / 1e-4 |
|---|---:|---:|---:|---:|
| C16-CONTROL final | 0.014632 | 0.014264 | 0.014990 | 146 |
| C4-REDUCED final | 0.019484 | 0.018800 | 0.019788 | 195 |
| C1-REDUCED final | 0.021720 | 0.020596 | 0.021574 | 217 |
| source (UOA U-LEAKY-LONG) | 0.015294 | 0.014808 | 0.015488 | 153 |

- **Met by none.** No arm meets it.
- **Progress.** C16's extra 6,000 updates moved the source from 153x to 146x above it. Reduction moved away from
  it.
- **Comparators.** Of the comparators, only INT8-SHARED at 1,024 B is below it (4.7e-6).

## 7. Byte accounting

| Arm | Payload / tile | Receiver one-time state | Sender one-time state | Bridge parameters (reduce / expand) | Amortized B / message at 504 / 3,072 / 64,512 |
|---|---:|---:|---:|---|---|
| C16 | 1,024 B | 154,052 B | 154,152 B | 272 / 272 | 1,330 / 1,074 / 1,026 |
| C4 | 256 B | 153,284 B | 153,336 B | 68 / 80 | 560 / 306 / 258 |
| C1 | 64 B | 153,092 B | 153,132 B | 17 / 32 | 368 / 114 / 66 |

- **Receiver.** Its state is the decoder parameters (140,120 B), the BN buffers (552 B), the int64 position-index
  buffers (12,288 B), the expand layer and the 4-B scale.
- **Comparators.** PCA-16 needs 69,632 B of shared state, SCALAR-4 16 B and INT8-SHARED 4 B.
- **Not modeled.** Framing, a method identifier, refresh and codec training cost.
- **No communication-saving claim** is made. A 64-B payload is a 1/64 bare payload ratio, not a 64-fold end-to-end
  saving.

## 8. Breakdowns (`analysis.json` `breakdowns`)

- **C1 final, held-out.** Per-tile RSE 0.880 to 1.002, and per-snapshot RSE 0.946 to 0.961.
- **C4 final, held-out.** Per-tile RSE 0.72 to 0.955, and per-snapshot 0.853 to 0.886.
- **Structure.** Layer 9 is easiest and layers 17 / 21 hardest in every arm. LoRA blocks are within about 0.03 of each
  other. No snapshot, client, row or block forms a failure pocket.

## 9. Verification

- **Frozen analyzer.** Exit 0, no evidence problem. The evaluation's torch-reference spot check agrees to 1.7e-16.
- **Independent verification** (`independent_verify.py`; all_ok; imports no torch and no package module).
  - Ledger roots are byte-identical.
  - All 57 payload byte checks pass: wire widths and bytes against the wire latents, input-PCA, SCALAR-4 and INT8
    re-encodings, and the source INT8 / FLOAT bytes.
  - Every per-tile sum was recomputed with math.fsum; the maximum relative deviation is 2.0e-16.
  - 21 / 21 agreements with the analysis: UPP outcomes, target distances and element counts.
- **Fresh-process replay** (`replay_final_states.py`, no optimizer): 126 / 126 bit-identical checks. The RNG is
  unchanged, and imports come from this package only.
- **Other bitwise checks.**
  - Each training child's step-0 and step-6,000 TRAIN evaluation, and INIT's step-zero sums, equal EVAL's.
  - The fresh UOA-source reproductions equal the UOA records.
- **Frozen package after the run** (`verify_manifest_after.out.txt`): 61 / 61 files and 12 / 12 external files are
  unchanged. Only the ledger mirror is new, by design.
- **Protected work** (`protected_fingerprint_comparison.out.txt`): identical before and after.
  - It covers the original dirty repository (HEAD `0ad2795`, 348 entries, paused SSBD), the UCX / UHO / UOA and both
    publication worktrees, every file (including ignored ones) under their diagnostic trees, and the UOA, UCX, UHO,
    UCOV, URP, URC and r1 run roots.
  - No closed protocol was relaunched or modified.

## 10. Discrepancies, deviations and process notes

- **Scientific evidence: none found.**
- **Session interruption.** The analyzer, independent verification and replay were each executed once, 11:38 to
  11:43 +08:00, just before the working session was interrupted and resumed. Their outputs were then read and
  checked; nothing was re-executed.
  The report tables, figures, protected-work comparison, manifest re-check and these documents were produced after
  the resumption.
- **Pre-freeze engineering (no effect on the frozen package).**
  - An engineering fixture test of the pilot runner (small model, temporary directory, deleted) preceded the three
    real pilots. It is not a pilot.
  - The config builder `checks/build_ulr_config.py` was reformatted by ruff after it had produced `ulr_config.json`.
    The change is formatting only, and the frozen manifest holds the reformatted bytes.
  - The pilot-measured code (`ulr_model.py`, `preflight/resource_pilot.py`) is unchanged since the pilots (hashes
    re-checked).
- **Analyzer artifact.** The C16 "fraction of damage recovered" is a division by a difference of about 1e-11, so it is
  meaningless. This report writes n/a.
- **No other post-hoc analysis.** `ulr_peek`-style viewing scripts in the session temporary folder only printed
  analysis fields; they computed nothing new and are not evidence.

## 11. What remains unknown (question 7)

- **Optimization.**
  - 6,000 adaptation updates from a PCA-initialized bridge is one budget, one learning rate and one seed.
  - The reduced arms were still improving slowly.
  - Whether a longer budget, training the reduced bottleneck from scratch, or another initialization would cross the
    input-PCA curve is not known.
- **The bottleneck family.** Only a 1x1 channel bottleneck at the encoder-decoder boundary was tested, with spatial
  4 x 4 kept and the cores at 16 channels.
  - The flat channel spectrum of the source latent suggests this boundary is a poor place to cut channels. That is
    a hypothesis.
  - Spatial reduction, other insertion points or an architecture trained for the bottleneck were not tested.
- **Unseen data.** EARLY / LATE were already observed during UCX / UOA. Results are exploratory, from one corpus and
  one seed. Tiles within a snapshot or client are not independent.
- **Complete LoRA updates.** Only 32 x 32 R3 rank-8 representation tiles were reconstructed.
  - No full A / B factors and no effective update DeltaW = s_lora (B_client A_client - B_ref A_ref), with its cross
    terms, were formed.
  - The R3 approximation error is separate.
- **Downstream utility.** Task-response fidelity, FAF benefit, cumulative communication and the cost of the codec
  were not measured. The innovation hypotheses are untested; their prerequisite is a learned codec that beats simple
  codecs at equal elements or bytes, and it was not met here.
- **Novelty.** None is claimed for PCA initialization, 1x1 projections, latent reduction or warm-start fine-tuning.

## 12. Files

- **This directory.**
  - Records and launch: `EXECUTION_AUTHORIZATION.md`, `prelaunch_verify.*`, `launch_scientific.ps1` and `launch/`.
  - Analysis and tables: `ulr_analyze.out.txt` and `analysis/` (`analysis.json`, `learning_curves.csv`,
    `checkpoints.csv`, `per_tile_metrics.csv`), `report_tables.*`.
  - Figures: `make_figures.py`, `element_count_vs_quality.svg` and `learning_curves_ulr.svg`.
  - Verification: `independent_verify.*`, `replay_final_states.*`, `verify_manifest_after.*` and
    `protected_fingerprint_*`.
  - Handoff: `REPORT.md`, `PUBLICATION_HANDOFF.md`, `CONTINUATION_CHECKPOINT.md` and `EVIDENCE_INVENTORY_SHA256.txt`
    (written last).
- **Package.** `diagnostics/ulr_20261015_v1/`: `protocol.md`, `UOA_CLARIFICATION_NOTE.md`, `SOURCE_INVENTORY.md`,
  code, checks, preflight, preparation and vendor.
- **Run root.** `D:/cgfed-runs/ulr_20261015_v1/`: the ledger, INIT (bridge init, step-zero sums), three arms (states
  at 0 / 1,000 / 3,000 / 6,000, curves, checkpoint sums), and EVAL (sums, predictions, wire latents, payload bytes).
