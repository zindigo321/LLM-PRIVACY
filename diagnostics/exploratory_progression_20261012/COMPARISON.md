# Uformer U-FIT vs historical ResNet-3 R3: derived comparison

Label: DERIVED. Every number below comes from recorded sums, read only after the source files' SHA-256 were checked
(`compare_records.py` → `comparison.json`). No model was rebuilt and no reconstruction was rerun. This is a check of
recorded arithmetic, not an independent numerical replay. The input tensors were not regenerated in this task.

Sources:
- Uformer: UID-20261012-v2 r1, arm U-FIT; `manifest_r1.json` SHA-256
  `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`; run root `D:/cgfed-runs/uid_20261012_v2_r1/`.
  All 42 entries of `FINAL_ARTIFACTS_SHA256.txt` match.
- ResNet-3: P5-A v2 + v2.1, invocation 1, control R3 `single_snapshot`; `results/phase5/p5a_v2/` (committed at
  `fcb6a60`, clean against HEAD).
  - `compare_records.py` checks the three R3 files it reads, and the R3 checkpoint whose header it reads, against
    `artifact_manifest.json`. The checkpoint is 1,996,508 B, SHA-256
    `dd46dbccba65c53b3c709ea46cedaaea9a470177d2d72dc2cfe1a19cb2dbbbc0`.
  - A separate check in this task found the live run root `D:/cgfed-runs/phase5_p5a_v2_seed1/` matching all 45
    manifest entries.

## Conditions

| | Uformer U-FIT | ResNet-3 R3 single snapshot |
|---|---|---|
| Input source | synthetic: `1.6·rand([1,1,32,32]) − 0.8`, data seed 20261012 | real LoRA representation `balanced_effective_delta_r8` (Qwen1.5-1.8B, r 8, q/k/v/o), position 39 (t 7, client 91) |
| Shape, elements | [1,1,32,32] fp32, 1,024 | [1,2048,1536] fp32, 3,145,728 |
| Hash | tensor SHA-256 `fc5e1283d1a44cb3bedaf89cff96818b1db0116e036e7cfad75c4a121138d722` | Phi SHA-256 `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`; snapshot file `09c95184a657fa778aef48e18196f95c568854476b75f2812d68dfd58126dd45` |
| Normalization | none (input already in [−0.8, 0.8]) | x / 0.015751632224572334 (training-population max-abs to 0.95); metrics in original space |
| Sampling strides | 1 at all six stages | 2 at all six stages |
| Latent / input elements | [64,32,32]: 65,536 / 1,024 = **64** (expansion) | [64,32,24]: 49,152 / 3,145,728 = **1/64** |
| Core blocks | 3 + 3 BasicUformer (2 LeWin blocks each) | 3 + 3 residual conv blocks |
| Parameters | 685,507 (sampling 24,833 + 24,746; cores 2 × 317,964) | 493,483 (sampling 24,833 + 24,746; cores 2 × 221,952) |
| Parameters per input element | 669.4 | 0.157 |
| Initialization | `manual_seed(1)` in `fork_rng`; PyTorch defaults, then upstream Uformer init for Linear/LayerNorm and position bias (trunc-normal std 0.02) | `manual_seed(derive_seed(1, "ae_init"))` = 453875208010999487 in `fork_rng`; PyTorch defaults |
| Optimizer | Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, wd 0 | identical |
| Training | 3,000 updates, batch 1, MSE in train mode; final state, eval mode | identical, but the loss is in normalized space |
| Device | CPU fp32, 24/24 threads, deterministic algorithms | RTX 4060 Laptop GPU fp32, deterministic, TF32 off |
| Wall time | 506.3 s (arm) | 316.0 s (charged GPU time) |

Shared by the two runs:
- the optimizer;
- the update budget;
- batch size 1;
- fp32 compute;
- the final-state eval-mode rule;
- the sampling-layer parameter shapes.

Different: input, scale, strides, bottleneck, core blocks, initialization, seed procedure and device.

## Final metrics against each run's own baselines

All values are final state, eval mode, in each run's original space. They come from recorded float64 sums; the full
values are in `comparison.json`.

| | Uformer U-FIT | ResNet-3 R3 |
|---|---|---|
| Recorded status (strict: RSE ≤ 0.01, cosine ≥ 0.99) | COMPLETED_FIT_FAIL | FIT FAIL |
| RSE = SSE / SSE(zero prediction) | 0.2727411031905583 | 0.9938081619334862 |
| Cosine | 0.8540472264633736 | 0.08718036691133635 |
| Norm ratio | 0.9002832321336198 | 0.12471139818255722 |
| SSE / SSE(scalar mean of the input) | 0.2728195664549294 | not recorded; ≥ 0.9938081619334862 |
| Relative SSE reduction vs zero prediction | 0.7272588968094418 | 0.0061918380665137995 |
| Relative SSE reduction vs scalar mean | 0.7271804335450707 | not recorded; ≤ 0.0061918380665137995 |
| Step 0 (post-init) eval RSE | 1.0059377103460292 | not recorded |

- The R3 record holds the AE's and zero predictor's SSE, SIG, HAT and DOT, but no element sum. The scalar-mean SSE is
  SIG − n·mean², which is at most SIG. Its value is therefore unknown, and only the bound shown can be stated. It is not
  computed here because that would need the payload, which this task does not reopen.
- Training curves (reported only; not gated):
  - Uformer pre-update train RSE: 1.878 at update 1, 0.351 at 1,000, 0.319 at 2,000, 0.272 at 3,000.
  - R3 train MSE over the zero-prediction MSE in normalized space (derived: recorded zero MSE / scale²; ignores
    float32 rounding of the scaled input): 12.29 at iteration 1, 1.013 at 1,000, 0.9998 at 2,000, 0.9937 at 3,000. So
    R3 also stayed at the trivial level in training, not only in eval.

## What the numbers support

- Under each run's own conditions, the Uformer U-FIT removed about 73% of the squared error of either constant
  baseline of its input. R3 removed about 0.6% relative to zero, and at most that relative to its scalar mean.
- U-FIT meets the post-hoc exploratory policy EP-20261012-v1 (`assessment_uid_r1_ufit.json`). Its recorded status
  stays COMPLETED_FIT_FAIL.
- The policy was not applied to R3, because its scope is the U-FIT record layout. R3's recorded RSE and cosine lie
  outside the policy's M1 and M2 thresholds.
- A relative reduction in squared error is not the fraction of information preserved. It says nothing about which
  structure of the input was kept, and it is not a compression, rate or distortion claim.
- Raw MSE is not compared across the two runs. The targets differ in scale by a factor of about 5×10⁴ in zero-baseline
  MSE (0.222 vs 4.28e-6), and the two losses are computed in different spaces.

## Why this is not a matched architecture comparison

The difference in outcome cannot be attributed to Uformer blocks versus residual blocks. At least these factors
differ, and any of them could explain it alone:

1. **Bottleneck.** U-FIT had 64 latent elements per input element, so an identity-like path exists in principle. R3 had
   1/64, so exact reconstruction of a generic input is impossible at that size. The two ratios differ by a factor of
   4,096.
2. **Geometry.** All strides are 1 in U-FIT and 2 in R3. U-FIT keeps the full 32×32 grid, while R3 reduces 2048×1536 to
   32×24.
3. **Input.** U-FIT used a 1,024-element i.i.d. uniform tensor. R3 used a 3.1-million-element real LoRA delta, with
   heavy-tailed normalized values (median |x| 0.078, max 0.889).
4. **Capacity per element.** About 669 parameters per input element in U-FIT against 0.157 in R3.
5. **Scale and normalization.** U-FIT used none. R3 used a global max-abs scale, and its loss is in normalized space.
6. **Initialization, seed procedure and device.** These are listed above; the R3 checkpoint exists, but no Uformer
   final weights were saved.

A matched comparison must hold the input, normalization, geometry, precision, optimizer, budget and device fixed, and
change only the core blocks. `NEXT_EXPERIMENT_SPEC.md` specifies that comparison against the existing Uformer reference.
