# URC-20261013-v1: stride-1 ResNet-3 comparator at the U-FIT conditions (protocol)

Status: frozen together with `manifest_urc.json`. Scientific execution needs a separate written approval of that
manifest's SHA-256.

## Question

Under the conditions of the UID-20261012-v2 r1 U-FIT arm, replace the BasicUformer cores with ResNet residual
blocks. How well does the model then fit the one fixed input, with one seed?

The Uformer remains the primary research target, and this arm is a diagnostic comparator. Its result does not gate
other exploratory work. The reference is the existing U-FIT record (`manifest_r1.json` SHA-256
`9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`). No Uformer arm is run.

## Plan

- **Arm**: exactly one, R-FIT. One invocation, one scientific start, no automatic retry, no resume.
  - The plan comes only from `urc_control.registered_plan`. Arm names U-CHECK and U-FIT are refused in any mode.
  - A start without an end record blocks every later invocation.
- **Model**: `ResNet3S1AutoEncoder` (`resnet3_s1.py`), variant `resnet3_stride1_diagnostic`.
  - It re-declares the production ResNet-3 (HEAD blob `cf38fd3f7749a2cdd45864e41f22d3ab0dc9e3db`) with all six
    strides set to 1 and `output_padding` 0. Latent [64,32,32], a 64:1 expansion.
  - 493,483 parameters and 1,941 buffer elements.
  - The production model is not used or changed. It refuses 32×32 inputs, because sides must be divisible by 64.
  - The geometry is explicit: the input must be exactly [B,1,32,32], and reflection padding needs every side > 3.
- **Shared settings**, read from the frozen UID config (raw SHA-256
  `d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd`) and checked against `urc_config.json`:
  - input: `uid_common.make_input`, data seed 20261012;
  - optimizer: Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 (amsgrad and maximize off by default);
  - 3,000 updates, batch 1, `mse_loss(decode(encode(x)), x)` in train mode;
  - model seed 1 via `fork_rng` + `manual_seed`;
  - CPU fp32, 24/24 threads, deterministic algorithms, torch 2.14.0+cu130.
- **Limits**:
  - arm: 1,800 s;
  - aggregate: 7,200 s;
  - RSS: min(4 GiB, 50% of available RAM at launch), sampled every 0.25 s; the child process tree is killed above it;
  - retained evidence: 512 MiB;
  - soft stop, kill and end margins as in r1.
- **Run root**: `D:/cgfed-runs/urc_20261013_v1/`. Ledger mirror: `scientific_ledger_mirror/` in this directory.

## Measurement

Identical to the U-FIT reference, from the same r1 code (`uid_r1_measure.py`, unchanged):
- step 0 on isolated copies, in both modes;
- a curve point at update 1, every 100 updates and the final update, with pre-update train loss and raw sums;
- gradient and parameter-change groups;
- final evaluation on two independent copies of the final state, loaded with the same RNG state, in eval and train
  mode;
- metrics as float64 sums (SSE, T, P, D, N) via the frozen `uid_common.metrics`.

Added:
- input and baseline equality with U-FIT is asserted before the optimizer exists;
- the sampling layers' initial tensors are hashed;
- `init_state.safetensors` and `final_state.safetensors` (all parameters and BatchNorm buffers) are saved, read back
  bit for bit, and their hashes checked.

## Outcomes and analysis (fixed now)

- **Status**: from the r1 rules, COMPLETED_PASS or COMPLETED_FIT_FAIL under the strict near-identity criteria
  (eval mode; RSE ≤ 0.01, cosine ≥ 0.99), or RESOURCE_STOP, EXECUTION_ERROR or INCOMPLETE.
- **Exploratory**: EP-20261012-v1, through the frozen `ep_classify.classify` (hash-pinned). This is its first
  pre-specified use, so for R-FIT it is not post hoc.
- **Matching**: `urc_analyze.py` compares the shared conditions exactly with the U-FIT record.
  - Any violation gives NOT_MATCHED, with the violated condition named.
  - The permitted differences below never cause NOT_MATCHED.
- **Comparison**: the signed final eval differences R-FIT − U-FIT in RSE and cosine, train mode alongside, and the
  resources of both runs.
  - One seed per architecture gives no variance estimate, so there is no significance test and no claim that one
    architecture is better.
- **Not permitted**: choosing a best point, recalibrating BatchNorm, extending the budget, re-seeding, or rerunning
  after the result is seen.
- Neither classification establishes generalization or operational compression.

## Permitted (intentional) differences from U-FIT

1. Core block architecture: 3 + 3 residual blocks, each Conv3×3–BN–ReLU–Conv3×3–BN plus identity, instead of
   3 + 3 BasicUformer layers with 2 LeWin blocks each.
2. Parameter count: 493,483 instead of 685,507. The sampling layers have identical shapes (24,833 + 24,746).
3. Core normalization: BatchNorm2d instead of LayerNorm. With batch 1, train-mode and eval-mode statistics differ, and
   both modes are reported.
4. Architecture-dependent initialization: PyTorch defaults only. The Uformer adds trunc-normal(0.02) init to its
   cores.
   - With fresh models and no update, the encoder sampling layers' initial tensors are equal: 21/21 parameters and
     21/21 buffers.
   - In the decoder sampling layers, all 18 buffers and the 12 BatchNorm parameters are equal. The 7 convolution
     weights and the head bias differ, because the cores consume different numbers of RNG draws.
   - No weights are copied between architectures.

## Interpretation limits

R-FIT can show whether, at these exact conditions, residual conv cores fit this one input to a materially different
degree than the BasicUformer cores did. It cannot show general architectural superiority. It shows no compression,
since the latent is a 64:1 expansion. It says nothing about generalization, real LoRA inputs, the paper's results or
downstream use. The permitted differences above remain confounds.
