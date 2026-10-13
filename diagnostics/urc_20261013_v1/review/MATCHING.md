# URC-v1 vs U-FIT: shared conditions, permitted differences, initialization

R-FIT and U-FIT are not "identical except for architecture". They share the conditions in section 1 and differ
intentionally in the four ways listed in section 2.

## 1. Shared conditions (a violation is NOT_MATCHED)

| Condition | How it is matched | Where it is checked |
|---|---|---|
| Input bytes | frozen `uid_common.make_input` with the frozen UID `input` block (data seed 20261012); SHA-256 `fc5e1283d1a44cb3bedaf89cff96818b1db0116e036e7cfad75c4a121138d722` | child, before the optimizer exists (refuses on any difference); `checks/check_reference_input.py`; `urc_analyze.py` |
| Normalization | none, as in U-FIT | same input path (no scaling step exists in either) |
| Baselines | frozen `uid_common.baselines`; zero-prediction MSE 0.22200250658157333, scalar-mean MSE 0.22193865837012977 | as above, exact float equality |
| Sampling geometry | six stages, all stride 1; latent [64,32,32]; sampling-layer shapes identical (encoder 24,833 and decoder 24,746 parameters) | `urc_control.resolve` (vs frozen U-FIT model block); `checks/test_urc_model.py`; `checks/check_init_comparison.py` (layouts) |
| Precision | fp32 compute, float64 metric reductions (frozen `uid_common.metrics`) | same code |
| Optimizer | Adam, the complete U-FIT settings: lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 (amsgrad/maximize off by default) | `urc_control.resolve`; child records `optimizer_defaults`; `urc_analyze.py` |
| Update budget | 3,000 updates, batch 1, `mse_loss(decode(encode(x)), x)` in train mode, final state | resolve; the r1 training loop unchanged |
| Measurement | step 0 on isolated copies; curve every 100; final train/eval on two copies from one state and RNG state; eval-mode verdict | the r1 `uid_r1_measure.py`, unchanged; training loop and final procedure unchanged (see `child_diff.patch`) |
| Execution | CPU only, 24/24 threads, deterministic algorithms, torch 2.14.0+cu130, model seed 1 via `fork_rng`; same soft-stop/kill margins, RSS guard and evidence cap | resolve; child CPU record; `urc_analyze.py` compares both invocation plans |

## 2. Permitted differences (never NOT_MATCHED)

1. **Core block architecture**: residual conv blocks instead of BasicUformer layers.
2. **Parameter count**: 493,483 instead of 685,507. The cores are 2 × 221,952 instead of 2 × 317,964.
3. **Core normalization**: BatchNorm2d instead of LayerNorm. R-FIT has 1,941 buffer elements, including the BatchNorm
   buffers of the six residual blocks.
4. **Architecture-dependent initialization**: PyTorch default initialization only, with no trunc-normal step.
   Observed values are below.

## 3. Initialization observed on fresh disposable models

`checks/check_init_comparison.py` builds each model once, as its arm builds it, with seed 1. It runs no forward pass,
creates no optimizer, makes no update and copies no weights. It hashes each corresponding tensor with
`uid_common.tensor_sha256` (layout-only). The fresh U-FIT build reproduces the recorded U-FIT initial state hash
`7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`.

| Stack | Layouts | Parameters equal | Buffers equal |
|---|---|---|---|
| encoder.sampling | all 42 tensors match | 21 / 21 | 21 / 21 |
| decoder.sampling | all 38 tensors match | 12 / 20 | 18 / 18 |

The decoder parameters that differ are the six ConvTranspose weights (`0, 3, 6, 9, 12, 15.weight`) and the head
`19.weight` / `19.bias`. All BatchNorm affine parameters and all buffers are equal; they are initialized
deterministically (1, 0, running statistics), not drawn from the RNG.

Interpretation:
- Both builds draw from the same RNG stream in the same order up to the end of the encoder sampling layers. After
  that, the cores consume different numbers of draws, so every RNG-drawn decoder tensor differs.
- The equal encoder tensors are an observation about these two builds. They are not a general guarantee from the
  seed.
- The full evidence is the check's JSON output (`evidence/<tag>/check_init_comparison.log`).
- No weights are copied to manufacture equality.
