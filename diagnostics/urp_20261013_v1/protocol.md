# URP-20261013-v1: Uformer reconstruction of one real LoRA-derived patch (protocol)

Status: frozen together with `manifest_urp.json`. Scientific execution needs a separate written approval of that
manifest's SHA-256.

## Question

Under the existing exploratory policy, can the tested Uformer variant learn to reconstruct one small real input
derived from LoRA weights?

The test is one fixed 32 × 32 patch of an archived R3 training snapshot, with the U-FIT model, seed, optimizer and
budget.

- This is a patch-level reconstruction diagnostic with the existing expanded latent: [64,32,32], 64 times as many
  elements as the patch.
- It is not a compression experiment.
- It tests neither generalization to unseen snapshots or patches nor whole-snapshot reconstruction.
- The Uformer is the primary target, as the supervisor requested. The better synthetic fit of R-FIT in URC-20261013-v1
  does not change that choice, and no ResNet arm is run.

## Background (closed, unchanged)

The synthetic comparison is settled (`diagnostics/synthetic_comparison_summary_20261013/STAGE_SUMMARY.md`):
- U-FIT and R-FIT both ended COMPLETED_FIT_FAIL under the strict near-identity criterion.
- Both passed EP-20261012-v1: post hoc for U-FIT, pre-specified for R-FIT.
- Under the matched workload, R-FIT had the lower RSE and the higher cosine.
- None of this establishes significance, general superiority, compression or downstream utility.

This protocol changes none of those records. It does not change EP-20261012-v1 (`POLICY.md` and `ep_classify.py` are
hash-pinned). It does not make the strict criterion a prerequisite.

## Input (`input/`; prepared and frozen before any model evaluation)

- **Source.** The R3 training snapshot of the historical P5-A single-snapshot control: representation
  `balanced_effective_delta_r8`, time index 7, client 91, P5-A position 39.
  - Source record: `results/phase5/p5a_v2/R3_balanced_effective_delta_r8_single_snapshot/p5a_record.json`, SHA-256
    `f60d3d15788b1945953d16394c39762b3a7a304e440d5bd63b96d87a23e97b2d`.
  - It is cross-checked against `results/phase5/p5a_v2/preflight/preflight_1.json`, SHA-256
    `b058eb1bf01d101e1800bdf05d7701b5ac3de515ea71cd542f70403ee5d48cee`.
- **Rebuilt by `input/prepare_input.py`, through the documented repository procedure.**
  - `index.jsonl`: SHA-256 `b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240`.
  - `load_states(verify=True)`:
    - end file `09c95184a657fa778aef48e18196f95c568854476b75f2812d68dfd58126dd45`;
    - start file `31b60754e8b55992c908e416f2652c8f2dbbbdee0030440fa2163a5528cd5a7a`;
    - both adapter content hashes.
  - Construction: `balanced_effective_delta(s=2, rank=8)`, then layout `layer_major_qkvo_AtB`.
  - The Phi tensor [1,2048,1536] fp32 has `tensor_sha256` `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`.
    This equals the archived `phi_sha256`. Its float64 Σx² equals the archived SIG exactly.
  - Every imported `cg_fedllm` module was unmodified against HEAD and the P5-A execution commit (`input_record.json`).
- **Positional rule.** `input/SELECTION_RULE.md`, SHA-256
  `e0b1df83a9ca79ac8e4682d25b6b56d1a8c76b17bd7d373684f48f2a31d7362d`. It was written at 11:43:51 +08:00, before the
  payload was opened (11:54:52), and is pinned by the builder.
  - The rule is the floor-centre tile of the aligned 32 × 32 grid: `Phi[0, 1024:1056, 768:800]`.
  - This is layer 12, in the q_proj A^T, q_proj B, k_proj A^T and k_proj B blocks, at hidden indices 1024–1055.
  - No other tile was extracted or evaluated.
- **Patch.** `input/patch_x.safetensors`, 4,552 bytes, file SHA-256
  `957704bf86fcce6d0b9c90b715e1a62435bbb96df65864d961cf23a3c8ae75fd`. It holds one tensor `x` of shape [1,1,32,32],
  fp32, in the original (Phi) space.
  - Tensor hash (`uid_common.tensor_sha256`): `92707ff85a521c92164bf90d2bb927d81eabbb9b3b2f107271ca2fafd75db98a`.
  - It has no exactly-zero element.
  - Input record: `input/input_record.json`, SHA-256 `bec71fdac597a6b340457e5330349da9f47f2d4d777757fbfe948126ba67b6c1`.
- **Normalization (different from the synthetic experiment).**
  - The synthetic input was used as is, with no normalizer.
  - Here the existing R3 definition is applied: `global_exact_maxabs_train`, with the recorded frozen scale
    `s_r = 0.015751632224572334`. That is `m_r / 0.95`, with `m_r = 0.014964050613343716` the exact training-population
    maximum (bitwise-checked in P5-A).
  - The arithmetic is the production `Normalizer`: `x_n = x / float32(s_r)` and `x̂ = y_n · float32(s_r)`, in fp32.
    It is re-declared in `urp_control.py` and checked bit for bit against the production class. The scale is not
    refitted on the patch.
  - Normalized patch tensor hash: `1d2f65aad35b9eb25e445d4b864ae1b1a894e4107ee358e61c602ffc898812bb`.
- **Range feasibility.**
  - max |x_n| = 0.8495237827301025, below the Tanh bound 1 with a margin of 0.150.
  - The Tanh-range ceiling reproduces the patch with RSE 1.4e-16.
  - The patch is not degenerate: it is finite, T > 0, its scalar-mean MSE is > 0, and max |x_n| < 1.

**Baselines (computed before any model, `uid_common.baselines`):**

| | original space | normalized space |
|---|---|---|
| T = Σx² | 0.004231302211738306 | 17.053869437280618 |
| zero-prediction MSE | 4.132131066150689e-06 | 0.016654169372344353 |
| scalar-mean MSE (one scalar, the patch mean) | 4.131472976536517e-06 | 0.016651517003540468 |
| scalar-mean RSE | 0.9998407384461826 | 0.99984073845146 |

## Plan

- **Arm.** Exactly one, U-PATCH: one invocation and one scientific start, with no automatic retry, no resume, no
  sweep and no control arm.
  - The plan comes only from `urp_control.registered_plan`.
  - U-CHECK, U-FIT and R-FIT are refused in any mode.
  - A start without an end record blocks every later invocation.
- **Model.**
  - It is built only by the frozen `uid_common.build_model(cfg, "U-FIT")` from the frozen U-FIT model configuration:
    the UID-20261012-v2 Uformer AE.
    - All six strides are 1, so the latent is [64,32,32].
    - 3 + 3 BasicUformer layers of depth 2, dim 64, 2 heads, window 8, LeFF mlp ratio 4.
    - ResNet sampling layers with BatchNorm and ReLU, and a Tanh head.
    - No encoder-decoder skip and no global shortcut.
    - 685,507 parameters.
  - The reconstruction path is unchanged: `x̂_n = decode(encode(x_n))`.
  - The model seed is 1. The child refuses unless the initial state hash equals the U-FIT record's,
    `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`.
- **Settings**, read from the frozen UID config (raw SHA-256
  `d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd`) and checked against `urp_config.json`:
  - CPU fp32, batch 1, 24/24 threads, deterministic algorithms, torch 2.14.0+cu130;
  - Adam: lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; amsgrad and maximize off by default;
  - exactly 3,000 planned updates of `mse_loss(decode(encode(x_n)), x_n)` in train mode, in normalized space, as in
    the production AE training loop.
- **Limits:**
  - arm: 1,800 s;
  - aggregate: 7,200 s;
  - RSS: min(4 GiB, 50% of available RAM at launch), sampled every 0.25 s; the child process tree is killed above it;
  - retained evidence: 512 MiB;
  - soft stop, kill and end margins as in r1.
- **Run root.** `D:/cgfed-runs/urp_20261013_v1/`. Ledger mirror: `scientific_ledger_mirror/` in this directory.

## Measurement (URC code, changed only in the data path)

- **Step 0.** On isolated copies, in both modes, with sums in both spaces.
- **Curve.** A point at update 1, every 100 updates and the final update. Each point has the pre-update train loss
  and normalized-space raw sums, gradient and parameter-change groups, and RSS.
- **Final evaluation.** Two independent copies of the final state, loaded with the same RNG state, in eval and train
  mode.
  - Original-space sums (SSE, T, P, D, N) of `x̂ = y_n · float32(s_r)` against `x`: the verdict basis.
  - Normalized-space sums of `y_n` against `x_n`: reported only.
- **States.** `init_state.safetensors` and `final_state.safetensors` hold every parameter and buffer: BatchNorm
  running statistics and counters, and the attention `relative_position_index`. Each is written exclusively, read
  back bit for bit, and its hash checked.
- **Checked before the optimizer exists:** the input file hash, both tensor hashes, the energies, the baselines in
  both spaces and max |x_n|.

## Outcomes and analysis (fixed now; reported separately)

- **Strict.** The r1 rules give COMPLETED_PASS or COMPLETED_FIT_FAIL under the strict near-identity criterion: eval
  mode, finite, RSE ≤ 0.01 and cosine ≥ 0.99, in the original space. A run can also end RESOURCE_STOP,
  EXECUTION_ERROR or INCOMPLETE. The strict criterion is reported, not a prerequisite.
- **Exploratory.** EP-20261012-v1 through the frozen `ep_classify.classify`, pre-specified here. It uses the
  original-space final eval sums and the patch's own original-space scalar-mean baseline.
  - RSE and cosine are invariant to the one global scale up to fp32 rounding.
  - The normalized-space values are reported alongside.
- **Context only.** The U-FIT synthetic numbers are listed for orientation. This is not a matched comparison: the
  input and the scale differ.
- **Not permitted:** choosing a best point, recalibrating BatchNorm, extending the budget, re-seeding, another patch,
  or rerunning after the result is seen.

## Interpretation limits

- A PASS under EP shows only that the final reconstruction of this one patch was clearly better than constant
  predictions.
- A FAIL does not show that the Uformer cannot fit real LoRA-derived data.
- One seed and one start give no variance estimate.
- The patch is 1,024 of the 3,145,728 Phi elements. Nothing here bears on the rest of the snapshot, other snapshots,
  compression, the paper's results or FAF utility.
- SSBD remains paused and is not affected.
