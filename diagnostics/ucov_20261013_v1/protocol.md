# UCOV-20261013-v1: Uformer input-coverage fitting on four more tiles of the same real snapshot (protocol)

Status: frozen together with `manifest_ucov.json`.

Authorization: the user approved this design on 2026-10-13, before the manifest existed, and asked for it to be run
once its verified frozen package matches the scope. The manifest digest is recorded when frozen. No earlier human
approval of that digest exists, and none is claimed.

This protocol supersedes, for this study, the draft `diagnostics/urp_20261013_v1_execution/NEXT_EXPERIMENT_PROPOSAL.md`.
That copy is sealed and is not rewritten.

## Question

Does the single-tile fitting capability observed in URP-20261013-v1 extend to four more fixed tiles of the same real
snapshot?

- Each arm trains its own fresh model on its own tile and is evaluated on that same tile. This is input-coverage
  fitting.
- It is not held-out evaluation of one trained model, and it is not compression: the latent [64,32,32] is a 64:1
  element expansion of each tile.

## Primary policy: UPP-20261013-v1 (`diagnostics/upp_20261013_v1/`, hash-pinned)

- **Pass.** A valid completed arm is a primary PASS when, from its final eval-mode reconstruction in the original
  space:
  - RSE ≤ 0.50;
  - cosine ≥ 0.65;
  - MSE ≤ 0.80 × min(zero-prediction MSE, scalar-mean MSE).
  The comparisons are inclusive, in float64, with no epsilon.
- **Non-passes.**
  - An undefined or non-finite required quantity fails.
  - Invalid, incomplete, interrupted, unverified or not-run execution is class U, an unknown outcome.
  - A registered degenerate tile is class F, an explicit non-pass.
- **Implied cosine bound.** With these definitions, RSE ≤ 0.50 implies cosine ≥ sqrt(0.5) ≈ 0.7071, apart from
  rounding. The requested 0.65 bound is kept exactly.
- **Disclosure.** The policy was chosen with knowledge of the URP-v1 result. It is prospective for this study only.
- **Single source.** The training child sets each completed arm's terminal status, COMPLETED_PASS or
  COMPLETED_FIT_FAIL, from `upp_classify.primary_metrics`. The analyzer recomputes the full classification with
  `upp_classify.classify`. The UID `criteria` are never read for the primary verdict.
- **Secondary references, reported only; neither can veto a primary PASS:**
  - historical strict near-identity: RSE ≤ 0.01 and cosine ≥ 0.99;
  - EP-20261012-v1: frozen `ep_classify.classify`.

**Aggregate.** Count over the four registered arms: P = PASS; F = FIT_FAIL, NONINFORMATIVE or DEGENERATE_INPUT;
U = UNVERIFIED, NOT_ELIGIBLE or NOT_RUN.

| Counts | Decision |
|---|---|
| P ≥ 3 | coverage criterion met |
| F ≥ 2 | coverage criterion not met |
| otherwise | inconclusive |

## Inputs (`input/`; registered and built before any model evaluation)

- **Source.** The same as URP-v1: R3 `balanced_effective_delta_r8`, t 7, client 91, P5-A position 39.
  - Phi is [1,2048,1536] fp32, rebuilt by the URP-v1 builder's provenance functions (imported unchanged).
  - `tensor_sha256` `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`, equal to the archived value;
    its SIG is equal too.
- **Rule.** `input/SELECTION_RULE.md`, SHA-256 `ddd58be9ec8ecae473a2ed48be749e8826f685470abf5c78d15d64f58bfe36f2`.
  - It was registered at 2026-10-13T20:23:39.610+08:00 (`SELECTION_RULE.registered.txt`), before the builder existed
    and before the payload was opened at 20:37:23.
  - It takes the quarter points of the aligned 64 × 48 grid of 32 × 32 tiles, in this fixed order:

| Arm | Slice (half-open) | Layer | Normalized max \|x_n\| | Tile file SHA-256 |
|---|---|---|---|---|
| U-TILE-1 | `Phi[0, 512:544, 384:416]` | 6 | 0.42174121737480164 | `b7d39da738313e83d6c36f5f1c73683b464257d27375017a6731b6b7be345a67` |
| U-TILE-2 | `Phi[0, 512:544, 1152:1184]` | 18 | 0.5362738966941833 | `db9ab2901b1a0e9f440ade425883f836b8d56f25a6cd4af16e7aaf147628ffb3` |
| U-TILE-3 | `Phi[0, 1536:1568, 384:416]` | 6 | 0.466398149728775 | `858c5e4499c76ddf878ff22e10e0cf27cb69788663714181efc490946ee70552` |
| U-TILE-4 | `Phi[0, 1536:1568, 1152:1184]` | 18 | 0.7001963257789612 | `95a51e95b19ef89a7b42c41ba86374f58bdbb4f1864925ca4434a276be2afe9e` |

- **Layout.** Each tile covers the q_proj A^T, q_proj B, k_proj A^T and k_proj B blocks of one layer. This was
  verified against the committed `layer_major_qkvo_AtB` block order.
- **Usability.** All four tiles are usable: finite, T > 0, non-constant, max |x_n| < 1, and no exactly-zero element.
  `input/inputs_record.json` (`b4ba2f96dff823c11524dbe448c22e2bdfbf161f6a974c19f003e1e52df5f57f`) holds the baselines
  in both spaces, the block energies and the range feasibility.
- **Not rerun.** The URP-v1 centre tile (32, 24) is not rerun. It was selected by a positional rule registered before
  its values were read, not for performance.
- **Normalization.** The frozen R3 scale 0.015751632224572334, with the production fp32 arithmetic and no refit.

## Plan

- **Invocation.** One invocation, with at most four scientific starts: U-TILE-1, U-TILE-2, U-TILE-3 and U-TILE-4, in
  that order, run one after another.
  - There is no retry, resume, state transfer, extra seed, replacement tile, extra arm or budget extension.
  - The plan comes only from `ucov_control.registered_plan`. The supervisor has no option to select, reorder, skip or
    repeat an arm.
- **Model.** Each arm builds a fresh frozen U-FIT Uformer with 685,507 parameters and a fresh Adam optimizer.
  - The frozen `uid_common.build_model(cfg, "U-FIT")` is used, through the frozen URP binding.
  - Seed 1. The child refuses unless the initial state equals
    `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`.
- **Settings.**
  - CPU fp32, batch 1, 24/24 threads, deterministic algorithms, torch 2.14.0+cu130.
  - Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0.
  - Exactly 3,000 planned updates of normalized-space MSE in train mode.
- **Limits.**
  - 1,800 s per arm and 9,000 s in aggregate, shared.
  - RSS min(4 GiB, 50% of available RAM at launch).
  - 512 MiB of retained evidence.
- **Continuation.**
  - The next registered arm runs after an ordinary COMPLETED_PASS or COMPLETED_FIT_FAIL, and after an arm's
    EXECUTION_ERROR. The invocation does not stop early because enough passes or failures have accumulated.
  - A guard kill (RSS or hard deadline), an INCOMPLETE arm, an interrupt, an exhausted aggregate budget or an
    exceeded evidence cap stops the invocation safely. The remaining arms are recorded NOT_RUN, with the reason.
  - A start without an end record blocks every later invocation.
- **Degenerate tiles.** A degenerate registered tile gets no start and is recorded NOT_RUN with that reason. None of
  the four tiles is degenerate.
- **Run root and mirror.** Run root `D:/cgfed-runs/ucov_20261013_v1/`; ledger mirror `scientific_ledger_mirror/` in
  this directory.

## Measurement (the URP-v1 code, per arm)

- **Unchanged from URP-v1:**
  - step 0 on isolated copies;
  - curve points at update 1, every 100 updates and the final update: pre-update train loss and sums, gradient
    groups, parameter change and RSS;
  - the final evaluation on two independent copies of the final state, in eval and train mode, with sums in both
    spaces;
  - the initial and final states with all parameters and buffers, each written exclusively and read back bit for bit.
- **Checked before the optimizer exists:** the input, both tensor hashes, the energies, the baselines in both spaces
  and max |x_n|.

## Analysis (`ucov_analyze.py`, fixed now)

1. Seal the run root and the mirror.
2. Check, per arm: the evidence, policy identity and hashes, status agreement, tile, initial state, modules and
   states.
3. Classify each arm under the primary policy, then aggregate.
4. Report the secondary references.
5. Report, for information only: normalized-space and train-mode quantities, curves and resources, with the URP-v1
   centre tile as context.

Not permitted after any result is seen:
- choosing a best point;
- recalibrating BatchNorm;
- changing a threshold;
- re-seeding;
- replacing or reordering a tile;
- rerunning.

## Interpretation limits

- **PASS.** A primary PASS shows only that a fresh model fit its own tile clearly better than any constant.
- **One seed per arm.** One seed and one start per arm give no variance estimate and no significance.
- **Failures.** Two failures would not show that the centre tile was uniquely favourable.
- **Not tested:** compression, held-out generalization, whole-snapshot reconstruction, the v/o blocks, other
  snapshots, downstream or FAF utility.
- **SSBD** remains paused and is not affected.
