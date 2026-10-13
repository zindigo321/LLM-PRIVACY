# Proposed next experiment: UHO-v1, held-out tile reconstruction by one shared Uformer (draft; not frozen, not run)

Written after UCOV-20261013-v1. This is a proposal only. It needs its own protocol, a frozen package and new
scientific authorization before any start.

## Why this question next

- **Settled for now: input coverage.** Under UPP-20261013-v1, five fixed q/k tiles of one snapshot were each fit
  clearly better than any constant by a model trained on that tile alone: four arms here, and URP-v1 as context.
  - RSE 0.194–0.227, cosine 0.879–0.898, MSE/best 0.19–0.23.
  - The spread was small.
- **Open: a model's behaviour on data it was not trained on.** Every result so far is train-on-x, evaluate-on-x. That
  is the gap between this line of diagnostics and the paper's use, where an encoder sees new LoRA updates.
- **The other open questions are kept separate.**
  - Compression, a strided latent below 1:1, changes the architecture. It is a later, separate study.
  - Seed variance and a longer budget refine the single-tile result. They do not address held-out behaviour.

## Hypothesis

H1: one fresh Uformer trained on a small fixed set of tiles of the R3 snapshot reconstructs other, unseen tiles of the
same snapshot clearly better than any constant. That means the primary policy UPP-20261013-v1 is met on the held-out
tiles.

H0: the held-out reconstruction is no better than about 0.80 × the best constant.

## Controlled change

- **What changes.** A single arm trains one model on a set of training tiles and is evaluated on disjoint held-out
  tiles.
  - The training data is a batch of 8 tiles: the 4 UCOV tiles, the URP centre tile and 3 more.
  - Batch size therefore goes from 1 to 8. This is the one structural change, and it is unavoidable for a multi-tile
    fit.
- **What stays fixed, from UCOV-v1:**
  - the frozen U-FIT Uformer (685,507 parameters, latent [64,32,32]), seed 1 and initial state `7a36411b…f6f7`;
  - CPU fp32, 24/24 threads, deterministic algorithms;
  - Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; 3,000 updates;
  - the frozen R3 scale and fp32 arithmetic, the loss in normalized space, the final-state evaluation in eval mode in
    the original space, the state saves, the curves and the ledgers.

## Input-selection rule (to be registered and hashed before any new value is read)

Source: the same rebuilt Phi (`485cbe5f…2546`) and the same 64 × 48 grid.
- **Training set (8 tiles).** The four UCOV quarter tiles, the URP centre (32, 24), and three positional tiles on the
  same grid: (16, 24), (48, 24) and (32, 12).
  - All have 64-column-aligned q/k block composition. The rule fixes them by position only.
- **Held-out set (4 tiles), never used for training.** The eighth points of the grid that are not in the training
  set, at tile rows 8 and 56, tile columns 6 and 42: (8, 6), (8, 42), (56, 6) and (56, 42).
  - These are layers 3 and 21, also q/k-aligned.
- **Fixed in advance.** The degeneracy treatment is the UCOV rule. No tile is replaced or reordered after values are
  seen.

## Measurements and decision criteria

- **Primary.** UPP-20261013-v1 on each held-out tile, using the final state in eval mode:
  - RSE ≤ 0.50, cosine ≥ 0.65 and MSE ≤ 0.80 × that tile's best constant MSE;
  - aggregate over the 4 held-out tiles with the same P/F/U rule: P ≥ 3 met; F ≥ 2 not met; otherwise inconclusive.
- **Reported only.**
  - The same metrics on each training tile: the in-sample fit, for comparison with UCOV's single-tile fits.
  - The gap between training and held-out RSE.
  - Curves, both secondary references, train-mode values and resources.
- **Interpretation.**
  - "Met" would show held-out reconstruction within one snapshot's q/k blocks with an expanded latent. It would show
    nothing about compression or other snapshots.
  - "Not met" would not show that the Uformer cannot generalize under other budgets or data amounts.

## Resources (estimated before proposing the larger batch)

- **Memory.** UCOV measured a peak working set of about 830 MB per batch-1 arm. The U-CHECK process with the same
  setup peaked at about 639 MB, which bounds the fixed overhead from above.
  - Scaling only the remaining approximately 191 MB of activations by the batch size gives at most about 2.2 GB at
    batch 8. That is under the min(4 GiB, 50% of available RAM) limit.
  - The latent of one batch is 8 × 64 × 32 × 32 fp32 = 2 MiB.
  - This is an estimate, not a measurement. The package would include a fixture-only check of the scaling.
- **Time.** About 8× the per-update compute of UCOV's 0.164 s, so about 1.3 s per update: roughly 66 min for 3,000
  updates, plus evaluation.
  - Proposed limits: 5,400 s for the arm and 7,200 s in aggregate. That is one start.
  - If the fixture check projects more than 5,400 s, the proposal must be revised rather than run with a cut budget.
- **Evidence.** Two 3.2 MB states, the curve and records: under 10 MB.

## Required authorization

- **New protocol.** For example UHO-YYYYMMDD-v1, with a hash-registered tile rule, an input build, a fixture
  memory-scaling check and a frozen package with its manifest.
- **Approval.** A written scientific authorization of exactly one invocation and one start, before or after the
  manifest exists, as the user prefers, with the digest recorded at freeze.
- **Not covered:** GPU, SSBD, FAF, other seeds, compression arms, other snapshots and v/o blocks.
