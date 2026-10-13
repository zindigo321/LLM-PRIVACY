# Proposed next experiment: URP-INPUT-COVERAGE (draft; not frozen, not approved, not launched)

Written after URP-20261013-v1. This is a proposal only. It needs its own protocol, a frozen package with a manifest,
and a separate written scientific approval of that manifest's SHA-256 before any start.

## Why this question next

What URP-v1 showed:
- One real patch was fit clearly better than any constant: EP PASS, with eval RSE 0.1935 and cosine 0.8981.
- It did not reach near-identity: strict FIT_FAIL.
- The loss was still falling at update 3,000.

Three questions remain open: input coverage, generalization and compression. They need different experiments and
are kept apart here.
- **Generalization** (held-out patches or snapshots) and **compression** (a bottleneck latent) are more informative
  only once fitting is known to hold beyond one hand-picked tile. One registered tile could be atypical, in either
  direction.
- **More updates or other optimizer settings on the same tile** would refine the near-identity question for that one
  tile. They would leave the coverage question open.

So the next question is whether the URP-v1 fit level holds for other real tiles. This experiment changes only which
tiles are fit; compression and generalization are not tested.

## Hypothesis

H1: under the URP-v1 conditions, the Uformer's single-tile fit carries over to other real R3 tiles. At least three of
the four new tiles reach EP PASS.

H0: the URP-v1 PASS is specific to the registered tile; two or more of the four new tiles fail EP.

This holds for tiles of the same snapshot only.

## Controlled change

- **Changes:** only the input tile.
- **Arms:** four arms, U-TILE-1 to U-TILE-4. Each trains a fresh model on one tile. There is no shared model,
  pooling or batch.
- **Fixed, exactly as URP-v1:**
  - the frozen `uid_common.build_model(cfg, "U-FIT")`, 685,507 parameters, latent [64,32,32] (64:1 expansion);
  - seed 1 and initial state `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7` in every arm;
  - CPU fp32, batch 1, 24/24 threads, deterministic algorithms;
  - Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0; exactly 3,000 updates per arm;
  - the frozen R3 scale 0.015751632224572334, with the production fp32 arithmetic;
  - loss in normalized space;
  - the curve schedule, the measurements, the state saves and the final-state evaluation.

## Input-selection rule (to be written and hashed before any payload value is read)

- **Source.** The same rebuilt Phi of R3, t 7, client 91, P5-A position 39, with Phi hash
  `485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546`. Same 64 × 48 grid of aligned 32 × 32 tiles.
- **Tiles.** Four positional tiles, chosen without seeing values. Each covers the same four 8-column blocks as the
  URP-v1 tile (q_proj A^T, q_proj B, k_proj A^T, k_proj B), at offset 0 within its layer:
  - (16, 12): rows 512–543, columns 384–415; layer 6;
  - (16, 36): rows 512–543, columns 1152–1183; layer 18;
  - (48, 12): rows 1536–1567, columns 384–415; layer 6;
  - (48, 36): rows 1536–1567, columns 1152–1183; layer 18.
- **Why these.** They are the quarter points of both grid axes, with the URP-v1 tile at the centre point. Layer and
  hidden index change; the block composition does not. Coverage of the v and o blocks would be a separate study.
- **Checked.** I checked these coordinates against the layout's column order: 64 columns per layer, in 8-column
  blocks qA^T qB kA^T kB vA^T vB oA^T oB. No tile value was read.
- **Degeneracy.** The URP-v1 rule, unchanged: a non-finite element, T = 0, a constant tile, or max |x_n| ≥ 1.
  - A degenerate tile is reported as such and counts as not passing.
  - It is never replaced.
  - The last condition cannot occur in this snapshot: the archived whole-Phi normalized max |x_n| is
    0.8891991972923279, from `input_record.json`. The other three are unknown until the tiles are read.
- **Excluded.** The registered URP-v1 tile is not rerun.

## Measurements and decision criteria

- **Per arm,** as URP-v1:
  - strict and EP-20261012-v1, both pre-specified and unchanged;
  - final eval and train metrics in both spaces;
  - both constant baselines;
  - the curve, gradients and parameter changes;
  - resources;
  - both saved states and a forward-only replay.
- **Decision**, fixed in advance. Each arm falls in one class:
  - P: EP PASS;
  - F: EP FAIL, UNINFORMATIVE or a degenerate tile;
  - U: NOT_ELIGIBLE or UNVERIFIED, an unknown outcome that is reported and never rerun.

  Then:
  - **Supports H1:** P ≥ 3.
  - **Supports H0:** F ≥ 2.
  - **Inconclusive:** otherwise, for example P = 2, F = 1, U = 1. The two supporting conditions cannot both hold,
    because P + F + U = 4.
- **Reported only, not gated:**
  - the median and range of final eval RSE across the five tiles, URP-v1 included as context;
  - each tile's energy and range;
  - any strict PASS.
- **Not permitted:**
  - choosing tiles after reading values;
  - replacing a tile;
  - choosing a best point;
  - BatchNorm recalibration;
  - changing a threshold;
  - extending the budget.

## Resources

- **Time.** About 530 s per arm: the URP-v1 arm took 528.1 s, at 0.174 s per update. Four arms run one after another
  take about 36 min.
  - Proposed limits: 1,800 s per arm and 9,000 s aggregate.
  - Background load is not controlled.
- **Memory.** One tile at a time keeps the URP-v1 footprint of about 0.83 GB peak working set, under the
  min(4 GiB, 50% of available RAM) limit.
  - The input itself is small: 4 KiB per tile, and the latent is 64 × 1,024 fp32 = 256 KiB.
  - Larger inputs are deliberately not proposed. A full Phi at stride 1 would need a [64,2048,1536] latent, 805 MB per
    activation copy before attention and LeFF intermediates. The attention would run over 3.1 M tokens. That is
    infeasible on this CPU budget and would also change the model's geometry.
- **Evidence.** About 6.4 MB per arm, so about 26 MB in total, under the 512 MiB cap.

## Required authorization

- **Freeze.** A new protocol, for example `URP-COVER-YYYYMMDD-v1`, with:
  - its tile rule;
  - an input build that writes four patch files;
  - a frozen package and its manifest.
- **Approval.** A written approval of exactly one invocation with four starts, bound to that manifest's SHA-256.
- **Not covered by this draft:** GPU, SSBD, FAF, other seeds, compression or generalization arms. Those need their
  own decisions.

## Later questions, deliberately not combined here

- **Seed variance on the URP-v1 tile:** for example, seeds 2–4.
- **Budget extension:** whether near-identity is reachable on one tile.
- **Held-out reconstruction (generalization):** train on some tiles, evaluate on others.
- **Compression:** a strided latent below 1:1.
