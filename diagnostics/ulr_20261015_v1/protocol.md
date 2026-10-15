# ULR-20261015-v1: Uformer Latent Reduction (protocol)

Status: frozen together with `manifest_ulr.json`.

## 1. Identity, authorization, boundary

- **Authorization.** The user's message of 2026-10-15 authorized a new isolated study, its implementation and
  verification, ONE TRAIN-only channel-PCA initialization, at most one synthetic pilot per arm, and exactly THREE
  scientific adaptation starts of 6,000 updates each.
  - The manifest digest is the identity executed under that advance authorization. No human approved the digest
    itself.
  - Not authorized: commit, push, publication, SSBD, LLM / FAF / GPU / DP runs, a fourth arm, another seed, another
    bottleneck family, QAT, and downstream experiments.
- **Code base.** Worktree `D:/projects-cd/cg-fedllm-ulr`, branch `ulr-20261015-v1`, at the verified `origin/main`
  `0238e3b774e0f97f171bc5cbc53f1ec06aa49c7e` (re-checked 2026-10-15), with no upstream.
  - The audited UOA package modules and their own vendored UCX / Uformer / UPP dependencies are copied byte for byte
    under `vendor/uoa_20261014_v1/` (`SOURCE_INVENTORY.md`).
  - Nothing is imported from another worktree, and `cg_fedllm` is never imported (checked in every process).
- **Boundary.**
  - UOA-20261014-v1 (manifest `c8b30ab08abcdae5a3564ece49e77c64d305856b8d0041b1b7da8eda135446ed`) and UCX-20261014-v1
    are closed. They are read only; their files, counters and conclusions are unchanged.
  - No ULR package, branch, run root or ledger existed before this one (checked 10:35 +08:00).

## 2. Question

Start from the same learned Uformer. When the transmitted latent shrinks from 256 to 64 and 16 fp32 elements:

- How much reconstruction quality and input-specific information remain?
- How much can a bounded end-to-end adaptation recover?

The PRIMARY outcome is the verified reduction of transmitted elements and the compressor's behavior. The normalized
MSE and its distance to 1e-4 (UOA `LOSS_CONTRACT.md`, reading R2) are SECONDARY outcomes.

## 3. The common source and the bridge

- **Source.** The UOA U-LEAKY-LONG final state after 12,000 updates, with every parameter and buffer:
  - `D:/cgfed-runs/uoa_20261014_v1/steps/U-LEAKY-LONG/state_12000.safetensors`;
  - file SHA-256 `ce8659b156670de1bc386ad1df48ce0497bbe450737476ec7fb2f0d696db3646`;
  - state SHA-256 `2f8e67bfcb3a3d797f93f8f52abbcd12498a50905c57ea0975092788f64df3ea`.

  These match the UOA training record and the UOA EVAL job. Its inherited 12,000-update training cost is disclosed
  with every adapted result.
- **Bridge.** At the communication boundary:
  - `h = encoder(x_n)`, the backbone internal latent [B, 16, 4, 4];
  - `z = reduce_c(h)`, the wire latent [B, c, 4, 4];
  - the payload is the 16 c fp32 values of z only;
  - `y_n = decoder(expand_c(parse(payload)))`.

  `reduce_c` and `expand_c` are 1x1 convolutions with bias. No activation, normalization, residual, bypass, extra
  branch or skip is added. The backbone structure is unchanged: sampling strides / channels, LeakyReLU(0.1),
  attention, LayerNorm, BN and Tanh. Its weights are trainable in every arm.
- **Arms** (input tile: 1,024 elements, 4,096 B fp32):

| Arm | c | Wire latent | Elements | Element ratio | fp32 payload |
|---|---:|---|---:|---:|---:|
| C16-CONTROL | 16 | [16, 4, 4] | 256 | 1/4 | 1,024 B |
| C4-REDUCED | 4 | [4, 4, 4] | 64 | 1/16 | 256 B |
| C1-REDUCED | 1 | [1, 4, 4] | 16 | 1/64 | 64 B |

- **Shapes.** The transformer cores still operate at [16, 4, 4] (`backbone_internal_shape`, recorded separately from
  `wire_latent_shape`). This tests a channel bottleneck at this boundary, not every Uformer design.
- **C16.** It is a matched, no-reduction parameterization control, not a pure continuation of the old model. Its
  bridge need not stay full-rank.

## 4. INIT: one TRAIN-only channel PCA and the zero-step checks

- **Data.** The exact UCX / UOA TRAIN tiles (1,440), the EARLY / LATE membership (168 / 336) and the TRAIN scale
  (fp32 0.013576022349298). They are verified by SHA-256 against the UCX records via the UOA configuration.
- **Source diagnostic.** On disposable deep copies of the source, on the four registered UCX / UOA TRAIN batches
  (seed 20261014): train-statistics against eval-statistics outputs and BN statistics.
  - It is recorded and never acted on. Any new suspected cause is a follow-up hypothesis.
- **Channel PCA (one fit).**
  - The TRAIN latents h of the source come from eval mode, every tile alone: [1440, 16, 4, 4]. They are reshaped to
    23,040 samples of 16 channels.
  - Centered float64 SVD: mu is the TRAIN mean, and V holds the principal directions as rows, in descending singular
    value order (LAPACK order), with this sign rule: each row is flipped so that its largest-|entry| component is
    positive, and ties go to the first index.
  - Near-degenerate singular values are reported.
  - For c in (16, 8, 4, 2, 1), computed in float64 and stored in fp32: W_reduce = V_c, b_reduce = -V_c mu,
    W_expand = V_c^T, b_expand = mu.
  - This channel PCA is over internal channels, not the input-tile PCA. It minimizes source-latent channel error and
    is not claimed optimal for reconstruction.
- **C16 identity check (fails closed).** On TRAIN, the C16 bridge must satisfy:
  - max |expand(reduce(h)) - h| <= 1e-4;
  - max |y_n(C16) - y_n(source)| <= 1e-4;
  - |pooled normalized MSE difference| / source <= 1e-4.

  Bitwise equality is not claimed.
- **Step-zero TRAIN evaluation.** Every c (16, 8, 4, 2, 1) is evaluated through the real wire path at 64 c B per
  tile. c = 8 and c = 2 are forward-only mappings from the same basis. They are not training arms, they cannot change
  the three registered arms, and they are never called adapted models.
- **Shared start.** Every arm holds tensors identical to the source backbone before adaptation, checked by hash. No
  whole-model initializer touches the loaded backbone.

## 5. Adaptation: exactly three starts

- **Start.** Each arm starts independently from the source state plus its own PCA bridge.
- **Optimizer.** A fresh Adam over every parameter: lr 2e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0, batch 8,
  exactly 6,000 updates (48,000 presentations, about 33.3 passes over TRAIN).
- **Schedule.** One newly registered schedule shared by the arms: concatenated `randperm(1440)` epochs from a CPU
  generator seeded 20261015, SHA-256 in `ulr_config.json`.
- **Unchanged from UOA.** The loss, the vendored mean normalized-space MSE in train mode, which passes through the
  bottleneck; the TRAIN scale; CPU fp32; 24 / 24 threads; deterministic algorithms.
- **Not used.** No scheduler, clipping, objective term, BN / GN change, Tanh removal, extra seed, QAT, per-tile
  normalization or dynamic rate.
- **Checkpoints.** States at 0 / 1,000 / 3,000 / 6,000 updates, each followed by a full-TRAIN evaluation through the
  wire path on an eval-mode copy. The training state, buffers, mode and RNG are checked unchanged.
- **Endpoint.** The final 6,000-update state is the endpoint. There is no best-checkpoint selection and no
  extension.
- **No pruning.** All three arms run within their budgets even if an earlier arm is poor; scientific conditions are
  never pruned on their errors.

## 6. Wire format and decoder isolation

- **Format.** The 16 c values of z, as row-major [c, 4, 4], explicit little-endian fp32: exactly 64 c bytes per tile
  (`ulr_model.wire_to_bytes` / `wire_from_bytes`).
- **Frozen validity rule.**
  - A payload of any other length is malformed and raises. The step then fails as EXECUTION_ERROR, not as a
    scientific result.
  - Non-finite values are transmitted unchanged and make the tile's prediction non-finite, which can never pass.
- **Old codec.** The old UCX / UOA 256-value latent codec is not used for any arm (test: it refuses 64 values).
- **Receiver.** Evaluation decodes with a standalone receiver: expand_c plus the frozen decoder, loaded strictly
  from only the `expand.*` and `base.decoder.*` tensors of a state file.
  - It receives only parsed payload bytes and the shared configuration (c, the TRAIN scale).
  - No x, h, encoder activation, per-tile side information, tile index or unsent skip reaches it.
  - A method identifier would be shared metadata. Its framing is not modeled.

## 7. Comparators, controls and metrics

- **Input PCA (hash-verified UCX TRAIN basis, leading rows; no new fit).**
  - PCA-256: 256 fp32 coefficients, 1,024 B.
  - PCA-64: 64 elements, 256 B.
  - PCA-16: 16 elements, 64 B.

  The shared state is the fp32 mean plus the fp32 rows.
- **256-B references.**
  - SCALAR-4 (UOA TRAIN-fitted levels): 1,024 2-bit indices.
  - The UOA source's INT8 latent (UOA scales): 256 int8 values.

  Both differ from 64 fp32 values.
- **1,024-B references.** INT8-SHARED (UCX step), and the unmodified source FLOAT latent (256 fp32).
- **64-B budget.** Nothing larger is labelled equal-byte there.
- **Controls.**
  - C-ZERO, C-MEAN, and C-ZLAT (the source's zero latent).
  - C-ZWIRE: an all-zero wire payload through each arm's receiver.
  - C-WRONG: the next snapshot's same-position payload in (time, client) order, cyclic. This is the UCX / UOA map;
    every pair crosses clients, checked at config time.
- **Evaluation scope.** After every state and comparator is fixed, EVAL evaluates each arm's step-zero and final
  states on TRAIN / EARLY / LATE. Intermediate checkpoints are TRAIN learning curves only.
- **Metrics.** float64 sums in the original and normalized spaces:
  - normalized MSE, raw MSE, SSE, target energy, RSE (no square root), cosine and norm ratio;
  - pooled values as ratios of math.fsum-pooled sums;
  - per-tile, snapshot, client, row, layer and LoRA-block breakdowns.

## 8. Policy and reporting

- **Policy.** UPP-20261014-v1 is unchanged and vendored by hash: valid evidence, pooled RSE <= 1.0 and cosine >= 0.1
  per held-out cohort. The two-cohort rule is applied separately to each arm's final state.
  - Step-zero states and comparators are classified descriptively.
  - The constant-baseline MSE condition is report-only.
  - No hidden pass-rate or degradation threshold is used.
- **Four separate conclusions.**
  - **A.** The verified element reduction.
  - **B.** UPP per arm.
  - **C.** Degradation, recovery and input dependence.
  - **D.** Comparators and the secondary 1e-4 distance.

  The smallest tested adapted element count that passes may be named, but only next to its errors and comparators.
  It is not proof of an optimal dimension, of useful communication or of paper reproduction.
- **Status.** EARLY / LATE were observed during UCX / UOA, so all results are exploratory. The three paired arms at
  one seed are not independent replications.

## 9. Execution, limits, restart rules

- **Steps.** One invocation of five steps, each a fresh child: INIT -> C16-CONTROL -> C4-REDUCED -> C1-REDUCED ->
  EVAL. This is the same two-root write-once ledger design as UOA.
  - Every child before EVAL is refused the held-out file and every UCX / UOA evaluation output by an audit hook.
- **Failures and restarts.** A step that does not complete ends the invocation, and later steps are NOT_RUN. There is
  no retry, rename, counter reset, step reduction, device switch or added experiment. Completed evidence remains
  deliverable. Resource or validity failures are reported as such, never as poor reconstruction.
- **Limits** (`ulr_spec.LIMITS`):
  - each neural arm: 1,800 s (soft stop 60 s, kill 20 s before);
  - INIT: 600 s;
  - EVAL: 1,500 s, and the post-run verification 500 s (2,000 s cumulative for evaluation and verification);
  - total scientific wall: 9,000 s including the pilots;
  - memory per child: min(4 GiB, 50% of available RAM before launch), with the neural floor from the pilots;
  - retained new evidence: 512 MiB.

  Counters: 1 invocation, 3 neural starts, 1 channel-PCA fit, 3 pilots (used: 3, all GO, 76.2 s), and no new input
  PCA or scalar-quantizer fit.

## 10. Analysis and verification

- **Analysis.** `ulr_analyze.py` (frozen) computes A-D, learning curves, breakdowns, byte accounting, diagnostics,
  resources and reproduction checks against the UOA records.
- **Independent verification** (`../ulr_20261015_v1_execution/independent_verify.py`). It is torch-free and imports
  no production metric, codec or classifier.
  - It recomputes the sums from the saved predictions and targets.
  - It re-decodes the PCA, SCALAR-4 and INT8 payloads.
  - It checks every wire payload length and its bytes against the saved wire latents.
  - It re-derives the UPP bounds and target distances.
- **Replay.** A fresh process with no optimizer re-runs the sender, and standalone receivers built from the saved
  states decode the saved payloads. Results are compared bit for bit.
- **Protected work.** It is compared against `preparation/protected_fingerprint_before.json`.

## 11. Interpretation limits

- **What a 16-element payload shows.** A 1/64 element ratio for a 1,024-element tile and a 1/64 bare fp32 payload
  ratio.
- **What it does not show.**
  - A 64-fold end-to-end communication saving: the decoder state, framing and refresh are excluded.
  - Faithful reproduction of the paper's full-tensor architecture.
  - Full LoRA effective-update fidelity, task-response fidelity, FAF benefit, or novelty.
- **No novelty claim.** None is made for PCA initialization, 1x1 projections, latent reduction or warm-start
  fine-tuning.
