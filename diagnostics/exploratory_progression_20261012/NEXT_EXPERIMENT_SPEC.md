# URC-v1: stride-1 ResNet-3 comparator at the Uformer U-FIT conditions (draft specification)

Status: DRAFT. Not frozen, not approved and not run. To run it, prepare a new package with its own protocol, config,
code and manifest (manifest written last), get one explicit approval, and use one invocation.

The Uformer remains the primary research target. The ResNet arm is a diagnostic comparator. This comparison does not
gate other exploratory work: Uformer follow-ups, such as the budget-only extension proposed in the r1 `REPORT.md` §6,
can be registered independently. Neither needs the strict near-identity threshold to be met first.

## Question

Under the U-FIT conditions of UID-20261012-v2 r1, change only the core blocks: BasicUformer → ResNet residual blocks.
Does the final-state fit of the one fixed input change, with one seed?

## Design

- **Arm**: R-FIT, a single arm. One scientific start, one invocation, seed 1, 3,000 updates. No retries, no extra
  seeds and no parameter search. There is no scientific U-CHECK arm. Pipeline sanity is covered by the engineering
  pre-checks below (fixture seeds only) and by reusing the frozen metric code.
- **Reference**: the existing U-FIT record (`manifest_r1.json` SHA-256
  `9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044`). No Uformer arm is rerun.
- **Limits**:
  - arm: 1,800 s, with the r1 soft-stop/kill mechanics (60 s eval reserve, 20 s end margin);
  - invocation: 3,600 s aggregate (an implementation choice: twice the arm limit, leaving time for setup and evidence);
  - RSS: min(4 GiB, 50% of available RAM at launch), sampled every 0.25 s; the child process tree is killed if it is
    exceeded;
  - retained evidence: 512 MiB at most.
- **Hardware**: CPU only (`CUDA_VISIBLE_DEVICES=-1`). Recorded hardware: i9-14900HX, about 32 GB of RAM. A
  read-only CIM query on 2026-10-12 reported 24 cores, 32 logical processors and 34,070,192,128 B of RAM. The RTX 4060
  Laptop GPU is present and unused.

## Conditions that must match the reference

They are checked at run time against the U-FIT `child_record.json`.

| Condition | Required value | Check |
|---|---|---|
| Input | `uid_common.make_input` from the frozen config: data seed 20261012, `1.6·rand([1,1,32,32]) − 0.8`, fp32 | tensor SHA-256 = `fc5e1283d1a44cb3bedaf89cff96818b1db0116e036e7cfad75c4a121138d722`; T = 227.3305667395311 exactly |
| Normalization | none | none applied |
| Baselines | zero prediction and one scalar mean | MSE 0.22200250658157333 and 0.22193865837012977 exactly |
| Geometry | six stages, all stride 1; latent [64,32,32] (64:1 expansion) | shape assertion |
| Precision | float32 compute, float64 metric reductions | the frozen `uid_common.metrics` |
| Optimizer | Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8, wd 0, all parameters | trainable coverage = optimizer coverage |
| Training | batch 1, `mse_loss(decode(encode(x)), x)`, train mode, 3,000 updates; curve every 100 updates | as in `uid_r1_child.run_arm` |
| Verdict state | final state, eval mode, two independent copies from the same RNG state; train mode reported | as in r1 |
| CPU | 24 intra-op / 24 inter-op threads, deterministic algorithms, torch 2.14.0+cu130 | the frozen `configure_cpu` result is recorded |
| Init procedure | `torch.random.fork_rng(devices=[])`, then `torch.manual_seed(1)`, then construction | as in `uid_common.build_model` |

The frozen `uid_common.py` (SHA-256 `f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230`) is imported
read-only after its hash is checked. It supplies `make_input`, `metrics`, `baselines`, `configure_cpu` and
`state_sha256`.
- `uid_common.load_config` accepts only protocol id UID-20261012-v2. The new package therefore loads its own config
  with its own loader.
- `make_input` receives the frozen `config.json` itself (raw SHA-256
  `d7ae76052d50776705f386655b0b501d403104d7d4ca118865a2b6ff3b71b0bd`, checked first).
- The r1 supervisor/child/measure mechanics are copied into the new package. The reviewed diff is limited to five
  changes: the arm definition, config loading, the model build, state saving and the reference assertions.
- Frozen r1 files are not edited.

If any condition cannot be met or is measured differently, R-FIT is still completed and reported. The comparison is
then labelled NOT_MATCHED, with the specific mismatch. No Uformer run is started to compensate; any new Uformer run
needs its own registration and approval.

## The model: an isolated stride-1 ResNet-3 adaptation

The production `ResNetAutoEncoder` at HEAD (`src/cg_fedllm/compression/autoencoder.py`, blob
`cf38fd3f7749a2cdd45864e41f22d3ab0dc9e3db`, also unchanged since the P5-A execution commit `6603bd8`) hardcodes
`stride=2` in all six encoder convolutions and `stride=2, output_padding=1` in all six decoder transposed convolutions.
That is 12 stride-2 layers. `check_input_hw` requires sides divisible by 2^6 = 64, so a 32×32 input is rejected
("AE input 32x32 must be divisible by 64", checked read-only on CPU). The production model is therefore not used, and
it is not modified.

- New file `resnet3_s1.py` in the new package. It imports neither `cg_fedllm` nor the uncommitted SSBD
  `StridedResNetAEConfig` that is in the working tree.
- It re-declares the HEAD ResNet-3 layer by layer:
  - encoder: ReflectionPad(1), Conv3×3 1→1 (no bias), BN, ReLU; six stages of Conv3×3 (no bias), BN, ReLU, channels
    1→2→4→8→16→32→64; three ResidualBlock(64);
  - decoder: three ResidualBlock(64); six stages of ConvTranspose3×3 (no bias), BN, ReLU; ReflectionPad(3),
    Conv7×7 1→1 (bias), Tanh.
- The single intervention: every stride is 1 and every `output_padding` is 0, the r1 Uformer rule
  `output_padding = stride − 1`.
- No skips and no global shortcut. `decode` receives only z.
- Submodules are named as in the Uformer AE: `encoder.sampling`, `encoder.core`, `decoder.core`, `decoder.sampling`,
  with `encode` / `decode` methods. This lets the copied `uid_r1_measure.groups` and `uid_common.reconstruct` work
  unchanged; the HEAD names `encoder.net.<i>` would raise a KeyError in `groups`. Construction order matches HEAD.
- The fixed name mapping to HEAD is:
  - `encoder.sampling.<i>` = `encoder.net.<i>` (i 0–21);
  - `encoder.core.<j>` = `encoder.net.<22+j>`;
  - `decoder.core.<j>` = `decoder.net.<j>`;
  - `decoder.sampling.<i>` = `decoder.net.<3+i>`.

**Engineering checks** (before freezing; fixture seeds only, never 20261012 or 1):
- (a) Rebuilt with stride 2, the class is state-dict-identical to HEAD `ResNetAutoEncoder` under the fixed name mapping
  (names and shapes). Its forward pass equals HEAD's in float64 on a 64×64 fixture input with the same weights.
- (b) With stride 1, the parameter and buffer shapes are unchanged. Strides change no weight shape.
- (c) [1,1,32,32] → z [1,64,32,32] → [1,1,32,32].
- (d) Gradients are nonzero and finite in all four groups after one backward pass.
- (e) A CPU timing probe: about 10 updates on a fixture input to estimate s/update. If 3,000 updates are projected to
  exceed about 1,500 s, report this before freezing; do not change the budget silently.

## Disclosed differences from the reference

| | Uformer U-FIT (reference) | R-FIT (planned) |
|---|---|---|
| Parameters | 685,507: sampling 24,833 + 24,746; cores 2 × 317,964 | 493,483: sampling 24,833 + 24,746; cores 2 × 221,952 (shapes from the verified R3 checkpoint header and the HEAD model) |
| Buffers | sampling-layer BN buffers plus the `relative_position_index` of each attention block | sampling-layer BN buffers plus the BN of 6 residual blocks; 1,941 buffer elements in total |
| Core normalization | LayerNorm in each LeWin block | BatchNorm in each residual block (batch 1: train/eval statistics differ; both modes reported) |
| Core initialization | PyTorch defaults, then trunc-normal(0.02) on Linear and the position-bias tables, LayerNorm (1, 0) | PyTorch defaults only (conv kaiming-uniform, BN (1, 0)) |
| RNG stream | `manual_seed(1)` in `fork_rng`; enc.sampling, enc.core, dec.core, dec.sampling built in that order | same procedure and order |

Initial values:
- Encoder sampling layers: built first in both models, with the same shapes and init calls, so they are expected to
  receive the same initial values. This is not verified. The run reports their initial L2 norm next to U-FIT's
  recorded `ref_l2` of 12.999809210476185; a match is consistent with equality but does not prove it.
- Decoder sampling layers: built after the cores, which consume different numbers of random draws, so their initial
  values differ. They cannot be copied either: the U-FIT initial state was not saved, only its hash
  `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`.
- The parameter difference (0.72×) is not equalized. The HEAD block definition is kept as the comparator.

## Records

These are required in addition to everything r1 records:
- `final_state.safetensors`: all parameters and buffers, saved after the final update and before the final
  evaluation. Its tensor-level `state_sha256` must equal `model_state_sha256_final`.
- `init_state.safetensors`: the initial state, with the same check.

Both together are about 4 MB, within the evidence cap. Each is saved once, and saving cannot overwrite.

The curve, gradient and parameter-change groups, the step-0 measurement on isolated copies, both baselines, RSS, wall
time and the end/invocation ledger are recorded as in r1.

## Analysis (fixed before the run)

- **Primary record**:
  - the frozen strict status (RSE ≤ 0.01, cosine ≥ 0.99, eval mode);
  - the EP-20261012-v1 outcome from `ep_classify.classify`. For this run it is pre-specified, because the policy
    predates it.
- **Comparison to the reference**, from recorded sums only:
  - final eval RSE, cosine and SSE / best-baseline SSE of each arm;
  - the differences in RSE and cosine (R-FIT − U-FIT).
  - The labels are descriptive only: one seed per architecture gives no variance estimate, so no significance and no
    "better" claim is made.
  - If the train-mode and eval-mode results differ materially, both are reported. The verdict basis stays eval mode.
- **Not permitted**: choosing a best checkpoint, recalibrating BatchNorm after training, extending the budget,
  re-seeding or rerunning after the result is seen.

## What R-FIT can and cannot show

- It can show whether, at these exact conditions, residual conv cores fit the fixed input to a materially different
  degree than the BasicUformer cores did, with one seed each.
- It cannot show architectural superiority in general, nor compression: the latent is a 64:1 expansion. It also says
  nothing about generalization, real LoRA inputs, the paper's results or downstream use.
- Parameter count, initialization and normalization differences (table above) remain confounds.
