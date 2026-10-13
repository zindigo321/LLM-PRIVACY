# URC-20261013-v1 result: stride-1 ResNet-3 (R-FIT) vs Uformer (U-FIT) at matched conditions

The single approved invocation ran on 2026-10-13 from 09:33:09 to 09:34:38 (+08:00). The implementer wrote this
report; it is not an independent acceptance. The approved package is `manifest_urc.json`, SHA-256
`b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6`. Before launch, 43/43 entries were verified, and they
were still intact after the run.

## Execution

| | R-FIT (this run) |
|---|---|
| Invocations / scientific starts | 1 / 1, both allowances used |
| Updates completed / planned | 3,000 / 3,000 |
| Status (strict criteria) | **COMPLETED_FIT_FAIL**; child, end record and invocation summary agree |
| Guard action, interruption | none / no |
| Supervisor and child exit codes | 0 / 0; stderr empty |
| Matching vs U-FIT | **MATCHED**: 0 violations in 35 shared conditions |
| EP-20261012-v1 (pre-specified for R-FIT) | **PASS** |

- **Shared conditions checked**: input bytes and baselines; CPU, threads, determinism and torch version; optimizer;
  update budget and curve schedule; every guard and timing setting of both invocation plans; measurement groups.
- **The four permitted differences** (core blocks, parameter count, core normalization, initialization) are listed
  separately and are not violations. They include the unequal decoder sampling-layer initial values.
- **Input and baselines were checked before the optimizer was created.** All six reference checks were exactly equal,
  including input SHA-256 `fc5e1283d1a44cb3bedaf89cff96818b1db0116e036e7cfad75c4a121138d722`.

## Final metrics (eval mode is the basis; full precision in `independent_verification.json`)

| | R-FIT | U-FIT (r1 reference) | Signed difference R − U |
|---|---|---|---|
| Eval RSE | 0.22426682990470648 | 0.2727411031905583 | **−0.0484742732858518** (favors R-FIT) |
| Eval cosine | 0.8835333171508959 | 0.8540472264633736 | **+0.02948609068752228** (favors R-FIT) |
| Train RSE | 0.2232009591742592 | 0.2720868857639887 | −0.04888592658972951 |
| Train cosine | 0.8843358254149193 | 0.8544958598853383 | +0.029839965529581014 |
| Eval norm ratio | 0.9535186900538957 | 0.9002832321336198 | |
| Eval SSE / scalar-mean SSE | 0.22433134789395942 | 0.2728195664549294 | |
| Step 0 eval RSE (post-init) | 1.03445159189013 | 1.0059377103460292 | |

- **Both metrics favor R-FIT**, so there is no tradeoff between them in this run. The eval SSE ratio R/U is
  0.8222700109415342.
- **Baselines**: the zero-prediction MSE is 0.22200250658157333 and the scalar-mean MSE is 0.22193865837012977. R-FIT's
  eval MSE is 0.04978779838194819, which removes about 78% of the scalar-mean baseline's squared error; U-FIT removed
  about 73%. This is error reduction on one input, not information preserved.
- **Train/eval gap**: the train-minus-eval output SSE is 0.0874, against an eval SSE of 51.0. As in U-FIT, the
  BatchNorm mode switch does not explain the fit failure.
- **Curve** (pre-update train RSE): 1.368 at update 1, 0.477 at 100, 0.295 at 1,000, 0.257 at 2,000, 0.222 at 3,000.
  - The pre-update train loss fell overall, but not monotonically at the end: 0.05025 at 2,600, 0.05021 at 2,700,
    0.04809 at 2,800, 0.04773 at 2,900, then 0.04928 at 3,000.
  - The lowest recorded point is 2,900. This is reported only; the verdict uses the final state, as for U-FIT.
  - U-FIT's last points were 0.06141 at 2,800, 0.06110 at 2,900 and 0.06038 at 3,000, its lowest.
- **Verification**: all 31 curve points are finite, with no non-finite gradient and no group whose gradient was zero.
  The parameter changes from init touched every element of all four groups.

## Verification (independent of `urc_analyze.py`)

- **Recomputed metrics**: `independent_verify.py` recomputed RSE, cosine, MSE and norm ratio from the recorded SSE, T,
  P, D and N of both runs. They are equal to the recorded values.
  - The identity SSE = P + T − 2D holds to a relative error of 1.1e-15 or less.
  - The signed differences, recomputed in float64 and from exact rationals, agree with the analysis script:
    δRSE −0.0484742732858518 in float64. The exact-rational route gives −0.04847427328585181, 2 ulp away; the
    float64 value carries the rounding of the two RSE divisions and the subtraction. δcos is +0.02948609068752228 by
    both routes.
- **EP**: re-implemented from POLICY.md, with no import, it gives PASS (M1, M2 and M3 all hold) and matches the frozen
  classifier.
- **Strict verdict**: re-derived as COMPLETED_FIT_FAIL.
- **Hashes**: all 15 entries of the sealed URC list match. All 42 entries of the r1 U-FIT list match. The ledger mirror
  is byte-identical to the primary, and the invocation names the approved manifest.
- **Replay**: `replay_final_state.py` loaded the saved final state into fresh models, with forward passes only and no
  optimizer. It reproduced every recorded eval-mode and train-mode sum bit for bit.

## Resources (timing scopes differ; see below)

| | R-FIT | U-FIT |
|---|---|---|
| Arm wall time (start record → end record) | 87.531 s | 506.344 s |
| Child process time (child start → record written) | 85.281 s | 503.938 s |
| Invocation aggregate (supervisor import → end) | 87.640 s | 510.516 s (includes 4.0 s of U-CHECK) |
| Sampled maximum RSS (supervisor, every 0.25 s) | 694,231,040 B | 822,517,760 B |
| Windows peak working set (child, at exit) | 698,499,072 B | 824,033,280 B |
| Retained run-root evidence | 4,075,460 B (includes two 1,996,636 B state files) | 83,280 B (no weights saved) |

- Both runs used the same machine, interpreter and thread settings. They ran one day apart, so background load is not
  controlled.
- The U-FIT aggregate includes its U-CHECK arm.
- The R-FIT time includes saving two state files. U-FIT saved none.
- The RSS values are sampled maxima; peaks shorter than 0.25 s can be missed.

## Saved states (parameters and BatchNorm buffers; 152 tensors, 493,483 parameter and 1,941 buffer elements each)

| | Path | File SHA-256 | State SHA-256 (`uid_common.state_sha256`) |
|---|---|---|---|
| Initial | `D:/cgfed-runs/urc_20261013_v1/arms/R-FIT/init_state.safetensors` | `d28ac9778aadf4c69416c40117ff73e84306b467ee831507abdb63b05c38e98a` | `00bd190fa22e1d257315b5ca5dd66a73d715a3cdfe7f5c6ddab184c539eafd6e` |
| Final | `D:/cgfed-runs/urc_20261013_v1/arms/R-FIT/final_state.safetensors` | `e03f6f3a581c4b8a48f0433591d842200682751d4beb3d480fe977a7dde35c58` | `c1149825552953545fccec464ddcde10440f9a56befd48aa130faaa749070e70` |

- Both files load strictly into a fresh model, and their recomputed state hashes equal the recorded ones.
- `num_batches_tracked` is 0 in the initial state and 3,000 in the final state.
- The initial state hash equals the fresh build hashed before launch by `check_init_comparison.py`.

## What this supports, and what it does not

Supported, narrowly: with this one input and seed 1, and with everything else held equal as listed, the stride-1
ResNet-3 fit the input somewhat better than the BasicUformer variant did.
- The final eval RSE was 0.224 against 0.273, and the cosine 0.884 against 0.854.
- It ran about 5.8 times faster per update on this CPU, with about 15% lower sampled peak memory.
- Neither model reached the strict near-identity criterion. Both pass the exploratory EP criterion: pre-specified for
  R-FIT, retrospective for U-FIT.

Not supported:
- **Significance**: one seed per architecture gives no variance estimate. The difference (about 0.05 in RSE) has not
  been compared with seed-to-seed variation, which was never measured.
- **Architectural superiority in general**: the permitted differences remain confounds. R-FIT has 0.72 times the
  parameters, BatchNorm cores, different core initialization, and different decoder sampling-layer initial values.
- **Compression**: the latent is a 64:1 expansion. Nothing here bears on generalization, real LoRA inputs, the paper's
  results or downstream use.
- **Whether either model would reach near-identity with more updates**: this was not tested. Both losses fell over
  the run as a whole, and both were noisy over the last 500 updates. R-FIT's final point was above its point at 2,900.

## Evidence (this directory unless stated)

- **Authorization and checks**: `EXECUTION_AUTHORIZATION.md`, `prelaunch_check.md`, `prelaunch_verify.py` / `.out.txt`.
- **Launch**: `launch/` holds `COMMANDS_AND_EXIT_CODES.txt` and the stdout/stderr of status before, scientific and
  status after. `launch_scientific.ps1` ran the PROPOSED_LAUNCH.md commands unchanged.
- **Analysis**: `analysis/analysis.json` (SHA-256 `728f7934f07966fd06bdabbea8bbcfdf529fa51899ced1c8736d7e8db1c59cf6`),
  `analysis/FINAL_ARTIFACTS_SHA256.txt` (the sealed run root and mirror) and `urc_analyze.out.txt`.
- **Independent checks**: `independent_verify.py`, `independent_verification.json`, `replay_final_state.py`,
  `replay_final_state.json` and their `.out.txt` files.
- **Run root**: `D:/cgfed-runs/urc_20261013_v1/`. Ledger mirror: `diagnostics/urc_20261013_v1/scientific_ledger_mirror/`.
- **Deviation**: the analysis evidence directory is `analysis/` here rather than this directory itself. That directory
  already held the authorization record, and the script refuses an existing directory. The procedure and script are
  unchanged; this is recorded in `EXECUTION_AUTHORIZATION.md` before launch.
