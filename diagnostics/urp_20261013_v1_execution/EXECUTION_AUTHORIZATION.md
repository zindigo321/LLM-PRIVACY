# URP-20261013-v1: execution authorization record

Recorded: 2026-10-13T16:34+08:00 (China Standard Time), the local time read just before this directory was created.
This directory is outside every manifest-covered path. It holds the authorization, launch and analysis evidence.

## Authorization (as given by the user in the current task message)

The user approved exactly ONE scientific invocation and ONE U-PATCH start under URP-20261013-v1, bound to:
- package: `diagnostics/urp_20261013_v1/`;
- manifest: `manifest_urp.json`, SHA-256 `1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1`;
- interpreter: `D:/conda_envs/cgfedllm/python.exe`;
- run root: `D:/cgfed-runs/urp_20261013_v1/`; ledger mirror: `diagnostics/urp_20261013_v1/scientific_ledger_mirror/`.

The approval covers only the unchanged package with its registered settings, input, dependencies, limits and execution
path. It supersedes the "PREPARE only" / "awaiting approval" statuses in `PROPOSED_LAUNCH.md` and
`CONTINUATION_CHECKPOINT.md` for this exact package only. Those files are left as written.

Registered conditions (from `protocol.md` and `resolved_config.json`; not changed here):
- input R3 `balanced_effective_delta_r8`, t 7, client 91, P5-A position 39; patch `Phi[0,1024:1056,768:800]`,
  [1,1,32,32] fp32, floor-centre rule; frozen scale 0.015751632224572334 with the production fp32 arithmetic;
- model: the frozen `uid_common.build_model(cfg, "U-FIT")`, 685,507 parameters, initial state
  `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`, latent [64,32,32];
- CPU fp32, batch 1, seed 1, 24/24 threads, deterministic algorithms; Adam lr 2e-4, betas (0.9, 0.999), eps 1e-8,
  weight decay 0; exactly 3,000 planned updates;
- limits: 1,800 s arm, 7,200 s aggregate, RSS min(4 GiB, 50% of available RAM at launch), 512 MiB retained evidence.

Not authorized: a retry, a resume, a second invocation or start, other arms or patches, seed or hyperparameter
changes, a budget extension, BatchNorm recalibration, any other scientific experiment, GPU work, SSBD, dependency
changes, commits or pushes. An analysis failure never authorizes another start.

## Planned evidence layout (fixed before launch)

- `prelaunch_verify.py`, `prelaunch_verify.out.txt`: read-only verification (manifest bytes, all entries, external
  bindings, pinned files, fresh resolution, the child's scientific config branch, interpreter, environment, memory);
  `prelaunch_check.md` summarizes it.
- `launch/`: written by `launch_scientific.ps1`, which runs the `PROPOSED_LAUNCH.md` commands unchanged through
  `Start-Process` so that stdout and stderr go to files, with start and end times and exit codes. It refuses to start
  if `launch/` already exists or if `--status` does not show 0 invocations, 0 starts and no run root.
- `analysis/`: created by `urp_analyze.py --evidence-dir <this directory>\analysis`, the path in `PROPOSED_LAUNCH.md`.
  The analyzer refuses an existing directory by design; `analysis/` does not exist before the analysis runs.
- `independent_verification.*`: my own recomputation from the recorded sums and the state files, separate from
  `urp_analyze.py` and `ep_classify.py`; and a forward-only replay of the saved final state (no optimizer).
- `REPORT.md`, `CONTINUATION_CHECKPOINT.md`, `NEXT_EXPERIMENT_PROPOSAL.md`, and `EVIDENCE_INVENTORY_SHA256.txt` last.
