# URC-20261013-v1: execution authorization record

Recorded: 2026-10-13T09:26:04+08:00 (China Standard Time), the local time read just before this file was written.
This directory is outside every manifest-covered path. It is the designated location for authorization, launch and
analysis evidence.

## Authorization (as given by the user)

The user approved exactly one URC-v1 scientific invocation with one R-FIT start, bound to:
- package: `diagnostics/urc_20261013_v1/`;
- manifest: `manifest_urc.json`, SHA-256 `b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6`;
- interpreter: `D:/conda_envs/cgfedllm/python.exe`;
- run root: `D:/cgfed-runs/urc_20261013_v1/`.

This is a bounded execution authorization based on the submitted engineering evidence, not an independent acceptance.

Approved choices:
- R-FIT only: exactly 3,000 planned updates; CPU fp32, batch 1, seed 1; the frozen optimizer, thread, determinism,
  input and normalization settings.
- An 1,800 s arm limit and a 7,200 s aggregate limit, with the frozen RSS, evidence-size and start-accounting rules.
- No U-CHECK or U-FIT execution.
- The disclosed differences: core blocks, parameter count, normalization and initialization, including the unequal
  decoder sampling parameters. Weights must not be changed to remove them.
- EP-20261012-v1 is pre-specified for R-FIT. Its application to U-FIT remains retrospective.

Not authorized:
- retry, resume, seed or hyperparameter changes, a budget extension, or another arm;
- GPU work, SSBD, commits or pushes.

An analysis failure never authorizes another start.

## Planned evidence layout (fixed before launch)

- `prelaunch_verify.py` and `prelaunch_verify.out.txt`: read-only verification; `prelaunch_check.md` summarizes it.
- `launch/`: the commands, start and end times and exit codes, and the stdout and stderr of `--status` (before and
  after) and of `--scientific`. Written by `launch_scientific.ps1`, which runs the PROPOSED_LAUNCH.md commands
  unchanged through `Start-Process` so that stdout and stderr go to files.
- `analysis/`: created by `urc_analyze.py --evidence-dir <this directory>\analysis`.
  - This deviates from the PROPOSED_LAUNCH.md path, which is this directory itself. The directory must exist before
    launch to hold this record, and `urc_analyze.py` refuses an existing directory by design.
  - The procedure and the script are unchanged.
- `independent_verification.*`: my own recomputation from the recorded sums and state files, separate from
  `urc_analyze.py`.
