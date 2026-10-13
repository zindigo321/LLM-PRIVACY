# UCOV-20261013-v1 working checkpoint (working notes; not evidence)

Task (user message, 2026-10-13 ~19:50 +08:00):
- implement the new primary Uformer policy;
- prepare, freeze, execute and analyze the four-tile input-coverage study;
- write a publication handoff.

No Git writes of any kind (no staging, commit, push, branch switch, reset, clean or stash).

## Baseline at 19:51–19:59 +08:00 (read-only checks)

- Branch `phase5-ssbd-r3-single`, HEAD `0ad2795c7d03f6d0ea76992ee83864a00efd6b03`.
- Working tree: 3 modified src files and the untracked SSBD files, `.claude/` and `diagnostics/`. All unrelated to
  this task; they are left untouched.
- URP-20261013-v1 is closed and intact:
  - its inventory `7550fc45…2aff` matches 54/54, and its manifest `1df7af08…1bd1` matches 61/61 plus 4 external;
  - recorded result: COMPLETED_FIT_FAIL, 3,000 updates, RSE 0.19345271823265606, cosine 0.8981045580444489,
    MSE 7.99371986840454e-07, EP-v1 PASS.
- No Python process was running, and no four-tile preparation existed.

## Paths owned by this task (new paths only; nothing else is edited)

- `diagnostics/upp_20261013_v1/`: the new primary policy UPP-20261013-v1 (`upp_policy.json`, which is
  authoritative), `POLICY.md`, `upp_classify.py`, `test_upp_classify.py`.
- `diagnostics/ucov_20261013_v1/`: the package for protocol UCOV-20261013-v1.
  - `protocol.md`, `ucov_config.json`, `resolved_config.json`, `ucov_control.py`, `ucov_child.py`,
    `ucov_supervise.py`, `ucov_analyze.py`;
  - `input/`, `checks/`, `review/`, `evidence/` and `manifest_ucov.json`;
  - `scientific_ledger_mirror/`, created by the run.
- `diagnostics/ucov_20261013_v1_execution/`: this checkpoint, the authorization record, pre-launch checks, launch,
  analysis, verification, figures, `REPORT.md`, the next-experiment proposal, `PUBLICATION_HANDOFF.md`, the
  repository-check logs and the inventory.
- Run root: `D:/cgfed-runs/ucov_20261013_v1/`.
- Memory notes under `C:/Users/acc04/.claude/projects/.../memory/`.

Not edited: anything under `src/`, `tests/`, `docs/`, `configs/`, `results/`, the README; any closed diagnostic
directory (uid_*, urc_*, urp_*, exploratory_progression_*, synthetic_comparison_*).

## Status

- [x] Policy UPP-20261013-v1, 133 tests pass, Ruff clean.
  - `upp_policy.json` `74b277b952e0af68dee00692cde6b95f8ed747d641843bbe98517c70d98c5770`.
  - `upp_classify.py` `31904623223cc18b216b97bf2cfaa27734487c38a93ebe57efca5de84abecbb1`.
  - Do not edit either file: `ucov_config.json` pins both hashes.
- [x] 20:23:39.610 +08:00: `SELECTION_RULE.md` registered (`ddd58be9…36f2`) before `prepare_inputs.py` existed.
- [x] 20:37:23–20:37:29: `prepare_inputs.py` (`84c6a812…7326`) run once, exit 0.
  - Phi hash equals the archived value, and so does SIG.
  - All four tiles are usable, with normalized max |x_n| 0.42 / 0.54 / 0.47 / 0.70.
  - `inputs_record.json` `b4ba2f96…f57f`.
- [x] Package code, fixtures and tests. `evidence/final1_20261013`: all 9 steps exit 0.
  - Tests: UPP 133, input rule 6, control 39, lifecycle 11, analyze 3.
  - Binding `all_ok`; status 0/0.
- [x] Repository gates (`repo_gates_20261013/`, PYTHONPATH=src): Ruff check and format exit 0; pytest -m "not gpu
  and not model": 445 passed, 2 deselected, 0 failed.
- [x] **FROZEN 21:18:48 +08:00:** `manifest_ucov.json` `3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824`.
  - 65 files plus 5 external, read back.
  - **No edits to any manifest-covered file from now on.**
- [x] Pre-launch check PASS, 16/16 (`prelaunch_verify.out.txt` `74316f86…6f09`).
- [x] **21:22:56 +08:00: `launch_scientific.ps1` STARTED ONCE (wrapper pid 82100). THE LAUNCH HAS HAPPENED.
  NEVER START IT AGAIN.**
  - **If `launch/` exists, the launch has already happened: never start it again.** Follow
    `launch/COMMANDS_AND_EXIT_CODES.txt` until its `LAUNCH_SCRIPT_END` line, then run the analyzer.
  - On recovery, inspect the processes and the ledger before doing anything.
- Progress, read from the records; not final until the analysis:
  - U-TILE-1 ended at about 21:31: COMPLETED_PASS, primary PASS, 3,000 updates, arm 491.9 s, no guard action.
    RSE 0.2267, cos 0.8794.
  - U-TILE-2 ended at about 21:39: COMPLETED_PASS, primary PASS, arm 497.6 s, no guard action. RSE 0.2160,
    cos 0.8855.
  - U-TILE-3 ended at about 21:48: COMPLETED_PASS, primary PASS, arm 499.0 s, no guard action. RSE 0.2148,
    cos 0.8861.
  - U-TILE-4 started straight after. The registered plan runs all four arms; no early stop at three passes.
- [x] 21:56:11.350: `LAUNCH_SCRIPT_END scientific_exit_code=0`. The scientific command ran 21:23:00.267–21:56:08.046.
  - Ledger: 1 invocation, 4 starts, all four COMPLETED_PASS; no guard action, no interruption, no global stop.
  - Every stderr is empty. **The invocation and start budget is consumed: 1/1 invocation, 4/4 starts.**
- [x] 21:57:11: frozen `ucov_analyze.py` → `analysis/`, exit 0.
  - `analysis.json` `ed717404…ce6d`, 40,080 B; sealed list 48 entries.
  - Aggregate: **COVERAGE_CRITERION_MET, P=4, F=0, U=0**.
- [x] `independent_verify.py --write` → `independent_verification.json` `1894ca82…8240`, `all_ok`.
  - Its console copy came from a print-only re-run, because the first capture failed on a PowerShell argument
    error; this is disclosed in `independent_verify.out.txt`.
- [x] 21:59:24–21:59:29: `replay_final_states.py` → `replay_final_states.json` `a8a60534…65e3`. All four arms are
  bit-identical, in both modes and both spaces; the run root is unchanged.
- [x] `make_curve_figure.py` → `learning_curves_ucov.svg` and `curves_ucov.csv`.
- [x] `REPORT.md`, checked by `verify_report_numbers.py`: 0 problems in 79.
- [x] `NEXT_EXPERIMENT_PROPOSAL.md`: UHO-v1, held-out tiles with one shared model. Not frozen, not run; it needs new
  authorization.
- [x] `PUBLICATION_HANDOFF.md`.

## FINAL STATE

- UCOV-20261013-v1 is CLOSED.
  - 1/1 invocation and 4/4 starts are consumed. Never relaunch, retry, resume or add a start.
  - Primary result: COVERAGE_CRITERION_MET, P = 4.
- The next permissible action is human review and publication through the other assistant. Any new scientific run
  needs its own protocol, package and authorization.
- SSBD remains paused. No Git write was made.
- [x] Written while the run was going (not run yet): `independent_verify.py`, `replay_final_states.py`,
  `make_curve_figure.py`, all Ruff-clean.
- [ ] After `LAUNCH_SCRIPT_END`, in order:
  1. Run `ucov_analyze.py --evidence-dir <this dir>\analysis`. It refuses an existing directory; if `analysis/`
     exists, inspect it.
  2. Run `independent_verify.py --write`.
  3. Run `replay_final_states.py`.
  4. Run `make_curve_figure.py`.
- [ ] Report, proposal, handoff and inventory.
