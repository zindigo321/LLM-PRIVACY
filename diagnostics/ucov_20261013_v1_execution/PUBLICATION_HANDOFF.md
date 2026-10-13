# Publication handoff: UPP-20261013-v1 policy and UCOV-20261013-v1 four-tile study

Status: complete and stable, written on 2026-10-13 after the analysis. No file listed here is still changing. This
assistant made no Git write of any kind. The branch is still `phase5-ssbd-r3-single`, HEAD
`0ad2795c7d03f6d0ea76992ee83864a00efd6b03`; neither was changed.

## Outcome (key paths)

- **Primary decision: coverage criterion MET.** UPP-20261013-v1 PASS in all four arms: P = 4, F = 0, U = 0.
  - Final eval RSE / cosine: U-TILE-1 0.2267 / 0.8794; U-TILE-2 0.2160 / 0.8855; U-TILE-3 0.2148 / 0.8861;
    U-TILE-4 0.1980 / 0.8956.
  - The historical strict criterion (0.01 / 0.99) fails in all four, and EP-20261012-v1 passes in all four. Both
    are reported only.
- **Report:** `diagnostics/ucov_20261013_v1_execution/REPORT.md`.
- **Machine-readable results:** `analysis/analysis.json`, `independent_verification.json` and
  `replay_final_states.json`.
- **Budget consumed:** 1 of 1 invocation and 4 of 4 starts. The protocol is closed.

## Files created by this task (all new; no existing file was modified)

| Path | Files | Status |
|---|---|---|
| `diagnostics/upp_20261013_v1/` | `upp_policy.json`, `upp_classify.py`, `POLICY.md`, `test_upp_classify.py` | complete; pinned by hash, so do not edit |
| `diagnostics/ucov_20261013_v1/` | the package. 65 files are covered by `manifest_ucov.json` (`3bc2ab30…e824`), plus the manifest itself and `scientific_ledger_mirror/` (10 records written by the run) | complete, frozen and closed |
| `diagnostics/ucov_20261013_v1_execution/` | authorization, pre-launch checks, launch logs, analysis, verification, replay, figure and CSV, report, proposal, this handoff, the checkpoint, `repo_gates_20261013/` and `EVIDENCE_INVENTORY_SHA256.txt` | complete; the inventory is written last |
| `D:/cgfed-runs/ucov_20261013_v1/` (outside the repository) | the ledger, and four arm directories with `child_record.json`, `curve.jsonl`, `job.json`, empty stdout and stderr, and the init and final `.safetensors` states | complete; about 25.7 MB |

Also changed outside the repository: the Claude memory notes under `C:/Users/acc04/.claude/projects/...` (not
project files).

## Suggested publication set (compact, about 0.66 MB)

- **Source and policy:** all of `diagnostics/upp_20261013_v1/`.
- **Package:** all of `diagnostics/ucov_20261013_v1/`, including:
  - `input/tile_{1..4}.safetensors` (4.4 KB each) and `checks/fixture_tile_{1..4}.safetensors` (1.2 KB each);
  - `evidence/final1_20261013/*.log`;
  - `scientific_ledger_mirror/`.
- **Evidence:** all of `diagnostics/ucov_20261013_v1_execution/`, including `repo_gates_20261013/*.log`.
- **Not in the repository:** the run-root state files. They are referenced by hash in `analysis/FINAL_ARTIFACTS_SHA256.txt`
  and `EVIDENCE_INVENTORY_SHA256.txt`.

The task did not add or change any test, CI, README, docs or config under `src/`, `tests/`, `docs/` or `configs/`. A
README or docs entry for Phase-5 diagnostics is a separate, optional documentation change for the publisher to
decide.

## Publication hazards (Git settings were not changed; please check before committing)

1. **Ignored evidence that the manifests bind.** `.gitignore` has `*.safetensors` (line 228) and `*.log` (line 59).
   These files are ignored but manifest-bound or evidence:
   - `diagnostics/ucov_20261013_v1/input/tile_{1..4}.safetensors` (runtime inputs, in the manifest);
   - `diagnostics/ucov_20261013_v1/checks/fixture_tile_{1..4}.safetensors` (in the manifest);
   - `diagnostics/ucov_20261013_v1/evidence/final1_20261013/*.log`: 10 logs in the manifest; the 11th file there is
     `SUMMARY.txt`, which is not ignored;
   - `diagnostics/ucov_20261013_v1_execution/repo_gates_20261013/*.log`.
   The closed URP-v1 package has the same issue (`input/patch_x.safetensors`, `checks/fixture_patch.safetensors`, its
   `evidence/**/*.log`). Leaving them out makes `verify_package` fail on a checkout, so add them explicitly (for
   example `git add -f`) or leave the package unpublished.
2. **Line endings.**
   - `core.autocrlf=true` is set locally, and `.gitattributes` has `* text=auto eol=lf` and `*.ps1 text eol=crlf`.
   - Every text file listed here is LF on disk; this was checked.
   - Git will convert the three `.ps1` files to CRLF in a checkout: `checks/run_all_checks.ps1` (manifest-bound) and
     the execution directory's `launch_scientific.ps1` and `run_analyzer.ps1`. Their committed blob stays LF, but a
     checkout on Windows will have different bytes. The URP-v1 manifest has the same `.ps1` issue.
   - Re-verify hashes against the committed blob (`git show :path | sha256sum`), not the working copy.
   - Any CRLF conversion of a `.json`, `.py` or `.md` file listed in a manifest invalidates the package.
3. **External artifacts.** A source-only checkout does not contain them. The package binds five external files by
   hash:
   - the U-FIT `child_record.json`, `d8150ec1b8a1ff61ffee1c9e7bf42114060794894305b3c66cf24228faff3ecb`;
   - the URP-v1 `child_record.json`, `45640ed18111e03935e4b9451bb02f364bea14bdbb0ccec1bd31f1faa24b9583`;
   - the R3 `index.jsonl`, `b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240`;
   - the end snapshot `t0007_c0091.safetensors`, `09c95184a657fa778aef48e18196f95c568854476b75f2812d68dfd58126dd45`;
   - the start snapshot, `31b60754e8b55992c908e416f2652c8f2dbbbdee0030440fa2163a5528cd5a7a`.

   The run-root states and records are under `D:/cgfed-runs/ucov_20261013_v1/`. The package cannot be re-verified or
   replayed without them, so do not describe a source checkout as self-contained.
4. **Dependencies on closed directories.** The package imports frozen files from four closed directories:
   `diagnostics/uid_20261012_v2/`, `uid_20261012_v2_r1/`, `urp_20261013_v1/` (its `urp_control.py`,
   `manifest_urp.json` and `input/prepare_input.py`) and `exploratory_progression_20261012/`. All of them are still
   untracked. Publishing UCOV without them breaks its identity checks.
5. **Stale editable install.** In `D:/conda_envs/cgfedllm` the editable install points at
   `.claude/worktrees/p5a-engineering-support/src`.
   - The package runs with PYTHONPATH unset and never imports `cg_fedllm`.
   - The repository gate ran with `PYTHONPATH=<repo>/src`.

## Unrelated existing changes to exclude (present before this task, untouched)

- **Modified:** `src/cg_fedllm/cli.py`, `src/cg_fedllm/compression/autoencoder.py` and
  `src/cg_fedllm/phase5/__init__.py`. These are the paused SSBD work.
- **Untracked:** `configs/phase5/ssbd_r3_single_seed1.yaml`, `src/cg_fedllm/phase5/ssbd*.py` (5 files),
  `tests/unit/ssbd_fixtures.py`, `tests/unit/test_ssbd_*.py` (8 files) and `.claude/`.
- **Other untracked diagnostics directories,** which are historical evidence from earlier tasks: `uid_*`, `urc_*`,
  `urp_*`, `exploratory_progression_*` and `synthetic_comparison_*`. Publish them only as their own reviewed changes.

## Tests and checks run (actual results)

- **Pre-freeze package checks.** `diagnostics/ucov_20261013_v1/evidence/final1_20261013/SUMMARY.txt`: all 9 steps
  exit 0.
  - Ruff check and format pass on the package and the policy directory.
  - Tests: UPP 133, input rule 6, control 39, lifecycle 11 (real subprocesses on fixtures) and analyze 3, all passed.
  - Registered binding `all_ok`; status 0/0.
- **Repository gates.** `repo_gates_20261013/`, with PYTHONPATH set to the repository `src`:
  - `ruff check src tests scripts`: exit 0;
  - `ruff format --check src tests scripts`: exit 0;
  - `pytest -q -m "not gpu and not model"`: 445 passed, 2 deselected, exit 0.
  - No pre-existing failure was observed.
- **After the run:**
  - the frozen analyzer, exit 0;
  - the independent verification, `all_ok`;
  - the forward-only replay, bit-identical in all four arms;
  - the report-number check, 0 problems in 79.

## Unresolved limitations

- One seed per arm and five tiles of one snapshot, all q/k blocks.
- A 64:1 expanded latent: no compression, no held-out generalization, no whole-snapshot reconstruction, no
  downstream or FAF claim.
- The primary policy was chosen with knowledge of URP-v1; that is disclosed in `POLICY.md`.
- The console copy of the first independent-verification run was lost to a PowerShell argument error. The record
  itself is intact, and this is disclosed in `REPORT.md`.
- SSBD remains paused.
