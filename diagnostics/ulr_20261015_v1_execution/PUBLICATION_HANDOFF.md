# Publication handoff: ULR-20261015-v1 (Uformer Latent Reduction)

Status: written after the run, for a later human-reviewed publication.

- This assistant made no Git write: no add, commit, push, PR or merge, and no change to `.gitignore` or
  `.gitattributes`.
- Worktree `D:/projects-cd/cg-fedllm-ulr`, branch `ulr-20261015-v1`, at the verified `origin/main` `0238e3b`, with no
  upstream.

## File scope (all new, untracked)

| Path | Content | Notes |
|---|---|---|
| `diagnostics/ulr_20261015_v1/` | frozen package (manifest `8db0e72f9f9315c105dd0e82b5bc189b15f078ff4b300b18222f666bd9b00bb6`, 61 files): protocol, UOA clarification note, source inventory, code, checks, preflight and preparation records, vendored byte copies, pre-freeze evidence logs | never edit; `scientific_ledger_mirror/` was written by the run and is outside the manifest |
| `diagnostics/ulr_20261015_v1_execution/` | authorization, pre-launch check, launch logs, analysis, independent verification, replay, protected-work comparison, manifest re-check, figures, tables, REPORT, this handoff, checkpoint, inventory | written after the freeze |

Nothing under `src/`, `tests/`, `configs/`, `results/`, `requirements/`, `.github/`, `README.md` or `docs/` was
changed.

**Dependencies.** UOA-20261014-v1 and UCX-20261014-v1 are still unpublished in their own worktrees. This package
vendors the needed UOA modules byte for byte, so its code does not depend on their publication. Its records do cite
their paths and hashes, and its runs read their artifacts.

Suggested commits (one logical change each; for a human to decide):

1. `feat(uformer): add the ULR-20261015-v1 latent reduction package`, plus the `.gitignore` / `.gitattributes` lines
   below.
2. `docs(uformer): record the ULR-20261015-v1 results`.

## Publication hazards

1. **Ignored but manifest-bound logs.** `.gitignore` ignores `*.log`, but `diagnostics/ulr_20261015_v1/evidence/final1_20261015/*.log`
   is listed in `manifest_ulr.json`. Add `!diagnostics/ulr_20261015_v1/evidence/**/*.log`, or use `git add -f`.
2. **Byte preservation.** `.gitattributes` has `* text=auto eol=lf` and `*.ps1 text eol=crlf`, but the
   manifest-bound `checks/run_all_checks.ps1` and the executed `launch_scientific.ps1` are LF.
   - Some captured outputs written by PowerShell (`*.out.txt`) have a UTF-8 BOM and CRLF.
   - Add the following, then re-verify against the committed blobs:
     ```text
     diagnostics/ulr_20261015_v1/** -text !eol
     diagnostics/ulr_20261015_v1_execution/** -text !eol
     ```
3. **External artifacts (not bundled; a checkout is not replay-complete).**
   - The ULR run root `D:/cgfed-runs/ulr_20261015_v1/` (167,668,070 B) holds every safetensors file: states, bridge
     init, sums, predictions, wire latents and payloads. `*.safetensors` is git-ignored.
   - The 12 bound UOA / UCX files are listed in `manifest_ulr.json` `external_bound_files`.
   - Without them, the analyzer, `independent_verify.py` and `replay_final_states.py` cannot be re-run.
4. **Absolute paths.** Records name `D:/projects-cd/cg-fedllm-ulr/...`, `D:/cgfed-runs/...` and the UOA / UCX
   worktrees. Never edit a frozen record; map paths in a new index if needed.
5. **Vendored copies.** `vendor/uoa_20261014_v1/` holds byte copies of UOA files and of UOA's own vendored UCX /
   Uformer / UPP files. ruff excludes `vendor/` on purpose.
6. **Stale editable install.** Every ULR process ran with PYTHONPATH unset, refused `cg_fedllm` and checked that every
   repository module came from this package.
7. **Post-run scripts.** `report_tables.py`, `make_figures.py`, `write_inventory.py` and `verify_manifest_after.py` are
   left as executed; run `ruff check` / `ruff format --check` on them before any commit. The two verification
   scripts are formatted and unchanged since before the launch.

## Verification commands (actual; PYTHONPATH unset)

- **Pre-freeze checks.** `powershell -NoProfile -File checks/run_all_checks.ps1 <tag>` from the package directory.
  - It runs ruff, the units (11), the lifecycle tests (6, real subprocesses on a synthetic fixture) and the
    UPP-20261014-v1 policy tests (118, in place and read-only in the UCX worktree).
  - `final1_20261015`: all exit 0.
- **Pre-launch.** `python -B prelaunch_verify.py 8db0e72f...0bb6`, giving all_ok.
- **Post-run, from this directory** (each was run once):
  - `python -B ../ulr_20261015_v1/ulr_analyze.py analysis`;
  - `python -B independent_verify.py independent_verification.json` (OPENBLAS_NUM_THREADS=1);
  - `python -B replay_final_states.py replay_final_states.json`;
  - `python -B verify_manifest_after.py`;
  - the protected-work fingerprint and `--compare`.

  The outputs are listed in `REPORT.md` section 9.

## Not done

Commit, push, publication, SSBD, FAF, LLM, GPU, DP, a fourth arm, another seed, another bottleneck family, QAT, and
downstream experiments.
