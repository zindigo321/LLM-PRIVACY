# URC-20261013-v1 continuation checkpoint (working notes; not evidence; outside the manifest)

Authorized 2026-10-13: implement, run targeted engineering checks, freeze. NOT authorized: scientific execution.

## Design (fixed)

- Directory `diagnostics/urc_20261013_v1/`; protocol id `URC-20261013-v1`; single arm `R-FIT`.
- Imported unchanged (hash-pinned): frozen `uid_common.py`, r1 `uid_r1_control.py`, r1 `uid_r1_measure.py`.
- Copied from r1 with reviewed diff: `urc_supervise.py` (from uid_r1_supervise.py), `urc_child.py` (from uid_r1_child.py).
- New: `resnet3_s1.py` (model), `urc_control.py` (identity, resolution, one-arm plan, build, state files),
  `urc_analyze.py` (post-run seal + analysis: EP, strict, matching, signed deltas), `urc_config.json`,
  `resolved_config.json` (built by checks/build_resolved_config.py), `protocol.md`, checks/, review/.
- Shared settings are read from the frozen UID config at resolution time; mechanics from execution_r1.json; evidence
  cap / RSS fraction / curve cadence from uid_r1_control constants. Aggregate 7,200 s (established; replaces the
  draft spec's 3,600 s).
- Run root `D:/cgfed-runs/urc_20261013_v1`; mirror `diagnostics/urc_20261013_v1/scientific_ledger_mirror`.

## Status

- [x] copied r1 supervisor/child (byte-identical copies before edits)
- [x] resnet3_s1.py (493,483 params, 1,941 buffer elements, latent [64,32,32], 0 stride-2 layers; ruff clean)
- [x] urc_config.json (all pinned hashes match disk)
- [x] urc_control.py (resolve, one-arm plan, verify_package, build_model, save/load_state)
- [x] urc_child.py edited (resolved-config loader, U.build_model, reference_checks before optimizer, sampling-init
  hashes, init/final state save + read-back, module allowlist + forbidden uformer/cg_fedllm + repo-local coverage)
- [x] urc_supervise.py edited (URC_DIR, U.check_plan in invoke, --approved-urc-manifest-sha256, status via urc_config,
  sys_path URC→frozen→r1, expected_config_sha256 in job)
- [x] checks/build_resolved_config.py; dry resolve OK (seed 1, Adam 2e-4 full, 3000, 1800 s, 24/24, aggregate 7200)
- [x] urc_analyze.py (seal, strict re-derivation, EP classify, MATCHED/NOT_MATCHED, signed deltas, resources;
  torch-free; ruff clean)
- [x] checks/fixture_config.json + fixture_run.py: smoke run on fixture OK (6 updates, COMPLETED_FIT_FAIL, states
  saved and hash-equal, coverage local = 5 listed modules). Fixed coverage false positive (torch.ops, __mp_main__).
- [x] checks/test_urc_model.py: 13 passed (HEAD blob loaded via git; stride-2 rebuild bit-equal to HEAD fwd).
- [x] resolved_config.json written (sha256 5db03929eb17c0c49dabd26d894b144cfa85e470bc9188549a87402886466608)
- [x] check_reference_input: all 29 comparisons equal, U-FIT record hash OK (no model built)
- [x] check_init_comparison: U-FIT fresh build reproduces recorded init hash; encoder.sampling 21/21 params + 21/21
  buffers equal; decoder.sampling 12/20 params equal (8 conv/convT weights + head bias differ; all BN equal), 18/18 buffers
- [x] test_urc_control: 25 passed. No URC run root / mirror exists.
- [x] test_urc_lifecycle: 7 passed (29 s)  - [x] test_urc_analyze: 2 passed (analyze() takes test-only overrides)
- [x] checks/run_all_checks.ps1 written
- [x] protocol.md written; review/child_diff.patch + supervise_diff.patch (regenerate after last code edit!)
- [x] urc_analyze: also compares the two invocation plans (cpu, budgets, guards, cadence, curve schedule) and
  verifies U-FIT files against the r1 final hash list; test_urc_analyze 2 passed
- [x] review/MATCHING.md  - [x] checks/build_manifest_urc.py  - [x] checks/probe_cpu_cost.py (engineering seeds:
  0.0177 s/fwd+bwd, ~53 s projected for 3,000 w/o Adam; bit-identical)
- [x] diffs regenerated after last code edit (09:10)  - [x] final checks: evidence/final_20261013, all 10 exit 0
  (47 tests; ref all_equal; init 21/21+21/21 enc, 12/20+18/18 dec; status 0 starts, no run root, no mirror)
- [x] manifest_urc.json built LAST: SHA-256 b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6
  (10,791 B, 43 files); read back: 43/43 entries match; only this checkpoint + the manifest are unlisted;
  U.verify_package(hash) OK; plan = one R-FIT start.
- [x] untouched evidence re-verified: UID final artifacts 42/42, manifest_r1 29/29, frozen UID 7/7, EP bytes intact;
  git working-tree diff identical to session start.
- [x] PROPOSED_LAUNCH.md (outside the manifest).
- FROZEN. NO EDITS to manifest-covered files. Scientific execution NOT authorized; nothing launched.
