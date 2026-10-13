# URP-20261013-v1 continuation checkpoint (working notes; not evidence; outside the manifest)

Task: PREPARE one real-patch Uformer diagnostic (authorized: preparation, input construction from the existing R3
snapshot, targeted engineering checks, freeze). NOT authorized: scientific training, GPU, SSBD, FAF, extra arms,
commits or pushes.

## Recovery (2026-10-13 10:45 +08:00)

- The previous turn ended with "API Error: Upstream response stream was interrupted". Its tool calls were reads only.
- Observed at 10:45: no URP directory, no file under `diagnostics/` newer than
  `urc_20261013_v1_execution/EVIDENCE_INVENTORY_SHA256.txt` (09:42), no new run root in `D:/cgfed-runs/`
  (only `uid_20261012_v2_r1`, `urc_20261013_v1` among diagnostics). Git status unchanged from session start.
- Found: URC `REPORT.md` cites analysis.json SHA-256 `728f7934…` = SHA-256 of the LF text printed by
  `urc_analyze.py`; the file on disk was written in text mode (CRLF), 12,735 bytes, SHA-256 `744ae6c4…` (as in
  `EVIDENCE_INVENTORY_SHA256.txt`). LF-normalized bytes (12,249) hash to `728f7934…`. Verified from bytes.
  Lesson for URP: write records in binary mode with LF and hash the file bytes.

## Design (fixed)

- Directory `diagnostics/urp_20261013_v1/`; protocol `URP-20261013-v1`; single arm `U-PATCH`; model = frozen
  `uid_common.build_model(cfg, "U-FIT")` (the U-FIT Uformer, seed 1), init state bound to the U-FIT record hash.
- Input: R3 t7 c91 (P5-A position 39), Phi via `balanced_effective_delta(s=2, rank=8)` + `layer_major_qkvo_AtB`,
  verified against the archived phi_sha256; patch rule fixed in `input/SELECTION_RULE.md` before any payload read.
- Normalization: frozen R3 `global_exact_maxabs_train` s_r = 0.015751632224572334 (fp32 cast, production arithmetic).
  Loss in normalized space; verdict (strict + EP) in original space; normalized-space sums reported.
- Runtime reuses frozen `uid_common`, `uformer_ae`, `uformer_blocks`, r1 `uid_r1_control`, `uid_r1_measure`;
  `urp_supervise.py` / `urp_child.py` / `urp_analyze.py` adapted from URC with reviewed diffs.
- Run root `D:/cgfed-runs/urp_20261013_v1`; mirror `diagnostics/urp_20261013_v1/scientific_ledger_mirror`.

## Status

- [x] 11:35 stage summary: `diagnostics/synthetic_comparison_summary_20261013/` (STAGE_SUMMARY.md, extract_summary.py,
  stage_summary.json f07039a5…7858; r1 42/42, URC 15/15, inventory 38/38 verified; U-FIT EP recomputed PASS).
- [x] 11:43:51 SELECTION_RULE.md written BEFORE any payload read (sha256 e0b1df83…362d).
- [x] checks/test_urp_input_rule.py: 10 passed (no payload).
- [x] 11:54:52 prepare_input.py (4ccef2ca…604b) run once, exit 0: Phi sha 485cbe5f… == archived, sig == archived;
  patch Phi[0,1024:1056,768:800] (layer 12 q/k), patch_x.safetensors 957704bf…75fd (4,552 B), input_record.json
  bec71fda…b6c1 (14,943 B); tensor sha orig 92707ff8…a98a, normalized 1d2f65aa…12bb; max|x_n| 0.8495; usable.
- [x] urp_child.py / urp_supervise.py created as byte copies of the URC files (then edited).
- [x] urp_control.py, urp_config.json, urp_child.py, urp_supervise.py, urp_analyze.py written; ruff clean.
  Dry resolve OK (seed 1, Adam 2e-4 full, 3000, 1800 s, 24/24, aggregate 7200, RSS 4 GiB/50 %, evidence 512 MiB).
- [x] checks/make_fixture_input.py -> checks/fixture_patch.safetensors (8244ff0f…ade9, seed 4242, 16x16, toy);
  checks/fixture_config.json (seed 7, r1 fixture Uformer, scale 0.0125); checks/fixture_run.py.
- [x] ~12:58 fixture smoke run in %TEMP%/urp_smoke1: COMPLETED_FIT_FAIL, 6 updates, states saved, modules local =
  6 listed. (No registered model or input used.)
- [x] tests written: test_urp_input_rule (10), test_urp_control (32), test_urp_lifecycle (6), test_urp_analyze (1);
  dev runs all pass (42 + 7). check_registered_binding.py: all_ok (init 7a36411b… == U-FIT record; latent
  [1,64,32,32]; output in Tanh range; no optimizer, no metric of the output).
- [x] resolved_config.json built (7f17fbae…5e8b) — regenerate only if urp_control/urp_config change.
- [x] no CPU probe: per-update cost is already measured for this exact model + shape (U-FIT 506.3 s / 3,000).
- [x] checks/run_all_checks.ps1 written.
- [x] protocol.md; review/child_diff.patch + supervise_diff.patch (URC -> URP, regenerated after the last code edit).
- [x] 13:48 final check run attempt 1: test_urp_lifecycle exit 1 (test-only import order; ModuleNotFoundError
  uid_common). Preserved as evidence/attempt1_failed_20261013/ with NOTE.txt. Test fixed; no runtime file changed.
- [x] 13:52 evidence/final2_20261013: all 8 steps exit 0 (10 + 32 + 6 + 1 tests; binding all_ok; status 0/0, no
  run root, no mirror). 13:54 ruff gate rerun after adding checks/build_manifest_urp.py (17 files, exit 0/0).
- [x] closed evidence re-verified 13:55: r1 list 42/42, URC sealed 15/15, URC inventory 38/38, frozen UID 7/7,
  r1 29/29, URC 43/43; EP bytes and P5-A record unchanged; git status unchanged outside diagnostics/.
- [x] DISCREPANCY found and handled: `diagnostics/uid_20261012_v2_r1/__pycache__/uid_r1_control.cpython-311.pyc`
  (26,915 B, sha256 7f0d0a1d…cf55) created 12:07:13 by my own command run without `-B` (the resolve probe at
  ~12:07). Not in any manifest or hash list; the source file and every listed entry are unchanged (29/29). Removed
  with its empty __pycache__ directory at 13:57.
- [x] 13:57:43 manifest_urp.json built LAST: SHA-256 1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1
  (16,432 B, 61 files + 4 external bound files). Read back: 61/61 hashes and sizes, 4/4 external, verify_package OK,
  only this checkpoint and the manifest unlisted, plan = one U-PATCH start.
- [x] PROPOSED_LAUNCH.md (outside the manifest). 13:58:20 `--status`: 0 invocations, 0 starts, NOT_STARTED, no run
  root, no mirror.
- FROZEN. NO EDITS to manifest-covered files. Scientific execution NOT authorized; nothing launched.

Next action (only after a separate written approval of the manifest hash): PROPOSED_LAUNCH.md commands.
