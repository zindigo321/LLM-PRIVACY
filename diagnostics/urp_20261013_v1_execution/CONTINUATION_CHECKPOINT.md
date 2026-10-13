# URP-20261013-v1 execution checkpoint (working notes; not evidence)

Authorization: ONE invocation and ONE U-PATCH start, manifest `1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1`
(`EXECUTION_AUTHORIZATION.md`). **Both are consumed. URP-20261013-v1 is CLOSED: never relaunch, retry, resume or add
a start under it.**

## What happened (verified from records)

- 16:34 +08:00: this directory created; authorization recorded.
- 16:49: pre-launch check PASS (`prelaunch_check.md`, `prelaunch_verify.out.txt` `50f05e3a…10a1`).
- 16:55: detach check (`process_detach_check.*`).
- **17:03:38–17:12:33: the ONE launch** (`launch/`). Scientific command 17:03:41.913–17:12:30.548, exit 0.
  - Recorded outcome: COMPLETED_FIT_FAIL; 3,000 of 3,000 updates; no guard action; not interrupted.
  - Ledger: 1 invocation, 1 start; the mirror is byte-identical.
- 17:36:42–17:36:44: frozen `urp_analyze.py` → `analysis/`, exit 0.
  - Evidence verified; strict COMPLETED_FIT_FAIL; EP-20261012-v1 PASS.
  - `analysis.json` SHA-256 `a4721424aaf4539daf55c8498344da6e0efbe2679c96d610780e2e0600722d5f`.
- `independent_verify.py` → `independent_verification.json`: all checks hold.
- `replay_final_state.py`: attempt 1 failed in its own module bookkeeping, with no record written; it is kept with a
  note. Attempt 2, 19:01:35–19:01:39, was bit-identical in both modes and both spaces, with no optimizer.
- `make_curve_figure.py` → `learning_curve_u_patch.svg` and `curve_u_patch.csv`.
- `REPORT.md`, checked by `verify_report_numbers.py`: 0 problems.
- `NEXT_EXPERIMENT_PROPOSAL.md`, `EVIDENCE_INVENTORY_SHA256.txt` (written last).

## Result

Final eval, original space:
- RSE 0.19345271823265606 and cosine 0.8981045580444489;
- MSE 7.99371986840454e-07, against the scalar-mean baseline 4.131472976536517e-06, a ratio of 0.1935.

Strict: FIT_FAIL. EP: PASS. Saved states:
- init `7a36411b…f6f7`;
- final `a3f1bf91f46faf09b0f461bf9641d4cf87a2748235f9c37abcc8a2d9fa907d3b`;
- both in `D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH/`.

## Next permissible action

Human review of `REPORT.md` and `NEXT_EXPERIMENT_PROPOSAL.md`. Any further scientific run, including the proposed
four-tile coverage study, needs a new protocol, a frozen package and a separate written approval of its manifest
hash. Nothing has been committed or pushed, and SSBD remains paused.
