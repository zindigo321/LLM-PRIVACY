# URP-20261013-v1: proposed scientific launch (NOT executed; needs a separate written approval of this hash)

Object of approval: `manifest_urp.json`, SHA-256
`1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1` (16,432 bytes, 61 files plus 4 external files
bound by hash). This file is outside the manifest because it cites the manifest's hash.

## Commands (PowerShell)

```powershell
Set-Location D:\projects-cd\cg-fedllm-reproduction\diagnostics\urp_20261013_v1
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue   # the supervisor refuses to run while PYTHONPATH is set
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
# read-only status (safe at any time; creates nothing)
& "D:/conda_envs/cgfedllm/python.exe" -B urp_supervise.py --status
# the one scientific invocation (only after approval)
& "D:/conda_envs/cgfedllm/python.exe" -B urp_supervise.py --scientific `
    --approved-urp-manifest-sha256 1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1
# after the invocation has ended: seal and analyze (read-only on the run roots; writes only the new directory)
& "D:/conda_envs/cgfedllm/python.exe" -B urp_analyze.py `
    --evidence-dir D:\projects-cd\cg-fedllm-reproduction\diagnostics\urp_20261013_v1_execution\analysis
```

## What runs

- One invocation and one start of the single arm U-PATCH. There is no retry and no resume.
- The model is the frozen U-FIT Uformer, seed 1, with the U-FIT initial state hash.
- The input is the registered patch `Phi[0, 1024:1056, 768:800]` of R3 t7 c91, normalized with the frozen R3 scale.
- CPU fp32, batch 1, 24/24 threads, deterministic algorithms.
- Adam with lr 2e-4, betas (0.9, 0.999), eps 1e-8 and weight decay 0, for 3,000 updates.
- Limits:
  - arm: 1,800 s;
  - aggregate: 7,200 s;
  - RSS: min(4 GiB, 50% of available RAM);
  - retained evidence: 512 MiB.
- U-CHECK, U-FIT and R-FIT cannot be selected.

## Refusal before training (EXECUTION_ERROR, no update)

The run refuses before the first update if any of these checks fails:
- any manifest entry, pinned file, or the fresh re-resolution of the settings and input;
- the input file hash, both tensor hashes, the energies, the baselines in both spaces, or max |x_n| < 1;
- the initial state, which must equal the U-FIT record;
- CPU fp32, the thread counts, determinism or the torch version;
- an imported module's hash, or an unlisted repository module (`cg_fedllm` is forbidden).

## Output and exit codes

- **Output.** Everything goes under `D:/cgfed-runs/urp_20261013_v1/`:
  - `ledger/`;
  - `arms/U-PATCH/`, holding `job.json`, `child_record.json`, `curve.jsonl`, `init_state.safetensors`,
    `final_state.safetensors` and the child's stdout and stderr.

  The ledger mirror is `diagnostics/urp_20261013_v1/scientific_ledger_mirror/`.
- **Exit codes.**
  - 0: U-PATCH ended COMPLETED_PASS or COMPLETED_FIT_FAIL.
  - 1: RESOURCE_STOP, EXECUTION_ERROR or INCOMPLETE.
  - 2: refused before any record (identity mismatch, ledger state, PYTHONPATH set, and similar).

## Expected duration

About 8–9 minutes. U-FIT, the same model at the same input shape, threads and budget, took 506.3 s of arm wall time on
this machine on 2026-10-12. This is an expectation, not a guarantee: background load is not controlled.

## State at freeze (2026-10-13, 13:57:43 +08:00)

- `--status` at 13:58: invocations 0, starts 0, U-PATCH NOT_STARTED. Neither the run root nor the ledger mirror
  exists.
- No registered URP scientific start has occurred.
- Every optimizer update so far ran on the fixture config: the toy 16×16 patch `checks/fixture_patch.safetensors`,
  model seed 7 and the small r1 fixture Uformer, in temporary directories.
- On the registered model with the registered patch, the only operation was one forward pass on a disposable copy in
  `check_registered_binding.py`. No optimizer was created there, and no metric of that output was computed.
