# URC-20261013-v1: proposed scientific launch (NOT executed; needs a separate written approval of this hash)

Object of approval: `manifest_urc.json`, SHA-256
`b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6` (10,791 bytes, 43 files).
This file is outside the manifest because it cites the manifest's hash.

## Commands (PowerShell)

```powershell
Set-Location D:\projects-cd\cg-fedllm-reproduction\diagnostics\urc_20261013_v1
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue   # the supervisor refuses to run while PYTHONPATH is set
$env:CUDA_VISIBLE_DEVICES = "-1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTHONDONTWRITEBYTECODE = "1"
# read-only status (safe at any time; creates nothing)
& "D:/conda_envs/cgfedllm/python.exe" -B urc_supervise.py --status
# the one scientific invocation (only after approval)
& "D:/conda_envs/cgfedllm/python.exe" -B urc_supervise.py --scientific `
    --approved-urc-manifest-sha256 b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6
# after the invocation has ended: seal and analyze (read-only on the run roots; writes only the new directory)
& "D:/conda_envs/cgfedllm/python.exe" -B urc_analyze.py `
    --evidence-dir D:\projects-cd\cg-fedllm-reproduction\diagnostics\urc_20261013_v1_execution
```

- **What runs**: one invocation, one start, the single arm R-FIT (3,000 updates, 1,800 s arm limit, 7,200 s
  aggregate, RSS min(4 GiB, 50% of available RAM), 512 MiB of evidence). No retry and no resume. U-CHECK and U-FIT
  cannot be selected.
- **Before training**, the run refuses with EXECUTION_ERROR if any of these fails:
  - any manifest entry, pinned file, or the fresh re-resolution of the shared settings;
  - the input or baseline equality with U-FIT;
  - CPU fp32, the thread counts, determinism or the torch version;
  - an imported module's hash, or an unlisted repository module.
- **Output**: `D:/cgfed-runs/urc_20261013_v1/`:
  - `ledger/`;
  - `arms/R-FIT/`, holding `job.json`, `child_record.json`, `curve.jsonl`, `init_state.safetensors`,
    `final_state.safetensors` and the child's stdout and stderr.
  - The ledger mirror is `diagnostics/urc_20261013_v1/scientific_ledger_mirror/`.
- **Exit codes**:
  - 0: R-FIT ended COMPLETED_PASS or COMPLETED_FIT_FAIL;
  - 1: RESOURCE_STOP, EXECUTION_ERROR or INCOMPLETE;
  - 2: refused before any record (identity mismatch, ledger state, PYTHONPATH set, and similar).
- **Expected duration**: about 1–2 minutes. This is an engineering estimate from forward+backward probes with
  engineering seeds (about 0.018–0.020 s per pass, without the Adam step and measurement overhead). It is not a
  guarantee.

## State at freeze (2026-10-13, about 09:12 +08:00)

- `--status`: invocations 0, starts 0, R-FIT NOT_STARTED. The run root and ledger mirror do not exist.
- No registered URC scientific start has occurred.
- Every optimizer update so far was made on the 16×16 fixture config, with seeds 7 and 4242, in temporary
  directories.
