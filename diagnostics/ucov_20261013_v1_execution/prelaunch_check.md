# UCOV-20261013-v1 pre-launch check (2026-10-13, about 21:25 +08:00)

Result: **PASS**. All 16 checks in `prelaunch_verify.out.txt` (11,471 bytes, SHA-256
`74316f862dd6400bbf1b8cec28a235118a55128506c8bf86f9ae83cf3cf16f09`) hold. The implementer did this check; it is not an
independent acceptance.

| Check | Result |
|---|---|
| Manifest | `3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824`, 17,562 bytes, equal to the frozen digest |
| Entries and externals | 65/65 and 5/5 match in SHA-256 and size; only the manifest itself is unlisted |
| Identity | `check_pinned` and `verify_package` pass; a fresh resolution equals `resolved_config.json` (`3db88e97…6337`); the child's scientific config branch accepts it |
| Primary policy | UPP-20261013-v1, `upp_policy.json` `74b277b9…5770`, thresholds 0.50 / 0.65 / 0.80. The UID `criteria` are carried only as a secondary reference |
| Plan | U-TILE-1..4 in order, one start each (4), 3,000 updates and 1,800 s per arm, 1 invocation, 9,000 s aggregate, CPU 24/24, run root `D:/cgfed-runs/ucov_20261013_v1` |
| Imports | All 9 runtime modules resolve to the manifest files with their approved hashes; nothing was imported to find this. `cg_fedllm` would resolve to the stale editable worktree, so PYTHONPATH stays unset; it is forbidden in the child |
| Dependencies | torch 2.14.0+cu130, safetensors 0.8.0, psutil 7.2.2, numpy 2.4.6, Python 3.11.16; CUDA unavailable |
| Environment | PYTHONPATH unset, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`, `-B`; no bytecode before or after |
| Ledger | 0 invocations, 0 starts; neither the run root nor the mirror exists |
| Processes | No Python process running; CPU load 2% |

The launch method is the one that worked for URP-v1: a detached, hidden PowerShell wrapper
(`process_detach_check.*` in the URP-v1 execution directory).
