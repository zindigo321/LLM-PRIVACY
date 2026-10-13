# URC-20261013-v1 pre-launch check (2026-10-13, about 09:30 +08:00)

Result: **PASS**. No mismatch that would prevent compliant execution was found. The implementer did this check; it is
not an independent acceptance.

| Check | Result | Basis |
|---|---|---|
| Manifest hash | `b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6`, equal to the approved value | `sha256sum`, `certutil`, Python `hashlib` (three independent tools) |
| Manifest entries | 43/43 match in SHA-256 and size; only `manifest_urc.json`, `PROPOSED_LAUNCH.md` and `CONTINUATION_CHECKPOINT.md` are outside it | `prelaunch_verify.out.txt` |
| Pinned dependencies | frozen `uid_common.py`, r1 `uid_r1_control.py` / `uid_r1_measure.py`, EP `ep_classify.py` / `POLICY.md` match | `urc_control.check_pinned` |
| Reference identities | frozen UID config (raw and canonical), UID manifest, `manifest_r1.json`, `execution_r1.json` and the U-FIT `child_record.json` match their pinned hashes | `urc_control.verify_package` |
| Resolved config | a fresh resolution equals `resolved_config.json` (`5db03929eb17c0c49dabd26d894b144cfa85e470bc9188549a87402886466608`); the child's scientific config branch accepts it | `prelaunch_verify.py` |
| Plan | exactly one R-FIT start, 3,000 updates, 1,800 s; 1 invocation; 7,200 s aggregate; CPU 24/24; run root `D:/cgfed-runs/urc_20261013_v1`; mirror `diagnostics/urc_20261013_v1/scientific_ledger_mirror` | `urc_control.registered_plan` |
| Imports | the child's six runtime modules are listed with their manifest hashes and checked again in the child | `expected_module_hashes` |
| Interpreter and environment | `D:\conda_envs\cgfedllm\python.exe` 3.11.16, equal to the plan; `PYTHONPATH` unset; `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` | `prelaunch_verify.out.txt` |
| Ledger (read-only `--status`) | invocations 0, starts 0, R-FIT NOT_STARTED; run root and mirror do not exist | 09:2x |
| Running workers | no Python process, and no process whose command line names urc_/uid_r1/uid_runner/ssbd/cgfed | `Get-CimInstance Win32_Process` |
| Memory and disk | 11,495,305,216 B available, so the RSS limit at launch is 4,294,967,296 B; 191 GB free on D: | `prelaunch_verify.out.txt` |
