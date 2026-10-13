# URP-20261013-v1 pre-launch check (2026-10-13, 16:49 +08:00)

Result: **PASS**. All 15 checks in `prelaunch_verify.out.txt` (11,648 bytes, SHA-256
`50f05e3a1b393bf5f47f998606147199ce2b9ed0380bcf1c3d870fb8225c10a1`) hold, and nothing found would prevent compliant
execution. The implementer did this check; it is not an independent acceptance.

| Check | Result | Basis |
|---|---|---|
| Repository | `D:/projects-cd/cg-fedllm-reproduction`, branch `phase5-ssbd-r3-single`, HEAD `0ad2795c7d03f6d0ea76992ee83864a00efd6b03`. Status unchanged from the session start: 3 modified and 15 untracked SSBD paths, `.claude/`, `diagnostics/`. This directory is new and untracked inside `diagnostics/` | `git status --porcelain=v1` |
| Manifest | `1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1`, 16,432 bytes, equal to the approved value | Python `hashlib`, `sha256sum`, `certutil`: three tools |
| Manifest entries | 61/61 match in SHA-256 and size. Unlisted: `CONTINUATION_CHECKPOINT.md`, `PROPOSED_LAUNCH.md` and the manifest itself, as at freeze | `prelaunch_verify.py` |
| External bindings | 4/4 match: the U-FIT `child_record.json`, the R3 `index.jsonl`, the end and start snapshot files | same |
| Pinned files and identity | `check_pinned` and `verify_package` pass: frozen UID config (raw `d7ae7605…b0bd`, canonical `bb0111fa…e5b3`), UID manifest, `manifest_r1.json`, `execution_r1.json`, the U-FIT record and its initial state hash `7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7`; EP `POLICY.md` and `ep_classify.py` | `urp_control` (the supervisor's own functions) |
| Resolved config | A fresh resolution equals `resolved_config.json` (`7f17fbae708d00b43daab7b3ac3645f13cd893be0e279abbd986d89854225e8b`). The child's scientific config branch accepts it | `urp_child.load_job_config` |
| Plan | One U-PATCH start, 3,000 updates, 1,800 s; 1 invocation; 7,200 s aggregate; CPU 24/24; soft stop 1,720 s and kill 1,780 s after the arm start; run root `D:/cgfed-runs/urp_20261013_v1`; mirror `diagnostics/urp_20261013_v1/scientific_ledger_mirror` | `registered_plan`, `arm_deadlines` |
| Imports | On the child's and the supervisor's `sys.path`, all 7 runtime modules resolve to the manifest-listed files with the approved SHA-256. Nothing was imported to find this | `importlib.machinery.PathFinder` |
| Stale editable install | `cg_fedllm` would still resolve to `.claude/worktrees/p5a-engineering-support/src` through the `__editable__` `.pth`. The child must not import it: it is in `FORBIDDEN_MODULES`, and any unlisted repository module refuses the run. So PYTHONPATH stays **unset**: the supervisor refuses it and the child checks that it is absent | `prelaunch_verify.out.txt` `imports` |
| Dependencies | torch 2.14.0+cu130 (from the env's site-packages), safetensors 0.8.0, psutil 7.2.2, numpy 2.4.6, Python 3.11.16: equal to the manifest. CUDA is unavailable with `CUDA_VISIBLE_DEVICES=-1` | distribution metadata, `torch.__version__` |
| Interpreter and environment | `D:\conda_envs\cgfedllm\python.exe`, equal to the plan, run with `-B`. `PYTHONPATH` is unset in the process, user and machine scopes. `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1`. No `__pycache__` or `.pyc` in the package or the closed directories it imports from, before or after | same |
| Ledger (read-only) | 0 invocations, 0 starts, U-PATCH NOT_STARTED, no records. Neither the run root nor the mirror exists | `K.ledger_status` |
| Running workers | No Python process at 16:16 +08:00 | `Get-CimInstance Win32_Process` |
| Memory, disk, CPU | 9,176,199,168 B available of 34,070,192,128 B. That puts the RSS limit at launch at 4,294,967,296 B, since 50% of available is more than 4 GiB. D: has 189,267,697,664 B free. The i9-14900HX has 24 physical and 32 logical CPUs; load was 16% at 16:16 | `psutil`, `Win32_Processor` |

## Launch method (one targeted engineering check)

- **Concern.** The expected run time, about 8 to 9 minutes, is close to the 10-minute limit of one tool call. An
  interrupted chat or tool call must neither stop the bounded run nor cause a second launch.
- **Check.** `process_detach_check.ps1` used no package file, model or run root; its output is
  `process_detach_check.txt`. The tool shell runs in a job with limit flags 0x800 (breakaway allowed, no
  kill-on-close). A PowerShell started from it with `Start-Process -WindowStyle Hidden` was still running 25 s after
  the tool shell had exited, and it exited on its own.
- **Choice.** `launch_scientific.ps1` is started that way. It runs the `PROPOSED_LAUNCH.md` commands unchanged, as the
  URC launch did. The supervisor still owns its child process tree and its RSS and deadline guards. A second launch
  is refused twice over: by the wrapper, because `launch/` exists, and by the ledger.

Not repeated, because recorded preparation evidence already covers them (`evidence/final2_20261013/`, all 8 steps exit
0): the input build, the registered-binding probe and the package test suites. No concrete unresolved concern called
for a new engineering check.
