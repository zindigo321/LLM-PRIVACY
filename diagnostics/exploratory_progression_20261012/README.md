# Exploratory progression analysis (2026-10-12)

This directory holds derived analysis of the closed UID-20261012-v2 r1 diagnostic and of the archived P5-A v2 R3
control. It changes no frozen package, protocol, manifest, record or status. No training was run.

| File | Content |
|---|---|
| `POLICY.md` | Exploratory progression policy EP-20261012-v1 and its rationale. It was selected after the U-FIT result was observed (disclosed in the policy). |
| `ep_classify.py` | The classifier (`classify`) and the extraction of U-FIT records. Extraction re-hashes all 42 entries of the r1 final hash list and the approved manifest before any value is used. |
| `test_ep_classify.py` | Targeted checks: threshold boundaries at one ulp, undefined and non-finite metrics, the zero baseline, outcome order, evidence tampering. |
| `assessment_uid_r1_ufit.json` | Machine-readable assessment written by `ep_classify.py`: EP outcome **PASS**, original status **COMPLETED_FIT_FAIL** (unchanged). Console output: `ep_classify.out.txt`. |
| `compare_records.py`, `comparison.json` | Derived numbers for both runs, from recorded sums after hash checks. Console output: `compare_records.out.txt`. |
| `COMPARISON.md` | Comparison report: conditions, metrics against each run's own baselines, why the comparison is not matched. |
| `NEXT_EXPERIMENT_SPEC.md` | Draft URC-v1: one CPU stride-1 ResNet-3 arm at the exact U-FIT conditions. Not frozen, not approved, not run. |
| `CONTINUATION_CHECKPOINT.md` | Working notes for resuming; not evidence. |

To reproduce, run from this directory with `D:/conda_envs/cgfedllm/python.exe` and the CPU environment
(`CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`):

```text
python -B -m pytest -q -p no:cacheprovider test_ep_classify.py
python -B ep_classify.py --out <new file>.json
python -B compare_records.py --out <new file>.json
```

The scripts never overwrite an output file. They import no model code: `compare_records.py` reads only the R3
checkpoint's safetensors header, after checking its hash.
