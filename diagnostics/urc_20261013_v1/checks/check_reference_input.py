"""ENGINEERING CHECK: the registered R-FIT input and baselines equal the U-FIT reference, before any run.

Builds the input with the frozen ``uid_common.make_input`` from the resolved configuration (the registered data seed
and settings, read from the frozen UID config) and computes both baselines with the frozen ``uid_common.baselines``.
No model is built, no forward pass and no optimizer are used. The comparison is exact (no tolerance) against the U-FIT
``child_record.json`` (hash-checked) and against the expected values in ``resolved_config.json``.

``python -B checks/check_reference_input.py``; prints a JSON report; exit 0 iff everything is equal.
"""

import json
import sys
from pathlib import Path

URC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URC))

import urc_control as U  # noqa: E402

K = U.K


def main() -> int:
    resolved = U.load_resolved()
    cpu = U.load_resolved()["cpu"]
    import uid_common as C  # frozen, hash-pinned

    U.check_pinned()
    C.configure_cpu(cpu["intra_op_threads"], cpu["inter_op_threads"])  # the registered CPU settings
    ref_path = Path(resolved["reference"]["child_record"])
    ref_ok = K.sha256_file(ref_path) == resolved["reference"]["child_record_sha256"]
    ref = K.load_json(ref_path)
    x = C.make_input(resolved)
    inp, base = C.describe_input(x), C.baselines(x)
    rows = {}
    for key in ("sha256", "shape", "dtype", "numel", "min", "max", "mean", "energy_sum_sq", "z_n"):
        rows[f"input.{key}"] = (inp[key], ref["input"][key])
    for b in ("zero_prediction", "shared_scalar_mean"):
        for key in ("sse", "target_energy", "pred_energy", "dot", "mse", "rse", "cosine", "n"):
            rows[f"baselines.{b}.{key}"] = (base[b][key], ref["baselines"][b][key])
    exp = resolved["expected"]
    rows["expected.input_sha256"] = (inp["sha256"], exp["input_sha256"])
    rows["expected.target_energy"] = (inp["energy_sum_sq"], exp["target_energy"])
    rows["expected.zero_prediction_mse"] = (base["zero_prediction"]["mse"], exp["zero_prediction_mse"])
    rows["expected.shared_scalar_mean_mse"] = (
        base["shared_scalar_mean"]["mse"],
        exp["shared_scalar_mean_mse"],
    )
    report = {
        "reference_child_record_hash_ok": ref_ok,
        "comparisons": {
            k: {"r_fit": a, "u_fit_or_expected": b, "equal": a == b} for k, (a, b) in rows.items()
        },
        "torch_modules_used": "uid_common.make_input, describe_input, baselines (no model, no optimizer)",
    }
    report["all_equal"] = ref_ok and all(v["equal"] for v in report["comparisons"].values())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
