"""ENGINEERING CHECK: sampling-layer initial values of R-FIT vs U-FIT, from fresh disposable models.

Builds each architecture once, exactly as its arm builds it (R-FIT: ``urc_control.build_model`` from the resolved
config; U-FIT: the frozen ``uid_common.build_model(cfg, "U-FIT")``, read-only, in this check process only), under the
registered model seed. No forward pass, no optimizer, no update, no weight copying: only the constructed tensors are
hashed (``uid_common.tensor_sha256``, name-independent here: the layout name is passed as ""), per corresponding layer.

Also verified against the record: the fresh U-FIT build reproduces the recorded initial state hash
``model_state_sha256_init`` of the U-FIT child record; the fresh R-FIT build is reported with its state hash.

``python -B checks/check_init_comparison.py``; prints JSON. Exit 0 when the check ran and the U-FIT build reproduces
its record; equality or inequality of the sampling layers is an observation, never an exit condition.
"""

import json
import sys
from pathlib import Path

URC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URC))

import urc_control as U  # noqa: E402

K = U.K


def layer_hashes(stack, C) -> list[dict]:
    """Per state tensor of a sampling stack: index-path, shape, dtype, layout-only hash (no name)."""
    return [
        {"key": k, "shape": list(v.shape), "dtype": str(v.dtype), "sha256": C.tensor_sha256(v, "")}
        for k, v in stack.state_dict().items()
    ]


def main() -> int:
    resolved = U.load_resolved()
    U.check_pinned()
    import uid_common as C

    cpu = resolved["cpu"]
    C.configure_cpu(cpu["intra_op_threads"], cpu["inter_op_threads"])
    uid_cfg = K.load_json(URC / resolved["reference"]["frozen_config"])
    ref = K.load_json(resolved["reference"]["child_record"])
    rfit = U.build_model(resolved)  # registered seed 1, fork_rng, PyTorch default init
    ufit = C.build_model(uid_cfg, "U-FIT")  # the frozen build, read-only
    import torch

    report = {
        "seed": resolved["model_init_seed"],
        "procedure": "fresh models; no forward, no optimizer, no update, no weight copying",
        "r_fit_state_sha256_init": C.state_sha256(rfit),
        "u_fit_state_sha256_init": C.state_sha256(ufit),
        "u_fit_recorded_state_sha256_init": ref["model_state_sha256_init"],
        "rng_state_untouched_after_builds": True,
        "stacks": {},
    }
    report["u_fit_build_reproduces_record"] = (
        report["u_fit_state_sha256_init"] == ref["model_state_sha256_init"]
    )
    rng0 = torch.get_rng_state()
    for side in ("encoder", "decoder"):
        a = layer_hashes(getattr(rfit, side).sampling, C)
        b = layer_hashes(getattr(ufit, side).sampling, C)
        rows = []
        for ra, rb in zip(a, b, strict=True):
            same_layout = (ra["key"], ra["shape"], ra["dtype"]) == (rb["key"], rb["shape"], rb["dtype"])
            rows.append(
                {
                    "key": ra["key"],
                    "shape": ra["shape"],
                    "dtype": ra["dtype"],
                    "layout_matches": same_layout,
                    "r_fit_sha256": ra["sha256"],
                    "u_fit_sha256": rb["sha256"],
                    "values_equal": same_layout and ra["sha256"] == rb["sha256"],
                    "kind": "buffer" if ra["key"].split(".")[-1] in BUFFERS else "parameter",
                }
            )
        report["stacks"][f"{side}.sampling"] = {
            "tensors": len(rows),
            "layouts_match": all(r["layout_matches"] for r in rows),
            "parameters_equal": sum(r["values_equal"] for r in rows if r["kind"] == "parameter"),
            "parameters_total": sum(r["kind"] == "parameter" for r in rows),
            "buffers_equal": sum(r["values_equal"] for r in rows if r["kind"] == "buffer"),
            "buffers_total": sum(r["kind"] == "buffer" for r in rows),
            "rows": rows,
        }
    report["rng_state_untouched_after_builds"] = torch.equal(rng0, torch.get_rng_state())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["u_fit_build_reproduces_record"] else 1


BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


if __name__ == "__main__":
    sys.exit(main())
