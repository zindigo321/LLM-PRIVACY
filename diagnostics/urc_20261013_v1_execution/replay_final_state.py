"""Read-only replay of the saved R-FIT final state: forward passes only, no optimizer, no update, no training.

``python -B replay_final_state.py`` writes ``replay_final_state.json`` (never overwritten) in this directory. With the
registered CPU settings (frozen ``configure_cpu``, 24/24 threads, deterministic algorithms), it rebuilds the registered
input with the frozen ``make_input`` and loads ``final_state.safetensors`` into two fresh models, as the child's final
evaluation did. It then computes eval-mode and train-mode metrics with the frozen ``uid_common.metrics`` and compares
every recorded sum exactly. This checks that the saved state is the evaluated state; it is not a new experiment.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
URC = HERE.parent / "urc_20261013_v1"
RUN = Path("D:/cgfed-runs/urc_20261013_v1/arms/R-FIT")
sys.path[:0] = [str(URC), str(URC.parent / "uid_20261012_v2"), str(URC.parent / "uid_20261012_v2_r1")]
KEYS = ("n", "sse", "target_energy", "pred_energy", "dot", "rse", "cosine", "mse", "max_abs_err", "finite")


def main() -> int:
    out_path = HERE / "replay_final_state.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    resolved = json.loads((URC / "resolved_config.json").read_text(encoding="utf-8"))
    import uid_common as C

    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import torch
    import urc_control as U

    rec = json.loads((RUN / "child_record.json").read_text(encoding="utf-8"))
    x = C.make_input(resolved)
    report = {"cpu": cpu, "input_sha256_equal": C.describe_input(x)["sha256"] == rec["input"]["sha256"], "modes": {}}
    for mode in ("eval", "train"):
        model, meta = U.load_state(RUN / "final_state.safetensors", resolved)  # strict load + state-hash check
        model.train(mode == "train")
        with torch.no_grad():
            m = C.metrics(C.reconstruct(model, x), x)
        r = rec["final"][f"{mode}_mode"]
        report["modes"][mode] = {
            "replayed": {k: m[k] for k in KEYS},
            "recorded": {k: r[k] for k in KEYS},
            "bit_identical": all(m[k] == r[k] for k in KEYS),
            "differences": {k: [m[k], r[k]] for k in KEYS if m[k] != r[k]},
        }
    report["optimizer_created"] = False
    report["all_bit_identical"] = report["input_sha256_equal"] and all(
        v["bit_identical"] for v in report["modes"].values()
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with out_path.open("x", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0 if report["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
