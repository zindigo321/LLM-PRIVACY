"""Read-only replay of the saved U-PATCH final state: forward passes only; no optimizer, no gradient, no update, no
BatchNorm recalibration, no parameter edit, nothing written to the run root.

``python -B replay_final_state.py`` from this directory, with PYTHONPATH unset. It writes ``replay_final_state.json``
here (exclusive, LF bytes). It follows the child's final evaluation (``urp_child.run_arm``) with the frozen code:
- CPU: the frozen ``configure_cpu`` with the registered 24/24 threads and deterministic algorithms, before any tensor;
- input: the frozen ``urp_control.load_input`` (patch bytes hashed, fp32 ``x / float32(s)``);
- model: two fresh models from ``urp_control.load_state`` (the frozen ``build_model`` plus a strict load of the file
  and a state-hash check), one per mode, each from the same CPU RNG state; the train-mode copy is a separate,
  disposable model, so its BatchNorm buffer updates cannot reach the eval-mode copy;
- metrics: the frozen ``uid_common.metrics`` in both spaces; every recorded sum is compared exactly (no tolerance).
This checks that the saved state is the evaluated state; it is not a new experiment.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
URP = HERE.parent / "urp_20261013_v1"
ARM = Path("D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH")
sys.path[:0] = [str(URP), str(URP.parent / "uid_20261012_v2"), str(URP.parent / "uid_20261012_v2_r1")]
KEYS = ("n", "sse", "target_energy", "pred_energy", "dot", "mse", "rse", "cosine", "norm_ratio", "max_abs_err",
        "finite")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(replayed: dict, recorded: dict) -> dict:
    diff = {k: [replayed[k], recorded[k]] for k in KEYS if replayed[k] != recorded[k]}
    return {"replayed": {k: replayed[k] for k in KEYS}, "bit_identical": not diff, "differences": diff}


def main() -> int:
    out_path = HERE / "replay_final_state.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    final_path = ARM / "final_state.safetensors"
    files_before = {p.name: sha(p) for p in sorted(ARM.iterdir()) if p.is_file()}
    import uid_common as C

    resolved = json.loads((URP / "resolved_config.json").read_text(encoding="utf-8"))
    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import torch
    import urp_control as U

    resolved = U.load_resolved()  # equal to a fresh resolution of the pinned sources
    rec = json.loads((ARM / "child_record.json").read_text(encoding="utf-8"))
    scale = float(resolved["input"]["normalization"]["scale_global"])
    x, x_n, patch_sha = U.load_input(resolved)
    report = {
        "recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": cpu,
        "input": {
            "patch_file_sha256_equal": patch_sha == rec["input"]["patch_file_sha256"],
            "original_sha256_equal": C.describe_input(x)["sha256"] == rec["input"]["original"]["sha256"],
            "normalized_sha256_equal": C.describe_input(x_n)["sha256"] == rec["input"]["normalized"]["sha256"],
        },
        "modes": {},
        "grad_enabled_during_forwards": False,
        "optimizer_created": False,
    }
    rng = torch.get_rng_state()
    outputs, states = {}, {}
    for mode in ("eval", "train"):  # the child's order; one fresh model per mode
        model, meta = U.load_state(final_path, resolved)  # strict load and state-hash check, read-only
        before = C.state_sha256(model)
        model.train(mode == "train")
        torch.set_rng_state(rng)
        with torch.no_grad():
            y_n = C.reconstruct(model, x_n)
        after = C.state_sha256(model)
        outputs[mode] = y_n
        states[mode] = {"loaded_state_sha256": before, "state_sha256_after_forward": after,
                        "state_changed_by_forward": after != before, "file_metadata_point": meta.get("point")}
        report["modes"][mode] = {
            "original": compare(C.metrics(U.denormalize(y_n, scale), x), rec["final"][f"{mode}_mode"]),
            "normalized": compare(C.metrics(y_n, x_n), rec["final"][f"{mode}_mode_normalized"]),
            **states[mode],
        }
    tme = float((outputs["train"].double() - outputs["eval"].double()).square().sum())
    report["train_minus_eval_output_sse_normalized"] = {
        "replayed": tme, "recorded": rec["final"]["train_minus_eval_output_sse_normalized"],
        "bit_identical": tme == rec["final"]["train_minus_eval_output_sse_normalized"]}
    report["loaded_state_equals_recorded_final"] = all(
        s["loaded_state_sha256"] == rec["model_state_sha256_final"] for s in states.values())
    report["eval_copy_unchanged_by_forward"] = not states["eval"]["state_changed_by_forward"]
    report["train_copy_changed_as_recorded"] = (
        states["train"]["state_changed_by_forward"] == rec["final"]["train_copy_state_changed_by_forward"])
    files_after = {p.name: sha(p) for p in sorted(ARM.iterdir()) if p.is_file()}
    report["run_root_arm_files_unchanged"] = files_before == files_after
    report["final_state_file_sha256"] = files_after["final_state.safetensors"]
    # the child's rule: absolute module files under the repository; relative pseudo-paths (torch._classes) skipped
    local = sorted(n for n, m in sys.modules.items() if isinstance(getattr(m, "__file__", None), str)
                   and Path(m.__file__).is_absolute()
                   and Path(m.__file__).resolve().is_relative_to(HERE.parents[1]) and n != "__main__")
    report["repository_modules_imported"] = local
    report["module_sha256"] = {n: sha(Path(sys.modules[n].__file__)) for n in local}
    report["cg_fedllm_imported"] = "cg_fedllm" in sys.modules
    report["all_bit_identical"] = (
        all(report["input"].values())
        and all(report["modes"][m][s]["bit_identical"] for m in ("eval", "train") for s in ("original", "normalized"))
        and report["train_minus_eval_output_sse_normalized"]["bit_identical"]
        and report["loaded_state_equals_recorded_final"] and report["eval_copy_unchanged_by_forward"]
        and report["train_copy_changed_as_recorded"] and report["run_root_arm_files_unchanged"]
        and not report["cg_fedllm_imported"])
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with out_path.open("xb") as fh:
        fh.write(data)
    print(data.decode("utf-8"), end="")
    print(f"replay_final_state.json {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")
    return 0 if report["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
