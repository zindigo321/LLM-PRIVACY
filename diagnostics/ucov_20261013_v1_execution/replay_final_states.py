"""Read-only replay of each completed arm's saved final state: forward passes only. No optimizer, no gradient, no
update, no BatchNorm recalibration, no parameter edit, and nothing written to the run root.

``python -B replay_final_states.py`` from this directory, with PYTHONPATH unset. It writes
``replay_final_states.json`` here (exclusive, LF bytes). It follows the child's final evaluation with the frozen code:
- CPU: the frozen ``configure_cpu``, 24/24 threads, deterministic algorithms, before any tensor;
- input: the frozen ``ucov_control.load_tile`` (tile bytes hashed; fp32 ``x / float32(s)``);
- model: per arm and per mode, a fresh model from the frozen ``urp_control.load_state``, which builds the frozen
  model, loads strictly and checks the state hash; each mode starts from the same CPU RNG state;
- separate copies: the train-mode copy is a separate disposable model, so its BatchNorm buffer updates cannot reach
  the eval-mode copy;
- metrics: the frozen ``uid_common.metrics`` in both spaces. Every recorded sum is compared exactly, with no
  tolerance.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
UCOV = HERE.parent / "ucov_20261013_v1"
RUN = Path("D:/cgfed-runs/ucov_20261013_v1/arms")
sys.path.insert(0, str(UCOV))
KEYS = (
    "n",
    "sse",
    "target_energy",
    "pred_energy",
    "dot",
    "mse",
    "rse",
    "cosine",
    "norm_ratio",
    "max_abs_err",
    "finite",
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(replayed: dict, recorded: dict) -> dict:
    diff = {k: [replayed[k], recorded[k]] for k in KEYS if replayed[k] != recorded[k]}
    return {"bit_identical": not diff, "differences": diff}


def main() -> int:
    out_path = HERE / "replay_final_states.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    files_before = {str(p): sha(p) for p in sorted(RUN.rglob("*")) if p.is_file()}
    import ucov_control as U  # torch-free at import

    resolved = json.loads((UCOV / "resolved_config.json").read_text(encoding="utf-8"))
    import uid_common as C

    cpu = C.configure_cpu(resolved["cpu"]["intra_op_threads"], resolved["cpu"]["inter_op_threads"])
    import torch

    resolved = U.load_resolved()  # equal to a fresh resolution of the pinned sources
    R = U.urp()
    scale = float(resolved["inputs"]["normalization"]["scale_global"])
    report = {
        "recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": cpu,
        "arms": {},
        "optimizer_created": False,
        "grad_enabled_during_forwards": False,
    }
    for arm in U.ARMS:
        rec_path = RUN / arm / "child_record.json"
        final_path = RUN / arm / "final_state.safetensors"
        if not rec_path.is_file() or not final_path.is_file():
            report["arms"][arm] = {"replayed": False, "reason": "no child record or no final state"}
            continue
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        x, x_n, digest = U.load_tile(resolved, arm)
        row = {
            "replayed": True,
            "tile_sha256_equal": digest == rec["input"]["patch_file_sha256"],
            "original_sha256_equal": C.describe_input(x)["sha256"] == rec["input"]["original"]["sha256"],
            "normalized_sha256_equal": C.describe_input(x_n)["sha256"]
            == rec["input"]["normalized"]["sha256"],
            "modes": {},
        }
        rng = torch.get_rng_state()
        outputs = {}
        for mode in ("eval", "train"):
            model, meta = R.load_state(final_path, resolved)
            before = C.state_sha256(model)
            model.train(mode == "train")
            torch.set_rng_state(rng)
            with torch.no_grad():
                y_n = C.reconstruct(model, x_n)
            after = C.state_sha256(model)
            outputs[mode] = y_n
            row["modes"][mode] = {
                "original": compare(C.metrics(R.denormalize(y_n, scale), x), rec["final"][f"{mode}_mode"]),
                "normalized": compare(C.metrics(y_n, x_n), rec["final"][f"{mode}_mode_normalized"]),
                "loaded_state_equals_recorded_final": before == rec["model_state_sha256_final"],
                "state_changed_by_forward": after != before,
                "metadata_point": meta.get("point"),
            }
            del model
        tme = float((outputs["train"].double() - outputs["eval"].double()).square().sum())
        row["train_minus_eval_bit_identical"] = tme == rec["final"]["train_minus_eval_output_sse_normalized"]
        row["eval_copy_unchanged"] = not row["modes"]["eval"]["state_changed_by_forward"]
        row["train_copy_changed_as_recorded"] = (
            row["modes"]["train"]["state_changed_by_forward"]
            == rec["final"]["train_copy_state_changed_by_forward"]
        )
        row["all_bit_identical"] = (
            row["tile_sha256_equal"]
            and row["original_sha256_equal"]
            and row["normalized_sha256_equal"]
            and row["train_minus_eval_bit_identical"]
            and row["eval_copy_unchanged"]
            and row["train_copy_changed_as_recorded"]
            and all(
                m["loaded_state_equals_recorded_final"]
                and m["original"]["bit_identical"]
                and m["normalized"]["bit_identical"]
                for m in row["modes"].values()
            )
        )
        report["arms"][arm] = row
    files_after = {str(p): sha(p) for p in sorted(RUN.rglob("*")) if p.is_file()}
    report["run_root_files_unchanged"] = files_before == files_after
    local = sorted(
        n
        for n, m in sys.modules.items()
        if isinstance(getattr(m, "__file__", None), str)
        and Path(m.__file__).is_absolute()
        and Path(m.__file__).resolve().is_relative_to(HERE.parents[1])
        and n != "__main__"
    )
    report["repository_modules_imported"] = local
    report["module_sha256"] = {n: sha(Path(sys.modules[n].__file__)) for n in local}
    report["cg_fedllm_imported"] = "cg_fedllm" in sys.modules
    replayed = [r for r in report["arms"].values() if r.get("replayed")]
    report["all_bit_identical"] = (
        bool(replayed)
        and all(r["all_bit_identical"] for r in replayed)
        and report["run_root_files_unchanged"]
        and not report["cg_fedllm_imported"]
    )
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with out_path.open("xb") as fh:
        fh.write(data)
    print(
        json.dumps(
            {
                "all_bit_identical": report["all_bit_identical"],
                "arms": {a: r.get("all_bit_identical", r.get("reason")) for a, r in report["arms"].items()},
            }
        )
    )
    print(f"replay_final_states.json {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")
    return 0 if report["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
