"""Derived comparison of the UID-20261012-v2 r1 U-FIT arm (Uformer) and the archived P5-A v2 R3 single-snapshot
control (ResNet-3), from recorded sums only. Every value read comes from a file whose SHA-256 is checked first; nothing
is re-run and no model is built. Writes ``comparison.json`` (never overwritten).

    python -B compare_records.py --out comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import ep_classify as E

RESULTS = E.REPO / "results" / "phase5" / "p5a_v2"
R3_DIR = "R3_balanced_effective_delta_r8_single_snapshot"
R3_FILES = (f"{R3_DIR}/p5a_record.json", f"{R3_DIR}/run_metadata.json", f"{R3_DIR}/config.resolved.yaml")
R3_N = 2048 * 1536  # frozen v2 protocol section 9: [1, 2048, 1536] fp32, n = 3,145,728
UFIT_LATENT, UFIT_INPUT = 64 * 32 * 32, 32 * 32


def verify_r3() -> list[str]:
    """Committed P5-A files must match the run-root entries of ``artifact_manifest.json``."""
    manifest = E.load(RESULTS / "artifact_manifest.json")
    entries = {e["path"].split("/", 1)[1]: e for e in manifest["run_root_artifacts"]}
    problems = []
    for rel in R3_FILES:
        p, e = RESULTS / rel, entries.get(rel)
        ok = (
            e is not None
            and p.is_file()
            and (E.sha256_file(p), p.stat().st_size) == (e["sha256"], e["bytes"])
        )
        if not ok:
            problems.append(f"{rel}: absent or differs from artifact_manifest.json")
    return problems


CKPT = Path("D:/cgfed-runs/phase5_p5a_v2_seed1") / R3_DIR / "autoencoder_p5a_final.safetensors"
BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def r3_checkpoint_groups() -> dict[str, Any]:
    """Parameter counts per group from the safetensors header of the R3 checkpoint, after its SHA-256 is checked
    against ``artifact_manifest.json``; the tensor data is not read. Module order of the production ResNet-3:
    encoder.net 0-21 stem and six stages, 22-24 residual blocks; decoder.net 0-2 residual blocks, then sampling."""
    manifest = E.load(RESULTS / "artifact_manifest.json")
    entry = next(e for e in manifest["run_root_artifacts"] if e["path"].endswith(f"{R3_DIR}/{CKPT.name}"))
    if not CKPT.is_file() or (E.sha256_file(CKPT), CKPT.stat().st_size) != (entry["sha256"], entry["bytes"]):
        return {"verified": False, "problems": [f"{CKPT}: absent or differs from artifact_manifest.json"]}
    with CKPT.open("rb") as fh:
        header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
    groups = dict.fromkeys(("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling"), 0)
    buffers = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        n = math.prod(spec["shape"])
        if name.endswith(BUFFERS):
            buffers += n
            continue
        side, _, idx = name.split(".")[:3]
        core = int(idx) >= 22 if side == "encoder" else int(idx) <= 2
        groups[f"{side}.{'core' if core else 'sampling'}"] += n
    return {
        "verified": True,
        "checkpoint_sha256": entry["sha256"],
        "groups": groups,
        "total": sum(groups.values()),
        "buffer_elements": buffers,
    }


def improvement(sse: float, base_sse: float) -> dict[str, Any]:
    """Relative reduction of squared error against one baseline of the same input: 1 - SSE / SSE_baseline."""
    ratio = sse / base_sse
    return {"sse_over_baseline": ratio, "relative_sse_reduction": 1.0 - ratio}


def ufit_side() -> dict[str, Any]:
    rec = E.assess_uid_r1()
    if rec["evidence"]["verified"] is not True:
        return {"verified": False, "problems": rec["evidence"]["problems"]}
    child = E.load(E.UID_RUN_ROOT / "arms" / "U-FIT" / "child_record.json")  # verified by assess_uid_r1
    fe, base = child["final"]["eval_mode"], child["baselines"]
    s0 = child["step0"]["post_init"]["eval"]
    return {
        "verified": True,
        "input": child["input"],
        "latent_shape": [64, 32, 32],
        "latent_over_input_elements": UFIT_LATENT / UFIT_INPUT,
        "parameters": {
            "groups": {g: v["numel"] for g, v in child["param_change_total"].items()},
            "total": child["trainable"]["parameters"],
        },
        "updates_done": child["updates_done"],
        "status": child["status"],
        "cpu": child["cpu"],
        "final_eval": {
            k: fe[k] for k in ("n", "sse", "target_energy", "pred_energy", "dot", "rse", "cosine", "mse")
        },
        "step0_eval": {"rse": s0["rse"], "cosine": s0["cosine"]},
        "vs_zero": improvement(fe["sse"], base["zero_prediction"]["sse"]),
        "vs_scalar_mean": improvement(fe["sse"], base["shared_scalar_mean"]["sse"]),
        "scalar_mean_baseline_recorded": True,
        "wall_s": E.load(E.UID_RUN_ROOT / "ledger" / "end_U-FIT.json")["arm_wall_s"],
        "curve_pre_update_train_rse": {
            k: v["pre_update_train_rse"]
            for k, v in E.load(E.UID_EVIDENCE / "analysis.json")["arms"]["U-FIT"]["curve_summary"].items()
            if k in ("1", "1000", "2000", "3000")
        },
    }


def r3_side() -> dict[str, Any]:
    import yaml  # PyYAML, pinned in the environment record of the P5-A run

    problems = verify_r3()
    if problems:
        return {"verified": False, "problems": problems}
    rec = E.load(RESULTS / R3_FILES[0])
    meta = E.load(RESULTS / R3_FILES[1])
    ae_cfg = yaml.safe_load((RESULTS / R3_FILES[2]).read_text(encoding="utf-8"))["autoencoder"]
    pooled = rec["evaluation"]["predictors"]["autoencoder"]["pooled"]
    zero = rec["evaluation"]["predictors"]["zero"]["pooled"]
    scale = rec["normalization"]["normalizer"]["scale_global"]
    n_from_mse, n_from_zero = pooled["sse"] / pooled["mse"], zero["sse"] / zero["mse"]
    if round(n_from_mse) != R3_N or round(n_from_zero) != R3_N or zero["sse"] != pooled["sig"]:
        return {
            "verified": False,
            "problems": [f"N or zero baseline inconsistent: {n_from_mse}, {n_from_zero}"],
        }
    zero_mse_normalized = pooled["zero_predictor_mse"] / scale**2  # derived; float32 scaling ignored
    curve = {c["iteration"]: c["train_mse_batch"] for c in rec["curve"]}
    return {
        "verified": True,
        "verified_against": "results/phase5/p5a_v2/artifact_manifest.json (run_root_artifacts)",
        "status": rec["status"],
        "representation": {"id": rec["representation"], "kind": rec["kind"], "layout": ae_cfg["layout"]},
        "input": {
            "shape": [1, 2048, 1536],
            "n": R3_N,
            "n_check": {"sse_over_mse": n_from_mse, "zero_sse_over_zero_mse": n_from_zero},
            "selected": rec["selected_records"][0],
            "normalized_abs_quantiles": {k: rec["normalized_range"]["all"][k] for k in ("p50", "p99", "max")},
        },
        "normalization": {"mode": ae_cfg["normalization"], "scale_global": scale},
        "latent_shape": rec["latent_shape"],
        "latent_over_input_elements": math.prod(rec["latent_shape"]) / R3_N,
        "parameters": r3_checkpoint_groups(),
        "optimizer": {
            k: ae_cfg[k]
            for k in ("learning_rate", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay")
        },
        "seeds": rec["seeds"],
        "iterations_done": rec["iterations_done"],
        "device": meta["environment"]["cuda"]["device_name"],
        "determinism": meta["determinism"],
        "gpu_wall_s_charged": rec["timing"]["gpu_wall_s_charged"],
        "final_eval": {
            k: pooled[k] for k in ("sse", "sig", "hat", "dot", "mse", "rse", "cosine", "norm_ratio")
        },
        "vs_zero": improvement(pooled["sse"], zero["sse"]),
        "scalar_mean_baseline_recorded": False,
        "vs_scalar_mean_bound": {
            "relative_sse_reduction_at_most": 1.0 - pooled["sse"] / zero["sse"],
            "reason": "the scalar-mean SSE is SIG - n * mean^2 <= SIG; the record has no element sum, so its value is unknown",
        },
        "train_mse_over_zero_mse_normalized": {
            str(k): curve[k] / zero_mse_normalized for k in (1, 1000, 2000, 3000)
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Derived Uformer U-FIT vs ResNet-3 R3 single-snapshot comparison"
    )
    ap.add_argument("--out", type=Path, required=True, help="new JSON file (never overwritten)")
    args = ap.parse_args(argv)
    doc = {
        "schema": "ep-comparison-v1",
        "label": "DERIVED from recorded sums; no reconstruction re-run",
        "uformer_u_fit": ufit_side(),
        "resnet3_r3_single_snapshot": r3_side(),
        "matched": False,
        "why_not_matched": "input source, scale, size, geometry (stride 1 vs stride 2), bottleneck (64:1 expansion vs "
        "1/64), device, seed procedure and parameter count all differ",
    }
    with args.out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for side in ("uformer_u_fit", "resnet3_r3_single_snapshot"):
        d = doc[side]
        print(side, "verified" if d["verified"] else f"UNVERIFIED {d['problems']}", d.get("vs_zero"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
