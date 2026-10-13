"""Independent verification of the URC-v1 R-FIT result (read-only; does not import urc_analyze or ep_classify).

``python -B independent_verify.py`` writes ``independent_verification.json`` (never overwritten) in this directory.
1. Hashes: the run root and mirror files against ``analysis/FINAL_ARTIFACTS_SHA256.txt``; the U-FIT files against
   the r1 ``FINAL_ARTIFACTS_SHA256.txt``; mirror == primary ledger byte for byte.
2. Metrics recomputed from the recorded sufficient statistics (SSE, T, P, D, N) of both runs, in float64 with the
   frozen formulas and also with exact rational arithmetic; the signed differences delta_RSE = R - U and
   delta_cosine = R - U are recomputed both ways, and the identity SSE = P + T - 2D is checked.
3. Status re-derived from the strict criteria; EP-20261012-v1 re-implemented here from POLICY.md (not imported).
4. State files: header parse, tensor count and element totals, BatchNorm counters, a fresh-model strict load and
   the state hash recomputed with the frozen ``uid_common.state_sha256``; init file vs ``check_init_comparison``.
5. Curve: 31 points, finite, and the final pre-update train loss vs the final train-mode MSE (reported only).
"""

import hashlib
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
URC = HERE.parent / "urc_20261013_v1"
RUN = Path("D:/cgfed-runs/urc_20261013_v1")
MIRROR = URC / "scientific_ledger_mirror"
UFIT_RUN = Path("D:/cgfed-runs/uid_20261012_v2_r1")
UID_LIST = HERE.parent / "uid_20261012_v2_r1_execution" / "FINAL_ARTIFACTS_SHA256.txt"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def check_list(listing: Path) -> dict:
    rows = [ln for ln in listing.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    bad = []
    for ln in rows:
        digest, size, name = re.split(r"\s+", ln.strip(), maxsplit=2)
        p = Path(name)
        if not p.is_file() or sha(p) != digest or p.stat().st_size != int(size):
            bad.append(name)
    return {"entries": len(rows), "mismatching": bad}


def metrics_float(m: dict) -> dict:
    t, p = m["target_energy"], m["pred_energy"]
    return {
        "rse": m["sse"] / t,
        "cosine": m["dot"] / math.sqrt(p * t),
        "mse": m["sse"] / m["n"],
        "norm_ratio": math.sqrt(p / t),
        "sse_identity_rel": abs((p + t - 2 * m["dot"]) - m["sse"]) / m["sse"],
    }


def exact(m: dict) -> dict:
    """Exact rationals of the recorded binary64 sums; cosine via exact sqrt-free comparison is not needed, so the
    exact value of cosine^2 is kept and cosine is rounded once at the end."""
    sse, t, p, d = (Fraction(m[k]) for k in ("sse", "target_energy", "pred_energy", "dot"))
    return {"rse": sse / t, "cos2": d * d / (p * t), "cos_sign": 1 if d > 0 else -1}


def ep_reimplemented(fe: dict, mean_mse: float, compliant: bool, verified: bool) -> dict:
    """EP-20261012-v1 from POLICY.md, written here independently of ep_classify.py."""
    if not verified:
        return {"outcome": "UNVERIFIED"}
    if not compliant:
        return {"outcome": "NOT_ELIGIBLE"}
    t, p = fe["target_energy"], fe["pred_energy"]
    vals = (fe["sse"], t, p, fe["dot"], mean_mse)
    if fe["finite"] is not True or not all(math.isfinite(v) for v in vals) or t <= 0 or p <= 0 or fe["sse"] < 0:
        return {"outcome": "FAIL", "why": "M0"}
    rse, cos, mse = fe["sse"] / t, fe["dot"] / math.sqrt(p * t), fe["sse"] / fe["n"]
    best = min(t / fe["n"], mean_mse)
    m1, m2 = rse <= 0.50, cos >= 0.80
    if best == 0:
        return {"outcome": "UNINFORMATIVE" if m1 and m2 else "FAIL"}
    m3 = mse <= 0.80 * best
    return {
        "outcome": "PASS" if m1 and m2 and m3 else "FAIL",
        "M1": [rse, m1],
        "M2": [cos, m2],
        "M3": [mse, 0.80 * best, m3],
        "mse_over_best": mse / best,
    }


def state_check(path: Path, rec_hash: str) -> dict:
    with path.open("rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    meta = header.pop("__metadata__", {})
    numel = {k: math.prod(v["shape"]) for k, v in header.items()}
    bufs = {k: v for k, v in numel.items() if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    sys.path[:0] = [str(URC), str(URC.parent / "uid_20261012_v2"), str(URC.parent / "uid_20261012_v2_r1")]
    import torch
    import uid_common as C
    import urc_control as U
    from safetensors.torch import load as st_load

    resolved = load(URC / "resolved_config.json")
    tensors = st_load(path.read_bytes())
    model = U.build_model(resolved)
    model.load_state_dict(tensors, strict=True)  # strict: every parameter and buffer present, no extra key
    nbt = sorted({int(v) for k, v in tensors.items() if k.endswith("num_batches_tracked")})
    return {
        "file_sha256": sha(path),
        "bytes": path.stat().st_size,
        "tensors": len(header),
        "parameter_elements": sum(v for k, v in numel.items() if k not in bufs),
        "buffer_elements": sum(bufs.values()),
        "metadata_state_sha256": meta.get("state_sha256"),
        "recomputed_state_sha256": C.state_sha256(model),
        "recorded_state_sha256": rec_hash,
        "hashes_agree": C.state_sha256(model) == meta.get("state_sha256") == rec_hash,
        "num_batches_tracked_values": nbt,
        "all_float_finite": all(bool(torch.isfinite(v).all()) for v in tensors.values() if v.is_floating_point()),
        "metadata_point": meta.get("point"),
    }


def main() -> int:
    out_path = HERE / "independent_verification.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    rec = load(RUN / "arms" / "R-FIT" / "child_record.json")
    ufit = load(UFIT_RUN / "arms" / "U-FIT" / "child_record.json")
    end, inv_end = load(RUN / "ledger" / "end_R-FIT.json"), load(RUN / "ledger" / "invocation_001_end.json")
    inv = load(RUN / "ledger" / "invocation_001.json")
    u_end = load(UFIT_RUN / "ledger" / "end_U-FIT.json")
    out: dict = {"hashes": {}}
    out["hashes"]["urc_sealed_list"] = check_list(HERE / "analysis" / "FINAL_ARTIFACTS_SHA256.txt")
    out["hashes"]["uid_r1_final_list"] = check_list(UID_LIST)
    names = sorted(p.name for p in (RUN / "ledger").iterdir())
    out["hashes"]["ledger_mirror_identical"] = names == sorted(p.name for p in MIRROR.iterdir()) and all(
        (RUN / "ledger" / n).read_bytes() == (MIRROR / n).read_bytes() for n in names
    )
    out["hashes"]["invocation_names_approved_manifest"] = (
        inv["identity"]["urc_manifest_sha256"]
        == sha(URC / "manifest_urc.json")
        == "b058adf35bf326516f17cb3d773222840b0d1f6605a0f6e4a535564e94588cb6"
    )
    out["counts"] = {
        "invocations": sum(1 for n in names if re.match(r"invocation_\d{3}\.json$", n)),
        "starts": sum(1 for n in names if n.startswith("start_")),
        "updates_done": rec["updates_done"],
        "status_child_end_summary": [rec["status"], end["status"], inv_end["arms"][0]["status"]],
        "guard_action": end["guard"]["action"],
        "interrupted": inv_end["interrupted"],
    }
    r_e, u_e = rec["final"]["eval_mode"], ufit["final"]["eval_mode"]
    r_t, u_t = rec["final"]["train_mode"], ufit["final"]["train_mode"]
    fm = {k: metrics_float(m) for k, m in (("r_eval", r_e), ("u_eval", u_e), ("r_train", r_t), ("u_train", u_t))}
    out["metrics_float64"] = fm
    out["metrics_equal_recorded"] = {
        k: fm[k]["rse"] == m["rse"] and fm[k]["cosine"] == m["cosine"]
        for k, m in (("r_eval", r_e), ("u_eval", u_e), ("r_train", r_t), ("u_train", u_t))
    }
    xr, xu = exact(r_e), exact(u_e)
    d_rse_exact = xr["rse"] - xu["rse"]
    d_cos_hi = math.sqrt(float(xr["cos2"])) - math.sqrt(float(xu["cos2"]))  # exact-ratio route, rounded at the end
    out["signed_differences"] = {
        "convention": "delta = R-FIT eval - U-FIT eval; delta_RSE < 0 favors R-FIT; delta_cosine > 0 favors R-FIT",
        "delta_rse_float64": fm["r_eval"]["rse"] - fm["u_eval"]["rse"],
        "delta_rse_exact_rational_rounded": float(d_rse_exact),
        "delta_cosine_float64": fm["r_eval"]["cosine"] - fm["u_eval"]["cosine"],
        "delta_cosine_from_exact_cos2": d_cos_hi,
        "delta_rse_train_float64": fm["r_train"]["rse"] - fm["u_train"]["rse"],
        "delta_cosine_train_float64": fm["r_train"]["cosine"] - fm["u_train"]["cosine"],
        "sse_ratio_r_over_u_eval": r_e["sse"] / u_e["sse"],
    }
    analysis = load(HERE / "analysis" / "analysis.json")
    c = analysis["comparison"]
    out["analysis_script_agrees"] = {
        "delta_rse": c["rse_difference_r_minus_u"] == out["signed_differences"]["delta_rse_float64"],
        "delta_cosine": c["cosine_difference_r_minus_u"] == out["signed_differences"]["delta_cosine_float64"],
        "matching_status": analysis["matching"]["status"],
        "ep_outcome": analysis["exploratory_ep"]["outcome"],
    }
    strict = r_e["finite"] and fm["r_eval"]["rse"] <= 0.01 and fm["r_eval"]["cosine"] >= 0.99
    out["strict_rederived"] = "COMPLETED_PASS" if strict else "COMPLETED_FIT_FAIL"
    rss_used = max(end["guard"]["sampled_max_rss_bytes"], rec["rss"]["os_peak_wset_bytes"])
    compliant = (
        rec["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL")
        and rec["updates_done"] == 3000
        and end["guard"]["action"] is None
        and inv_end["interrupted"] is False
        and end["arm_wall_s"] <= 1800
        and inv_end["aggregate_wall_s"] <= 7200
        and out["counts"]["starts"] <= 1
        and rss_used <= inv["rss_limit_bytes"]
        and inv_end["evidence_bytes_run_root"] <= 512 * 2**20
    )
    verified = (
        not out["hashes"]["urc_sealed_list"]["mismatching"]
        and not out["hashes"]["uid_r1_final_list"]["mismatching"]
        and out["hashes"]["ledger_mirror_identical"]
        and out["hashes"]["invocation_names_approved_manifest"]
    )
    out["ep_reimplemented"] = ep_reimplemented(r_e, rec["baselines"]["shared_scalar_mean"]["mse"], compliant, verified)
    out["baselines"] = {
        "r_fit_mse": fm["r_eval"]["mse"],
        "zero_mse": r_e["target_energy"] / r_e["n"],
        "scalar_mean_mse": rec["baselines"]["shared_scalar_mean"]["mse"],
        "sse_over_zero_sse": r_e["sse"] / rec["baselines"]["zero_prediction"]["sse"],
        "sse_over_scalar_mean_sse": r_e["sse"] / rec["baselines"]["shared_scalar_mean"]["sse"],
        "u_fit_sse_over_scalar_mean_sse": u_e["sse"] / ufit["baselines"]["shared_scalar_mean"]["sse"],
    }
    out["states"] = {
        "init": state_check(RUN / "arms" / "R-FIT" / "init_state.safetensors", rec["model_state_sha256_init"]),
        "final": state_check(RUN / "arms" / "R-FIT" / "final_state.safetensors", rec["model_state_sha256_final"]),
    }
    out["states"]["init_equals_prelaunch_fresh_build"] = (
        rec["model_state_sha256_init"] == "00bd190fa22e1d257315b5ca5dd66a73d715a3cdfe7f5c6ddab184c539eafd6e"
    )
    curve = [json.loads(x) for x in (RUN / "arms" / "R-FIT" / "curve.jsonl").read_text().splitlines() if x.strip()]
    out["curve"] = {
        "points": len(curve),
        "updates": [curve[0]["update"], curve[-1]["update"]],
        "all_finite": all(math.isfinite(c["pre_update_train_loss"]) for c in curve),
        "grad_nonfinite_total": sum(g["nonfinite"] for c in curve for g in c["grad"].values()),
        "zero_grad_groups": sorted({n for c in curve for n, g in c["grad"].items() if g["grad_l2"] == 0}),
        "pre_update_train_rse": {str(c["update"]): c["pre_update_train"]["rse"] for c in curve
                                 if c["update"] in (1, 100, 500, 1000, 1500, 2000, 2500, 3000)},
        "min_pre_update_loss": min((c["pre_update_train_loss"], c["update"]) for c in curve),
    }
    out["resources"] = {
        "r_fit": {"arm_wall_s": end["arm_wall_s"], "child_compute_s": rec["child_monotonic_s"]["end"]
                  - rec["child_monotonic_s"]["begin"], "guard_sampled_max_rss": end["guard"]["sampled_max_rss_bytes"],
                  "os_peak_wset": rec["rss"]["os_peak_wset_bytes"], "aggregate_wall_s": inv_end["aggregate_wall_s"]},
        "u_fit": {"arm_wall_s": u_end["arm_wall_s"], "child_compute_s": ufit["child_monotonic_s"]["end"]
                  - ufit["child_monotonic_s"]["begin"], "guard_sampled_max_rss": u_end["guard"]["sampled_max_rss_bytes"],
                  "os_peak_wset": ufit["rss"]["os_peak_wset_bytes"]},
    }
    text = json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with out_path.open("x", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
