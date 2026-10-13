"""Post-run analysis (read-only on the run root): independent metric recomputation from the recorded reductions,
verdict re-derivation from the frozen criteria, curve and gradient summary. Writes analysis.json and curve_u_fit.csv
in this directory only. Imports no model code and performs no forward pass.
"""

import csv
import json
import math
from pathlib import Path

RUN = Path("D:/cgfed-runs/uid_20261012_v2_r1")
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE.parent / "uid_20261012_v2" / "config.json").read_text(encoding="utf-8"))
CRIT = CFG["criteria"]


def recompute(m: dict) -> dict:
    """RSE, cosine, norm ratio, q_n from SSE, energies, dot and N only; compared with the recorded values."""
    rse = m["sse"] / m["target_energy"]
    cos = m["dot"] / math.sqrt(m["pred_energy"] * m["target_energy"]) if m["pred_energy"] > 0 else None
    ratio = math.sqrt(m["pred_energy"] / m["target_energy"])
    q_n = (m["sse"] / m["n"]) / (m["target_energy"] / m["n"])
    # Cauchy-Schwarz consistency of the recorded sums: SSE = P + T - 2 D
    sse_from_parts = m["pred_energy"] + m["target_energy"] - 2 * m["dot"]
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)  # noqa: E731
    return {
        "rse": rse, "cosine": cos, "norm_ratio": ratio, "q_n": q_n,
        "max_rel_diff_vs_recorded": max(rel(rse, m["rse"]), rel(ratio, m["norm_ratio"]), rel(q_n, m["q_n"]),
                                        0.0 if cos is None else rel(cos, m["cosine"])),
        "sse_identity_rel_diff": rel(sse_from_parts, m["sse"]),
        "passes_frozen_criteria": bool(m["finite"] and cos is not None and rse <= CRIT["max_rse"]
                                       and cos >= CRIT["min_cosine"]),
    }


def main() -> None:
    out = {"criteria": CRIT, "arms": {}}
    for arm in ("U-CHECK", "U-FIT"):
        rec = json.loads((RUN / "arms" / arm / "child_record.json").read_text(encoding="utf-8"))
        end = json.loads((RUN / "ledger" / f"end_{arm}.json").read_text(encoding="utf-8"))
        curve = [json.loads(line) for line in (RUN / "arms" / arm / "curve.jsonl").read_text().splitlines()]
        a = {
            "status_ledger": end["status"], "status_child": rec["status"], "updates_done": rec["updates_done"],
            "arm_wall_s": end["arm_wall_s"], "child_wall_s": rec["child_monotonic_s"]["end"] - rec["child_monotonic_s"]["begin"],
            "guard_sampled_max_rss_bytes": end["guard"]["sampled_max_rss_bytes"], "guard_samples": end["guard"]["samples"],
            "child_sampled_max_rss_bytes": rec["rss"]["sampled_max_rss_bytes"], "os_peak_wset_bytes": rec["rss"]["os_peak_wset_bytes"],
            "final_eval": recompute(rec["final"]["eval_mode"]), "final_train": recompute(rec["final"]["train_mode"]),
            "step0_eval": recompute(rec["step0"]["post_init"]["eval"]), "step0_train": recompute(rec["step0"]["post_init"]["train"]),
            "baseline_zero_rse": rec["baselines"]["zero_prediction"]["rse"],
            "baseline_shared_mean_rse": rec["baselines"]["shared_scalar_mean"]["rse"],
            "train_minus_eval_output_sse": rec["final"]["train_minus_eval_output_sse"],
            "curve_points": [c["update"] for c in curve],
            "verdict_rederived": "COMPLETED_PASS" if recompute(rec["final"]["eval_mode"])["passes_frozen_criteria"]
            else "COMPLETED_FIT_FAIL",
        }
        a["verdict_matches_record"] = a["verdict_rederived"] == rec["status"] == end["status"]
        pts = {c["update"]: c for c in curve}
        a["curve_summary"] = {
            k: {"pre_update_train_loss": pts[k]["pre_update_train_loss"], "pre_update_train_rse": pts[k]["pre_update_train"]["rse"],
                "pre_update_train_cosine": pts[k]["pre_update_train"]["cosine"]}
            for k in sorted(pts) if k in (1, 100, 500, 1000, 1500, 2000, 2500, 3000) or k == max(pts)
        }
        nonfinite = sum(g["nonfinite"] for c in curve for g in c["grad"].values())
        zero_grad_groups = sorted({name for c in curve for name, g in c["grad"].items() if g["grad_l2"] == 0})
        a["grad_nonfinite_total"], a["grad_zero_groups"] = nonfinite, zero_grad_groups
        a["grad_l2_first_last"] = {name: (curve[0]["grad"][name]["grad_l2"], curve[-1]["grad"][name]["grad_l2"])
                                   for name in curve[0]["grad"]}
        a["param_change_total"] = rec["param_change_total"]
        if arm == "U-FIT":
            losses = [c["pre_update_train_loss"] for c in curve]
            best = min(range(len(losses)), key=losses.__getitem__)
            a["curve_min_pre_update_loss"] = {"update": curve[best]["update"], "loss": losses[best],
                                              "note": "reported only; the protocol uses the final state, never the best point"}
            with (HERE / "curve_u_fit.csv").open("x", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                groups = list(curve[0]["grad"])
                w.writerow(["update", "pre_update_train_loss", "pre_update_train_rse", "pre_update_train_cosine",
                            *[f"grad_l2_{g}" for g in groups], *[f"delta_from_init_l2_{g}" for g in groups], "rss_bytes"])
                for c in curve:
                    w.writerow([c["update"], c["pre_update_train_loss"], c["pre_update_train"]["rse"],
                                c["pre_update_train"]["cosine"], *[c["grad"][g]["grad_l2"] for g in groups],
                                *[c["param_change_from_init"][g]["delta_l2"] for g in groups], c["rss_bytes_at_point"]])
        out["arms"][arm] = a
    inv_end = json.loads((RUN / "ledger" / "invocation_001_end.json").read_text(encoding="utf-8"))
    out["invocation"] = {"aggregate_wall_s": inv_end["aggregate_wall_s"], "evidence_bytes": inv_end["evidence_bytes_run_root"],
                         "evidence_within_cap": inv_end["evidence_within_cap"], "interrupted": inv_end["interrupted"]}
    (HERE / "analysis.json").open("x", encoding="utf-8").write(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
