"""ULR-20261015-v1: render the report tables from ``analysis/analysis.json`` (no new computation).

``python -B report_tables.py`` writes ``report_tables.md`` next to this file (exclusively) and prints it.
"""

from __future__ import annotations

import json
from pathlib import Path

EXE = Path(__file__).resolve().parent
COH = ("TRAIN", "EARLY", "LATE")
ARMS = ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED")


def f(v, d: int = 4) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
        return f"{v:.3e}"
    return f"{v:.{d}f}"


def main() -> int:
    a = json.loads((EXE / "analysis" / "analysis.json").read_text("utf-8"))
    m, L = a["methods"], []
    er = a["A_element_reduction"]
    L += ["## A. Element reduction (observed in EVAL)", "", "| Arm state | Wire latent | Elements | Payload B | Element ratio | Receiver tensors / B | Verified |", "|---|---|---:|---:|---:|---|---|"]
    for k, r in er["arms"].items():
        rc = r["receiver"] or {}
        L.append(f"| {k} | {r['wire_latent_shape_observed']} | {r['wire_latent_numel_observed']} | {r['payload_bytes_observed']} | {r['element_ratio']:.4f} | {rc.get('tensors')} / {rc.get('bytes')} | {f(r['verified'])} |")
    L += ["", f"Step-zero payload bytes by c (INIT, TRAIN): {er['step_zero_payload_bytes']}; all verified: {f(er['all_verified'])}"]
    u = a["B_upp"]
    L += ["", "## B. UPP-20261014-v1 (vendored pinned classifier)", "", "| Unit | Role | EARLY | LATE | Study | TRAIN | Tiles PASS E / L |", "|---|---|---|---|---|---|---|"]
    for name, r in u.items():
        if isinstance(r, dict) and "cohorts" in r:
            co = r["cohorts"]
            L.append(f"| {name} | {r['role']} | {co['EARLY']['pooled_outcome']} | {co['LATE']['pooled_outcome']} | {r.get('study')} | {co['TRAIN']['pooled_outcome']} | "
                     f"{co['EARLY']['tiles']['PASS']}/{sum(co['EARLY']['tiles'].values())}, {co['LATE']['tiles']['PASS']}/{sum(co['LATE']['tiles'].values())} |")  # fmt: skip
    L += ["", f"Smallest tested adapted element count with study PASS: {u.get('smallest_tested_adapted_element_count_with_study_pass')}"]
    L += ["", "## C1. Main results (pooled; normalized MSE direct for neural, raw / s32^2 for codecs)", ""]
    L += ["| Method | Elements | Bytes | TRAIN RSE | TRAIN nMSE | EARLY RSE / cos | EARLY nMSE | LATE RSE / cos | LATE nMSE |", "|---|---:|---:|---:|---:|---|---:|---|---:|"]
    order = ("SOURCE-LEAKY:FLOAT", *(f"{x}:{p}" for x in ARMS for p in ("STEP0", "FINAL")), "PCA-256", "PCA-64", "PCA-16", "SCALAR-4", "SOURCE-LEAKY:INT8LAT", "INT8-SHARED")
    for k in order:
        r = {g: m[g][k] for g in COH}
        L.append(f"| {k} | {r['TRAIN'].get('elements')} | {r['TRAIN'].get('payload_bytes')} | {f(r['TRAIN']['rse'])} | {f(r['TRAIN']['normalized_mse'], 6)} | "
                 f"{f(r['EARLY']['rse'])} / {f(r['EARLY']['cosine'])} | {f(r['EARLY']['normalized_mse'], 6)} | {f(r['LATE']['rse'])} / {f(r['LATE']['cosine'])} | {f(r['LATE']['normalized_mse'], 6)} |")  # fmt: skip
    d = a["C_damage_recovery"]
    L += ["", "## C2. Damage (step 0 / source), recovery (final / step 0), final / final C16 (normalized MSE ratios)", ""]
    L += ["| Arm | Cohort | step0 / source | final / step0 | fraction of damage recovered | final / source | final / C16 final | paired: final lower than step0 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        for g in COH:
            r = d[arm][g]
            pf = r["paired_final_vs_step0"]
            L.append(f"| {arm} | {g} | {r['damage_ratio_step0_over_source']:.4f} | {r['recovery_ratio_final_over_step0']:.4f} | {f(r['fraction_of_damage_recovered'])} | "
                     f"{r['final_over_source']:.4f} | {r['final_over_c16_final']:.4f} | {pf['a_lower']}/{pf['tiles']} |")  # fmt: skip
    lc = a["C_learning_curves"]
    L += ["", "## C3. TRAIN learning curves (eval mode, wire path)", "", "| Arm | 0 | 1,000 | 3,000 | 6,000 (normalized MSE / RSE / cos) |", "|---|---|---|---|---|"]
    for arm in ARMS:
        cells = [f"{lc[arm][u]['train_normalized_mse_direct']:.6f} / {lc[arm][u]['train_rse']:.4f} / {lc[arm][u]['train_cosine']:.4f}" for u in ("0", "1000", "3000", "6000")]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    s0 = a["step_zero_all_widths_train"]
    L += ["", "Step-zero TRAIN by width (PCA bridge, no adaptation; c = 8 and 2 forward-only): " + "; ".join(
        f"c={c} ({16 * int(c)} el.): RSE {v['train_rse']:.4f}, cos {v['train_cosine']:.4f}" for c, v in sorted(s0.items(), key=lambda kv: -int(kv[0])))]
    ind = a["C_input_dependence"]
    L += ["", "## C4. Input dependence (held-out; tiles where the correct payload has lower SSE; pooled SSE ratio correct / wrong)", ""]
    L += ["| Method | EARLY vs C-WRONG | LATE vs C-WRONG | EARLY vs zero wire / latent | LATE vs zero wire / latent |", "|---|---|---|---|---|"]
    for k in (*(f"{x}:FINAL" for x in ARMS), *(f"{x}:STEP0" for x in ARMS), "SOURCE-LEAKY:FLOAT", "PCA-256", "PCA-64", "PCA-16", "SCALAR-4"):
        e, la = ind["EARLY"]["paired"][k], ind["LATE"]["paired"][k]
        z = "vs_zero_wire" if "vs_zero_wire" in e else ("vs_zero_latent" if "vs_zero_latent" in e else None)
        zc = (lambda r: f"{r[z]['a_lower']}/{r[z]['tiles']} ({r[z]['pooled_sse_ratio_a_over_b']:.3f})") if z else (lambda r: "-")
        L.append(f"| {k} | {e['vs_wrong']['a_lower']}/{e['vs_wrong']['tiles']} ({e['vs_wrong']['pooled_sse_ratio_a_over_b']:.3f}) | "
                 f"{la['vs_wrong']['a_lower']}/{la['vs_wrong']['tiles']} ({la['vs_wrong']['pooled_sse_ratio_a_over_b']:.3f}) | {zc(e)} | {zc(la)} |")  # fmt: skip
    c = a["D_comparators"]
    L += ["", "## D1. Matching element counts (final arm vs input PCA with the same number of fp32 elements)", ""]
    L += ["| Arm | PCA | Cohort | arm RSE | PCA RSE | arm nMSE | PCA nMSE | tiles arm lower | pooled SSE arm / PCA |", "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for arm, rows in c["matching_elements"].items():
        for g in COH:
            r = rows[g]
            pp = r["paired_arm_vs_pca"]
            L.append(f"| {arm} | {r['pca_name']} | {g} | {f(r['arm']['rse'])} | {f(r['pca']['rse'])} | {f(r['arm']['normalized_mse'], 6)} | {f(r['pca']['normalized_mse'], 6)} | {pp['a_lower']}/{pp['tiles']} | {pp['pooled_sse_ratio_a_over_b']:.4f} |")
    L += ["", "## D2. Matching bytes (ranking by pooled SSE; element kinds differ)", ""]
    for budget, rows in c["matching_bytes"].items():
        for g in COH:
            r = rows[g]
            L.append(f"- {budget} {g}: " + " < ".join(f"{k} (RSE {r['methods'][k]['rse']:.4f}; {r['methods'][k]['elements']} {r['methods'][k]['element_kind']})" for k in r["ranking_by_pooled_sse"]))
    t = a["D_secondary_target"]
    L += ["", "## D3. Secondary target: normalized MSE vs 1e-4", "", "| Method | TRAIN | EARLY | LATE | x target (TRAIN) |", "|---|---:|---:|---:|---:|"]
    for k in (*(f"{x}:FINAL" for x in ARMS), "SOURCE-LEAKY:FLOAT", "PCA-16", "PCA-64", "PCA-256", "SCALAR-4", "INT8-SHARED"):
        L.append(f"| {k} | {f(t[k]['TRAIN']['normalized_mse'], 6)} | {f(t[k]['EARLY']['normalized_mse'], 6)} | {f(t[k]['LATE']['normalized_mse'], 6)} | {t[k]['TRAIN']['factor_above_target']:.1f} |")
    by = a["bytes"]
    L += ["", "## E. Bytes", "", "| Arm | Payload B | Receiver one-time B | Sender one-time B | Bridge parameters (reduce / expand) | Amortized B at 504 / 3,072 / 64,512 |", "|---|---:|---:|---:|---|---|"]
    for arm, r in by["arms"].items():
        am = r["amortized_bytes_per_message"]
        L.append(f"| {arm} | {r['payload_bytes']} | {r['receiver_one_time_bytes']} | {r['sender_one_time_bytes']} | {r['bridge_parameters']['reduce']} / {r['bridge_parameters']['expand']} | {am['504']:.0f} / {am['3072']:.0f} / {am['64512']:.0f} |")
    text = "\n".join(["# ULR-20261015-v1 report tables (generated from analysis/analysis.json)", "", *L, ""])
    (EXE / "report_tables.md").open("x", encoding="utf-8", newline="\n").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
