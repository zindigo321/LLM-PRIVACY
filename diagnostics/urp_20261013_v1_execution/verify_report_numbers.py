"""Check the numbers quoted in REPORT.md against the evidence files (read-only; prints a table).

``python -B verify_report_numbers.py``. Each check names the source value and the text as it appears in REPORT.md.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM = Path("D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    rep = (HERE / "REPORT.md").read_text(encoding="utf-8")
    iv, an = load(HERE / "independent_verification.json"), load(HERE / "analysis" / "analysis.json")
    ch = load(ARM / "child_record.json")
    pts = [json.loads(x) for x in (ARM / "curve.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    loss = {p["update"]: p["pre_update_train_loss"] for p in pts}
    fe, ft = ch["final"]["eval_mode"], ch["final"]["train_mode"]
    base = ch["baselines"]["original"]["shared_scalar_mean"]["mse"]
    last = [k for k in sorted(loss) if k >= 2000]  # "over the last 1,000 updates": intervals 2,000-2,100 ... 2,900-3,000
    drops = [1 - loss[b] / loss[a] for a, b in zip(last, last[1:])]
    grads = [iv["curve"]["grad_l2_first_last"][g][0] / iv["curve"]["grad_l2_first_last"][g][1]
             for g in iv["curve"]["grad_l2_first_last"]]
    rc = iv["curve"]["relative_change_from_init_at_3000"].values()
    t = iv["timing"]
    checks = [
        ("rse/bound", fe["rse"] / 0.01, "19.3 times the bound", lambda v: f"{v:.1f}" == "19.3"),
        ("cos shortfall", 0.99 - fe["cosine"], "0.0919 short", lambda v: f"{v:.4f}" == "0.0919"),
        ("removed share", 1 - fe["mse"] / base, "removed 80.7%", lambda v: f"{100 * v:.1f}" == "80.7"),
        ("mse/best", fe["mse"] / base, "19.3% of the best", lambda v: f"{100 * v:.1f}" == "19.3"),
        ("wrapper", t["wrapper_elapsed_s"], "535.223", lambda v: f"{v:.3f}" == "535.223"),
        ("command", t["scientific_command_wall_s"], "528.635", lambda v: f"{v:.3f}" == "528.635"),
        ("aggregate", t["supervisor_aggregate_s"], "528.281", lambda v: f"{v:.3f}" == "528.281"),
        ("arm", t["arm_wall_s"], "528.140", lambda v: f"{v:.3f}" == "528.140"),
        ("child", t["child_process_s"], "525.593", lambda v: f"{v:.3f}" == "525.593"),
        ("loop", t["train_loop_point1_to_point3000_s"], "522.016", lambda v: f"{v:.3f}" == "522.016"),
        ("s/update", iv["curve"]["mean_s_per_update"], "0.174 s per update", lambda v: f"{v:.3f}" == "0.174"),
        ("final phase", t["final_phase_after_point3000_s"], "0.890", lambda v: f"{v:.3f}" == "0.890"),
        ("ratio 1000", iv["curve"]["loss_ratio_3000_over_2000"], "0.870", lambda v: f"{v:.3f}" == "0.870"),
        ("ratio 500", iv["curve"]["loss_ratio_3000_over_2500"], "0.937", lambda v: f"{v:.3f}" == "0.937"),
        ("drop min", min(drops), "0.14%", lambda v: f"{100 * v:.2f}" == "0.14"),
        ("drop max", max(drops), "2.2%", lambda v: f"{100 * v:.1f}" == "2.2"),
        ("drop min is last", drops.index(min(drops)), "The smallest was the last\n    interval, 2,900 to 3,000",
         lambda v: v == len(drops) - 1),
        ("slope", iv["curve"]["loglog_slope_2000_to_3000"], "is −0.34", lambda v: f"{v:.2f}" == "-0.34"),
        ("grad fall", [min(grads), max(grads)], "24 to 49 times", lambda v: [round(x) for x in v] == [24, 49]),
        ("rel change", [min(rc), max(rc)], "3.0% to 5.2%", lambda v: [f"{100 * x:.1f}" for x in v] == ["3.0", "5.2"]),
        ("tm/eval", iv["derived"]["train_minus_eval_output_sse_over_eval_sse_normalized"], "1.1% of the eval SSE",
         lambda v: f"{100 * v:.1f}" == "1.1"),
        ("norm vs orig", iv["metrics"]["normalized_vs_original_rel_diff"]["eval_mode"]["rse"], "relative 6e-9",
         lambda v: f"{v:.0e}" == "6e-09"),
        ("identity", max(iv["metrics"][k]["sse_identity_rel_err"] for k in
                         ("eval_mode", "eval_mode_normalized", "train_mode", "train_mode_normalized")),
         "within 1.9e-15", lambda v: v <= 1.9e-15),
        ("margin M1", an["exploratory_ep"]["checks"]["M1"]["margin"], "−0.3065", lambda v: f"{v:.4f}" == "-0.3065"),
        ("margin M2", an["exploratory_ep"]["checks"]["M2"]["margin"], "+0.0981", lambda v: f"{v:.4f}" == "0.0981"),
        ("mean", ch["input"]["original"]["mean"], "−2.5653e-05", lambda v: f"{v:.4e}" == "-2.5653e-05"),
        ("curve 1/100/1000/2000/3000", [loss[k] for k in (1, 100, 1000, 2000, 3000)],
         "MSE: 0.2138 at update 1, 0.01192 at 100, 0.004448 at 1,000, 0.003673 at 2,000 and 0.003197 at 3,000",
         lambda v: [f"{x:.4g}" for x in v] == ["0.2138", "0.01192", "0.004448", "0.003673", "0.003197"]),
        ("rse 1/100/1000/2000/3000", [pts[i]["pre_update_train"]["rse"] for i in (0, 1, 10, 20, 30)],
         "Pre-update RSE: 12.84, 0.716, 0.267, 0.2206 and 0.1919",
         lambda v: [f"{v[0]:.2f}", f"{v[1]:.3f}", f"{v[2]:.3f}", f"{v[3]:.4f}", f"{v[4]:.4f}"]
         == ["12.84", "0.716", "0.267", "0.2206", "0.1919"]),
        ("full-run max drop (not quoted as the range)", max(1 - loss[b] / loss[a] for a, b in
                                                            zip(sorted(loss)[1:], sorted(loss)[2:])),
         "Over the last 1,000 updates", lambda v: v > 0.022),
        ("ufit ctx", (iv["context_u_fit"]["u_fit_final_eval"]["rse"], iv["context_u_fit"]["u_fit_final_eval"]["cosine"]),
         "0.2727 / 0.8540", lambda v: (f"{v[0]:.4f}", f"{v[1]:.4f}") == ("0.2727", "0.8540")),
    ]
    exact = [fe["rse"], fe["cosine"], ft["rse"], ft["cosine"], fe["mse"], ft["mse"], fe["norm_ratio"],
             ft["norm_ratio"], fe["max_abs_err"], ft["max_abs_err"], ch["final"]["eval_mode_normalized"]["rse"],
             ch["final"]["eval_mode_normalized"]["cosine"], ch["final"]["train_mode_normalized"]["rse"],
             ch["final"]["train_mode_normalized"]["cosine"], an["exploratory_ep"]["checks"]["M3"]["threshold"],
             an["exploratory_ep"]["quantities"]["mse_over_best_baseline"],
             ch["step0"]["post_init"]["eval"]["original"]["rse"], ch["step0"]["post_init"]["eval"]["original"]["cosine"],
             4.132131066150689e-06, 4.131472976536517e-06, 0.016654169372344353, 0.016651517003540468]
    bad = 0
    for name, value, text, ok in checks:
        good = ok(value) and text in rep
        bad += not good
        print(f"{'ok ' if good else 'BAD'} {name}: {value!r} -> {text!r} (in report: {text in rep})")
    for v in exact:
        good = repr(v) in rep
        bad += not good
        print(f"{'ok ' if good else 'BAD'} exact {v!r} in report: {good}")
    ints = {"823,758,848": iv["resources"]["guard_sampled_max_rss_bytes"],
            "828,366,848": iv["resources"]["child_sampled_max_rss_bytes"],
            "828,628,992": iv["resources"]["child_os_peak_wset_bytes"],
            "4,294,967,296": iv["resources"]["rss_limit_bytes"],
            "9,453,146,112": iv["resources"]["available_ram_bytes_at_launch"],
            "6,416,844": iv["resources"]["evidence_bytes_run_root"],
            "536,870,912": iv["resources"]["evidence_cap_bytes"], "1,930": iv["resources"]["guard_samples"],
            "7,150": 7150, "16,432": iv["integrity"]["manifest_bytes"]}
    for text, v in ints.items():
        good = f"{v:,}" == text and text in rep
        bad += not good
        print(f"{'ok ' if good else 'BAD'} {text} == {v}")
    hashes = [an_sha for an_sha in ("a4721424aaf4539daf55c8498344da6e0efbe2679c96d610780e2e0600722d5f",)]
    import hashlib

    good = hashlib.sha256((HERE / "analysis" / "analysis.json").read_bytes()).hexdigest() == hashes[0]
    bad += not good
    print(f"{'ok ' if good else 'BAD'} analysis.json sha256 as quoted: {good}")
    print(f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
