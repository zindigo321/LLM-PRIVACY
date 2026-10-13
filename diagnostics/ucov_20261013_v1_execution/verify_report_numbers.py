"""Check the numbers quoted in REPORT.md against the evidence files (read-only; prints a table; exit 1 on a problem).

``python -B verify_report_numbers.py``. Each check pairs a value from the evidence with the text that must appear in
REPORT.md.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path("D:/cgfed-runs/ucov_20261013_v1")
ARMS = ("U-TILE-1", "U-TILE-2", "U-TILE-3", "U-TILE-4")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    rep = (HERE / "REPORT.md").read_text(encoding="utf-8")
    an, iv = load(HERE / "analysis" / "analysis.json"), load(HERE / "independent_verification.json")
    inv_end = load(RUN / "ledger" / "invocation_001_end.json")
    checks = []
    for arm in ARMS:
        a, q = an["arms"][arm], an["arms"][arm]["primary"]["quantities"]
        row = f"| {arm} |"
        line = next((ln for ln in rep.splitlines() if ln.startswith(row)), "")
        for v, fmt in (
            (q["rse"], ".5f"),
            (q["cosine"], ".5f"),
            (q["mse"], ".5e"),
            (q["zero_mse"], ".5e"),
            (q["scalar_mean_mse"], ".5e"),
            (q["mse_over_best_baseline"], ".5f"),
        ):
            s = format(v, fmt).replace("e-0", "e-0")
            checks.append((f"{arm} {s}", s in line))
        checks.append((f"{arm} improvement", f"{100 * q['baseline_improvement']:.1f}%" in line))
        checks.append((f"{arm} primary PASS", a["primary"]["outcome"] == "PASS" and "**PASS**" in line))
        checks.append((f"{arm} strict FAIL", a["secondary"]["historical_strict"]["outcome"] == "FAIL"))
        checks.append((f"{arm} EP PASS", a["secondary"]["ep_v1"]["outcome"] == "PASS"))
        res = a["resources"]
        rline = next((ln for ln in rep.splitlines() if ln.startswith(row) and "B |" in ln), "")
        for v in (res["arm_wall_s"], res["child_process_s"], iv["arms"][arm]["curve"]["train_loop_s"]):
            checks.append((f"{arm} {v:.3f}", f"{v:.3f}" in rline))
        checks.append((f"{arm} s/upd", f"({iv['arms'][arm]['curve']['mean_s_per_update']:.3f})" in rline))
        for v in (res["guard"]["sampled_max_rss_bytes"], res["os_peak_wset_bytes"]):
            checks.append((f"{arm} {v:,}", f"{v:,} B" in rline))
        tr = a["reported_only"]["final_train_original"]
        checks.append((f"{arm} train rse", f"{tr['rse']:.4f}" in rep))
    checks += [
        (
            "aggregate met",
            an["aggregate"]["decision"] == "COVERAGE_CRITERION_MET" and "P = 4, F = 0, U = 0" in rep,
        ),
        ("aggregate s", f"{inv_end['aggregate_wall_s']:,.3f}" in rep),
        ("evidence bytes", f"{inv_end['evidence_bytes_run_root']:,} B" in rep),
        (
            "analysis sha",
            hashlib.sha256((HERE / "analysis" / "analysis.json").read_bytes()).hexdigest() in rep,
        ),
        ("analysis bytes", f"{(HERE / 'analysis' / 'analysis.json').stat().st_size:,} bytes" in rep),
        ("manifest sha", "3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824" in rep),
        ("sealed entries", f"covers {iv['sealed']['entries']} files" in rep),
        ("4/4 starts", iv["ledger"]["starts"] == 4 and "4 of 4 starts" in rep),
        (
            "min rse margin",
            f"at least {min(0.5 - an['arms'][x]['primary']['quantities']['rse'] for x in ARMS):.3f}" in rep,
        ),
        (
            "min cos margin",
            f"at least {min(an['arms'][x]['primary']['quantities']['cosine'] - 0.65 for x in ARMS):.3f}"
            in rep,
        ),
        (
            "max ratio",
            f"MSE/best ≤ {max(an['arms'][x]['primary']['quantities']['mse_over_best_baseline'] for x in ARMS):.3f}"
            in rep,
        ),
    ]
    bad = [c for c in checks if not c[1]]
    for name, ok in checks:
        print(("ok  " if ok else "BAD ") + name)
    print(f"{len(bad)} problem(s) in {len(checks)} checks")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
