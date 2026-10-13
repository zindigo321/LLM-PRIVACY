"""Derived stage summary of the closed synthetic comparison: U-FIT (UID-20261012-v2 r1) and R-FIT (URC-20261013-v1).

Read-only. Every number is read from the saved records after their bytes match the hash lists sealed at the end of
each run: r1 ``FINAL_ARTIFACTS_SHA256.txt`` (U-FIT run root), URC ``analysis/FINAL_ARTIFACTS_SHA256.txt`` (R-FIT run
root and mirror) and URC ``EVIDENCE_INVENTORY_SHA256.txt`` (URC analysis and report). The U-FIT exploratory outcome is
recomputed with the frozen EP classifier from the verified r1 records. No model code is imported. Writes one new file
(LF bytes) and refuses to write if any file read fails its hash check.

    python -B extract_summary.py --out stage_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
EP_DIR = DIAG / "exploratory_progression_20261012"
PINNED = {
    EP_DIR / "ep_classify.py": "f7efd10029e8e50071b9a5013b23eefb1d2e3707d07d05e80fda4a690e160cd1",
    EP_DIR / "POLICY.md": "fe14a225d16fed0373314eb89291035a2c8175bb17f37a7a03cf21c8e7fca592",
}
sys.path.insert(0, str(EP_DIR))
import ep_classify as E  # noqa: E402  (frozen classifier; its bytes are checked in main before any use)

U_RUN = Path("D:/cgfed-runs/uid_20261012_v2_r1")
R_RUN = Path("D:/cgfed-runs/urc_20261013_v1")
U_LIST = DIAG / "uid_20261012_v2_r1_execution" / "FINAL_ARTIFACTS_SHA256.txt"
R_LIST = DIAG / "urc_20261013_v1_execution" / "analysis" / "FINAL_ARTIFACTS_SHA256.txt"
R_INVENTORY = DIAG / "urc_20261013_v1_execution" / "EVIDENCE_INVENTORY_SHA256.txt"
R_ANALYSIS = DIAG / "urc_20261013_v1_execution" / "analysis" / "analysis.json"
R_REPORT = DIAG / "urc_20261013_v1_execution" / "REPORT.md"
U_ASSESSMENT = EP_DIR / "assessment_uid_r1_ufit.json"
CITED_ANALYSIS_SHA256 = "728f7934f07966fd06bdabbea8bbcfdf529fa51899ced1c8736d7e8db1c59cf6"  # URC REPORT.md
TIMING_SCOPE = (
    "arm_wall_s = supervisor time.monotonic() from immediately before the arm's start record is written until "
    "the child process has exited (popen.wait). It includes the child's interpreter start, imports, CPU setup, "
    "model build, input and baselines, the step-0 measurement, the 3,000 updates with curve-point measurements, "
    "the final evaluation and the child record; for R-FIT also writing and reading back two state files. It "
    "excludes package verification (charged to the invocation aggregate) and the end record."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def paths(run: Path, arm: str) -> dict[str, Path]:
    led = run / "ledger"
    return {
        "child": run / "arms" / arm / "child_record.json",
        "end": led / f"end_{arm}.json",
        "invocation": led / "invocation_001.json",
        "invocation_end": led / "invocation_001_end.json",
    }


def arm_summary(p: dict[str, Path]) -> dict:
    c, end, inv, inv_end = (E.load(p[k]) for k in ("child", "end", "invocation", "invocation_end"))
    fe, ft = c["final"]["eval_mode"], c["final"]["train_mode"]
    t, pe = fe["target_energy"], fe["pred_energy"]
    rse, cos = fe["sse"] / t, fe["dot"] / math.sqrt(pe * t)
    return {
        "status": c["status"],
        "status_end_record": end["status"],
        "updates_done": c["updates_done"],
        "parameters": c["trainable"]["parameters"],
        "input_sha256": c["input"]["sha256"],
        "final_eval": {
            "rse": fe["rse"],
            "cosine": fe["cosine"],
            "norm_ratio": fe["norm_ratio"],
            "mse": fe["mse"],
            "recomputed_from_sums_equal": (rse, cos) == (fe["rse"], fe["cosine"]),
        },
        "final_train": {"rse": ft["rse"], "cosine": ft["cosine"]},
        "baselines": {
            "zero_mse": c["baselines"]["zero_prediction"]["mse"],
            "scalar_mean_mse": c["baselines"]["shared_scalar_mean"]["mse"],
        },
        "timing": {
            "arm_wall_s": end["arm_wall_s"],
            "child_process_s": c["child_monotonic_s"]["end"] - c["child_monotonic_s"]["begin"],
            "invocation_aggregate_wall_s": inv_end["aggregate_wall_s"],
            "invocation_record_wall_clock_utc": inv["wall_clock_utc"],
            "threads_intra_inter": [c["cpu"]["intra_op_threads"], c["cpu"]["inter_op_threads"]],
            "state_files_saved": "init_state_file" in c and "final_state_file" in c,
        },
        "memory": {
            "supervisor_sampled_max_rss_bytes": end["guard"]["sampled_max_rss_bytes"],
            "os_peak_wset_bytes": c["rss"]["os_peak_wset_bytes"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="derived stage summary of the closed synthetic comparison")
    ap.add_argument("--out", type=Path, required=True, help="new JSON file (never overwritten)")
    args = ap.parse_args()
    bad = [str(k) for k, v in PINNED.items() if E.sha256_file(k) != v]
    if bad:
        raise SystemExit(f"frozen EP files changed: {bad}")
    up, rp = paths(U_RUN, "U-FIT"), paths(R_RUN, "R-FIT")
    u_rows, u_bad = E.verify_files(U_LIST, list(up.values()))
    r_rows, r_bad = E.verify_files(R_LIST, list(rp.values()))
    i_rows, i_bad = E.verify_files(R_INVENTORY, [R_ANALYSIS, R_REPORT])
    problems = u_bad + r_bad + i_bad
    if problems:
        print(json.dumps(problems, indent=2))
        return 1
    u, r = arm_summary(up), arm_summary(rp)
    analysis = E.load(R_ANALYSIS)
    ep_u = E.classify(E.assess_uid_r1())  # re-hashes all r1 list entries and the r1 manifest before any value
    stored = E.load(U_ASSESSMENT)
    raw = R_ANALYSIS.read_bytes()
    doc = {
        "schema": "synthetic-comparison-stage-summary-v1",
        "label": "DERIVED from sealed records; no run reopened, nothing recomputed by a model",
        "verified": {
            "r1_final_list": {"path": str(U_LIST), "entries": len(u_rows), "mismatching": 0},
            "urc_sealed_list": {"path": str(R_LIST), "entries": len(r_rows), "mismatching": 0},
            "urc_evidence_inventory": {"path": str(R_INVENTORY), "entries": len(i_rows), "mismatching": 0},
            "frozen_ep_files": {str(k): v for k, v in PINNED.items()},
        },
        "u_fit": u,
        "r_fit": r,
        "strict_near_identity": {
            "criteria": "eval mode, finite, RSE <= 0.01 and cosine >= 0.99 (frozen UID config)",
            "u_fit": u["status"],
            "r_fit": r["status"],
        },
        "exploratory_ep_20261012_v1": {
            "u_fit_recomputed": ep_u["outcome"],
            "u_fit_stored_assessment": stored["exploratory"]["outcome"],
            "u_fit_stored_assessment_sha256": E.sha256_file(U_ASSESSMENT),
            "u_fit_basis": "post hoc (policy selected after the U-FIT result was seen)",
            "r_fit": analysis["exploratory_ep"]["outcome"],
            "r_fit_basis": "pre-specified in URC-20261013-v1",
        },
        "matched_comparison": {
            "matching": analysis["matching"]["status"],
            "violations": analysis["matching"]["violations"],
            "shared_conditions_checked": len(analysis["matching"]["shared_conditions"]),
            "permitted_differences": analysis["matching"]["permitted_differences"],
            "delta_rse_r_minus_u": r["final_eval"]["rse"] - u["final_eval"]["rse"],
            "delta_cosine_r_minus_u": r["final_eval"]["cosine"] - u["final_eval"]["cosine"],
            "analysis_script_deltas": [
                analysis["comparison"]["rse_difference_r_minus_u"],
                analysis["comparison"]["cosine_difference_r_minus_u"],
            ],
            "seeds_per_architecture": 1,
        },
        "timing": {
            "scope": TIMING_SCOPE,
            "arm_wall_ratio_u_over_r": u["timing"]["arm_wall_s"] / r["timing"]["arm_wall_s"],
            "background_load": "not controlled and not recorded; same machine, interpreter, torch build and 24/24 "
            "threads; the two invocations started at the UTC times recorded above (different calendar days)",
        },
        "clarifications": {
            "urc_analysis_json_hash": {
                "cited_in_report": CITED_ANALYSIS_SHA256,
                "cited_value_present_in_report": CITED_ANALYSIS_SHA256
                in R_REPORT.read_text(encoding="utf-8"),
                "file_bytes": len(raw),
                "file_sha256": sha256(raw),
                "crlf_line_endings": raw.count(b"\r\n"),
                "lf_normalized_sha256": sha256(raw.replace(b"\r\n", b"\n")),
                "lf_normalized_equals_cited": sha256(raw.replace(b"\r\n", b"\n")) == CITED_ANALYSIS_SHA256,
            },
        },
        "not_established": [
            "statistical significance (one seed per architecture; seed-to-seed variation never measured)",
            "general architectural superiority (the permitted differences remain confounds)",
            "compression (latent [64, 32, 32] is a 64:1 element expansion)",
            "generalization, real-LoRA behaviour or downstream utility",
        ],
    }
    data = (json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.out.open("xb") as fh:
        fh.write(data)
    print(f"wrote {args.out} ({len(data)} bytes) sha256={sha256(data)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
