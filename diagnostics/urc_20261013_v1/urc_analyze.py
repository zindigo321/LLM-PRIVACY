"""URC-20261013-v1 post-run analysis (read-only on both run roots; never run before the scientific invocation ends).

``python -B urc_analyze.py --evidence-dir <new dir>`` does, in order:
1. seal: write ``FINAL_ARTIFACTS_SHA256.txt`` (SHA-256 and size of every file in the URC run root and ledger mirror);
2. strict: re-derive the frozen near-identity verdict (RSE <= 0.01, cosine >= 0.99, eval mode) from recorded sums;
3. exploratory: apply the frozen EP-20261012-v1 classifier (``ep_classify.classify``, hash-pinned) to R-FIT;
4. matching: shared conditions vs the U-FIT reference record (MATCHED / NOT_MATCHED), permitted differences listed;
5. comparison: signed differences R-FIT - U-FIT in final eval RSE and cosine, with resource usage of both runs.

Imports no model code, runs no forward pass and writes only new files in ``--evidence-dir``. No significance or
architectural-superiority claim is computed: one seed per architecture gives no variance estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

URC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(URC_DIR))

import urc_control as U  # noqa: E402

K = U.K
sys.path.insert(0, str(U.EP_DIR))
import ep_classify as E  # noqa: E402  (frozen EP classifier; hash-pinned in urc_control.PINNED_SHA256)

SUMS = ("finite", "n", "sse", "target_energy", "pred_energy", "dot")
SHARED_RECORD_FIELDS = {  # dotted path in both child records -> condition name
    "input.sha256": "input bytes",
    "input.shape": "input shape",
    "input.dtype": "input dtype",
    "input.energy_sum_sq": "target energy",
    "baselines.zero_prediction.mse": "zero-prediction MSE",
    "baselines.shared_scalar_mean.mse": "scalar-mean MSE",
    "cpu.intra_op_threads": "intra-op threads",
    "cpu.inter_op_threads": "inter-op threads",
    "cpu.deterministic_algorithms": "deterministic algorithms",
    "cpu.cuda_available": "CUDA available (must be false)",
    "cpu.torch_version": "torch version",
    "updates_done": "updates done",
    "verdict_basis": "verdict basis",
    "final.copies_loaded_identical_state": "final copies loaded one state",
    "step0.isolation_verified": "step-0 isolation verified",
    "env.CUDA_VISIBLE_DEVICES": "CUDA_VISIBLE_DEVICES",
    "env.PYTHONPATH": "PYTHONPATH (must be unset)",
}
PERMITTED_DIFFERENCES = [
    "core block architecture: 3 + 3 ResNet residual blocks (conv, BatchNorm) instead of 3 + 3 BasicUformer layers",
    "parameter count: 493,483 instead of 685,507 (sampling-layer shapes identical, 24,833 + 24,746)",
    "core normalization layers: BatchNorm2d in the residual blocks instead of LayerNorm in the LeWin blocks",
    "architecture-dependent initialization: PyTorch defaults only; the Uformer cores add trunc-normal(0.02) init; "
    "the RNG stream after the encoder sampling layers differs, so decoder sampling-layer initial values differ",
]


def get(rec: dict, dotted: str) -> Any:
    for part in dotted.split("."):
        rec = rec[part]
    return rec


def seal(evidence: Path, roots: list[Path]) -> Path:
    out = evidence / "FINAL_ARTIFACTS_SHA256.txt"
    lines = [f"recorded: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    for root in roots:
        for p in sorted(q for q in root.rglob("*") if q.is_file()):
            lines.append(f"{K.sha256_file(p)} {p.stat().st_size:>8}  {p.resolve()}")
    with out.open("x", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


PLAN_KEYS = (  # execution settings compared between the two invocation plans
    "aggregate_seconds",
    "cpu",
    "curve_every",
    "end_margin_s",
    "eval_reserve_s",
    "evidence_cap_bytes",
    "interrupt_grace_s",
    "kill_wait_s",
    "poll_interval_s",
    "python_executable",
    "rss_available_fraction",
    "rss_cap_bytes",
    "torch_version",
)


def execution_conditions(r_inv: dict, u_inv: dict, r_curve: list, u_curve: list) -> dict[str, tuple]:
    """Plan settings, the arm's update/time budget and the curve-point schedule of both invocations."""
    rp, up = r_inv["plan"], u_inv["plan"]
    u_arm = next(a for a in up["arms"] if a["arm"] == "U-FIT")
    out = {f"plan.{k}": (rp.get(k, "absent"), up.get(k, "absent")) for k in PLAN_KEYS}
    out["arm max_updates"] = (rp["arms"][0]["max_updates"], u_arm["max_updates"])
    out["arm max_seconds"] = (rp["arms"][0]["max_seconds"], u_arm["max_seconds"])
    out["curve-point schedule"] = (r_curve, u_curve)
    return out


def curve_points(path: Path) -> list[int]:
    return [
        json.loads(line)["update"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def matching(rfit: dict, ufit: dict, resolved: dict, extra: dict | None = None) -> dict[str, Any]:
    """Shared conditions compared exactly; any violation is NOT_MATCHED. Permitted differences are listed apart."""
    rows = [
        {"condition": name, "field": "invocation plan / curve", "r_fit": a, "u_fit": b, "equal": a == b}
        for name, (a, b) in (extra or {}).items()
    ]
    for path, name in SHARED_RECORD_FIELDS.items():
        try:
            a, b = get(rfit, path), get(ufit, path)
        except (KeyError, TypeError):
            a, b = "absent", "absent"
        rows.append(
            {"condition": name, "field": path, "r_fit": a, "u_fit": b, "equal": a == b and a != "absent"}
        )
    ufit_opt = K.load_json(URC_DIR / resolved["reference"]["frozen_config"])["u_fit"]["optimizer"]
    rows.append(
        {
            "condition": "optimizer (complete settings)",
            "field": "resolved optimizer vs frozen U-FIT optimizer",
            "r_fit": resolved["optimizer"],
            "u_fit": ufit_opt,
            "equal": resolved["optimizer"] == ufit_opt,
        }
    )
    groups = list(rfit["param_change_total"])
    rows.append(
        {
            "condition": "measurement groups",
            "field": "param_change_total keys",
            "r_fit": groups,
            "u_fit": list(ufit["param_change_total"]),
            "equal": groups == list(ufit["param_change_total"]),
        }
    )
    violations = [r["condition"] for r in rows if not r["equal"]]
    return {
        "status": "MATCHED" if not violations else "NOT_MATCHED",
        "violations": violations,
        "shared_conditions": rows,
        "permitted_differences": PERMITTED_DIFFERENCES,
    }


def reference_artifact_problems(resolved: dict, ufit_path: Path) -> list[str]:
    """Every U-FIT run-root file read here must match the r1 final hash list (``FINAL_ARTIFACTS_SHA256.txt``)."""
    listing = URC_DIR / resolved["reference"]["final_artifacts_list"]
    rows, problems = E.verify_files(
        listing,
        [
            ufit_path,
            ufit_path.parent / "curve.jsonl",
            ufit_path.parents[2] / "ledger" / "end_U-FIT.json",
            ufit_path.parents[2] / "ledger" / "invocation_001.json",
            ufit_path.parents[2] / "ledger" / "invocation_001_end.json",
        ],
    )
    return [f"U-FIT reference: {p}" for p in problems]


def resources(end: dict, child: dict, inv_end: dict) -> dict[str, Any]:
    return {
        "arm_wall_s": end["arm_wall_s"],
        "aggregate_wall_s": inv_end["aggregate_wall_s"],
        "guard_sampled_max_rss_bytes": end["guard"]["sampled_max_rss_bytes"],
        "child_sampled_max_rss_bytes": child["rss"]["sampled_max_rss_bytes"],
        "os_peak_wset_bytes": child["rss"]["os_peak_wset_bytes"],
        "evidence_bytes_run_root": inv_end["evidence_bytes_run_root"],
        "updates_done": child.get("updates_done"),
        "seconds_per_update": end["arm_wall_s"] / child["updates_done"]
        if child.get("updates_done")
        else None,
    }


def classifier_input(
    rfit: dict, end: dict, inv: dict, inv_end: dict, resolved: dict, ev: dict, starts: int
) -> dict:
    """The EP classifier's input for R-FIT, built like ``ep_classify.uid_r1_input`` from verified records."""
    lim = resolved["limits"]
    rss = (
        end["guard"]["sampled_max_rss_bytes"],
        rfit["rss"]["sampled_max_rss_bytes"],
        rfit["rss"]["os_peak_wset_bytes"],
    )
    return {
        "evidence": ev,
        "run": {
            "status": rfit["status"],
            "updates_done": rfit.get("updates_done"),
            "updates_planned": resolved["max_updates"],
            "prerequisite": None,  # URC-v1 has no prerequisite arm
            "interrupted": inv_end["interrupted"],
            "guard_actions": [] if end["guard"]["action"] is None else [end["guard"]["action"]],
            "limits": {
                "arm_wall_s": {"used": end["arm_wall_s"], "limit": resolved["max_seconds"]},
                "aggregate_wall_s": {"used": inv_end["aggregate_wall_s"], "limit": lim["aggregate_seconds"]},
                "training_starts": {"used": starts, "limit": lim["max_training_starts"]},
                "invocations": {"used": 1, "limit": lim["max_invocations"]},
                "rss_bytes": {"used": max(v for v in rss if v is not None), "limit": inv["rss_limit_bytes"]},
                "evidence_bytes": {
                    "used": inv_end["evidence_bytes_run_root"],
                    "limit": lim["evidence_cap_bytes"],
                },
            },
        },
        "final_eval": {k: rfit["final"]["eval_mode"][k] for k in SUMS},
        "baselines": {"scalar_mean_mse": rfit["baselines"]["shared_scalar_mean"]["mse"]},
    }


def final_metrics(m: dict) -> dict[str, Any]:
    """RSE and cosine recomputed from the recorded sums (the frozen formulas), compared with the recorded values."""
    t, p = m["target_energy"], m["pred_energy"]
    rse = m["sse"] / t if t > 0 else None
    cos = m["dot"] / math.sqrt(p * t) if p > 0 and t > 0 else None
    return {"rse": rse, "cosine": cos, "recorded_equal": (rse, cos) == (m["rse"], m["cosine"])}


def analyze(
    resolved: dict,
    evidence: Path,
    run_root: Path | None = None,
    mirror: Path | None = None,
    arm: str = U.ARM,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Scientific use: defaults only (the registered run root, mirror, arm and the frozen manifest). The overrides
    exist for the engineering test on a fixture run; they never change a registered path."""
    run_root = Path(run_root or resolved["execution"]["run_root"])
    mirror = Path(mirror or resolved["execution"]["ledger_mirror"])
    sealed = seal(evidence, [run_root, mirror])
    ledger = K.Ledger(run_root / "ledger", mirror).read()  # refuses if the two roots disagree
    starts = sum(1 for n in ledger if n.startswith("start_"))
    inv, inv_end, end = (
        ledger["invocation_001.json"],
        ledger["invocation_001_end.json"],
        ledger[f"end_{arm}.json"],
    )
    rfit = K.load_json(run_root / "arms" / arm / "child_record.json")
    ref = resolved["reference"]
    ufit_path = Path(ref["child_record"])
    problems = []
    if K.sha256_file(ufit_path) != ref["child_record_sha256"]:
        problems.append("U-FIT child_record.json differs from its pinned hash")
    problems += reference_artifact_problems(resolved, ufit_path)
    if manifest_sha256 is None:
        manifest_sha256 = K.sha256_file(URC_DIR / U.MANIFEST_NAME)
    if inv["identity"].get("urc_manifest_sha256") != manifest_sha256:
        problems.append("the invocation names another URC manifest")
    if not (rfit["status"] == end["status"] == inv_end["arms"][0]["status"]):
        problems.append(
            f"status disagreement: {rfit['status']}, {end['status']}, {inv_end['arms'][0]['status']}"
        )
    ufit = K.load_json(ufit_path)
    ev = {"verified": not problems, "problems": problems, "sealed_list": str(sealed)}
    out: dict[str, Any] = {
        "schema": "urc-v1-analysis",
        "protocol_id": U.PROTOCOL_ID,
        "evidence": ev,
        "status_recorded": rfit["status"],
        "resources": {"r_fit": resources(end, rfit, inv_end)},
    }
    u_end = K.load_json(ufit_path.parents[2] / "ledger" / "end_U-FIT.json")
    u_inv_end = K.load_json(ufit_path.parents[2] / "ledger" / "invocation_001_end.json")
    out["resources"]["u_fit_reference"] = resources(u_end, ufit, u_inv_end)
    if rfit["status"] not in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL"):
        out["note"] = "R-FIT did not complete; no metric comparison is made"
        return out
    r, u = final_metrics(rfit["final"]["eval_mode"]), final_metrics(ufit["final"]["eval_mode"])
    strict = (
        "COMPLETED_PASS"
        if C_passes(rfit["final"]["eval_mode"], resolved["criteria"])
        else "COMPLETED_FIT_FAIL"
    )
    out["strict_near_identity"] = {
        "criteria": resolved["criteria"],
        "rederived_status": strict,
        "matches_record": strict == rfit["status"],
    }
    out["exploratory_ep"] = E.classify(classifier_input(rfit, end, inv, inv_end, resolved, ev, starts))
    u_inv = K.load_json(ufit_path.parents[2] / "ledger" / "invocation_001.json")
    extra = execution_conditions(
        inv,
        u_inv,
        curve_points(run_root / "arms" / arm / "curve.jsonl"),
        curve_points(ufit_path.parent / "curve.jsonl"),
    )
    out["matching"] = matching(rfit, ufit, resolved, extra)
    out["comparison"] = {
        "label": "descriptive, one seed per architecture; no significance or superiority claim",
        "r_fit_final_eval": r,
        "u_fit_final_eval": u,
        "rse_difference_r_minus_u": r["rse"] - u["rse"],
        "cosine_difference_r_minus_u": r["cosine"] - u["cosine"],
        "r_fit_final_train": final_metrics(rfit["final"]["train_mode"]),
        "u_fit_final_train": final_metrics(ufit["final"]["train_mode"]),
        "valid_as_matched_comparison": out["matching"]["status"] == "MATCHED" and ev["verified"],
    }
    return out


def C_passes(m: dict, criteria: dict) -> bool:  # noqa: N802  (the frozen uid_common.passes rule, torch-free)
    if not m["finite"] or m["rse"] is None or m["cosine"] is None:
        return False
    if not (math.isfinite(m["rse"]) and math.isfinite(m["cosine"])):
        return False
    return m["rse"] <= float(criteria["max_rse"]) and m["cosine"] >= float(criteria["min_cosine"])


def main() -> int:
    ap = argparse.ArgumentParser(description="URC-v1 post-run seal and analysis (read-only on the run roots)")
    ap.add_argument("--evidence-dir", type=Path, required=True, help="new directory for the analysis outputs")
    args = ap.parse_args()
    U.check_pinned()
    resolved = U.load_resolved()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    out = analyze(resolved, args.evidence_dir)
    text = json.dumps(K.json_safe(out), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with (args.evidence_dir / "analysis.json").open("x", encoding="utf-8") as fh:
        fh.write(text)
    print(f"analysis.json sha256={hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    print(json.dumps({k: out.get(k) for k in ("status_recorded", "strict_near_identity")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
