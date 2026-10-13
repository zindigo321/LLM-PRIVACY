"""URP-20261013-v1 post-run analysis (read-only on the run root and mirror; never run before the invocation ends).

``python -B urp_analyze.py --evidence-dir <new dir>`` does, in order:
1. seal: write ``FINAL_ARTIFACTS_SHA256.txt`` (SHA-256 and size of every file in the URP run root and ledger mirror);
2. evidence: the ledger roots agree, the invocation names the approved URP manifest, child / end / summary statuses
   agree, the run's input, model binding and module hashes are the registered ones, and both state files have their
   recorded hashes;
3. strict: re-derive the frozen near-identity verdict (eval mode, original space; RSE <= 0.01, cosine >= 0.99);
4. exploratory: apply the frozen EP-20261012-v1 classifier (``ep_classify.classify``, hash-pinned) to U-PATCH, in the
   original space, with the patch's own zero and scalar-mean baselines;
5. reported only: the same quantities in normalized space and in train mode, the curve, and resources. The U-FIT
   synthetic result is listed for context; it is not a matched comparison (other input, other scale).

Imports no model code, runs no forward pass and writes only new files in ``--evidence-dir``.
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

URP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(URP_DIR))

import urp_control as U  # noqa: E402

K = U.K
sys.path.insert(0, str(U.EP_DIR))
import ep_classify as E  # noqa: E402  (frozen EP classifier; hash-pinned in urp_control.PINNED_SHA256)

SUMS = ("finite", "n", "sse", "target_energy", "pred_energy", "dot")
NOT_ESTABLISHED = [
    "compression (latent [64, 32, 32] is a 64:1 element expansion of the 32 x 32 patch)",
    "generalization to other patches, other snapshots or unseen snapshots",
    "reconstruction of the whole snapshot, or any property of the Phi tensor outside the patch",
    "statistical significance or seed stability (one seed, one start)",
    "superiority of Uformer or of any architecture; downstream or FAF utility",
]


def seal(evidence: Path, roots: list[Path]) -> Path:
    out = evidence / "FINAL_ARTIFACTS_SHA256.txt"
    lines = [f"recorded: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"]
    for root in roots:
        for p in sorted(q for q in root.rglob("*") if q.is_file()):
            lines.append(f"{K.sha256_file(p)} {p.stat().st_size:>8}  {p.resolve()}")
    with out.open("xb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("utf-8"))
    return out


def recomputed(m: dict) -> dict[str, Any]:
    """RSE and cosine recomputed from the recorded sums (the frozen formulas), compared with the recorded values."""
    t, p = m["target_energy"], m["pred_energy"]
    rse = m["sse"] / t if t > 0 else None
    cos = m["dot"] / math.sqrt(p * t) if p > 0 and t > 0 else None
    return {"rse": rse, "cosine": cos, "recorded_equal": (rse, cos) == (m["rse"], m["cosine"])}


def strict_status(m: dict, criteria: dict) -> str:  # the frozen uid_common.passes rule, torch-free
    if not m["finite"] or m["rse"] is None or m["cosine"] is None:
        return "COMPLETED_FIT_FAIL"
    if not (math.isfinite(m["rse"]) and math.isfinite(m["cosine"])):
        return "COMPLETED_FIT_FAIL"
    ok = m["rse"] <= float(criteria["max_rse"]) and m["cosine"] >= float(criteria["min_cosine"])
    return "COMPLETED_PASS" if ok else "COMPLETED_FIT_FAIL"


def run_problems(
    child: dict, inv: dict, end: dict, inv_end: dict, resolved: dict, manifest_sha256: str, arm: str
):
    """Evidence problems of the run (V1 of EP-20261012-v1 as applied to URP)."""
    problems = []
    if inv["identity"].get("urp_manifest_sha256") != manifest_sha256:
        problems.append("the invocation names another URP manifest")
    if not (child["status"] == end["status"] == inv_end["arms"][0]["status"]):
        problems.append(
            f"status disagreement: {child['status']}, {end['status']}, {inv_end['arms'][0]['status']}"
        )
    if child.get("input", {}).get("patch_file_sha256") != resolved["input"]["patch_file_sha256"]:
        problems.append("the run read another input patch")
    if not child.get("input_checks", {}).get("all_equal"):
        problems.append("the run's input checks did not all hold")
    if child.get("model_state_sha256_init") != resolved["reference"]["model_state_sha256_init"]:
        problems.append("the run's initial state is not the U-FIT initial state")
    if child.get("modules", {}).get("checked_against_approved") is not True:
        problems.append("the run's modules were not checked against the approved hashes")
    for point in ("init", "final"):
        info = child.get(f"{point}_state_file")
        key = "model_state_sha256_init" if point == "init" else "model_state_sha256_final"
        path = Path(resolved["execution"]["run_root"]) / "arms" / arm / f"{point}_state.safetensors"
        if info is None or not path.is_file():
            problems.append(f"{point} state file missing")
        elif K.sha256_file(path) != info["file_sha256"] or info["state_sha256"] != child.get(key):
            problems.append(f"{point} state file differs from its record")
    return problems


def classifier_input(child: dict, end: dict, inv: dict, inv_end: dict, resolved: dict, ev: dict, starts: int):
    """The EP classifier's input for U-PATCH, built like ``ep_classify.uid_r1_input``: original-space final eval sums
    and the patch's original-space scalar-mean baseline."""
    lim = resolved["limits"]
    rss = (
        end["guard"]["sampled_max_rss_bytes"],
        child["rss"]["sampled_max_rss_bytes"],
        child["rss"]["os_peak_wset_bytes"],
    )
    return {
        "evidence": ev,
        "run": {
            "status": child["status"],
            "updates_done": child.get("updates_done"),
            "updates_planned": resolved["max_updates"],
            "prerequisite": None,  # URP-v1 has no prerequisite arm
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
        "final_eval": {k: child["final"]["eval_mode"][k] for k in SUMS},
        "baselines": {"scalar_mean_mse": child["baselines"]["original"]["shared_scalar_mean"]["mse"]},
    }


def curve_summary(path: Path) -> dict[str, Any]:
    pts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    losses = [(p["pre_update_train_loss"], p["update"]) for p in pts]
    return {
        "points": len(pts),
        "updates": [p["update"] for p in pts],
        "pre_update_train_loss_normalized": {str(p["update"]): p["pre_update_train_loss"] for p in pts},
        "pre_update_train_rse_normalized": {str(p["update"]): p["pre_update_train"]["rse"] for p in pts},
        "min_pre_update_loss": list(min(losses)) if losses else None,
        "all_finite": all(math.isfinite(v) for v, _ in losses),
        "grad_nonfinite_total": sum(g["nonfinite"] for p in pts for g in p["grad"].values()),
    }


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
    child = K.load_json(run_root / "arms" / arm / "child_record.json")
    if manifest_sha256 is None:
        manifest_sha256 = K.sha256_file(URP_DIR / U.MANIFEST_NAME)
    resolved_paths = {**resolved, "execution": {**resolved["execution"], "run_root": str(run_root)}}
    problems = run_problems(child, inv, end, inv_end, resolved_paths, manifest_sha256, arm)
    ev = {"verified": not problems, "problems": problems, "sealed_list": str(sealed)}
    out: dict[str, Any] = {
        "schema": "urp-v1-analysis",
        "protocol_id": U.PROTOCOL_ID,
        "evidence": ev,
        "status_recorded": child["status"],
        "resources": {
            "arm_wall_s": end["arm_wall_s"],
            "arm_wall_scope": "supervisor monotonic clock, from just before the start record to the child's exit",
            "child_process_s": child["child_monotonic_s"]["end"] - child["child_monotonic_s"]["begin"],
            "aggregate_wall_s": inv_end["aggregate_wall_s"],
            "guard_sampled_max_rss_bytes": end["guard"]["sampled_max_rss_bytes"],
            "os_peak_wset_bytes": child["rss"]["os_peak_wset_bytes"],
            "evidence_bytes_run_root": inv_end["evidence_bytes_run_root"],
            "rss_limit_bytes": inv["rss_limit_bytes"],
        },
    }
    if child["status"] not in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL"):
        out["note"] = "U-PATCH did not complete; no metric is assessed"
        return out
    fe = child["final"]["eval_mode"]
    strict = strict_status(fe, resolved["criteria"])
    out["strict_near_identity"] = {
        "criteria": resolved["criteria"],
        "space": "original (Phi values)",
        "rederived_status": strict,
        "matches_record": strict == child["status"],
    }
    out["exploratory_ep"] = E.classify(classifier_input(child, end, inv, inv_end, resolved, ev, starts))
    out["exploratory_ep"]["space"] = "original (Phi values); baselines of the patch itself"
    base = child["baselines"]
    out["reported_only"] = {
        "final_eval_original": {**recomputed(fe), "mse": fe["mse"], "norm_ratio": fe["norm_ratio"]},
        "final_eval_normalized": recomputed(child["final"]["eval_mode_normalized"]),
        "final_train_original": recomputed(child["final"]["train_mode"]),
        "final_train_normalized": recomputed(child["final"]["train_mode_normalized"]),
        "train_minus_eval_output_sse_normalized": child["final"]["train_minus_eval_output_sse_normalized"],
        "baselines_mse": {
            space: {k: base[space][k]["mse"] for k in ("zero_prediction", "shared_scalar_mean")}
            for space in ("original", "normalized")
        },
        "step0_eval_rse": {
            s: child["step0"]["post_init"]["eval"][s]["rse"] for s in ("original", "normalized")
        },
        "curve": curve_summary(run_root / "arms" / arm / "curve.jsonl"),
        "u_fit_synthetic_context": {
            "label": "not matched: other input (synthetic), other scale; listed for orientation only",
            "u_fit_final_eval_rse": 0.2727411031905583,
            "u_fit_final_eval_cosine": 0.8540472264633736,
        },
    }
    out["does_not_establish"] = NOT_ESTABLISHED
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="URP-v1 post-run seal and analysis (read-only on the run roots)")
    ap.add_argument("--evidence-dir", type=Path, required=True, help="new directory for the analysis outputs")
    args = ap.parse_args()
    U.check_pinned()
    resolved = U.load_resolved()
    args.evidence_dir.mkdir(parents=True, exist_ok=False)
    out = analyze(resolved, args.evidence_dir)
    data = (json.dumps(K.json_safe(out), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with (args.evidence_dir / "analysis.json").open("xb") as fh:
        fh.write(data)
    print(f"analysis.json {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()} (file bytes)")
    summary = {k: out.get(k) for k in ("status_recorded", "strict_near_identity")}
    summary["ep_outcome"] = out.get("exploratory_ep", {}).get("outcome")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
