"""UCOV-20261013-v1 post-run analysis (read-only on the run root and mirror; never run before the invocation ends).

``python -B ucov_analyze.py --evidence-dir <new dir>`` runs these steps in order:
1. **Seal.** Write ``FINAL_ARTIFACTS_SHA256.txt``: SHA-256 and size of every file in the run root and the ledger
   mirror.
2. **Evidence, per arm.**
   - The ledger roots agree, and the invocation names the frozen UCOV manifest and the primary policy hashes.
   - Child, end and summary statuses agree.
   - The run's tile, initial state, module and policy hashes are the registered ones.
   - Both state files have their recorded hashes.
3. **Primary.** UPP-20261013-v1 through ``upp_classify.classify`` (hash-pinned), per arm, and the four-arm
   ``aggregate``. These decide the study.
4. **Secondary references, reported only.**
   - The historical strict near-identity criterion (0.01 / 0.99).
   - The frozen EP-20261012-v1 ``ep_classify.classify``.
   Neither can veto a primary PASS.
5. **Reported only.** Normalized-space and train-mode quantities, curves, resources, and the URP-v1 centre tile for
   context.

The analyzer imports no model code, runs no forward pass, and writes only new files in ``--evidence-dir``.
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

UCOV_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UCOV_DIR))

import ucov_control as U  # noqa: E402

K = U.K
sys.path.insert(0, str(U.EP_DIR))
import ep_classify as E  # noqa: E402  (frozen EP classifier; hash-pinned in ucov_control.PINNED_SHA256)
import upp_classify as P  # noqa: E402  (primary policy classifier; hash-pinned)

SUMS = ("finite", "n", "sse", "target_energy", "pred_energy", "dot")
URP_CONTEXT = {
    "label": "context only: the URP-20261013-v1 centre tile (32, 24), a separate closed experiment; not a fifth arm",
    "final_eval_rse": 0.19345271823265606,
    "final_eval_cosine": 0.8981045580444489,
    "final_eval_mse": 7.99371986840454e-07,
    "recorded_status_strict": "COMPLETED_FIT_FAIL",
    "ep_v1_outcome_prespecified": "PASS",
    "upp_v1_outcome_retrospective": "PASS (retrospective application; does not replace its recorded verdicts)",
}
NOT_ESTABLISHED = [
    "compression (latent [64, 32, 32] is a 64:1 element expansion of each 32 x 32 tile)",
    "held-out generalization: each arm trains its own model on its own tile and is evaluated on that same tile",
    "reconstruction of the whole snapshot, other snapshots, other layers' v/o blocks or any other tile",
    "statistical significance or seed stability (one seed, one start per arm)",
    "superiority of Uformer or of any architecture; downstream or FAF utility",
    "that a failing tile shows the URP-v1 centre tile was uniquely favourable",
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


def strict_reference(m: dict, criteria: dict) -> str:
    """The historical strict near-identity verdict (frozen uid_common.passes rule), torch-free; reported only."""
    if not m["finite"] or m["rse"] is None or m["cosine"] is None:
        return "FAIL"
    if not (math.isfinite(m["rse"]) and math.isfinite(m["cosine"])):
        return "FAIL"
    ok = m["rse"] <= float(criteria["max_rse"]) and m["cosine"] >= float(criteria["min_cosine"])
    return "PASS" if ok else "FAIL"


def recomputed(m: dict) -> dict[str, Any]:
    t, p = m["target_energy"], m["pred_energy"]
    rse = m["sse"] / t if t > 0 else None
    cos = m["dot"] / math.sqrt(p * t) if p > 0 and t > 0 else None
    return {"rse": rse, "cosine": cos, "recorded_equal": (rse, cos) == (m["rse"], m["cosine"])}


def arm_problems(
    arm: str,
    child: dict | None,
    inv: dict,
    end: dict | None,
    summary_arm: dict,
    resolved: dict,
    run_root: Path,
    manifest_sha256: str,
) -> list[str]:
    """Evidence problems of one started arm (the invocation-level identity is checked for every arm in ``analyze``)."""
    problems = []
    pol = resolved["primary_policy"]
    if child is None or end is None:
        return [*problems, "child record or end record missing"]
    if not (child["status"] == end["status"] == summary_arm["status"]):
        problems.append(f"status disagreement: {child['status']}, {end['status']}, {summary_arm['status']}")
    tile = resolved["inputs"]["tiles"][arm]
    if child.get("input", {}).get("patch_file_sha256") != tile["patch_file_sha256"]:
        problems.append("the run read another tile")
    if not child.get("input_checks", {}).get("all_equal"):
        problems.append("the run's input checks did not all hold")
    if child.get("model_state_sha256_init") != resolved["reference"]["model_state_sha256_init"]:
        problems.append("the run's initial state is not the U-FIT initial state")
    if child.get("modules", {}).get("checked_against_approved") is not True:
        problems.append("the run's modules were not checked against the approved hashes")
    if child["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL"):
        cp = child.get("primary_policy") or {}
        if (cp.get("policy_id"), cp.get("policy_sha256")) != (pol["policy_id"], pol["policy_sha256"]):
            problems.append("the child applied another primary policy")
        for point in ("init", "final"):
            info = child.get(f"{point}_state_file")
            key = "model_state_sha256_init" if point == "init" else "model_state_sha256_final"
            path = run_root / "arms" / arm / f"{point}_state.safetensors"
            if info is None or not path.is_file():
                problems.append(f"{point} state file missing")
            elif K.sha256_file(path) != info["file_sha256"] or info["state_sha256"] != child.get(key):
                problems.append(f"{point} state file differs from its record")
    return problems


def limits_block(
    arm_end: dict | None, child: dict | None, inv: dict, inv_end: dict, resolved: dict, starts: int
):
    lim = resolved["limits"]
    rss_vals = [] if arm_end is None else [arm_end["guard"]["sampled_max_rss_bytes"]]
    if child is not None and child.get("rss"):
        rss_vals += [child["rss"]["sampled_max_rss_bytes"], child["rss"]["os_peak_wset_bytes"]]
    rss_used = max((v for v in rss_vals if v is not None), default=None)
    return {
        "arm_wall_s": {
            "used": None if arm_end is None else arm_end.get("arm_wall_s"),
            "limit": resolved["max_seconds"],
        },
        "aggregate_wall_s": {"used": inv_end["aggregate_wall_s"], "limit": lim["aggregate_seconds"]},
        "training_starts": {"used": starts, "limit": lim["max_training_starts"]},
        "invocations": {"used": 1, "limit": lim["max_invocations"]},
        "rss_bytes": {"used": rss_used, "limit": inv["rss_limit_bytes"]},
        "evidence_bytes": {"used": inv_end["evidence_bytes_run_root"], "limit": lim["evidence_cap_bytes"]},
    }


def primary_input(arm, child, arm_end, summary_arm, inv, inv_end, resolved, ev, starts) -> dict[str, Any]:
    """The UPP-20261013-v1 classifier input of one registered arm (started or not)."""
    exp = resolved["inputs_expected"][arm]
    started = arm_end is not None
    rec: dict[str, Any] = {
        "evidence": ev,
        "input": {
            "all_finite": exp["degeneracy_checks"].get("all_finite"),
            "target_energy": exp["original_target_energy"],
            "scalar_mean_mse": exp["original_scalar_mean_mse"],
            "normalized_max_abs": exp["normalized_max_abs"],
        },
        "run": {
            "started": started,
            "status": summary_arm["status"]
            if not started
            else (child or {}).get("status", arm_end["status"]),
            "updates_done": None if child is None else child.get("updates_done"),
            "updates_planned": resolved["max_updates"],
            "interrupted": inv_end["interrupted"],
            "guard_actions": []
            if arm_end is None or arm_end["guard"]["action"] is None
            else [arm_end["guard"]["action"]],
            "limits": limits_block(arm_end, child, inv, inv_end, resolved, starts),
        },
        "baselines": {
            "zero_prediction_mse": exp["original_zero_prediction_mse"],
            "scalar_mean_mse": exp["original_scalar_mean_mse"],
        },
    }
    if child is not None and child.get("final"):
        rec["final_eval"] = {k: child["final"]["eval_mode"][k] for k in SUMS}
        b = child["baselines"]["original"]
        if (b["zero_prediction"]["mse"], b["shared_scalar_mean"]["mse"]) != (
            exp["original_zero_prediction_mse"],
            exp["original_scalar_mean_mse"],
        ):
            rec["evidence"] = {
                **ev,
                "verified": False,
                "problems": [*ev["problems"], "recorded baselines differ from the registered ones"],
            }
    return rec


def ep_input(prim: dict) -> dict:
    """The frozen EP-20261012-v1 classifier input for the same arm (secondary reference; no prerequisite)."""
    run = prim["run"]
    return {
        "evidence": prim["evidence"],
        "run": {
            **{
                k: run[k]
                for k in (
                    "status",
                    "updates_done",
                    "updates_planned",
                    "interrupted",
                    "guard_actions",
                    "limits",
                )
            },
            "prerequisite": None,
        },
        "final_eval": prim.get("final_eval", {}),
        "baselines": {"scalar_mean_mse": prim["baselines"]["scalar_mean_mse"]},
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
    manifest_sha256: str | None = None,
    arms: tuple[str, ...] = U.ARMS,
) -> dict[str, Any]:
    """Scientific use: defaults only (the registered run root, mirror and the frozen manifest). The overrides exist
    for the engineering test on a fixture run; they never change a registered path."""
    run_root = Path(run_root or resolved["execution"]["run_root"])
    mirror = Path(mirror or resolved["execution"]["ledger_mirror"])
    sealed = seal(evidence, [run_root, mirror])
    ledger = K.Ledger(run_root / "ledger", mirror).read()  # refuses if the two roots disagree
    starts = sum(1 for n in ledger if n.startswith("start_"))
    inv, inv_end = ledger["invocation_001.json"], ledger["invocation_001_end.json"]
    if manifest_sha256 is None:
        manifest_sha256 = K.sha256_file(UCOV_DIR / U.MANIFEST_NAME)
    policy = U.load_primary_policy(resolved["primary_policy"])
    by_arm = {a["arm"]: a for a in inv_end["arms"]}
    global_problems = []  # invocation-level identity: applies to every registered arm, started or not
    if inv["identity"].get("ucov_manifest_sha256") != manifest_sha256:
        global_problems.append("the invocation names another UCOV manifest")
    if inv["identity"].get("primary_policy", {}).get("policy_sha256") != policy["policy_sha256"]:
        global_problems.append("the invocation names another primary policy")
    if inv.get("plan", {}).get("primary_policy_sha256") != policy["policy_sha256"]:
        global_problems.append("the invocation plan names another primary policy")
    if [a["arm"] for a in inv_end["arms"]] != list(arms):
        global_problems.append(
            f"the invocation summary lists {[a['arm'] for a in inv_end['arms']]}, not {list(arms)}"
        )
    out: dict[str, Any] = {
        "schema": "ucov-v1-analysis",
        "protocol_id": U.PROTOCOL_ID,
        "primary_policy": {
            "policy_id": policy["policy_id"],
            "policy_sha256": policy["policy_sha256"],
            "classifier_sha256": resolved["primary_policy"]["classifier_sha256"],
            "thresholds": policy["thresholds"],
        },
        "sealed_list": str(sealed),
        "ledger": {"records": sorted(ledger), "starts": starts, "invocations": 1},
        "global_stop": inv_end.get("global_stop"),
        "arms": {},
    }
    outcomes = {}
    for arm in arms:
        summary_arm = by_arm.get(arm, {"arm": arm, "status": "NOT_RUN", "reason": "absent from the summary"})
        arm_end = ledger.get(f"end_{arm}.json")
        started = f"start_{arm}.json" in ledger
        child_path = run_root / "arms" / arm / "child_record.json"
        child = K.load_json(child_path) if child_path.is_file() else None
        problems = list(global_problems)
        if started:
            problems += arm_problems(
                arm, child, inv, arm_end, summary_arm, resolved, run_root, manifest_sha256
            )
        elif summary_arm["status"] != "NOT_RUN":
            problems.append(f"no start record but summary status {summary_arm['status']}")
        ev = {"verified": not problems, "problems": problems}
        prim_in = primary_input(
            arm, child, arm_end if started else None, summary_arm, inv, inv_end, resolved, ev, starts
        )
        prim = P.classify(prim_in, policy)
        outcomes[arm] = prim["outcome"]
        row: dict[str, Any] = {
            "slice": resolved["inputs_expected"][arm]["slice"],
            "started": started,
            "recorded_status": summary_arm["status"],
            "not_run_reason": summary_arm.get("reason"),
            "primary": prim,
            "evidence": ev,
        }
        if child is not None and child.get("final"):
            fe = child["final"]["eval_mode"]
            child_primary = (child.get("primary_verdict") or {}).get("outcome")
            row["primary_consistency"] = {
                "child_primary_outcome": child_primary,
                "analysis_metric_outcome": prim["outcome"] if prim["class"] != "U" else None,
                "agree": prim["class"] == "U" or child_primary == prim["outcome"],
            }
            row["secondary"] = {
                "historical_strict": {
                    "criteria": resolved["secondary_references"]["historical_strict_criteria"],
                    "outcome": strict_reference(
                        fe, resolved["secondary_references"]["historical_strict_criteria"]
                    ),
                    "role": "reported only; cannot veto the primary verdict",
                },
                "ep_v1": {
                    **E.classify(ep_input(prim_in)),
                    "role": "reported only; cannot veto the primary verdict",
                },
            }
            base = child["baselines"]
            row["reported_only"] = {
                "final_eval_original": {**recomputed(fe), "mse": fe["mse"], "norm_ratio": fe["norm_ratio"]},
                "final_eval_normalized": recomputed(child["final"]["eval_mode_normalized"]),
                "final_train_original": recomputed(child["final"]["train_mode"]),
                "final_train_normalized": recomputed(child["final"]["train_mode_normalized"]),
                "train_minus_eval_output_sse_normalized": child["final"][
                    "train_minus_eval_output_sse_normalized"
                ],
                "baselines_mse": {
                    s: {k: base[s][k]["mse"] for k in ("zero_prediction", "shared_scalar_mean")}
                    for s in ("original", "normalized")
                },
                "step0_eval_rse": {
                    s: child["step0"]["post_init"]["eval"][s]["rse"] for s in ("original", "normalized")
                },
                "curve": curve_summary(run_root / "arms" / arm / "curve.jsonl"),
            }
        if arm_end is not None:
            row["resources"] = {
                "arm_wall_s": arm_end.get("arm_wall_s"),
                "child_process_s": None
                if child is None
                else child["child_monotonic_s"]["end"] - child["child_monotonic_s"]["begin"],
                "guard": arm_end.get("guard"),
                "os_peak_wset_bytes": None if child is None else child["rss"]["os_peak_wset_bytes"],
                "updates_done": None if child is None else child.get("updates_done"),
            }
        out["arms"][arm] = row
    out["aggregate"] = P.aggregate(outcomes, len(arms), policy)
    out["resources"] = {
        "aggregate_wall_s": inv_end["aggregate_wall_s"],
        "aggregate_limit_s": resolved["limits"]["aggregate_seconds"],
        "evidence_bytes_run_root": inv_end["evidence_bytes_run_root"],
        "evidence_cap_bytes": resolved["limits"]["evidence_cap_bytes"],
        "rss_limit_bytes": inv["rss_limit_bytes"],
        "interrupted": inv_end["interrupted"],
    }
    out["context_urp_v1_centre_tile"] = URP_CONTEXT
    out["does_not_establish"] = NOT_ESTABLISHED
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="UCOV-v1 post-run seal and analysis (read-only on the run roots)"
    )
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
    summary = {
        "aggregate": out["aggregate"]["decision"],
        "counts": out["aggregate"]["counts"],
        "arms": {a: r["primary"]["outcome"] for a, r in out["arms"].items()},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
