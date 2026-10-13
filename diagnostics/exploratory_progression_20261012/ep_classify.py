"""Exploratory progression classifier EP-20261012-v1 (``POLICY.md`` in this directory): derived analysis only.

``classify`` is a pure function of recorded quantities. ``assess_uid_r1`` gathers them from the UID-20261012-v2 r1
records after re-hashing every artifact on the run's final hash list. Nothing here imports model code, runs a forward
pass or edits a record; ``main`` writes one new assessment file and refuses to overwrite one.

    python -B ep_classify.py --out assessment_uid_r1_ufit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

POLICY_ID = "EP-20261012-v1"
MAX_RSE = 0.50
MIN_COSINE = 0.80
BASELINE_FACTOR = 0.80
OUTCOMES = ("UNVERIFIED", "NOT_ELIGIBLE", "FAIL", "UNINFORMATIVE", "PASS")
COMPLETED = ("COMPLETED_PASS", "COMPLETED_FIT_FAIL")
STATS = ("sse", "target_energy", "pred_energy", "dot")
REQUIRED = {
    "evidence": ("verified", "problems"),
    "run": (
        "status",
        "updates_done",
        "updates_planned",
        "prerequisite",
        "interrupted",
        "guard_actions",
        "limits",
    ),
    "final_eval": ("finite", "n", *STATS),
    "baselines": ("scalar_mean_mse",),
}
REQUIRED_LIMITS = (
    "arm_wall_s",
    "aggregate_wall_s",
    "training_starts",
    "invocations",
    "rss_bytes",
    "evidence_bytes",
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UID_EVIDENCE = REPO / "diagnostics" / "uid_20261012_v2_r1_execution"
UID_PACKAGE = REPO / "diagnostics" / "uid_20261012_v2_r1"
UID_RUN_ROOT = Path("D:/cgfed-runs/uid_20261012_v2_r1")
UID_MANIFEST_SHA256 = "9d3aba9a8d5d1157325959a3efeec15c2f872a8e8267cc8f14668b42f01a8044"


def real(v: Any) -> float | None:
    """A finite real number as float; None for an absent, null, boolean, non-numeric or non-finite value."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:
        return None
    return f if math.isfinite(f) else None


def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def compare(value: float, op: str, threshold: float) -> dict[str, Any]:
    """One inclusive float64 comparison against a literal threshold, no epsilon (as ``p5a_metrics.compare``)."""
    ok = value <= threshold if op == "<=" else value >= threshold
    return {"value": value, "operator": op, "threshold": threshold, "margin": value - threshold, "pass": ok}


def missing_keys(rec: Mapping[str, Any]) -> list[str]:
    """Required keys that are absent. A key that is present with a null or non-finite value is a recorded but
    undefined quantity (M0 or V2), not an unrecorded one."""
    out = []
    for section, keys in REQUIRED.items():
        sec = rec.get(section)
        if not isinstance(sec, Mapping):
            out.append(section)
            continue
        out += [f"{section}.{k}" for k in keys if k not in sec]
    limits = rec["run"].get("limits") if isinstance(rec.get("run"), Mapping) else None
    if isinstance(limits, Mapping):
        out += [f"run.limits.{k}" for k in REQUIRED_LIMITS if k not in limits]
    return out


def compliance_problems(run: Mapping[str, Any]) -> list[str]:
    """V2: a completed status, every planned update done, the prerequisite passed, no interruption or guard action,
    and used <= limit for every required limit."""
    problems = []
    if run["status"] not in COMPLETED:
        problems.append(f"status {run['status']!r} is not a completed status")
    done, planned = run["updates_done"], run["updates_planned"]
    if not (is_int(done) and is_int(planned) and planned > 0 and done == planned):
        problems.append(f"updates done {done!r} != updates planned {planned!r}")
    pre = run["prerequisite"]
    if pre is not None and not (isinstance(pre, Mapping) and pre.get("status") == "COMPLETED_PASS"):
        problems.append(f"prerequisite control not COMPLETED_PASS: {pre!r}")
    if run["interrupted"] is not False:
        problems.append(f"interrupted = {run['interrupted']!r}, not false")
    if run["guard_actions"] != []:
        problems.append(f"guard action(s) recorded: {run['guard_actions']!r}")
    limits = run["limits"]
    if not isinstance(limits, Mapping):
        return [*problems, f"limits = {limits!r} is not a mapping"]
    for name in REQUIRED_LIMITS:
        entry = limits[name]
        used = real(entry.get("used")) if isinstance(entry, Mapping) else None
        cap = real(entry.get("limit")) if isinstance(entry, Mapping) else None
        if used is None or cap is None:
            problems.append(f"limit {name}: used and limit must be finite numbers, got {entry!r}")
        elif not used <= cap:
            problems.append(f"limit {name}: used {used!r} > limit {cap!r}")
    return problems


def quantities(fe: Mapping[str, Any], scalar_mean_mse: Any) -> tuple[dict[str, Any], list[str]]:
    """M0: the policy's quantities from the recorded float64 sums, and every reason one of them is undefined."""
    sse, t, p, d = (real(fe[k]) for k in STATS)
    n, mean_mse = fe["n"], real(scalar_mean_mse)
    bad = [] if fe["finite"] is True else [f"reconstruction finite = {fe['finite']!r}, not true"]
    bad += [f"{k} = {fe[k]!r} is not a finite number" for k, v in zip(STATS, (sse, t, p, d)) if v is None]
    if mean_mse is None:
        bad.append(f"scalar_mean_mse = {scalar_mean_mse!r} is not a finite number")
    if not (is_int(n) and n > 0):
        bad.append(f"n = {n!r} is not a positive integer")
    if sse is not None and sse < 0:
        bad.append(f"sse = {sse!r} < 0")
    if mean_mse is not None and mean_mse < 0:
        bad.append(f"scalar_mean_mse = {mean_mse!r} < 0")
    if t is not None and t <= 0:
        bad.append(f"target energy {t!r} <= 0: RSE undefined")
    if p is not None and p <= 0:
        bad.append(f"prediction energy {p!r} <= 0: cosine undefined")
    denom = math.sqrt(p * t) if not bad else 0.0  # the frozen uid_common formula, so values match the record
    if not bad and not (0 < denom < math.inf):
        bad.append(f"sqrt(pred_energy * target_energy) = {denom!r} is not a finite positive number")
    if bad:
        return {}, bad
    q = {"rse": sse / t, "cosine": d / denom, "mse": sse / n, "zero_mse": t / n, "scalar_mean_mse": mean_mse}
    q["best_baseline"] = "scalar_mean" if mean_mse < q["zero_mse"] else "zero"
    q["best_baseline_mse"] = min(mean_mse, q["zero_mse"])
    bad = [
        f"{k} = {q[k]!r} is not finite" for k in ("rse", "cosine", "mse", "zero_mse") if real(q[k]) is None
    ]
    if not bad and q["best_baseline_mse"] > 0:
        q["mse_over_best_baseline"] = q["mse"] / q["best_baseline_mse"]
    return q, bad


def classify(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The single EP-20261012-v1 outcome of one assessment input, in the order fixed by ``POLICY.md``."""
    out: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "thresholds": {"max_rse": MAX_RSE, "min_cosine": MIN_COSINE, "baseline_factor": BASELINE_FACTOR},
        "checks": {},
    }
    missing = missing_keys(rec)
    if missing:
        reasons = [f"required quantity not recorded: {k}" for k in missing]
        ev = rec.get("evidence")
        if isinstance(ev, Mapping):
            reasons += [str(p) for p in ev.get("problems") or []]
        out["checks"]["V1"] = {"pass": False, "problems": reasons}
        return {**out, "outcome": "UNVERIFIED", "reasons": reasons}
    ev = rec["evidence"]
    v1 = [str(p) for p in ev["problems"] or []]
    if ev["verified"] is not True:
        v1.append(f"evidence verified = {ev['verified']!r}, not true")
    out["checks"]["V1"] = {"pass": not v1, "problems": v1}
    if v1:
        return {**out, "outcome": "UNVERIFIED", "reasons": v1}
    problems = compliance_problems(rec["run"])
    out["checks"]["V2"] = {"pass": not problems, "problems": problems}
    if problems:
        return {**out, "outcome": "NOT_ELIGIBLE", "reasons": problems}
    q, undefined = quantities(rec["final_eval"], rec["baselines"]["scalar_mean_mse"])
    out["checks"]["M0"] = {"pass": not undefined, "problems": undefined}
    if undefined:
        return {**out, "outcome": "FAIL", "reasons": ["undefined required quantity cannot pass", *undefined]}
    out["quantities"] = q
    m1 = out["checks"]["M1"] = compare(q["rse"], "<=", MAX_RSE)
    m2 = out["checks"]["M2"] = compare(q["cosine"], ">=", MIN_COSINE)
    reasons = [f"{k} fails" for k, m in (("M1", m1), ("M2", m2)) if not m["pass"]]
    if q["best_baseline_mse"] == 0:
        out["checks"]["M3"] = {"pass": False, "uninformative": True, "best_baseline_mse": 0.0}
        if reasons:
            return {**out, "outcome": "FAIL", "reasons": reasons}
        reason = "best baseline MSE is 0: the baseline-improvement assessment is uninformative"
        return {**out, "outcome": "UNINFORMATIVE", "reasons": [reason]}
    m3 = out["checks"]["M3"] = compare(q["mse"], "<=", BASELINE_FACTOR * q["best_baseline_mse"])
    reasons += [] if m3["pass"] else ["M3 fails"]
    return {**out, "outcome": "FAIL" if reasons else "PASS", "reasons": reasons}


# ---- UID-20261012-v2 r1 record extraction ----------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_list(path: Path) -> list[tuple[str, int, Path]]:
    """Entries ``<sha256> <bytes> <path>`` of a FINAL_ARTIFACTS_SHA256.txt; the first line is the time stamp."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            digest, size, name = re.split(r"\s+", line.strip(), maxsplit=2)
            rows.append((digest, int(size), Path(name)))
    return rows


def verify_files(listing: Path, needed: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Re-hash every listed file; every file the assessment reads must be on the list."""
    rows, problems = [], []
    entries = hash_list(listing)
    for digest, size, path in entries:
        actual = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "sha256": digest,
                "bytes": size,
                "match": (actual, actual_size) == (digest, size),
            }
        )
        if not rows[-1]["match"]:
            problems.append(f"hash or size mismatch: {path} (actual {actual}, {actual_size} bytes)")
    listed = {os.path.normcase(os.path.abspath(p)) for _, _, p in entries}
    problems += [
        f"read but not on the hash list: {p}"
        for p in needed
        if os.path.normcase(os.path.abspath(p)) not in listed
    ]
    return rows, problems


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_package(package: Path, reads: list[str]) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """``manifest_r1.json`` must have the approved SHA-256, and each package file read must match its entry.
    Returns the hashes checked, the manifest's path -> SHA-256 entries and the problems."""
    manifest = package / "manifest_r1.json"
    actual = sha256_file(manifest) if manifest.is_file() else None
    if actual != UID_MANIFEST_SHA256:
        return (
            {"manifest_r1_sha256": actual},
            {},
            [f"manifest_r1.json SHA-256 {actual} != {UID_MANIFEST_SHA256}"],
        )
    entries = {e["path"]: e["sha256"] for e in load(manifest)["files"]}
    files, problems = {}, []
    for rel in reads:
        p = package / rel
        files[rel] = sha256_file(p) if p.is_file() else None
        if files[rel] is None or entries.get(rel) != files[rel]:
            problems.append(f"{rel}: absent from the approved manifest or SHA-256 differs ({files[rel]})")
    return {"manifest_r1_sha256": actual, "files": files}, entries, problems


def disagreements(values: Mapping[str, tuple[Any, ...]]) -> list[str]:
    """Each named quantity must be exactly equal (float ==, no tolerance) in every record that states it."""
    return [f"records disagree on {k}: {v!r}" for k, v in values.items() if any(x != v[0] for x in v[1:])]


def strict_status(m: Mapping[str, Any], criteria: Mapping[str, Any]) -> str:
    """The frozen UID verdict re-derived from recorded sums: finite, RSE <= max_rse, cosine >= min_cosine."""
    t, p = real(m.get("target_energy")), real(m.get("pred_energy"))
    sse, dot = real(m.get("sse")), real(m.get("dot"))
    if m.get("finite") is not True or None in (t, p, sse, dot) or t <= 0 or p <= 0:
        return "COMPLETED_FIT_FAIL"
    ok = sse / t <= float(criteria["max_rse"]) and dot / math.sqrt(p * t) >= float(criteria["min_cosine"])
    return "COMPLETED_PASS" if ok else "COMPLETED_FIT_FAIL"


FROZEN = "../uid_20261012_v2/"
PACKAGE_READS = [f"{FROZEN}config.json", f"{FROZEN}uid_common.py", "execution_r1.json"]
MODULE_ENTRIES = {
    "uformer_ae": f"{FROZEN}uformer_ae.py",
    "uformer_blocks": f"{FROZEN}uformer_blocks.py",
    "uid_common": f"{FROZEN}uid_common.py",
    "uid_r1_child": "uid_r1_child.py",
    "uid_r1_control": "uid_r1_control.py",
    "uid_r1_measure": "uid_r1_measure.py",
}
SUMS = ("finite", "n", *STATS)


def assess_uid_r1(
    evidence: Path = UID_EVIDENCE, package: Path = UID_PACKAGE, run_root: Path = UID_RUN_ROOT
) -> dict[str, Any]:
    """The classifier input for the U-FIT arm, read from the r1 records only after every file is verified."""
    ledger = run_root / "ledger"
    reads = {
        "child_fit": run_root / "arms" / "U-FIT" / "child_record.json",
        "child_check": run_root / "arms" / "U-CHECK" / "child_record.json",
        "end_fit": ledger / "end_U-FIT.json",
        "end_check": ledger / "end_U-CHECK.json",
        "start_fit": ledger / "start_U-FIT.json",
        "invocation": ledger / "invocation_001.json",
        "invocation_end": ledger / "invocation_001_end.json",
    }
    starts = sorted(ledger.glob("start_*.json"))
    invocations = sorted(p for p in ledger.glob("invocation_*.json") if not p.stem.endswith("_end"))
    listing = evidence / "FINAL_ARTIFACTS_SHA256.txt"
    ev: dict[str, Any] = {"hash_list": str(listing), "problems": []}
    if not listing.is_file():
        return {"evidence": {**ev, "verified": False, "problems": [f"hash list missing: {listing}"]}}
    rows, problems = verify_files(listing, [*reads.values(), *starts, *invocations])
    pkg, entries, more = verify_package(package, PACKAGE_READS)
    ev.update(
        hash_list_sha256=sha256_file(listing),
        files_on_list=len(rows),
        files_matching=sum(r["match"] for r in rows),
        package=pkg,
    )
    if problems or more:  # numbers are never taken from unverified files
        return {"evidence": {**ev, "verified": False, "problems": problems + more}}
    r = {k: load(p) for k, p in reads.items()}
    cfg = load(package / PACKAGE_READS[0])
    return uid_r1_input(r, cfg, entries, ev, len(starts), len(invocations))


def uid_r1_input(
    r: Mapping[str, Any],
    cfg: Mapping[str, Any],
    entries: Mapping[str, str],
    ev: dict[str, Any],
    starts: int,
    invs: int,
) -> dict[str, Any]:
    """Cross-check identity and agreement of the verified records, then map them onto the classifier input."""
    fit, check, inv, inv_end = r["child_fit"], r["child_check"], r["invocation"], r["invocation_end"]
    end_fit, end_check = r["end_fit"], r["end_check"]
    fe = fit["final"]["eval_mode"]
    arms_end = {a["arm"]: a for a in inv_end["arms"]}
    plan_fit = next(a for a in inv["plan"]["arms"] if a["arm"] == "U-FIT")
    problems = [
        f"module {name}: run hash {fit['modules']['sha256'].get(name)} != approved {entries.get(rel)}"
        for name, rel in MODULE_ENTRIES.items()
        if fit["modules"]["sha256"].get(name) != entries.get(rel)
    ]
    if inv["identity"]["r1_manifest_sha256"] != UID_MANIFEST_SHA256:
        problems.append(f"invocation names manifest {inv['identity']['r1_manifest_sha256']}")
    if (
        fit["config_sha256"]["raw"] != entries[PACKAGE_READS[0]]
        or fit["modules"]["checked_against_approved"] is not True
    ):
        problems.append("run config or module check differs from the approved package")
    problems += disagreements(
        {
            "U-FIT status": (
                fit["status"],
                end_fit["status"],
                arms_end["U-FIT"]["status"],
                strict_status(fe, cfg["criteria"]),
            ),
            "U-CHECK status": (
                check["status"],
                end_check["status"],
                arms_end["U-CHECK"]["status"],
                strict_status(check["final"]["eval_mode"], cfg["criteria"]),
            ),
            "U-FIT updates done": (
                fit["updates_done"],
                end_fit["updates_done"],
                arms_end["U-FIT"]["updates_done"],
            ),
            "U-FIT planned updates": (
                cfg["u_fit"]["max_updates"],
                plan_fit["max_updates"],
                r["start_fit"]["max_updates"],
            ),
            "N": (fe["n"], fit["input"]["numel"], fit["baselines"]["shared_scalar_mean"]["n"]),
            "target energy": (
                fe["target_energy"],
                fit["input"]["energy_sum_sq"],
                fit["baselines"]["zero_prediction"]["sse"],
            ),
        }
    )
    if problems:
        return {"evidence": {**ev, "verified": False, "problems": problems}}
    rss = (
        end_fit["guard"]["sampled_max_rss_bytes"],
        fit["rss"]["sampled_max_rss_bytes"],
        fit["rss"]["os_peak_wset_bytes"],
    )
    return {
        "evidence": {**ev, "verified": True, "problems": []},
        "run": {
            "status": fit["status"],
            "updates_done": fit["updates_done"],
            "updates_planned": cfg["u_fit"]["max_updates"],
            "prerequisite": {"control": "U-CHECK", "status": end_check["status"]},
            "interrupted": inv_end["interrupted"],
            "guard_actions": [
                e["guard"]["action"] for e in (end_check, end_fit) if e["guard"]["action"] is not None
            ],
            "limits": {
                "arm_wall_s": {"used": end_fit["arm_wall_s"], "limit": cfg["u_fit"]["max_seconds"]},
                "aggregate_wall_s": {
                    "used": inv_end["aggregate_wall_s"],
                    "limit": cfg["global_limits"]["max_scientific_seconds"],
                },
                "training_starts": {"used": starts, "limit": cfg["global_limits"]["max_training_starts"]},
                "invocations": {"used": invs, "limit": inv["plan"]["max_invocations"]},
                "rss_bytes": {"used": max(rss), "limit": inv["rss_limit_bytes"]},
                "evidence_bytes": {
                    "used": inv_end["evidence_bytes_run_root"],
                    "limit": inv["plan"]["evidence_cap_bytes"],
                },
            },
        },
        "final_eval": {k: fe[k] for k in SUMS},
        "baselines": {"scalar_mean_mse": fit["baselines"]["shared_scalar_mean"]["mse"]},
        "reported_only": {
            "final_train_mode": {k: fit["final"]["train_mode"][k] for k in SUMS},
            "frozen_criteria": cfg["criteria"],
            "input_sha256": fit["input"]["sha256"],
            "model_state_sha256_final": fit["model_state_sha256_final"],
        },
    }


DISCLOSURE = (
    "EP-20261012-v1 was selected on 2026-10-12 AFTER the UID-20261012-v2 U-FIT result was observed; for this run it "
    "is a post-hoc exploratory classification, not a pre-registered test. The recorded COMPLETED_FIT_FAIL status and "
    "the frozen strict criteria (RSE <= 0.01, cosine >= 0.99) are unchanged and remain the standard for near-identity "
    "reconstruction."
)
NOT_ESTABLISHED = ["identity reconstruction", "compression", "generalization", "downstream usefulness"]


def build_assessment(rec: Mapping[str, Any]) -> dict[str, Any]:
    run = rec.get("run") or {}
    reported = rec.get("reported_only") or {}
    train = None
    if "final_train_mode" in reported:  # same quantities, never gated
        tq, tbad = quantities(reported["final_train_mode"], rec["baselines"]["scalar_mean_mse"])
        train = {"undefined": tbad} if tbad else tq
    return {
        "schema": "ep-assessment-v1",
        "label": "DERIVED: exploratory, post-hoc",
        "policy": {
            "id": POLICY_ID,
            "policy_sha256": sha256_file(HERE / "POLICY.md"),
            "classifier_sha256": sha256_file(Path(__file__).resolve()),
        },
        "disclosure": DISCLOSURE,
        "subject": {
            "protocol_id": "UID-20261012-v2",
            "revision_id": "UID-20261012-v2-r1",
            "arm": "U-FIT",
            "manifest_r1_sha256": UID_MANIFEST_SHA256,
            "evidence_dir": "diagnostics/uid_20261012_v2_r1_execution",
            "run_root": UID_RUN_ROOT.as_posix(),
        },
        "original_outcome": {
            "status": run.get("status", "UNVERIFIED: not read from unverified records"),
            "criteria": reported.get("frozen_criteria"),
            "changed_by_this_assessment": False,
        },
        "exploratory": classify(rec),
        "final_train_mode_reported_only": train,
        "does_not_establish": NOT_ESTABLISHED,
        "classifier_input": rec,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EP-20261012-v1 assessment of the UID-20261012-v2 r1 U-FIT arm")
    ap.add_argument("--out", type=Path, required=True, help="new JSON file (never overwritten)")
    args = ap.parse_args(argv)
    doc = build_assessment(assess_uid_r1())
    text = json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.out.open("x", encoding="utf-8") as fh:
        fh.write(text)
    ex = doc["exploratory"]
    print(f"{POLICY_ID}: {ex['outcome']} (original status {doc['original_outcome']['status']})")
    for reason in ex["reasons"]:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# @@CONTINUE@@
