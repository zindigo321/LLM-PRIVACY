"""Primary policy UPP-20261013-v1 for Uformer reconstruction diagnostics (``POLICY.md``; values in ``upp_policy.json``).

Pure, standard library only: no torch, no model code, no record is edited. ``load_policy`` reads the authoritative
JSON and refuses if its content is not the content this classifier was written for. ``classify`` gives the one
outcome of one arm; ``aggregate`` gives the multi-arm decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "upp_policy.json"
POLICY_ID = "UPP-20261013-v1"
EXPECTED_THRESHOLDS = {"max_rse": 0.5, "min_cosine": 0.65, "baseline_factor": 0.8}
OUTCOMES = ("UNVERIFIED", "DEGENERATE_INPUT", "NOT_RUN", "NOT_ELIGIBLE", "FIT_FAIL", "NONINFORMATIVE", "PASS")
CLASSES = {
    "PASS": "P",
    "FIT_FAIL": "F",
    "NONINFORMATIVE": "F",
    "DEGENERATE_INPUT": "F",
    "UNVERIFIED": "U",
    "NOT_ELIGIBLE": "U",
    "NOT_RUN": "U",
}
COMPLETED = ("COMPLETED_PASS", "COMPLETED_FIT_FAIL", "COMPLETED")
STATS = ("sse", "target_energy", "pred_energy", "dot")
REQUIRED_LIMITS = (
    "arm_wall_s",
    "aggregate_wall_s",
    "training_starts",
    "invocations",
    "rss_bytes",
    "evidence_bytes",
)


class PolicyError(RuntimeError):
    """The policy file is missing, malformed, or not the content this classifier implements."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    """The authoritative policy, with its file SHA-256; refuses any content this classifier does not implement."""
    data = Path(path).read_bytes()
    pol = json.loads(data.decode("utf-8"))
    if pol.get("policy_id") != POLICY_ID:
        raise PolicyError(f"policy id {pol.get('policy_id')!r} is not {POLICY_ID}")
    if pol.get("thresholds") != EXPECTED_THRESHOLDS:
        raise PolicyError(f"thresholds {pol.get('thresholds')!r} are not {EXPECTED_THRESHOLDS}")
    if tuple(pol.get("outcome_order", ())) != OUTCOMES or pol.get("outcome_classes") != CLASSES:
        raise PolicyError("outcome order or classes differ from this classifier")
    return {**pol, "policy_sha256": hashlib.sha256(data).hexdigest()}


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
    """One inclusive float64 comparison against a literal threshold, no epsilon."""
    ok = value <= threshold if op == "<=" else value >= threshold
    return {"value": value, "operator": op, "threshold": threshold, "margin": value - threshold, "pass": ok}


def degenerate(desc: Mapping[str, Any]) -> list[str]:
    """The registered-input degeneracy conditions; ``desc`` holds all_finite, target_energy, scalar_mean_mse and
    normalized_max_abs of the input. Missing or non-numeric values count as degenerate-unknown and are named."""
    out = []
    if desc.get("all_finite") is not True:
        out.append("an input element is not finite")
    t, m, mx = (
        real(desc.get("target_energy")),
        real(desc.get("scalar_mean_mse")),
        real(desc.get("normalized_max_abs")),
    )
    if t is None or t <= 0:
        out.append(f"target energy {desc.get('target_energy')!r} is not positive")
    if m is None or m <= 0:
        out.append(f"scalar-mean MSE {desc.get('scalar_mean_mse')!r} is not positive (constant input)")
    if mx is None or mx >= 1.0:
        out.append(f"normalized max |x_n| {desc.get('normalized_max_abs')!r} is not below 1")
    return out


def compliance(run: Mapping[str, Any]) -> list[str]:
    """Valid completion: a completed status, every planned update, no interruption or guard action, every limit."""
    problems = []
    if run.get("status") not in COMPLETED:
        problems.append(f"status {run.get('status')!r} is not a completed status")
    done, planned = run.get("updates_done"), run.get("updates_planned")
    if not (is_int(done) and is_int(planned) and planned > 0 and done == planned):
        problems.append(f"updates done {done!r} != updates planned {planned!r}")
    if run.get("interrupted") is not False:
        problems.append(f"interrupted = {run.get('interrupted')!r}, not false")
    if run.get("guard_actions") != []:
        problems.append(f"guard action(s) recorded: {run.get('guard_actions')!r}")
    limits = run.get("limits")
    if not isinstance(limits, Mapping):
        return [*problems, f"limits = {limits!r} is not a mapping"]
    for name in REQUIRED_LIMITS:
        entry = limits.get(name)
        used = real(entry.get("used")) if isinstance(entry, Mapping) else None
        cap = real(entry.get("limit")) if isinstance(entry, Mapping) else None
        if used is None or cap is None:
            problems.append(f"limit {name}: used and limit must be finite numbers, got {entry!r}")
        elif not used <= cap:
            problems.append(f"limit {name}: used {used!r} > limit {cap!r}")
    return problems


def quantities(fe: Mapping[str, Any], zero_mse: Any, mean_mse: Any) -> tuple[dict[str, Any], list[str]]:
    """The policy's quantities from the recorded float64 sums, and every reason one of them is undefined."""
    vals = {k: real(fe.get(k)) for k in STATS}
    n, z, m = fe.get("n"), real(zero_mse), real(mean_mse)
    bad = [] if fe.get("finite") is True else [f"reconstruction finite = {fe.get('finite')!r}, not true"]
    bad += [f"{k} = {fe.get(k)!r} is not a finite number" for k, v in vals.items() if v is None]
    bad += [
        f"{name} = {v!r} is not a finite number"
        for name, v, r in (("zero_prediction_mse", zero_mse, z), ("scalar_mean_mse", mean_mse, m))
        if r is None
    ]
    if not (is_int(n) and n > 0):
        bad.append(f"n = {n!r} is not a positive integer")
    sse, t, p, d = (vals[k] for k in STATS)
    if sse is not None and sse < 0:
        bad.append(f"sse = {sse!r} < 0")
    for name, v in (("zero_prediction_mse", z), ("scalar_mean_mse", m)):
        if v is not None and v < 0:
            bad.append(f"{name} = {v!r} < 0")
    if t is not None and t <= 0:
        bad.append(f"target energy {t!r} <= 0: RSE undefined")
    if p is not None and p <= 0:
        bad.append(f"prediction energy {p!r} <= 0: cosine undefined")
    if bad:
        return {}, bad
    denom = math.sqrt(p * t)  # the frozen uid_common.metrics formula, so values match the record
    if not 0 < denom < math.inf:
        return {}, [f"sqrt(pred_energy * target_energy) = {denom!r} is not a finite positive number"]
    q = {"rse": sse / t, "cosine": d / denom, "mse": sse / n, "zero_mse": z, "scalar_mean_mse": m}
    q["best_baseline"] = "scalar_mean" if m < z else "zero"
    q["best_baseline_mse"] = min(m, z)
    bad = [f"{k} = {q[k]!r} is not finite" for k in ("rse", "cosine", "mse") if real(q[k]) is None]
    if not bad and q["best_baseline_mse"] > 0:
        q["mse_over_best_baseline"] = q["mse"] / q["best_baseline_mse"]
        q["baseline_improvement"] = 1.0 - q["mse_over_best_baseline"]
    return q, bad


REQUIRED = {
    "evidence": ("verified", "problems"),
    "input": ("all_finite", "target_energy", "scalar_mean_mse", "normalized_max_abs"),
    "run": ("started", "status", "updates_done", "updates_planned", "interrupted", "guard_actions", "limits"),
    "final_eval": ("finite", "n", *STATS),
    "baselines": ("zero_prediction_mse", "scalar_mean_mse"),
}


def missing_keys(rec: Mapping[str, Any]) -> list[str]:
    """Required keys that are absent. ``final_eval`` is required only for a started, completed run."""
    out = []
    for section, keys in REQUIRED.items():
        if section == "final_eval":
            run = rec.get("run") if isinstance(rec.get("run"), Mapping) else {}
            if not (run.get("started") is True and run.get("status") in COMPLETED):
                continue
        sec = rec.get(section)
        if not isinstance(sec, Mapping):
            out.append(section)
            continue
        out += [f"{section}.{k}" for k in keys if k not in sec]
    return out


def classify(rec: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """The single UPP-20261013-v1 outcome of one arm, in the order fixed by ``POLICY.md``.

    ``rec`` holds ``evidence`` {verified, problems}, ``input`` (degeneracy quantities of the registered input),
    ``run`` {started, status, updates_done, updates_planned, interrupted, guard_actions, limits}, ``final_eval``
    (the recorded final eval-mode sums in the original space) and ``baselines`` (original-space MSEs).
    """
    if policy.get("policy_id") != POLICY_ID or policy.get("thresholds") != EXPECTED_THRESHOLDS:
        raise PolicyError("classify was given a policy other than UPP-20261013-v1")
    th = policy["thresholds"]
    out: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "policy_sha256": policy.get("policy_sha256"),
        "thresholds": dict(th),
        "checks": {},
    }

    def done(outcome: str, reasons: list[str]) -> dict[str, Any]:
        return {**out, "outcome": outcome, "class": CLASSES[outcome], "reasons": reasons}

    missing = missing_keys(rec)
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), Mapping) else {}
    if missing:
        reasons = [f"required quantity not recorded: {k}" for k in missing] + [
            str(p) for p in ev.get("problems") or []
        ]
        out["checks"]["evidence"] = {"pass": False, "problems": reasons}
        return done("UNVERIFIED", reasons)
    v1 = [str(p) for p in ev["problems"] or []]
    if ev["verified"] is not True:
        v1.append(f"evidence verified = {ev['verified']!r}, not true")
    out["checks"]["evidence"] = {"pass": not v1, "problems": v1}
    if v1:
        return done("UNVERIFIED", v1)
    deg = degenerate(rec["input"])
    out["checks"]["input"] = {"pass": not deg, "problems": deg}
    if deg:
        return done("DEGENERATE_INPUT", deg)
    run = rec["run"]
    if run["started"] is not True:
        out["checks"]["run"] = {"pass": False, "problems": ["the arm was never started"]}
        return done("NOT_RUN", ["the arm was never started"])
    comp = compliance(run)
    out["checks"]["run"] = {"pass": not comp, "problems": comp}
    if comp:
        return done("NOT_ELIGIBLE", comp)
    b = rec["baselines"]
    m = primary_metrics(rec["final_eval"], b["zero_prediction_mse"], b["scalar_mean_mse"], policy)
    out["checks"].update(m["checks"])
    if "quantities" in m:
        out["quantities"] = m["quantities"]
    return done(m["outcome"], m["reasons"])


def primary_metrics(
    fe: Mapping[str, Any], zero_mse: Any, mean_mse: Any, policy: Mapping[str, Any]
) -> dict[str, Any]:
    """The metric part of the policy for a valid completed run: PASS, FIT_FAIL or NONINFORMATIVE from the final
    eval-mode sums and the two baselines. ``classify`` uses it after the validity checks; the training child uses it
    to set the terminal status of a completed arm, so both apply the same function to the same recorded numbers."""
    if policy.get("policy_id") != POLICY_ID or policy.get("thresholds") != EXPECTED_THRESHOLDS:
        raise PolicyError("primary_metrics was given a policy other than UPP-20261013-v1")
    th = policy["thresholds"]
    checks: dict[str, Any] = {}
    q, undefined = quantities(fe, zero_mse, mean_mse)
    checks["defined"] = {"pass": not undefined, "problems": undefined}
    if undefined:
        reasons = ["undefined or non-finite required quantity cannot pass", *undefined]
        return {"outcome": "FIT_FAIL", "reasons": reasons, "checks": checks}
    rse = checks["rse"] = compare(q["rse"], "<=", th["max_rse"])
    cos = checks["cosine"] = compare(q["cosine"], ">=", th["min_cosine"])
    reasons = [f"{k} fails" for k, c in (("rse", rse), ("cosine", cos)) if not c["pass"]]
    if q["best_baseline_mse"] == 0:
        checks["baseline"] = {"pass": False, "noninformative": True, "best_baseline_mse": 0.0}
        if not reasons:
            reasons = ["best baseline MSE is 0: the baseline condition carries no information"]
            return {"outcome": "NONINFORMATIVE", "reasons": reasons, "checks": checks, "quantities": q}
        return {"outcome": "FIT_FAIL", "reasons": reasons, "checks": checks, "quantities": q}
    base = checks["baseline"] = compare(q["mse"], "<=", th["baseline_factor"] * q["best_baseline_mse"])
    reasons += [] if base["pass"] else ["baseline fails"]
    return {
        "outcome": "FIT_FAIL" if reasons else "PASS",
        "reasons": reasons,
        "checks": checks,
        "quantities": q,
    }


def aggregate(outcomes: Mapping[str, str], n_registered: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    """The pre-specified multi-arm decision: P >= 3 met, F >= 2 not met, otherwise inconclusive (four arms).

    ``outcomes`` maps every registered arm to its outcome; an arm missing from it is counted as U."""
    if policy.get("policy_id") != POLICY_ID:
        raise PolicyError("aggregate was given a policy other than UPP-20261013-v1")
    if n_registered != 4:
        raise PolicyError(f"the aggregate rule is fixed for 4 registered arms, got {n_registered}")
    if len(outcomes) > n_registered or any(o not in CLASSES for o in outcomes.values()):
        raise PolicyError(f"invalid outcomes {dict(outcomes)!r}")
    classes = {arm: CLASSES[o] for arm, o in outcomes.items()}
    counts = {c: sum(1 for v in classes.values() if v == c) for c in ("P", "F", "U")}
    counts["U"] += n_registered - len(outcomes)
    if counts["P"] >= 3:
        decision = "COVERAGE_CRITERION_MET"
    elif counts["F"] >= 2:
        decision = "COVERAGE_CRITERION_NOT_MET"
    else:
        decision = "INCONCLUSIVE"
    return {
        "policy_id": POLICY_ID,
        "classes": classes,
        "counts": counts,
        "n_registered": n_registered,
        "rule": "P >= 3: met; F >= 2: not met; otherwise inconclusive",
        "decision": decision,
    }
