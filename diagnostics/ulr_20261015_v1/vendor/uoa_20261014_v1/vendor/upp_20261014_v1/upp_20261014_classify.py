"""Primary policy UPP-20261014-v1 for Uformer reconstruction diagnostics (``POLICY.md``; values in ``upp_policy.json``).

Pure, standard library only: no torch, no numpy, no model code, and no import of any earlier policy or classifier.
``load_policy`` reads the authoritative JSON and refuses if its content is not the content this classifier was written
for. ``classify`` gives the outcome of one unit (for example one tile of one method); ``classify_pooled`` gives the
outcome of a pooled unit (for example a cohort of tiles) from the raw sums of its registered members;
``study_summary`` combines the two registered held-out cohorts of a study.

Only RSE and cosine decide. The MSE over the best constant baseline is computed when it is defined and is reported
only: it is never an input to an outcome, and neither is its absence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
POLICY_PATH = HERE / "upp_policy.json"
POLICY_ID = "UPP-20261014-v1"
EXPECTED_THRESHOLDS = {"max_rse": 1.0, "min_cosine": 0.1}
OUTCOMES = ("UNVERIFIED", "DEGENERATE_INPUT", "NOT_RUN", "NOT_ELIGIBLE", "FIT_FAIL", "PASS")
CLASSES = {
    "PASS": "P",
    "FIT_FAIL": "F",
    "DEGENERATE_INPUT": "F",
    "UNVERIFIED": "U",
    "NOT_ELIGIBLE": "U",
    "NOT_RUN": "U",
}
# terminal statuses that mean "the registered procedure ran to completion" (the older two carried old verdicts)
COMPLETED = ("COMPLETED", "COMPLETED_PASS", "COMPLETED_FIT_FAIL")
STATS = ("sse", "target_energy", "pred_energy", "dot")
STUDY_DECISIONS = ("PASS", "PARTIAL", "FAIL", "INCOMPLETE")
STUDY_RULE = (
    "over the two registered held-out cohorts: INCOMPLETE if a cohort is missing or not valid (class U); "
    "otherwise PASS if both cohorts PASS, PARTIAL if exactly one does, FAIL if neither does"
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


def check_policy(policy: Mapping[str, Any], who: str) -> Mapping[str, Any]:
    if policy.get("policy_id") != POLICY_ID or policy.get("thresholds") != EXPECTED_THRESHOLDS:
        raise PolicyError(f"{who} was given a policy other than {POLICY_ID}")
    return policy["thresholds"]


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
    if op not in ("<=", ">="):
        raise ValueError(f"operator {op!r}")
    ok = value <= threshold if op == "<=" else value >= threshold
    return {"value": value, "operator": op, "threshold": threshold, "margin": value - threshold, "pass": ok}


def degenerate(desc: Mapping[str, Any]) -> list[str]:
    """The input conditions under which RSE is undefined: a non-finite element, or a target energy T that is not a
    finite positive number. Missing or non-numeric values are named. Nothing else about the input is a condition."""
    out = []
    if desc.get("all_finite") is not True:
        out.append(f"input all_finite = {desc.get('all_finite')!r}: an input element is not finite")
    t = real(desc.get("target_energy"))
    if t is None or t <= 0:
        out.append(
            f"target energy {desc.get('target_energy')!r} is not a finite positive number: RSE undefined"
        )
    return out


def compliance(run: Mapping[str, Any], required_limits: Sequence[str]) -> list[str]:
    """Valid completion: a completed status, every planned update (a method without optimizer updates registers 0),
    no interruption or guard action, and every limit recorded and respected (inclusive)."""
    problems = []
    if run.get("status") not in COMPLETED:
        problems.append(f"status {run.get('status')!r} is not a completed status")
    done, planned = run.get("updates_done"), run.get("updates_planned")
    if not (is_int(done) and is_int(planned) and planned >= 0 and done == planned):
        problems.append(f"updates done {done!r} != updates planned {planned!r}")
    if run.get("interrupted") is not False:
        problems.append(f"interrupted = {run.get('interrupted')!r}, not false")
    if run.get("guard_actions") != []:
        problems.append(f"guard action(s) recorded: {run.get('guard_actions')!r}")
    limits = run.get("limits")
    if not isinstance(limits, Mapping) or not limits:
        return [*problems, f"limits = {limits!r} is not a non-empty mapping"]
    problems += [f"limit {name}: not recorded" for name in required_limits if name not in limits]
    for name, entry in limits.items():
        used = real(entry.get("used")) if isinstance(entry, Mapping) else None
        cap = real(entry.get("limit")) if isinstance(entry, Mapping) else None
        if used is None or cap is None:
            problems.append(f"limit {name}: used and limit must be finite numbers, got {entry!r}")
        elif not used <= cap:
            problems.append(f"limit {name}: used {used!r} > limit {cap!r}")
    return problems


def quantities(fe: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """RSE, cosine, MSE and norm ratio from the recorded float64 sums, and every reason one of them is undefined."""
    vals = {k: real(fe.get(k)) for k in STATS}
    n = fe.get("n")
    bad = [] if fe.get("finite") is True else [f"prediction finite = {fe.get('finite')!r}, not true"]
    bad += [f"{k} = {fe.get(k)!r} is not a finite number" for k, v in vals.items() if v is None]
    if not (is_int(n) and n > 0):
        bad.append(f"n = {n!r} is not a positive integer")
    sse, t, p, d = (vals[k] for k in STATS)
    if sse is not None and sse < 0:
        bad.append(f"sse = {sse!r} < 0")
    if t is not None and t <= 0:
        bad.append(f"target energy {t!r} <= 0: RSE undefined")
    if p is not None and p <= 0:
        bad.append(f"prediction energy {p!r} <= 0: cosine undefined")
    if bad:
        return {}, bad
    denom = math.sqrt(p * t)
    if not 0 < denom < math.inf:
        return {}, [f"sqrt(pred_energy * target_energy) = {denom!r} is not a finite positive number"]
    q = {"rse": sse / t, "cosine": d / denom, "mse": sse / n, "norm_ratio": math.sqrt(p / t)}
    return q, [f"{k} = {q[k]!r} is not finite" for k in ("rse", "cosine", "mse") if real(q[k]) is None]


def report_only_baseline(q: Mapping[str, Any], baselines: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    """MSE / min(zero-prediction MSE, scalar-mean MSE) and the historical UPP-20261013-v1 baseline condition, when
    they are defined. Reported only: the caller never reads this to decide an outcome, and this never raises."""
    factor = (policy.get("report_only") or {}).get("historical_baseline_factor")
    out: dict[str, Any] = {"role": "report only; never an input to the outcome", "defined": False}
    if not isinstance(baselines, Mapping) or real(q.get("mse")) is None:
        return {**out, "reason": "no baselines recorded or MSE undefined"}
    z, m = real(baselines.get("zero_prediction_mse")), real(baselines.get("scalar_mean_mse"))
    if z is None or m is None or z < 0 or m < 0:
        return {**out, "reason": f"baselines {dict(baselines)!r} are not finite non-negative numbers"}
    best = min(z, m)
    out.update(
        zero_prediction_mse=z,
        scalar_mean_mse=m,
        best_baseline="scalar_mean" if m < z else "zero",
        best_baseline_mse=best,
    )
    if best == 0:
        return {**out, "reason": "best constant baseline MSE is 0: the ratio is undefined"}
    ratio = q["mse"] / best
    out.update(defined=True, mse_over_best_baseline=ratio)
    if real(factor) is not None:
        out["historical_baseline_condition_holds"] = ratio <= factor
        out["historical_baseline_factor"] = factor
    return out


def primary_metrics(
    fe: Mapping[str, Any], policy: Mapping[str, Any], baselines: Any = None
) -> dict[str, Any]:
    """The metric part of the policy for a valid unit: PASS or FIT_FAIL from the final eval-mode sums alone.
    ``baselines`` only feeds the report-only ratio."""
    th = check_policy(policy, "primary_metrics")
    checks: dict[str, Any] = {}
    q, undefined = quantities(fe)
    checks["defined"] = {"pass": not undefined, "problems": undefined}
    if undefined:
        reasons = ["undefined or non-finite required quantity cannot pass", *undefined]
        return {"outcome": "FIT_FAIL", "reasons": reasons, "checks": checks}
    rse = checks["rse"] = compare(q["rse"], "<=", th["max_rse"])
    cos = checks["cosine"] = compare(q["cosine"], ">=", th["min_cosine"])
    reasons = [f"{k} fails" for k, c in (("rse", rse), ("cosine", cos)) if not c["pass"]]
    return {
        "outcome": "FIT_FAIL" if reasons else "PASS",
        "reasons": reasons,
        "checks": checks,
        "quantities": q,
        "report_only": report_only_baseline(q, baselines, policy),
    }


REQUIRED = {
    "evidence": ("verified", "problems"),
    "input": ("all_finite", "target_energy"),
    "run": ("started", "status", "updates_done", "updates_planned", "interrupted", "guard_actions", "limits"),
    "final_eval": ("finite", "n", *STATS),
}


def _completed(rec: Mapping[str, Any]) -> bool:
    run = rec.get("run") if isinstance(rec.get("run"), Mapping) else {}
    return run.get("started") is True and run.get("status") in COMPLETED


def missing_keys(rec: Mapping[str, Any], sections: Sequence[str] = tuple(REQUIRED)) -> list[str]:
    """Required keys that are absent. ``final_eval`` is required only for a started, completed run."""
    out = []
    for section in sections:
        if section == "final_eval" and not _completed(rec):
            continue
        sec = rec.get(section)
        if not isinstance(sec, Mapping):
            out.append(section)
            continue
        out += [f"{section}.{k}" for k in REQUIRED[section] if k not in sec]
    return out


def _evidence(rec: Mapping[str, Any], missing: list[str]) -> list[str]:
    ev = rec.get("evidence") if isinstance(rec.get("evidence"), Mapping) else {}
    problems = [f"required quantity not recorded: {k}" for k in missing]
    problems += [str(p) for p in ev.get("problems") or []]
    if not missing and ev.get("verified") is not True:
        problems.append(f"evidence verified = {ev.get('verified')!r}, not true")
    return problems


def _decided(policy: Mapping[str, Any], unit: str) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "policy_sha256": policy.get("policy_sha256"),
        "thresholds": dict(policy["thresholds"]),
        "unit": unit,
        "checks": {},
    }


def classify(
    rec: Mapping[str, Any], policy: Mapping[str, Any], required_limits: Sequence[str]
) -> dict[str, Any]:
    """The single UPP-20261014-v1 outcome of one unit, in the order fixed by ``POLICY.md``.

    ``rec`` holds ``evidence`` {verified, problems}, ``input`` {all_finite, target_energy}, ``run`` {started, status,
    updates_done, updates_planned, interrupted, guard_actions, limits}, ``final_eval`` (the recorded final eval-mode
    sums in the original space) and optionally ``baselines`` (original-space MSEs, report only).
    ``required_limits`` names the limits the study registered; each must be recorded.
    """
    check_policy(policy, "classify")
    out = _decided(policy, "single")

    def done(outcome: str, reasons: list[str]) -> dict[str, Any]:
        return {**out, "outcome": outcome, "class": CLASSES[outcome], "reasons": reasons}

    problems = _evidence(rec, missing_keys(rec))
    out["checks"]["evidence"] = {"pass": not problems, "problems": problems}
    if problems:
        return done("UNVERIFIED", problems)
    deg = degenerate(rec["input"])
    out["checks"]["input"] = {"pass": not deg, "problems": deg}
    if deg:
        return done("DEGENERATE_INPUT", deg)
    run = rec["run"]
    if run["started"] is not True:
        out["checks"]["run"] = {"pass": False, "problems": ["the unit was never started"]}
        return done("NOT_RUN", ["the unit was never started"])
    comp = compliance(run, required_limits)
    out["checks"]["run"] = {"pass": not comp, "problems": comp}
    if comp:
        return done("NOT_ELIGIBLE", comp)
    m = primary_metrics(rec["final_eval"], policy, rec.get("baselines"))
    out["checks"].update(m["checks"])
    for key in ("quantities", "report_only"):
        if key in m:
            out[key] = m[key]
    return done(m["outcome"], m["reasons"])


def pool(final_evals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pooled sums of registered members: ``math.fsum`` of each float64 sum and the total N. The pool is finite only
    if every member is finite; a missing or non-numeric member sum makes that pooled sum None (undefined)."""
    out: dict[str, Any] = {"members": len(final_evals)}
    out["finite"] = bool(final_evals) and all(fe.get("finite") is True for fe in final_evals)
    for k in STATS:
        vals = [real(fe.get(k)) for fe in final_evals]
        out[k] = None if not vals or any(v is None for v in vals) else math.fsum(vals)
    ns = [fe.get("n") for fe in final_evals]
    out["n"] = sum(ns) if ns and all(is_int(n) and n > 0 for n in ns) else None
    return out


def classify_pooled(
    rec: Mapping[str, Any], policy: Mapping[str, Any], required_limits: Sequence[str]
) -> dict[str, Any]:
    """The UPP-20261014-v1 outcome of a pooled unit, with the same bounds as a single unit.

    ``rec`` holds ``evidence``, ``run`` (the run that produced every member), ``registered_members`` (the labels fixed
    before any value was read) and ``members`` {label: {"input": {...}, "final_eval": {...}}}. Every registered member
    must be present and no other; the pooled RSE and cosine are ratios of the pooled sums (not means of ratios).
    """
    check_policy(policy, "classify_pooled")
    out = _decided(policy, "pooled")

    def done(outcome: str, reasons: list[str]) -> dict[str, Any]:
        return {**out, "outcome": outcome, "class": CLASSES[outcome], "reasons": reasons}

    problems = _evidence(rec, missing_keys(rec, ("evidence", "run")))
    registered, members = rec.get("registered_members"), rec.get("members")
    if not (isinstance(registered, list) and registered and len(set(registered)) == len(registered)):
        problems.append(f"registered members {registered!r} are not a non-empty list of distinct labels")
    elif not isinstance(members, Mapping):
        problems.append("members are not recorded")
    else:
        absent = [m for m in registered if m not in members]
        extra = sorted(m for m in members if m not in registered)
        problems += [f"registered member {m} not recorded" for m in absent]
        problems += [f"member {m} is not registered" for m in extra]
        for label in registered:
            if label in members:
                sub = {**rec, **members[label]} if isinstance(members[label], Mapping) else {}
                problems += [f"{label}: {k}" for k in missing_keys(sub, ("input", "final_eval"))]
    out["checks"]["evidence"] = {"pass": not problems, "problems": problems}
    if problems:
        return done("UNVERIFIED", problems)
    inputs = [members[m]["input"] for m in registered]
    energies = [real(i.get("target_energy")) for i in inputs]
    deg = degenerate(
        {
            "all_finite": all(i.get("all_finite") is True for i in inputs),
            "target_energy": None if any(e is None for e in energies) else math.fsum(energies),
        }
    )
    out["checks"]["input"] = {"pass": not deg, "problems": deg, "members": len(inputs)}
    if deg:
        return done("DEGENERATE_INPUT", deg)
    run = rec["run"]
    if run["started"] is not True:
        out["checks"]["run"] = {"pass": False, "problems": ["the unit was never started"]}
        return done("NOT_RUN", ["the unit was never started"])
    comp = compliance(run, required_limits)
    out["checks"]["run"] = {"pass": not comp, "problems": comp}
    if comp:
        return done("NOT_ELIGIBLE", comp)
    pooled = out["pooled_sums"] = pool([members[m]["final_eval"] for m in registered])
    m = primary_metrics(pooled, policy, None)
    out["checks"].update(m["checks"])
    if "quantities" in m:
        out["quantities"] = m["quantities"]
    return done(m["outcome"], m["reasons"])


def study_summary(
    cohort_outcomes: Mapping[str, str], registered: Sequence[str], policy: Mapping[str, Any]
) -> dict[str, Any]:
    """The registered study decision over exactly two held-out cohorts (``STUDY_RULE``). A registered cohort missing
    from ``cohort_outcomes`` counts as not valid."""
    check_policy(policy, "study_summary")
    if len(registered) != 2 or len(set(registered)) != 2:
        raise PolicyError(f"the study rule is fixed for two registered cohorts, got {list(registered)!r}")
    if any(c not in registered for c in cohort_outcomes) or any(
        o not in CLASSES for o in cohort_outcomes.values()
    ):
        raise PolicyError(f"invalid cohort outcomes {dict(cohort_outcomes)!r}")
    classes = {c: CLASSES[cohort_outcomes[c]] if c in cohort_outcomes else "U" for c in registered}
    passed = [c for c in registered if classes[c] == "P"]
    if any(v == "U" for v in classes.values()):
        decision = "INCOMPLETE"
    else:
        decision = {2: "PASS", 1: "PARTIAL", 0: "FAIL"}[len(passed)]
    return {
        "policy_id": POLICY_ID,
        "registered_cohorts": list(registered),
        "classes": classes,
        "passed": passed,
        "missing": [c for c in registered if c not in cohort_outcomes],
        "rule": STUDY_RULE,
        "decision": decision,
    }
