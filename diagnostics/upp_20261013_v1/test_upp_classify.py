"""Tests of UPP-20261013-v1 (``upp_classify.py``). Standard library and pytest only; no torch.

Two kinds of test, kept apart:
- SCALAR: the threshold comparison itself (``compare``) and the aggregate rule, on bare numbers; boundary values are
  the literal thresholds and their adjacent float64 values.
- REALIZABLE: complete classifier inputs whose sums (SSE, T, P, D, N) come from actual vectors, computed with math.fsum
  as uid_common.metrics does in float64; these are the quantities a run can record.
"""

import copy
import itertools
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import upp_classify as P  # noqa: E402

POL = P.load_policy()
EP_DIR = HERE.parent / "exploratory_progression_20261012"


# ---- helpers: realizable records ---------------------------------------------------------------------------
def sums(pred: list[float], x: list[float]) -> dict:
    return {
        "finite": all(math.isfinite(v) for v in pred),
        "n": len(x),
        "sse": math.fsum((a - b) ** 2 for a, b in zip(pred, x)),
        "target_energy": math.fsum(b * b for b in x),
        "pred_energy": math.fsum(a * a for a in pred),
        "dot": math.fsum(a * b for a, b in zip(pred, x)),
    }


def baselines(x: list[float]) -> dict:
    mean = math.fsum(x) / len(x)
    return {
        "zero_prediction_mse": math.fsum(b * b for b in x) / len(x),
        "scalar_mean_mse": math.fsum((b - mean) ** 2 for b in x) / len(x),
    }


X = [((i * 37) % 23 - 11) / 11.0 for i in range(64)]  # a fixed non-constant input with mean near 0


def record(pred: list[float], x: list[float] = X, **over) -> dict:
    rec = {
        "evidence": {"verified": True, "problems": []},
        "input": {
            "all_finite": all(math.isfinite(v) for v in x),
            "target_energy": math.fsum(b * b for b in x),
            "scalar_mean_mse": baselines(x)["scalar_mean_mse"],
            "normalized_max_abs": 0.9,
        },
        "run": {
            "started": True,
            "status": "COMPLETED_FIT_FAIL",
            "updates_done": 3000,
            "updates_planned": 3000,
            "interrupted": False,
            "guard_actions": [],
            "limits": {k: {"used": 1, "limit": 2} for k in P.REQUIRED_LIMITS},
        },
        "final_eval": sums(pred, x),
        "baselines": baselines(x),
    }
    for path, value in over.items():
        sec, key = path.split("__")
        rec[sec][key] = value
    return rec


def scaled(c: float, x: list[float] = X) -> list[float]:
    return [c * v for v in x]


# ---- the policy file is the authority ------------------------------------------------------------------------
def test_policy_file_values_and_identity():
    raw = json.loads(P.POLICY_PATH.read_text(encoding="utf-8"))
    assert raw["policy_id"] == "UPP-20261013-v1"
    assert raw["thresholds"] == {"max_rse": 0.5, "min_cosine": 0.65, "baseline_factor": 0.8}
    assert POL["policy_sha256"] == P.sha256_file(P.POLICY_PATH) and len(POL["policy_sha256"]) == 64
    assert b"\r\n" not in P.POLICY_PATH.read_bytes()


@pytest.mark.parametrize("edit", [("max_rse", 0.51), ("min_cosine", 0.8), ("baseline_factor", 1.0)])
def test_a_policy_file_with_other_values_is_refused(tmp_path, edit):
    raw = json.loads(P.POLICY_PATH.read_text(encoding="utf-8"))
    raw["thresholds"][edit[0]] = edit[1]
    (tmp_path / "p.json").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(P.PolicyError):
        P.load_policy(tmp_path / "p.json")


def test_classify_refuses_a_foreign_policy():
    with pytest.raises(P.PolicyError):
        P.classify(record(scaled(0.9)), {**POL, "policy_id": "EP-20261012-v1"})
    with pytest.raises(P.PolicyError):
        P.primary_metrics({}, 1.0, 1.0, {**POL, "thresholds": {"max_rse": 0.01, "min_cosine": 0.99}})


def test_the_historical_uid_criteria_are_not_the_primary_thresholds():
    uid = json.loads((HERE.parent / "uid_20261012_v2" / "config.json").read_text(encoding="utf-8"))
    assert uid["criteria"] == {"max_rse": 0.01, "min_cosine": 0.99}  # unchanged, secondary reference only
    assert POL["thresholds"]["max_rse"] != uid["criteria"]["max_rse"]
    assert POL["thresholds"]["min_cosine"] != uid["criteria"]["min_cosine"]
    assert POL["secondary_references"]["historical_strict"]["max_rse"] == uid["criteria"]["max_rse"]


# ---- SCALAR: inclusive boundaries and the adjacent float64 values -------------------------------------------
@pytest.mark.parametrize(("op", "threshold"), [("<=", 0.5), (">=", 0.65), ("<=", 0.8 * 3.3e-06)])
def test_boundaries_are_inclusive_and_the_next_float_fails(op, threshold):
    assert P.compare(threshold, op, threshold)["pass"] is True
    outside = math.nextafter(threshold, math.inf if op == "<=" else -math.inf)
    inside = math.nextafter(threshold, -math.inf if op == "<=" else math.inf)
    assert P.compare(outside, op, threshold)["pass"] is False
    assert P.compare(inside, op, threshold)["pass"] is True


def test_scalar_boundaries_through_classify_with_exact_quantities():
    # SCALAR (not realizable by construction): sums chosen so that RSE, cosine and MSE land exactly on the bounds
    base = {"zero_prediction_mse": 1.0, "scalar_mean_mse": 1.0}
    rec = record(scaled(0.9))
    rec["baselines"] = base
    # T = 1, N = 1: RSE = SSE, MSE = SSE; P and D give cosine exactly 0.65 when D = 0.65 and P = 1
    rec["final_eval"] = {
        "finite": True,
        "n": 1,
        "sse": 0.5,
        "target_energy": 1.0,
        "pred_energy": 1.0,
        "dot": 0.65,
    }
    out = P.classify(rec, POL)
    assert (
        out["outcome"] == "PASS"
        and out["checks"]["rse"]["margin"] == 0.0
        and out["checks"]["cosine"]["margin"] == 0.0
    )
    rec["final_eval"]["sse"] = math.nextafter(0.5, 1.0)
    assert P.classify(rec, POL)["outcome"] == "FIT_FAIL"
    rec["final_eval"]["sse"], rec["final_eval"]["dot"] = 0.5, math.nextafter(0.65, 0.0)
    assert P.classify(rec, POL)["reasons"] == ["cosine fails"]
    rec["final_eval"] = {
        "finite": True,
        "n": 1,
        "sse": 0.4,
        "target_energy": 1.0,
        "pred_energy": 1.0,
        "dot": 0.9,
    }
    rec["baselines"] = {"zero_prediction_mse": 0.5, "scalar_mean_mse": 0.5}  # 0.8 x 0.5 = 0.4: on the bound
    assert P.classify(rec, POL)["outcome"] == "PASS"
    rec["final_eval"]["sse"] = math.nextafter(0.4, 1.0)
    assert P.classify(rec, POL)["reasons"] == ["baseline fails"]


def test_noninformative_is_only_reachable_with_a_zero_best_baseline_record():
    # SCALAR / structural: for a consistent record a zero best baseline means a degenerate input, which is
    # classified first; NONINFORMATIVE needs baselines that disagree with the input description
    rec = record(scaled(0.9))
    rec["baselines"] = {"zero_prediction_mse": 0.0, "scalar_mean_mse": 0.0}
    out = P.classify(rec, POL)
    assert out["outcome"] == "NONINFORMATIVE" and out["class"] == "F"


# ---- REALIZABLE: complete inputs from actual vectors ---------------------------------------------------------
def orthogonal(x: list[float]) -> list[float]:
    """y with y . x = 0 exactly and ||y|| = ||x|| exactly: pairs (u, v) -> (v, -u)."""
    y = []
    for i in range(0, len(x), 2):
        y += [x[i + 1], -x[i]]
    return y


Y = orthogonal(X)


def mix(a: float, b: float) -> list[float]:
    return [a * u + b * v for u, v in zip(X, Y)]


def test_realizable_inputs_are_what_they_claim():
    assert math.fsum(u * v for u, v in zip(X, Y)) == 0.0
    assert math.fsum(v * v for v in Y) == math.fsum(v * v for v in X)
    assert baselines(X)["scalar_mean_mse"] < baselines(X)["zero_prediction_mse"]


def ep_input(rec: dict) -> dict:
    """The frozen EP-20261012-v1 classifier input for the same record (no prerequisite)."""
    return {
        "evidence": rec["evidence"],
        "run": {
            **{
                k: rec["run"][k]
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
        "final_eval": rec["final_eval"],
        "baselines": {"scalar_mean_mse": rec["baselines"]["scalar_mean_mse"]},
    }


def test_primary_pass_despite_historical_strict_and_ep_v1_failure():
    sys.path.insert(0, str(EP_DIR))
    import ep_classify as E  # frozen, standard library only

    rec = record(mix(0.75, 0.6))  # RSE 0.4225, cosine 0.7809: inside the new bounds, outside EP-v1 M2
    out = P.classify(rec, POL)
    q = out["quantities"]
    assert out["outcome"] == "PASS" and out["class"] == "P"
    assert 0.42 < q["rse"] < 0.43 and 0.78 < q["cosine"] < 0.79
    ep = E.classify(ep_input(rec))
    assert (
        ep["outcome"] == "FAIL" and ep["checks"]["M2"]["pass"] is False and ep["checks"]["M1"]["pass"] is True
    )
    strict = q["rse"] <= 0.01 and q["cosine"] >= 0.99
    assert strict is False  # historical strict fails as well; neither reference vetoes the primary PASS


def test_rse_failure_with_perfect_direction():
    out = P.classify(record(scaled(0.25)), POL)  # RSE 0.5625, cosine 1
    assert out["outcome"] == "FIT_FAIL" and out["reasons"] == ["rse fails"] and out["class"] == "F"
    assert out["checks"]["cosine"]["pass"] is True


def test_near_rse_boundary_from_real_vectors():
    c = 1 - math.sqrt(0.5)  # RSE = (1 - c)^2 = 0.5 up to rounding
    inside, outside = P.classify(record(scaled(c + 1e-6)), POL), P.classify(record(scaled(c - 1e-6)), POL)
    assert inside["outcome"] == "PASS" and 0 < 0.5 - inside["quantities"]["rse"] < 2e-6
    assert outside["outcome"] == "FIT_FAIL" and 0 < outside["quantities"]["rse"] - 0.5 < 2e-6


def test_cosine_failure_comes_with_an_rse_failure():
    out = P.classify(record(mix(0.5, 0.7)), POL)  # cosine 0.581, RSE 0.74
    assert out["outcome"] == "FIT_FAIL" and set(out["reasons"]) == {"rse fails", "cosine fails"}


def test_rse_bound_implies_cosine_above_0_7071_on_realizable_records():
    import random

    rng = random.Random(20261013)
    seen = 0
    for _ in range(4000):
        a, b = rng.uniform(0, 1.5), rng.uniform(-1, 1)
        noise = [rng.gauss(0, 0.05) for _ in X]
        out = P.classify(record([v + e for v, e in zip(mix(a, b), noise)]), POL)
        q = out.get("quantities")
        if q and q["rse"] <= 0.5:
            seen += 1
            assert q["cosine"] >= math.sqrt(0.5) - 1e-12 and out["checks"]["cosine"]["pass"] is True
    assert seen > 500


def test_baseline_condition_binds_for_an_input_with_a_large_mean():
    big = [10.0 + v for v in X]
    mean = math.fsum(big) / len(big)
    flat = P.classify(record([mean] * len(big), x=big), POL)  # the scalar-mean prediction itself
    assert flat["checks"]["rse"]["pass"] and flat["checks"]["cosine"]["pass"]
    assert flat["outcome"] == "FIT_FAIL" and flat["reasons"] == ["baseline fails"]
    assert flat["quantities"]["best_baseline"] == "scalar_mean"
    half = P.classify(record([mean + 0.5 * (v - mean) for v in big], x=big), POL)
    assert half["outcome"] == "PASS" and abs(half["quantities"]["mse_over_best_baseline"] - 0.25) < 1e-12


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "nan", None])
def test_nonfinite_or_unrecorded_sums_fail_and_never_pass(bad):
    rec = record(scaled(0.9))
    rec["final_eval"]["dot"] = bad
    out = P.classify(rec, POL)
    assert out["outcome"] == "FIT_FAIL" and out["checks"]["defined"]["pass"] is False


def test_nonfinite_reconstruction_from_real_vectors_fails():
    pred = scaled(0.9)
    pred[5] = float("nan")
    out = P.classify(record(pred), POL)
    assert (
        out["outcome"] == "FIT_FAIL"
        and "undefined or non-finite required quantity cannot pass" in out["reasons"]
    )


def test_zero_prediction_gives_an_undefined_cosine_and_fails():
    out = P.classify(record([0.0] * len(X)), POL)
    assert out["outcome"] == "FIT_FAIL" and any("cosine undefined" in r for r in out["reasons"])


# ---- validity: unknown outcomes are not fitting results ------------------------------------------------------
@pytest.mark.parametrize(
    ("edit", "outcome"),
    [
        ({"evidence__verified": False}, "UNVERIFIED"),
        ({"evidence__problems": ["ledger roots disagree"]}, "UNVERIFIED"),
        ({"run__status": "RESOURCE_STOP"}, "NOT_ELIGIBLE"),
        ({"run__status": "EXECUTION_ERROR"}, "NOT_ELIGIBLE"),
        ({"run__status": "INCOMPLETE"}, "NOT_ELIGIBLE"),
        ({"run__updates_done": 2999}, "NOT_ELIGIBLE"),
        ({"run__interrupted": True}, "NOT_ELIGIBLE"),
        ({"run__guard_actions": ["killed: RSS limit"]}, "NOT_ELIGIBLE"),
        ({"run__started": False}, "NOT_RUN"),
    ],
)
def test_invalid_incomplete_or_unverified_execution_is_class_u_even_with_good_metrics(edit, outcome):
    rec = record(scaled(0.95), **edit)
    out = P.classify(rec, POL)
    assert out["outcome"] == outcome and out["class"] == "U"
    assert "quantities" not in out  # metrics are never assessed for an unknown outcome


def test_an_exceeded_limit_is_not_eligible():
    rec = record(scaled(0.95))
    rec["run"]["limits"]["arm_wall_s"] = {"used": 1800.000001, "limit": 1800.0}
    assert P.classify(rec, POL)["outcome"] == "NOT_ELIGIBLE"
    rec["run"]["limits"]["arm_wall_s"] = {"used": 1800.0, "limit": 1800.0}  # inclusive
    assert P.classify(rec, POL)["outcome"] == "PASS"


def test_missing_sections_are_unverified():
    rec = record(scaled(0.95))
    del rec["baselines"]
    assert P.classify(rec, POL)["outcome"] == "UNVERIFIED"
    rec = record(scaled(0.95))
    del rec["final_eval"]["sse"]
    assert P.classify(rec, POL)["outcome"] == "UNVERIFIED"


def test_a_never_started_arm_needs_no_final_eval():
    rec = record(scaled(0.95), run__started=False, run__status=None)
    del rec["final_eval"]
    assert P.classify(rec, POL)["outcome"] == "NOT_RUN"


@pytest.mark.parametrize(
    "edit",
    [
        {"input__all_finite": False},
        {"input__target_energy": 0.0},
        {"input__scalar_mean_mse": 0.0},
        {"input__normalized_max_abs": 1.0},
        {"input__normalized_max_abs": float("nan")},
    ],
)
def test_degenerate_registered_input_is_an_explicit_nonpass(edit):
    rec = record(scaled(0.95), **edit, run__started=False)
    out = P.classify(rec, POL)
    assert out["outcome"] == "DEGENERATE_INPUT" and out["class"] == "F"


def test_degeneracy_of_real_vectors():
    const = [0.003] * 64
    d = P.degenerate(
        {
            "all_finite": True,
            "target_energy": math.fsum(v * v for v in const),
            "scalar_mean_mse": baselines(const)["scalar_mean_mse"],
            "normalized_max_abs": 0.2,
        }
    )
    assert d == ["scalar-mean MSE 0.0 is not positive (constant input)"]
    zero = [0.0] * 64
    assert (
        len(
            P.degenerate(
                {"all_finite": True, "target_energy": 0.0, "scalar_mean_mse": 0.0, "normalized_max_abs": 0.0}
            )
        )
        == 2
        and baselines(zero)["scalar_mean_mse"] == 0.0
    )


# ---- aggregate: every P/F/U combination of four arms ---------------------------------------------------------
REP = {"P": "PASS", "F": "FIT_FAIL", "U": "NOT_ELIGIBLE"}


@pytest.mark.parametrize("combo", sorted(set(itertools.product("PFU", repeat=4))))
def test_aggregate_all_81_assignments(combo):
    arms = {f"U-TILE-{i + 1}": REP[c] for i, c in enumerate(combo)}
    out = P.aggregate(arms, 4, POL)
    p, f = combo.count("P"), combo.count("F")
    assert out["counts"] == {"P": p, "F": f, "U": 4 - p - f}
    expected = (
        "COVERAGE_CRITERION_MET" if p >= 3 else "COVERAGE_CRITERION_NOT_MET" if f >= 2 else "INCONCLUSIVE"
    )
    assert out["decision"] == expected


def test_aggregate_counts_degenerate_and_noninformative_as_f_and_missing_arms_as_u():
    out = P.aggregate(
        {"U-TILE-1": "PASS", "U-TILE-2": "DEGENERATE_INPUT", "U-TILE-3": "NONINFORMATIVE"}, 4, POL
    )
    assert out["counts"] == {"P": 1, "F": 2, "U": 1} and out["decision"] == "COVERAGE_CRITERION_NOT_MET"
    out = P.aggregate(
        {"U-TILE-1": "PASS", "U-TILE-2": "PASS", "U-TILE-3": "NOT_RUN", "U-TILE-4": "UNVERIFIED"}, 4, POL
    )
    assert out["counts"] == {"P": 2, "F": 0, "U": 2} and out["decision"] == "INCONCLUSIVE"


def test_aggregate_refuses_other_arm_counts_and_unknown_outcomes():
    with pytest.raises(P.PolicyError):
        P.aggregate({"a": "PASS"}, 5, POL)
    with pytest.raises(P.PolicyError):
        P.aggregate({"a": "COMPLETED_PASS"}, 4, POL)


def test_classify_does_not_mutate_its_input():
    rec = record(mix(0.75, 0.6))
    before = copy.deepcopy(rec)
    P.classify(rec, POL)
    assert rec == before


@pytest.mark.parametrize(
    "pred",
    [mix(0.75, 0.6), scaled(0.25), mix(0.5, 0.7), scaled(0.95), [0.0] * len(X), scaled(1 - math.sqrt(0.5))],
)
def test_primary_metrics_is_the_metric_part_of_classify(pred):
    rec = record(pred)
    full = P.classify(rec, POL)
    b = rec["baselines"]
    part = P.primary_metrics(rec["final_eval"], b["zero_prediction_mse"], b["scalar_mean_mse"], POL)
    assert part["outcome"] == full["outcome"] and part["reasons"] == full["reasons"]
    assert part.get("quantities") == full.get("quantities")
    assert all(full["checks"][k] == v for k, v in part["checks"].items())


def test_outcome_and_class_tables_match_the_policy_file():
    assert tuple(POL["outcome_order"]) == P.OUTCOMES and POL["outcome_classes"] == P.CLASSES
    assert {POL["outcome_classes"][o] for o in P.OUTCOMES} == {"P", "F", "U"}
