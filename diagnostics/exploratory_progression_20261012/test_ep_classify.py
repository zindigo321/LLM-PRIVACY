"""Targeted checks of ``ep_classify``: thresholds at and beyond the boundary, undefined metrics, the zero-baseline
case, outcome order and evidence verification. Synthetic inputs and temporary files only; nothing registered.

    python -B -m pytest -q -p no:cacheprovider test_ep_classify.py
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ep_classify as E  # noqa: E402

N = 100


def make(rse: float, cosine: float, target: float = 50.0, mean_mse: float = 0.45) -> dict:
    """A classifier input whose sums reproduce ``rse`` and ``cosine``: SSE = rse * T and D = cosine * sqrt(P * T) with
    P = T. The zero-prediction MSE is T / N = 0.5 by default."""
    return {
        "evidence": {"verified": True, "problems": []},
        "run": {
            "status": "COMPLETED_FIT_FAIL",
            "updates_done": 3000,
            "updates_planned": 3000,
            "prerequisite": {"control": "U-CHECK", "status": "COMPLETED_PASS"},
            "interrupted": False,
            "guard_actions": [],
            "limits": {k: {"used": 1, "limit": 2} for k in E.REQUIRED_LIMITS},
        },
        "final_eval": {
            "finite": True,
            "n": N,
            "sse": rse * target,
            "target_energy": target,
            "pred_energy": target,
            "dot": cosine * target,
        },
        "baselines": {"scalar_mean_mse": mean_mse},
    }


def outcome(rec: dict) -> str:
    return E.classify(rec)["outcome"]


def test_pass_well_inside():
    res = E.classify(make(0.27, 0.85))
    assert res["outcome"] == "PASS" and res["reasons"] == []
    assert res["quantities"]["best_baseline"] == "scalar_mean"


def test_rse_and_cosine_exactly_at_threshold_pass():
    res = E.classify(make(0.5, 0.8, mean_mse=1.0))  # zero MSE 0.5 is the best baseline; MSE 0.25 <= 0.4
    assert res["quantities"]["rse"] == 0.5 and res["quantities"]["cosine"] == 0.8
    assert res["checks"]["M1"]["pass"] and res["checks"]["M2"]["pass"] and res["outcome"] == "PASS"


def test_one_ulp_beyond_threshold_fails():
    # unit energies, so SSE = RSE and D = cosine exactly (with T = 50, nextafter(0.8, 0) * 50 rounds to 40.0)
    for rse, cos, check in ((math.nextafter(0.5, 1.0), 0.85, "M1"), (0.27, math.nextafter(0.8, 0.0), "M2")):
        res = E.classify(make(rse, cos, target=1.0, mean_mse=1.0))
        assert res["quantities"]["rse"] == rse and res["quantities"]["cosine"] == cos
        assert not res["checks"][check]["pass"] and res["outcome"] == "FAIL"


def test_baseline_factor_boundary():
    # MSE = 0.27 * 50 / 100 = 0.135; the bound 0.8 * best equals it exactly when best = 0.135 / 0.8
    rec = make(0.27, 0.85)
    mse = rec["final_eval"]["sse"] / N
    best = mse / 0.8
    rec["baselines"]["scalar_mean_mse"] = best
    if not mse <= 0.8 * best:  # float64 rounding: step to the smallest best that satisfies the bound exactly
        best = math.nextafter(best, math.inf)
        rec["baselines"]["scalar_mean_mse"] = best
    assert mse <= 0.8 * best and E.classify(rec)["checks"]["M3"]["pass"]
    lower = best
    while mse <= 0.8 * lower:  # the largest best that fails is at most a few ulps below
        lower = math.nextafter(lower, 0.0)
    rec["baselines"]["scalar_mean_mse"] = lower
    res = E.classify(rec)
    assert not res["checks"]["M3"]["pass"] and res["outcome"] == "FAIL"


def test_m3_fails_when_a_constant_explains_the_input():
    # large-mean input: RSE 0.27 looks good, but a scalar mean does better than the model
    res = E.classify(make(0.27, 0.85, mean_mse=0.1))
    assert res["checks"]["M1"]["pass"] and not res["checks"]["M3"]["pass"] and res["outcome"] == "FAIL"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sse", math.nan),
        ("sse", math.inf),
        ("sse", None),
        ("dot", "0.4"),
        ("target_energy", 0.0),
        ("pred_energy", 0.0),
        ("n", 0),
        ("n", 100.0),
        ("finite", False),
        ("finite", None),
    ],
)
def test_undefined_or_nonfinite_required_quantity_fails(field, value):
    rec = make(0.27, 0.85)
    rec["final_eval"][field] = value
    res = E.classify(rec)
    assert res["outcome"] == "FAIL" and not res["checks"]["M0"]["pass"] and "quantities" not in res


@pytest.mark.parametrize("value", [math.nan, None, -1e-9])
def test_undefined_scalar_mean_baseline_fails(value):
    rec = make(0.27, 0.85)
    rec["baselines"]["scalar_mean_mse"] = value
    assert outcome(rec) == "FAIL"


def test_zero_best_baseline_is_uninformative_not_pass():
    res = E.classify(make(0.27, 0.85, mean_mse=0.0))
    assert res["outcome"] == "UNINFORMATIVE" and res["checks"]["M3"]["uninformative"] is True
    assert outcome(make(0.6, 0.85, mean_mse=0.0)) == "FAIL"  # a failing absolute gate still fails


def test_missing_key_is_unverified():
    rec = make(0.27, 0.85)
    del rec["baselines"]["scalar_mean_mse"]
    assert outcome(rec) == "UNVERIFIED"
    rec = make(0.27, 0.85)
    del rec["run"]["limits"]["rss_bytes"]
    assert outcome(rec) == "UNVERIFIED"
    assert outcome({}) == "UNVERIFIED"


def test_unverified_evidence_precedes_good_metrics():
    rec = make(0.0, 1.0)
    rec["evidence"] = {"verified": False, "problems": ["hash mismatch"]}
    res = E.classify(rec)
    assert res["outcome"] == "UNVERIFIED" and "quantities" not in res
    rec["evidence"] = {"verified": "yes", "problems": []}
    assert outcome(rec) == "UNVERIFIED"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "RESOURCE_STOP"),
        (("status",), "EXECUTION_ERROR"),
        (("updates_done",), 2999),
        (("prerequisite",), {"control": "U-CHECK", "status": "COMPLETED_FIT_FAIL"}),
        (("interrupted",), True),
        (("guard_actions",), ["killed: RSS limit"]),
        (("limits", "rss_bytes"), {"used": 3, "limit": 2}),
        (("limits", "arm_wall_s"), {"used": math.nan, "limit": 1800}),
        (("limits", "evidence_bytes"), {"used": 1, "limit": None}),
    ],
)
def test_noncompliant_run_is_not_eligible(path, value):
    rec = make(0.27, 0.85)
    target = rec["run"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert outcome(rec) == "NOT_ELIGIBLE"


def test_limit_exactly_reached_is_compliant():
    rec = make(0.27, 0.85)
    rec["run"]["limits"]["training_starts"] = {"used": 2, "limit": 2}
    assert outcome(rec) == "PASS"


def test_classify_does_not_mutate_its_input():
    rec = make(0.27, 0.85)
    before = copy.deepcopy(rec)
    E.classify(rec)
    assert rec == before


def test_verify_files_detects_tampering_and_unlisted_reads(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b"{}")
    b.write_bytes(b"[]")
    digest = hashlib.sha256(b"{}").hexdigest()
    listing = tmp_path / "list.txt"
    listing.write_text(f"recorded: now\n{digest}  2  {a}\n", encoding="utf-8")
    rows, problems = E.verify_files(listing, [a])
    assert problems == [] and rows[0]["match"]
    _, problems = E.verify_files(listing, [a, b])
    assert problems == [f"read but not on the hash list: {b}"]
    a.write_bytes(b"{ }")
    _, problems = E.verify_files(listing, [a])
    assert len(problems) == 1 and problems[0].startswith("hash or size mismatch")


def test_assess_uid_r1_with_missing_evidence_is_unverified(tmp_path):
    rec = E.assess_uid_r1(evidence=tmp_path, package=tmp_path, run_root=tmp_path)
    assert rec["evidence"]["verified"] is False
    assert E.classify(rec)["outcome"] == "UNVERIFIED"
