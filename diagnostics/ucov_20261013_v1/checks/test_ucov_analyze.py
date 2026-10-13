"""ENGINEERING CHECK: the post-run analysis on real fixture runs (toy tiles, fixture seed and scale).

A fixture invocation names no UCOV manifest and uses other inputs and another initial state, so the evidence is
UNVERIFIED with those problems named and every arm is class U. The wiring is checked as well:
- the policy identity;
- the primary verdicts recomputed by the analyzer's metric path, which must equal the child's;
- the secondary references;
- the aggregate.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCOV))

import ucov_analyze as A  # noqa: E402
import ucov_control as U  # noqa: E402
import upp_classify as P  # noqa: E402

ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
FIXTURE = json.loads((UCOV / "checks" / "fixture_config.json").read_text(encoding="utf-8"))
ARMS = [{"arm": a, "max_updates": 6, "max_seconds": 120.0} for a in U.ARMS]


def fixture_run(tmp_path, arms, hooks=None):
    args = [
        sys.executable,
        "-B",
        str(UCOV / "checks" / "fixture_run.py"),
        str(tmp_path),
        json.dumps(arms),
        "{}",
    ]
    if hooks is not None:
        args.append(json.dumps(hooks))
    p = subprocess.run(args, cwd=UCOV, env=ENV, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def fixture_resolved():
    """The fixture config completed with the registered (real) resolution fields the analyzer reads."""
    real = U.resolve(U.load_ucov_config())
    return {
        **real,
        "inputs": FIXTURE["inputs"],
        "inputs_expected": FIXTURE["inputs_expected"],
        "max_updates": 6,
        "max_seconds": 120.0,
        "limits": {
            **real["limits"],
            "max_training_starts": 4,
            "aggregate_seconds": 600.0,
            "evidence_cap_bytes": 64 * 2**20,
        },
    }


def test_fixture_run_analysis_is_unverified_with_named_problems(tmp_path):
    fixture_run(tmp_path, ARMS)
    evidence = tmp_path / "analysis"
    evidence.mkdir()
    out = A.analyze(
        fixture_resolved(), evidence, tmp_path / "run", tmp_path / "mirror", manifest_sha256="fixture"
    )
    assert out["primary_policy"]["policy_id"] == "UPP-20261013-v1"
    assert out["primary_policy"]["policy_sha256"] == FIXTURE["primary_policy"]["policy_sha256"]
    for row in out["arms"].values():
        assert row["primary"]["outcome"] == "UNVERIFIED" and row["primary"]["class"] == "U"
        probs = row["evidence"]["problems"]
        assert "the invocation names another UCOV manifest" in probs
        assert "the run's initial state is not the U-FIT initial state" in probs
        assert row["secondary"]["ep_v1"]["outcome"] == "UNVERIFIED"
        assert row["secondary"]["historical_strict"]["role"].startswith("reported only")
    assert (
        out["aggregate"]["counts"] == {"P": 0, "F": 0, "U": 4}
        and out["aggregate"]["decision"] == "INCONCLUSIVE"
    )
    sealed = (evidence / "FINAL_ARTIFACTS_SHA256.txt").read_bytes()
    assert b"\r\n" not in sealed
    n_files = sum(1 for q in tmp_path.rglob("*") if q.is_file() and "analysis" not in q.parts)
    assert len(sealed.decode("utf-8").splitlines()) - 1 == n_files
    json.dumps(out, allow_nan=False)


def test_with_evidence_marked_valid_the_analyzer_reproduces_the_childs_primary_verdict(tmp_path):
    fixture_run(tmp_path, ARMS)
    resolved = fixture_resolved()
    ledger = U.K.Ledger(tmp_path / "run" / "ledger", tmp_path / "mirror").read()
    inv, inv_end = ledger["invocation_001.json"], ledger["invocation_001_end.json"]
    by_arm = {a["arm"]: a for a in inv_end["arms"]}
    policy = U.load_primary_policy(resolved["primary_policy"])
    for arm in U.ARMS:
        child = json.loads((tmp_path / "run/arms" / arm / "child_record.json").read_text(encoding="utf-8"))
        ok = {"verified": True, "problems": []}
        prim_in = A.primary_input(
            arm, child, ledger[f"end_{arm}.json"], by_arm[arm], inv, inv_end, resolved, ok, 4
        )
        prim = P.classify(prim_in, policy)
        assert prim["class"] in ("P", "F")
        assert prim["outcome"] == child["primary_verdict"]["outcome"]  # one function, one set of numbers
        assert (
            prim_in["final_eval"]["target_energy"] == child["input"]["original"]["energy_sum_sq"]
        )  # original space
        assert child["status"] == ("COMPLETED_PASS" if prim["outcome"] == "PASS" else "COMPLETED_FIT_FAIL")


def test_degenerate_and_not_run_arms_get_their_classes_from_the_same_inputs(tmp_path):
    s = fixture_run(tmp_path, [ARMS[0], {**ARMS[1], "usable": False}, ARMS[2], ARMS[3]])
    assert s["arms"][1]["status"] == "NOT_RUN" and "degenerate" in s["arms"][1]["reason"]
    resolved = fixture_resolved()
    resolved["inputs_expected"] = {
        **resolved["inputs_expected"],
        "U-TILE-2": {
            **resolved["inputs_expected"]["U-TILE-2"],
            "usable": False,
            "degeneracy_checks": {"all_finite": True},
            "original_scalar_mean_mse": 0.0,
        },
    }
    ledger = U.K.Ledger(tmp_path / "run" / "ledger", tmp_path / "mirror").read()
    inv, inv_end = ledger["invocation_001.json"], ledger["invocation_001_end.json"]
    by_arm = {a["arm"]: a for a in inv_end["arms"]}
    policy = U.load_primary_policy(resolved["primary_policy"])
    ok = {"verified": True, "problems": []}
    deg = P.classify(
        A.primary_input("U-TILE-2", None, None, by_arm["U-TILE-2"], inv, inv_end, resolved, ok, 3), policy
    )
    assert (
        deg["outcome"] == "DEGENERATE_INPUT" and deg["class"] == "F"
    )  # a registered degenerate input: F, not U
    # the same arm with a usable registered input but never started is NOT_RUN (U)
    usable = {**resolved, "inputs_expected": fixture_resolved()["inputs_expected"]}
    nr = P.classify(
        A.primary_input("U-TILE-2", None, None, by_arm["U-TILE-2"], inv, inv_end, usable, ok, 3), policy
    )
    assert nr["outcome"] == "NOT_RUN" and nr["class"] == "U"
    # through the full analyzer the fixture evidence is unverified, so every arm is U there
    evidence = tmp_path / "analysis"
    evidence.mkdir()
    out = A.analyze(resolved, evidence, tmp_path / "run", tmp_path / "mirror", manifest_sha256="fixture")
    assert out["arms"]["U-TILE-2"]["started"] is False and out["arms"]["U-TILE-2"]["primary"]["class"] == "U"
    assert out["ledger"]["starts"] == 3
