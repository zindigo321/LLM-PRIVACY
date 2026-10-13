"""ENGINEERING CHECK: the post-run analysis on a real fixture run (toy 16x16 patch, fixture seed and scale).

The fixture invocation names no URP manifest and uses another input and initial state, so the evidence must be
UNVERIFIED with those problems named. The strict re-derivation, the classifier wiring (original space, the patch's own
baselines) and the reported-only sections are still produced. The EP classifier itself (``ep_classify``) was reviewed
with its own 34 targeted tests and is not re-tested here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP))

import urp_analyze as A  # noqa: E402
import urp_control as U  # noqa: E402

ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def fixture_run(tmp_path):
    arm = {"arm": "FX-UPATCH", "max_updates": 6, "max_seconds": 120.0}
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            str(URP / "checks" / "fixture_run.py"),
            str(tmp_path),
            json.dumps([arm]),
            "{}",
        ],
        cwd=URP,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert p.returncode == 0, p.stderr


def test_analysis_of_a_fixture_run(tmp_path):
    fixture_run(tmp_path)
    evidence = tmp_path / "analysis"
    evidence.mkdir()
    resolved = U.load_resolved()
    out = A.analyze(
        resolved, evidence, tmp_path / "run", tmp_path / "mirror", arm="FX-UPATCH", manifest_sha256="fixture"
    )
    problems = out["evidence"]["problems"]
    assert out["evidence"]["verified"] is False
    assert "the invocation names another URP manifest" in problems
    assert "the run read another input patch" in problems
    assert "the run's initial state is not the U-FIT initial state" in problems
    assert not any("state file" in q for q in problems)  # both fixture state files match their records
    sealed = (evidence / "FINAL_ARTIFACTS_SHA256.txt").read_bytes()
    assert b"\r\n" not in sealed
    n_files = sum(1 for q in tmp_path.rglob("*") if q.is_file() and "analysis" not in q.parts)
    assert len(sealed.decode("utf-8").splitlines()) - 1 == n_files
    assert out["strict_near_identity"]["matches_record"] is True
    assert out["strict_near_identity"]["space"].startswith("original")
    assert out["exploratory_ep"]["outcome"] == "UNVERIFIED"  # unverified evidence is never PASS
    child = json.loads((tmp_path / "run/arms/FX-UPATCH/child_record.json").read_text(encoding="utf-8"))
    ci = A.classifier_input(
        child,
        json.loads((tmp_path / "run/ledger/end_FX-UPATCH.json").read_text(encoding="utf-8")),
        json.loads((tmp_path / "run/ledger/invocation_001.json").read_text(encoding="utf-8")),
        json.loads((tmp_path / "run/ledger/invocation_001_end.json").read_text(encoding="utf-8")),
        resolved,
        {"verified": True, "problems": []},
        1,
    )
    assert ci["final_eval"]["target_energy"] == child["input"]["original"]["energy_sum_sq"]  # original space
    assert ci["baselines"]["scalar_mean_mse"] == child["baselines"]["original"]["shared_scalar_mean"]["mse"]
    r = out["reported_only"]
    assert r["final_eval_original"]["recorded_equal"] and r["final_eval_normalized"]["recorded_equal"]
    assert r["curve"]["points"] == 4 and r["curve"]["all_finite"] is True
    assert "not matched" in r["u_fit_synthetic_context"]["label"]
    assert any("compression" in s for s in out["does_not_establish"])
    json.dumps(out, allow_nan=False)
