"""ENGINEERING CHECK: the post-run analysis on a real fixture run (16x16 fixture input, fixture seeds).

The fixture run cannot match the U-FIT reference (other input, other geometry): the analysis must say NOT_MATCHED,
name the violated shared conditions, still apply the strict and EP classifications, and compute signed differences.
The EP classifier itself (``ep_classify``) was reviewed with its own 34 targeted tests and is not re-tested here.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

URC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URC))

import urc_analyze as A  # noqa: E402
import urc_control as U  # noqa: E402

ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}


def test_analysis_of_a_fixture_run(tmp_path):
    arm = {"arm": "FX-RFIT", "max_updates": 6, "max_seconds": 120.0}
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            str(URC / "checks" / "fixture_run.py"),
            str(tmp_path),
            json.dumps([arm]),
            "{}",
        ],
        cwd=URC,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert p.returncode == 0, p.stderr
    evidence = tmp_path / "analysis"
    evidence.mkdir()
    resolved = U.load_resolved()
    out = A.analyze(
        resolved, evidence, tmp_path / "run", tmp_path / "mirror", arm="FX-RFIT", manifest_sha256="fixture"
    )
    # evidence: the fixture invocation names no URC manifest, so the evidence is (correctly) not verified
    assert out["evidence"]["verified"] is False and "another URC manifest" in out["evidence"]["problems"][0]
    sealed = (evidence / "FINAL_ARTIFACTS_SHA256.txt").read_text(encoding="utf-8").splitlines()
    assert len(sealed) - 1 == sum(1 for q in tmp_path.rglob("*") if q.is_file() and "analysis" not in q.parts)
    # strict verdict re-derived and equal to the record
    assert out["strict_near_identity"]["matches_record"] is True
    # EP: unverified evidence is never PASS
    assert out["exploratory_ep"]["outcome"] == "UNVERIFIED"
    # matching: the fixture differs from U-FIT in input bytes, shape and baselines -> NOT_MATCHED, named
    m = out["matching"]
    assert m["status"] == "NOT_MATCHED"
    assert {"input bytes", "input shape", "target energy", "zero-prediction MSE"} <= set(m["violations"])
    assert "updates done" in m["violations"]  # 6 fixture updates vs 3,000
    # execution settings of the two invocation plans: the fixture plan differs (threads, budget, cadence)
    assert {"plan.cpu", "arm max_updates", "plan.curve_every", "curve-point schedule"} <= set(m["violations"])
    assert "plan.python_executable" not in m["violations"] and "plan.torch_version" not in m["violations"]
    assert out["evidence"]["problems"] == [
        "the invocation names another URC manifest"
    ]  # U-FIT files verified
    assert len(m["permitted_differences"]) == 4
    # comparison: signed differences from recorded sums, explicitly not a valid matched comparison
    c = out["comparison"]
    assert c["rse_difference_r_minus_u"] == c["r_fit_final_eval"]["rse"] - c["u_fit_final_eval"]["rse"]
    assert (
        c["cosine_difference_r_minus_u"] == c["r_fit_final_eval"]["cosine"] - c["u_fit_final_eval"]["cosine"]
    )
    assert c["r_fit_final_eval"]["recorded_equal"] and c["u_fit_final_eval"]["recorded_equal"]
    assert c["valid_as_matched_comparison"] is False and "no significance" in c["label"]
    assert out["resources"]["u_fit_reference"]["arm_wall_s"] == 506.3440000000046
    json.dumps(out, allow_nan=False)


def test_matching_accepts_identical_records_and_flags_one_difference():
    resolved = U.load_resolved()
    ref = json.loads(Path(resolved["reference"]["child_record"]).read_text(encoding="utf-8"))
    same = A.matching(ref, ref, resolved)
    assert same["status"] == "MATCHED" and same["violations"] == []
    other = json.loads(json.dumps(ref))
    other["cpu"]["intra_op_threads"] = 2
    diff = A.matching(other, ref, resolved)
    assert diff["status"] == "NOT_MATCHED" and diff["violations"] == ["intra-op threads"]
