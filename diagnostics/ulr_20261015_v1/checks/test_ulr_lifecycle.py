"""ULR-20261015-v1 lifecycle tests: the real supervisor, all five children and the analyzer, in subprocesses, on
the synthetic engineering fixture. Checks the completed path, independent arm starts from one backbone, real payload
lengths, held-out isolation, the second-invocation refusal and the failure path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "checks"))
from make_fixture import make_fixture  # noqa: E402


def env() -> dict:
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"})
    return e


def supervise(cfg: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(PKG / "ulr_supervise.py"), "--fixture", str(cfg)],
                          capture_output=True, text=True, cwd=str(PKG), env=env(), timeout=900)  # fmt: skip


def load(p: Path) -> dict:
    return json.loads(p.read_text("utf-8"))


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("ulr_fixture_ok")
    cfg = make_fixture(root)
    return root, cfg, supervise(cfg)


def test_all_steps_complete_with_empty_stderr(completed):
    root, _, r = completed
    assert r.returncode == 0, r.stdout + r.stderr
    s = json.loads(r.stdout)
    assert set(s["steps"].values()) == {"COMPLETED"} and s["counters"] == {
        "neural_starts": 3,
        "channel_pca_fits": 1,
    }
    assert s["ledger_after"]["invocation_closed"] is True
    for p in (root / "run" / "steps").rglob("*.stderr.txt"):
        assert p.stat().st_size == 0, p


def test_arms_start_from_one_backbone_with_their_own_bridge(completed):
    root, cfg, _ = completed
    c = load(cfg)
    recs = {
        a: load(root / f"run/steps/{a}/train_record.json")
        for a in ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED")
    }
    assert {r["start"]["backbone_sha256"] for r in recs.values()} == {
        c["uoa"]["source_state"]["state_sha256"]
    }
    assert all(r["start"]["bridge_equals_persisted_init"] for r in recs.values())
    assert len({r["start"]["state_sha256"] for r in recs.values()}) == 3
    assert [r["model"]["wire_latent_numel"] for r in recs.values()] == [256, 64, 16]
    assert {r["schedule"]["sha256"] for r in recs.values()} == {c["schedule_sha256"]}
    for r in recs.values():
        assert [k["update"] for k in r["checkpoints"]] == [0, 10, 20, 30] and r["updates_done"] == 30
        assert r["trainable"]["tensors_missing_from_optimizer"] == 0
        assert {tuple(k["train_eval"]["payload_bytes"]) for k in r["checkpoints"]} == {
            (r["model"]["payload_bytes"],)
        }


def test_init_identity_check_and_isolation(completed):
    root, cfg, _ = completed
    ini = load(root / "run/steps/INIT/init_record.json")
    assert ini["step_zero_train"]["16"]["identity_check"]["within_tolerance"] is True
    assert sorted(ini["step_zero_train"], key=int) == ["1", "2", "4", "8", "16"]
    assert {v["payload_bytes"] for v in ini["step_zero_train"].values()} == {64, 128, 256, 512, 1024}
    c = load(cfg)
    for s, name in (("INIT", "init"), ("C4-REDUCED", "train")):
        job = load(root / f"run/steps/{s}/{name}_job.json")
        assert c["ucx"]["heldout_tiles"]["path"] in job["forbidden_paths"]
        assert set(c["uoa"]["eval_outputs"]) <= set(job["forbidden_paths"])


def test_eval_payloads_receivers_and_analyzer(completed):
    root, cfg, _ = completed
    ev = load(root / "run/steps/EVAL/eval_record.json")
    w = ev["payload_bytes_per_tile"]
    assert w["C16-CONTROL:FINAL"] == 1024 and w["C4-REDUCED:FINAL"] == 256 and w["C1-REDUCED:FINAL"] == 64
    assert (
        w["PCA-16"] == 64 and w["PCA-64"] == 256 and w["SCALAR-4"] == 256 and w["SOURCE-LEAKY:INT8LAT"] == 256
    )
    assert ev["wire_latent_shapes"]["C1-REDUCED:FINAL"] == [1, 4, 4]
    assert ev["isolation"]["global_rng_unchanged"] and not ev["isolation"]["any_model_in_training_mode"]
    r = subprocess.run([sys.executable, "-B", str(PKG / "ulr_analyze.py"), str(root / "analysis"), str(cfg)],
                       capture_output=True, text=True, cwd=str(PKG), env=env(), timeout=600)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    a = load(root / "analysis" / "analysis.json")
    assert a["evidence_problems"] == [] and a["A_element_reduction"]["all_verified"] is True
    for arm in ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED"):
        assert a["B_upp"][f"{arm}:FINAL"]["study"] in ("PASS", "PARTIAL", "FAIL")
        lc = a["C_learning_curves"][arm]
        assert (
            lc["0"]["train_child_eval_bit_identical_to_eval_step"]
            and lc["30"]["train_child_eval_bit_identical_to_eval_step"]
        )
        assert lc["0"]["init_step_zero_bit_identical_to_eval_step"]
    assert a["reproduction_vs_uoa"] == {"available": False}


def test_second_invocation_is_refused(completed):
    _, cfg, _ = completed
    r = supervise(cfg)
    assert r.returncode == 2 and "already records an invocation" in r.stderr


def test_failed_arm_ends_the_invocation(tmp_path):
    cfg = make_fixture(tmp_path, hooks={"C4-REDUCED": {"raise_at_update": 5}})
    r = supervise(cfg)
    s = json.loads(r.stdout)
    assert r.returncode == 1 and s["steps"]["INIT"] == s["steps"]["C16-CONTROL"] == "COMPLETED"
    assert (
        s["steps"]["C4-REDUCED"] == "EXECUTION_ERROR"
        and [s["steps"][k] for k in ("C1-REDUCED", "EVAL")] == ["NOT_RUN"] * 2
    )
    assert s["counters"]["neural_starts"] == 2
