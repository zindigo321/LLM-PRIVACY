"""ENGINEERING FIXTURES: real-subprocess lifecycle tests of the URC supervisor and child (fixture config only).

Each test runs ``checks/fixture_run.py`` as a real process with the fixture's own seeds, 16x16 input and narrow
model, in pytest's ``tmp_path``. Optimizer updates happen only on these fixture models. The r1 lifecycle suite
(guards, deadlines, interrupts, aggregate budget; ``uid_20261012_v2_r1/evidence/final4_20261012``) covers the
supervision code that is copied unchanged; these tests cover what URC-v1 changed or added.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

URC = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN = URC / "checks" / "fixture_run.py"
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
FIT = {"arm": "FX-RFIT", "max_updates": 6, "max_seconds": 120.0}
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")


def run(tmp, arms, over=None, hooks=None, timeout=240):
    args = [PY, "-B", str(RUN), str(tmp), json.dumps(arms), json.dumps(over or {})]
    if hooks is not None:
        args.append(json.dumps(hooks))
    p = subprocess.run(args, cwd=URC, env=ENV, capture_output=True, text=True, timeout=timeout)
    summary = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    return p, summary


def arm_dir(tmp, arm="FX-RFIT"):
    return tmp / "run" / "arms" / arm


def child_record(tmp):
    return json.loads((arm_dir(tmp) / "child_record.json").read_text(encoding="utf-8"))


def test_completed_arm_records_states_curve_and_measurements(tmp_path):
    p, s = run(tmp_path, [FIT])
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") and a["updates_done"] == 6
    rec = child_record(tmp_path)
    assert rec["cpu"]["applied_as_requested"] is True and rec["cpu"]["deterministic_algorithms"] is True
    assert rec["step0"]["isolation_verified"] is True and set(rec["step0"]["post_init"]) == {"eval", "train"}
    curve = [json.loads(x) for x in (arm_dir(tmp_path) / "curve.jsonl").read_text().splitlines()]
    assert [c["update"] for c in curve] == [1, 2, 4, 6]
    for c in curve:
        for g in GROUPS:
            assert c["grad"][g]["grad_l2"] > 0 and c["grad"][g]["nonfinite"] == 0
            assert c["param_change_this_update"][g]["delta_l2"] > 0
        assert {"sse", "target_energy", "pred_energy", "dot", "n"} <= set(c["pre_update_train"])
    f = rec["final"]
    assert f["copies_loaded_identical_state"] is True and f["train_copy_state_changed_by_forward"] is True
    assert {"sse", "target_energy", "pred_energy", "dot", "n"} <= set(f["eval_mode"]) and rec[
        "verdict_basis"
    ] == "eval_mode"
    # initial and final states: saved, hash-equal to the recorded state hashes, and loadable bit for bit
    sys.path.insert(0, str(URC))
    import torch
    import urc_control as U

    fixture = json.loads((URC / "checks" / "fixture_config.json").read_text(encoding="utf-8"))
    for point, key in (("init", "model_state_sha256_init"), ("final", "model_state_sha256_final")):
        info = rec[f"{point}_state_file"]
        assert info["state_sha256"] == rec[key]
        model, meta = U.load_state(arm_dir(tmp_path) / f"{point}_state.safetensors", fixture)
        assert meta["point"] == point and meta["state_sha256"] == rec[key]
    final_model, _ = U.load_state(arm_dir(tmp_path) / "final_state.safetensors", fixture)
    bn = [v for k, v in final_model.state_dict().items() if k.endswith("num_batches_tracked")]
    assert bn and all(int(v) == 6 for v in bn)  # BatchNorm counters saved: one per update
    init_model, _ = U.load_state(arm_dir(tmp_path) / "init_state.safetensors", fixture)
    init_bn = [v for k, v in init_model.state_dict().items() if k.endswith("num_batches_tracked")]
    assert all(int(v) == 0 for v in init_bn)  # the initial state precedes every update
    moved = [
        k
        for k, v in final_model.state_dict().items()
        if v.is_floating_point() and not torch.equal(v, init_model.state_dict()[k])
    ]
    assert moved  # training changed the saved state
    # runtime-file coverage: only the listed modules were imported; Uformer and cg_fedllm were not
    assert rec["modules"]["local"] == [
        "resnet3_s1",
        "uid_common",
        "uid_r1_control",
        "uid_r1_measure",
        "urc_control",
    ]
    assert rec["modules_at_end"]["local"] == rec["modules"]["local"]
    assert rec["precision"]["input_dtype_cpu"] and rec["precision"]["model_float_tensors_fp32_cpu"]
    assert rec["precision"]["default_dtype"] == "torch.float32"
    assert set(rec["sampling_init_tensor_sha256"]) == {"encoder.sampling", "decoder.sampling"}
    # one start, mirrored ledger
    names = sorted(q.name for q in (tmp_path / "run" / "ledger").iterdir())
    assert names == [
        "end_FX-RFIT.json",
        "invocation_001.json",
        "invocation_001_end.json",
        "start_FX-RFIT.json",
    ]
    for n in names:
        assert (tmp_path / "run/ledger" / n).read_bytes() == (tmp_path / "mirror" / n).read_bytes()


def test_second_invocation_is_refused_and_writes_nothing(tmp_path):
    p, _ = run(tmp_path, [FIT])
    assert p.returncode == 0, p.stderr
    before = sorted(
        (q.relative_to(tmp_path).as_posix(), q.read_bytes()) for q in tmp_path.rglob("*") if q.is_file()
    )
    p2, _ = run(tmp_path, [FIT])
    assert p2.returncode == 2 and "1 invocation(s) already recorded" in p2.stderr
    after = sorted(
        (q.relative_to(tmp_path).as_posix(), q.read_bytes()) for q in tmp_path.rglob("*") if q.is_file()
    )
    assert after == before


def test_uid_arm_names_are_refused_by_the_supervisor(tmp_path):
    for arm in ("U-FIT", "U-CHECK"):
        p, _ = run(tmp_path / arm, [{**FIT, "arm": arm}])
        assert p.returncode == 2 and "UID-20261012-v2 arm" in p.stderr
        assert not (tmp_path / arm / "run").exists()  # refused before any record


def test_child_refuses_a_module_with_an_unapproved_hash(tmp_path):
    names = ("uid_common", "uid_r1_control", "uid_r1_measure", "urc_control", "resnet3_s1", "urc_child")
    p, s = run(tmp_path, [FIT], over={"expected_module_sha256": {n: "0" * 64 for n in names}})
    assert p.returncode == 0, p.stderr
    assert s["arms"][0]["status"] == "EXECUTION_ERROR" and s["arms"][0]["updates_done"] is None
    rec = child_record(tmp_path)
    assert "without the approved SHA-256" in rec["error"]["message"]
    assert not (arm_dir(tmp_path) / "curve.jsonl").read_text()
    assert not (arm_dir(tmp_path) / "init_state.safetensors").exists()  # refused before any state was written


def test_child_error_is_an_execution_error(tmp_path):
    p, s = run(tmp_path, [FIT], hooks={"raise_at_update": 3})
    assert p.returncode == 0, p.stderr
    assert s["arms"][0]["status"] == "EXECUTION_ERROR"
    assert "fixture error at update 3" in child_record(tmp_path)["error"]["message"]
    assert not (arm_dir(tmp_path) / "final_state.safetensors").exists()


def test_soft_stop_is_a_resource_stop(tmp_path):
    p, s = run(tmp_path, [{**FIT, "max_seconds": 6.0}], hooks={"sleep_each_update_s": 1.5})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["arm_wall_s"] <= 6.0


def test_rss_guard_kills_the_child(tmp_path):
    t = time.monotonic()
    p, s = run(tmp_path, [FIT], over={"rss_cap_bytes": 700 * 2**20}, hooks={"alloc_mib_at_update": [2, 1024]})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["guard"]["action"] == "killed: RSS limit"
    assert time.monotonic() - t < 120
