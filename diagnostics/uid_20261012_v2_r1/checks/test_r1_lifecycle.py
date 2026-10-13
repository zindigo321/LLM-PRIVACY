"""ENGINEERING FIXTURES: real-subprocess lifecycle tests of the r1 supervisor and child.

Every test runs ``checks/fixture_run.py`` (or the child) as a real process on the fixture config with its own
seeds, input and small models, in pytest's ``tmp_path``. Optimizer updates happen only on these fixture models.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN = R1 / "checks" / "fixture_run.py"
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
FIT = {"arm": "FX-FIT", "model_arm": "U-FIT", "max_updates": 6, "max_seconds": 120.0}
CHK = {"arm": "FX-CHECK", "model_arm": "U-CHECK", "max_updates": 40, "max_seconds": 120.0}


def run(tmp, arms, over=None, hooks=None, timeout=240):
    args = [PY, "-B", str(RUN), str(tmp), json.dumps(arms), json.dumps(over or {})]
    if hooks is not None:
        args.append(json.dumps(hooks))
    p = subprocess.run(args, cwd=R1, env=ENV, capture_output=True, text=True, timeout=timeout)
    summary = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    return p, summary


def ledger(tmp):
    return {p.name: json.loads(p.read_text()) for p in sorted((tmp / "run" / "ledger").glob("*.json"))}


def arm_dir(tmp, arm):
    return tmp / "run" / "arms" / arm


def test_completed_arm_records_required_fields_and_cpu(tmp_path):
    p, s = run(tmp_path, [FIT])
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") and a["updates_done"] == 6
    rec = json.loads((arm_dir(tmp_path, "FX-FIT") / "child_record.json").read_text())
    # CPU configuration observed in the child, before any tensor work
    assert rec["cpu"]["applied_as_requested"] is True and rec["cpu"]["deterministic_algorithms"] is True
    assert (rec["cpu"]["intra_op_threads"], rec["cpu"]["inter_op_threads"]) == (2, 1)
    assert rec["cpu"]["cuda_available"] is False
    # step 0 on disposable copies, both modes, isolation verified
    assert rec["step0"]["isolation_verified"] is True and set(rec["step0"]["post_init"]) == {"eval", "train"}
    # curve: update 1, every 2 (fixture), final; pre-update losses separate from post-update evaluation
    curve = [
        json.loads(line) for line in (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").read_text().splitlines()
    ]
    assert [c["update"] for c in curve] == [1, 2, 4, 6]
    for c in curve:
        assert {
            "pre_update_train_loss",
            "pre_update_train",
            "grad",
            "param_change_this_update",
            "param_change_from_init",
            "rss_bytes_at_point",
        } <= set(c)
        for g in ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling"):
            assert c["grad"][g]["grad_l2"] > 0 and c["grad"][g]["nonfinite"] == 0
            assert c["param_change_this_update"][g]["delta_l2"] > 0
    assert rec["trainable"]["tensors_missing_from_optimizer"] == 0
    assert rec["trainable"]["in_optimizer"] == rec["trainable"]["parameters"]
    # final: two copies of one final state, eval verdict, train mode separate
    f = rec["final"]
    assert f["copies_loaded_identical_state"] is True and {"eval_mode", "train_mode"} <= set(f)
    assert f["train_copy_state_changed_by_forward"] is True  # BN buffers moved in the train copy only
    assert rec["verdict_basis"] == "eval_mode"
    assert rec["rss"]["samples"] >= 1 and rec["rss"]["sampled_max_rss_bytes"] > 0
    assert a["guard"]["samples"] >= 1 and a["guard"]["action"] is None
    # ledger: invocation, start, end, invocation end; the mirror is identical
    assert set(ledger(tmp_path)) == {
        "invocation_001.json",
        "start_FX-FIT.json",
        "end_FX-FIT.json",
        "invocation_001_end.json",
    }
    for name in ledger(tmp_path):
        assert (tmp_path / "run/ledger" / name).read_bytes() == (tmp_path / "mirror" / name).read_bytes()


def test_second_invocation_is_refused(tmp_path):
    p, _ = run(tmp_path, [CHK])
    assert p.returncode == 0, p.stderr
    p2, _ = run(tmp_path, [CHK])
    assert p2.returncode == 2 and "1 invocation(s) already recorded" in p2.stderr


def test_per_arm_soft_stop_is_a_resource_stop(tmp_path):
    arm = {**FIT, "max_updates": 6, "max_seconds": 6.0}
    p, s = run(tmp_path, [arm], hooks={"sleep_each_update_s": 1.5})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["child_status"] == "RESOURCE_STOP"
    assert a["updates_done"] is None or a["updates_done"] < 6
    assert a["arm_wall_s"] <= 6.0  # arm limit from its start record


def test_hang_inside_an_update_is_killed_at_the_hard_deadline(tmp_path):
    arm = {**FIT, "max_seconds": 8.0}
    t = time.monotonic()
    p, s = run(tmp_path, [arm], hooks={"hang_at_update": 2})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["guard"]["action"] == "killed: hard deadline"
    assert a["child_record_present"] is False and a["arm_wall_s"] <= 8.0
    assert time.monotonic() - t < 60


def test_hang_in_final_evaluation_is_killed(tmp_path):
    arm = {**FIT, "max_seconds": 10.0}
    p, s = run(tmp_path, [arm], hooks={"hang_in_final_eval": True})
    assert p.returncode == 0, p.stderr
    assert (
        s["arms"][0]["status"] == "RESOURCE_STOP"
        and s["arms"][0]["guard"]["action"] == "killed: hard deadline"
    )


def test_aggregate_budget_is_shared_across_two_children(tmp_path):
    over = {"aggregate_seconds": 14.0, "end_margin_s": 1.0, "eval_reserve_s": 2.0}
    a1, a2 = {**CHK, "max_seconds": 60.0}, {**FIT, "max_seconds": 60.0}
    p, s = run(tmp_path, [a1, a2], over=over, hooks={"sleep_each_update_s": 0.25})
    assert p.returncode == 0, p.stderr
    second = s["arms"][1]
    # the second child gets only what remains of the shared 14 s, not a fresh budget
    assert second["status"] in ("RESOURCE_STOP", "NOT_RUN")
    assert s["aggregate_wall_s"] <= 14.0 + 1.0


def test_rss_guard_kills_the_child(tmp_path):
    p, s = run(tmp_path, [FIT], over={"rss_cap_bytes": 700 * 2**20}, hooks={"alloc_mib_at_update": [2, 1024]})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["guard"]["action"] == "killed: RSS limit"
    assert a["guard"]["sampled_max_rss_bytes"] > 700 * 2**20


def test_child_error_is_an_execution_error(tmp_path):
    p, s = run(tmp_path, [FIT], hooks={"raise_at_update": 3})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "EXECUTION_ERROR" and a["child_status"] == "EXECUTION_ERROR"
    rec = json.loads((arm_dir(tmp_path, "FX-FIT") / "child_record.json").read_text())
    assert "fixture error at update 3" in rec["error"]["message"]


def test_supervisor_interrupt_writes_end_records(tmp_path):
    args = [
        PY,
        "-B",
        str(RUN),
        str(tmp_path),
        json.dumps([{**FIT, "max_updates": 6}]),
        "{}",
        json.dumps({"sleep_each_update_s": 2.0}),
    ]
    proc = subprocess.Popen(
        args,
        cwd=R1,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    deadline = time.monotonic() + 120
    while (
        not (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").exists()
        or not (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").read_text()
    ):
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.2)
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    out, err = proc.communicate(timeout=120)
    lg = ledger(tmp_path)
    assert lg["end_FX-FIT.json"]["status"] == "INCOMPLETE", (out, err)
    assert lg["invocation_001_end.json"]["interrupted"] is True
    rec = json.loads((arm_dir(tmp_path, "FX-FIT") / "child_record.json").read_text())
    assert rec["status"] == "INCOMPLETE"


def test_killed_supervisor_leaves_an_unfinished_start_that_blocks_reentry(tmp_path):
    args = [
        PY,
        "-B",
        str(RUN),
        str(tmp_path),
        json.dumps([{**FIT, "max_updates": 6}]),
        "{}",
        json.dumps({"sleep_each_update_s": 3.0}),
    ]
    proc = subprocess.Popen(args, cwd=R1, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 120
    while (
        not (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").exists()
        or not (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").read_text()
    ):
        assert time.monotonic() < deadline and proc.poll() is None  # wait until the child is training
        time.sleep(0.2)
    proc.kill()  # the supervisor dies abruptly and writes nothing more
    proc.communicate(timeout=60)
    job = str(arm_dir(tmp_path, "FX-FIT") / "job.json")
    import psutil

    def orphans():
        found = []
        for p in psutil.process_iter(["cmdline"]):
            try:
                if job in " ".join(p.info["cmdline"] or []):
                    found.append(p.pid)
            except psutil.Error:
                pass
        return found

    t = time.monotonic()
    while orphans() and time.monotonic() - t < 30:  # the child's watchdog notices the dead supervisor
        time.sleep(0.2)
    assert orphans() == []
    assert not (arm_dir(tmp_path, "FX-FIT") / "child_record.json").exists()  # killed child: no record
    lg = ledger(tmp_path)
    assert "start_FX-FIT.json" in lg and "end_FX-FIT.json" not in lg  # the start survives, unfinished
    before = sorted(
        (q.relative_to(tmp_path).as_posix(), q.read_bytes()) for q in tmp_path.rglob("*") if q.is_file()
    )
    p2, _ = run(tmp_path, [FIT])
    assert p2.returncode == 2 and "INCOMPLETE prior start" in p2.stderr
    after = sorted(
        (q.relative_to(tmp_path).as_posix(), q.read_bytes()) for q in tmp_path.rglob("*") if q.is_file()
    )
    assert after == before  # the refused re-entry wrote nothing and re-ran nothing


def test_child_refuses_a_module_with_an_unapproved_hash(tmp_path):
    """The scientific binding of imported modules to their approved hashes, exercised on a fixture arm."""
    over = {
        "expected_module_sha256": {
            n: "0" * 64
            for n in (
                "uid_common",
                "uformer_ae",
                "uformer_blocks",
                "uid_r1_control",
                "uid_r1_measure",
                "uid_r1_child",
            )
        }
    }
    p, s = run(tmp_path, [FIT], over=over)
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "EXECUTION_ERROR" and a["updates_done"] is None
    rec = json.loads((arm_dir(tmp_path, "FX-FIT") / "child_record.json").read_text())
    assert "without the approved SHA-256" in rec["error"]["message"]
    assert not (arm_dir(tmp_path, "FX-FIT") / "curve.jsonl").read_text()  # refused before any update


def test_aggregate_clock_starts_before_the_invocation_record(tmp_path):
    p, s = run(tmp_path, [CHK])
    assert p.returncode == 0, p.stderr
    inv = ledger(tmp_path)["invocation_001.json"]
    assert inv["monotonic_t0"] < inv["monotonic_at_record"]  # setup before the record is charged
    assert "before package verification" in inv["aggregate_clock_start"]


def test_registered_config_is_refused_in_engineering_mode(tmp_path):
    sys.path.insert(0, str(R1))
    import uid_r1_control as K

    reg = K.load_json(K.FROZEN_DIR / "config.json")
    try:
        K.fixture_plan(reg, tmp_path, [FIT])
        raise AssertionError("registered config accepted")
    except K.ControlError:
        pass
