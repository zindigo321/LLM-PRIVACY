"""ENGINEERING FIXTURES: real-subprocess lifecycle tests of the UCOV supervisor and child (fixture config only).

Each test runs ``checks/fixture_run.py`` as a real process in pytest's ``tmp_path``. The fixture has:
- its own seed (7), its four toy 16x16 tiles, a fixture scale and a small Uformer;
- the REAL pinned primary policy UPP-20261013-v1.

Optimizer updates happen only on these fixture models and toy tiles.

Covered:
- four-arm serial accounting;
- continuation after an ordinary metric failure and after an execution error;
- no duplicate start, and refusal of a second invocation;
- interruption (CTRL_BREAK) and a killed supervisor;
- the exhausted aggregate budget;
- a global guard stop with the remaining arms recorded NOT_RUN;
- policy identity through job, child record and status;
- states and replay.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN = UCOV / "checks" / "fixture_run.py"
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
sys.path.insert(0, str(UCOV))

import ucov_control as U  # noqa: E402  (no torch; puts the URP, UPP, frozen UID and r1 directories on sys.path)

ARMS = [{"arm": a, "max_updates": 6, "max_seconds": 120.0} for a in U.ARMS]
FIXTURE = json.loads((UCOV / "checks" / "fixture_config.json").read_text(encoding="utf-8"))
POLICY_SHA = FIXTURE["primary_policy"]["policy_sha256"]


def run(tmp, arms, over=None, hooks=None, timeout=600):
    args = [PY, "-B", str(RUN), str(tmp), json.dumps(arms), json.dumps(over or {})]
    if hooks is not None:
        args.append(json.dumps(hooks))
    p = subprocess.run(args, cwd=UCOV, env=ENV, capture_output=True, text=True, timeout=timeout)
    summary = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    return p, summary


def arm_dir(tmp, arm):
    return tmp / "run" / "arms" / arm


def record(tmp, arm):
    return json.loads((arm_dir(tmp, arm) / "child_record.json").read_text(encoding="utf-8"))


def ledger(tmp):
    root = tmp / "run" / "ledger"
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in root.iterdir()}


def files(tmp):
    return sorted((q.relative_to(tmp).as_posix(), q.read_bytes()) for q in tmp.rglob("*") if q.is_file())


def test_four_arms_run_serially_with_the_primary_policy(tmp_path):
    import uid_common as C

    C.configure_cpu(2, 1)  # the fixture plan's threads, for the bit-exact replay; before any tensor work here
    import torch

    p, s = run(tmp_path, ARMS)
    assert p.returncode == 0, p.stderr
    assert [a["arm"] for a in s["arms"]] == list(U.ARMS) and s["starts"] == 4 and s["global_stop"] is None
    lg = ledger(tmp_path)
    assert sorted(n for n in lg if n.startswith("start_")) == [f"start_{a}.json" for a in U.ARMS]
    starts = [lg[f"start_{a}.json"]["monotonic_s"] for a in U.ARMS]
    ends = [lg[f"end_{a}.json"]["monotonic_s"]["end"] for a in U.ARMS]
    assert all(
        e < s2 for e, s2 in zip(ends, starts[1:])
    )  # strictly serial: each arm ends before the next starts
    inits = set()
    for a in s["arms"]:
        rec = record(tmp_path, a["arm"])
        assert a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") and a["updates_done"] == 6
        v = rec["primary_verdict"]
        assert rec["primary_policy"] == {
            "policy_id": "UPP-20261013-v1",
            "policy_sha256": POLICY_SHA,
            "thresholds": {"max_rse": 0.5, "min_cosine": 0.65, "baseline_factor": 0.8},
        }
        assert (a["status"] == "COMPLETED_PASS") == (v["outcome"] == "PASS") and a["primary_outcome"] == v[
            "outcome"
        ]
        assert (
            json.loads((arm_dir(tmp_path, a["arm"]) / "job.json").read_text())["primary_policy_sha256"]
            == POLICY_SHA
        )
        assert rec["input"]["patch_file_sha256"] == FIXTURE["inputs"]["tiles"][a["arm"]]["patch_file_sha256"]
        inits.add(rec["model_state_sha256_init"])
        assert rec["modules"]["local"] == [
            "ucov_control",
            "uformer_ae",
            "uformer_blocks",
            "uid_common",
            "uid_r1_control",
            "uid_r1_measure",
            "upp_classify",
            "urp_control",
        ]
        # replay: the saved final state reproduces the recorded eval sums exactly (forward only, toy tile)
        model, meta = U.urp().load_state(arm_dir(tmp_path, a["arm"]) / "final_state.safetensors", FIXTURE)
        assert meta["state_sha256"] == rec["model_state_sha256_final"] == C.state_sha256(model)
        x, x_n, _ = U.load_tile(FIXTURE, a["arm"])
        model.eval()
        with torch.no_grad():
            y = U.urp().denormalize(
                C.reconstruct(model, x_n), FIXTURE["inputs"]["normalization"]["scale_global"]
            )
        rep, fe = C.metrics(y, x), rec["final"]["eval_mode"]
        assert all(rep[k] == fe[k] for k in ("sse", "target_energy", "pred_energy", "dot", "n"))
    assert len(inits) == 1  # identical initial state in every arm (fresh model, same seed)
    for n in lg:
        assert (tmp_path / "run/ledger" / n).read_bytes() == (tmp_path / "mirror" / n).read_bytes()


def test_later_arms_continue_after_an_execution_error_in_an_earlier_arm(tmp_path):
    # the fixture hook applies to every child; use a tiny per-arm budget hook instead: raise only at update 3, which
    # all arms reach, so every arm records EXECUTION_ERROR and every arm is still started (no global stop)
    p, s = run(tmp_path, ARMS, hooks={"raise_at_update": 3})
    assert p.returncode == 0, p.stderr
    assert [a["status"] for a in s["arms"]] == ["EXECUTION_ERROR"] * 4 and s["starts"] == 4
    assert s["global_stop"] is None
    assert "fixture error at update 3" in record(tmp_path, "U-TILE-4")["error"]["message"]


def test_later_arms_continue_after_an_ordinary_primary_fit_fail(tmp_path):
    # six updates on a toy tile from the fixture init cannot reach RSE <= 0.5 (step-0 RSE is about 1): a valid,
    # completed primary FIT_FAIL in every arm, and the next arms still run
    p, s = run(tmp_path, ARMS)
    assert p.returncode == 0, p.stderr
    assert s["starts"] == 4 and s["global_stop"] is None
    fails = [a for a in s["arms"] if a["status"] == "COMPLETED_FIT_FAIL"]
    assert fails, [a["status"] for a in s["arms"]]
    for a in fails:
        v = record(tmp_path, a["arm"])["primary_verdict"]
        assert v["outcome"] == "FIT_FAIL" and a["primary_outcome"] == "FIT_FAIL"
    first_fail = U.ARMS.index(fails[0]["arm"])
    assert all(a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") for a in s["arms"][first_fail:])


def test_second_invocation_is_refused_and_writes_nothing(tmp_path):
    p, _ = run(tmp_path, ARMS[:1])
    assert p.returncode == 0, p.stderr
    before = files(tmp_path)
    p2, _ = run(tmp_path, ARMS[:1])
    assert p2.returncode == 2 and "1 invocation(s) already recorded" in p2.stderr
    assert files(tmp_path) == before


def test_duplicate_or_closed_arm_plans_are_refused_before_any_record(tmp_path):
    for name, arms in (("dup", [ARMS[0], ARMS[0]]), ("closed", [{**ARMS[0], "arm": "U-PATCH"}])):
        p, _ = run(tmp_path / name, arms)
        assert p.returncode == 2 and ("repeats an arm" in p.stderr or "closed protocol" in p.stderr)
        assert not (tmp_path / name / "run").exists()


def test_rss_guard_is_a_global_stop_and_later_arms_are_not_run(tmp_path):
    p, s = run(
        tmp_path, ARMS[:2], over={"rss_cap_bytes": 700 * 2**20}, hooks={"alloc_mib_at_update": [2, 1024]}
    )
    assert p.returncode == 0, p.stderr
    first, second = s["arms"]
    assert first["status"] == "RESOURCE_STOP" and first["guard"]["action"] == "killed: RSS limit"
    assert second["status"] == "NOT_RUN" and "RESOURCE_STOP" in second["reason"]
    assert s["starts"] == 1 and "start_U-TILE-2.json" not in ledger(tmp_path)


def test_exhausted_aggregate_budget_stops_safely_and_refuses_the_remaining_arms(tmp_path):
    # 6 updates x 2 s per arm cannot fit four times into 20 s: the shared budget ends the invocation. Either an arm
    # is soft-stopped by the aggregate deadline (RESOURCE_STOP, a global stop) or the next start is refused as
    # exhausted; in both cases every later arm is NOT_RUN and there is never a start beyond the budget
    over = {"aggregate_seconds": 20.0, "end_margin_s": 1.0, "eval_reserve_s": 2.0}
    p, s = run(
        tmp_path, [{**a, "max_seconds": 60.0} for a in ARMS], over=over, hooks={"sleep_each_update_s": 2.0}
    )
    assert p.returncode == 0, p.stderr
    statuses = [a["status"] for a in s["arms"]]
    ran = [st for st in statuses if st != "NOT_RUN"]
    assert ran and statuses[len(ran) :] == ["NOT_RUN"] * (4 - len(ran)) and len(ran) < 4
    assert s["starts"] == len(ran) and s["global_stop"] is not None
    assert s["aggregate_wall_s"] <= 20.0 + 1.0
    assert all(
        "aggregate budget exhausted" in a["reason"] or "RESOURCE_STOP" in a["reason"]
        for a in s["arms"]
        if a["status"] == "NOT_RUN"
    )


def test_supervisor_interrupt_records_incomplete_and_not_run(tmp_path):
    args = [
        PY,
        "-B",
        str(RUN),
        str(tmp_path),
        json.dumps(ARMS),
        "{}",
        json.dumps({"sleep_each_update_s": 1.5}),
    ]
    proc = subprocess.Popen(
        args,
        cwd=UCOV,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    curve = arm_dir(tmp_path, "U-TILE-1") / "curve.jsonl"
    deadline = time.monotonic() + 180
    while not curve.exists() or not curve.read_text():
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.2)
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    out, err = proc.communicate(timeout=180)
    lg = ledger(tmp_path)
    assert lg["end_U-TILE-1.json"]["status"] == "INCOMPLETE", (out, err)
    end = lg["invocation_001_end.json"]
    assert end["interrupted"] is True and end["starts"] == 1
    assert [a["status"] for a in end["arms"][1:]] == ["NOT_RUN"] * 3
    assert all("interrupted" in a["reason"] for a in end["arms"][1:])


def test_killed_supervisor_leaves_an_unfinished_start_that_blocks_reentry(tmp_path):
    import psutil

    args = [
        PY,
        "-B",
        str(RUN),
        str(tmp_path),
        json.dumps(ARMS),
        "{}",
        json.dumps({"sleep_each_update_s": 3.0}),
    ]
    proc = subprocess.Popen(
        args, cwd=UCOV, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    curve = arm_dir(tmp_path, "U-TILE-1") / "curve.jsonl"
    deadline = time.monotonic() + 180
    while not curve.exists() or not curve.read_text():
        assert time.monotonic() < deadline and proc.poll() is None
        time.sleep(0.2)
    proc.kill()  # the supervisor dies abruptly and writes nothing more
    proc.communicate(timeout=60)
    job = str(arm_dir(tmp_path, "U-TILE-1") / "job.json")

    def orphans():
        found = []
        for q in psutil.process_iter(["cmdline"]):
            try:
                if job in " ".join(q.info["cmdline"] or []):
                    found.append(q.pid)
            except psutil.Error:
                pass
        return found

    t = time.monotonic()
    while orphans() and time.monotonic() - t < 30:  # the child's watchdog notices the dead supervisor
        time.sleep(0.2)
    assert orphans() == []
    lg = ledger(tmp_path)
    assert "start_U-TILE-1.json" in lg and "end_U-TILE-1.json" not in lg and "start_U-TILE-2.json" not in lg
    before = files(tmp_path)
    p2, _ = run(tmp_path, ARMS)
    assert p2.returncode == 2 and "INCOMPLETE prior start" in p2.stderr
    assert files(tmp_path) == before  # the refused re-entry wrote nothing and re-ran nothing


def test_child_refuses_a_module_with_an_unapproved_hash(tmp_path):
    names = [
        "uid_common",
        "uformer_ae",
        "uformer_blocks",
        "uid_r1_control",
        "uid_r1_measure",
        "urp_control",
        "upp_classify",
        "ucov_control",
        "ucov_child",
    ]
    p, s = run(tmp_path, ARMS[:1], over={"expected_module_sha256": {n: "0" * 64 for n in names}})
    assert p.returncode == 0, p.stderr
    assert s["arms"][0]["status"] == "EXECUTION_ERROR"
    assert "without the approved SHA-256" in record(tmp_path, "U-TILE-1")["error"]["message"]
    assert not (arm_dir(tmp_path, "U-TILE-1") / "init_state.safetensors").exists()


def test_a_degenerate_registered_arm_gets_no_start(tmp_path):
    arms = [ARMS[0], {**ARMS[1], "usable": False}, ARMS[2]]
    p, s = run(tmp_path, arms)
    assert p.returncode == 0, p.stderr
    assert s["arms"][1]["status"] == "NOT_RUN" and "degenerate" in s["arms"][1]["reason"]
    assert s["starts"] == 2 and "start_U-TILE-2.json" not in ledger(tmp_path)
    assert s["arms"][2]["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL")
