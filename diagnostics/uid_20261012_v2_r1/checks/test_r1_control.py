"""Engineering tests of uid_r1_control (ledger, start accounting, deadlines, registered plan). No torch, no
registered model or input; every ledger lives under pytest's tmp_path."""

import json
import sys
from pathlib import Path

import pytest

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))

import uid_r1_control as K  # noqa: E402

PLAN = {
    "arms": [{"arm": "A"}, {"arm": "B"}],
    "prerequisite": {"B": "A"},
    "max_training_starts": 2,
    "max_invocations": 1,
}


def ledger(tmp_path: Path) -> K.Ledger:
    return K.Ledger(tmp_path / "primary" / "ledger", tmp_path / "mirror")


def test_records_are_write_once_and_mirrored(tmp_path):
    lg = ledger(tmp_path)
    lg.write("start_A.json", {"arm": "A"})
    assert (tmp_path / "primary/ledger/start_A.json").read_bytes() == (
        tmp_path / "mirror/start_A.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        lg.write("start_A.json", {"arm": "A", "again": True})
    assert json.loads((tmp_path / "mirror/start_A.json").read_text())["arm"] == "A"


def test_duplicate_and_excess_starts_are_refused(tmp_path):
    lg = ledger(tmp_path)
    K.check_can_start(lg.read(), PLAN, "A")
    lg.write("start_A.json", {"arm": "A"})
    lg.write("end_A.json", {"arm": "A", "status": "COMPLETED_PASS"})
    with pytest.raises(K.ControlError, match="already has a start record"):
        K.check_can_start(lg.read(), PLAN, "A")
    K.check_can_start(lg.read(), PLAN, "B")
    lg.write("start_B.json", {"arm": "B"})
    lg.write("end_B.json", {"arm": "B", "status": "COMPLETED_FIT_FAIL"})
    plan3 = {**PLAN, "arms": PLAN["arms"] + [{"arm": "C"}], "prerequisite": {}}
    with pytest.raises(K.ControlError, match="2 start\\(s\\) recorded; max 2"):
        K.check_can_start(lg.read(), plan3, "C")


def test_prerequisite_must_pass(tmp_path):
    lg = ledger(tmp_path)
    with pytest.raises(K.ControlError, match="requires A COMPLETED_PASS, found NOT_STARTED"):
        K.check_can_start(lg.read(), PLAN, "B")
    lg.write("start_A.json", {"arm": "A"})
    lg.write("end_A.json", {"arm": "A", "status": "COMPLETED_FIT_FAIL"})
    with pytest.raises(K.ControlError, match="requires A COMPLETED_PASS, found COMPLETED_FIT_FAIL"):
        K.check_can_start(lg.read(), PLAN, "B")


def test_unfinished_start_blocks_reentry_and_new_invocations(tmp_path):
    lg = ledger(tmp_path)
    lg.write("start_A.json", {"arm": "A"})  # the process died before an end record
    records = lg.read()
    assert K.summarize(records, ["A", "B"])["arms"]["A"] == "STARTED_UNFINISHED"
    with pytest.raises(K.ControlError, match="INCOMPLETE prior start"):
        K.check_can_open(records, PLAN)
    with pytest.raises(K.ControlError):
        K.check_can_start(records, PLAN, "A")


def test_one_invocation_only(tmp_path):
    lg = ledger(tmp_path)
    K.check_can_open(lg.read(), PLAN)
    lg.write("invocation_001.json", {"n": 1})
    with pytest.raises(K.ControlError, match="1 invocation\\(s\\) already recorded; max 1"):
        K.check_can_open(lg.read(), PLAN)


def test_removing_the_run_root_does_not_reset_the_count(tmp_path):
    lg = ledger(tmp_path)
    lg.write("invocation_001.json", {"n": 1})
    lg.write("start_A.json", {"arm": "A"})
    for p in (tmp_path / "primary/ledger").iterdir():
        p.unlink()
    with pytest.raises(K.ControlError, match="ledger roots disagree"):
        lg.read()


def test_partial_or_foreign_records_block(tmp_path):
    lg = ledger(tmp_path)
    (tmp_path / "primary/ledger").mkdir(parents=True)
    (tmp_path / "mirror").mkdir()
    for root in (tmp_path / "primary/ledger", tmp_path / "mirror"):
        (root / "start_A.json").write_bytes(b'{"arm": "A"')  # torn write
    with pytest.raises(K.ControlError, match="unreadable ledger record"):
        lg.read()
    for root in (tmp_path / "primary/ledger", tmp_path / "mirror"):
        (root / "start_A.json").unlink()
        (root / "notes.txt").write_text("x")
    with pytest.raises(K.ControlError, match="unexpected entry"):
        lg.read()


def test_mirror_must_differ_from_primary(tmp_path):
    with pytest.raises(K.ControlError, match="must be different"):
        K.Ledger(tmp_path / "x", tmp_path / "x")


def test_deadlines_share_one_aggregate_clock():
    # first arm: its own limit binds
    assert K.arm_deadlines(0.0, 1.0, 300.0, 7200.0, 20.0, 60.0) == (221.0, 281.0)
    # second arm starting late: the remaining aggregate budget binds, not a fresh budget
    soft, kill = K.arm_deadlines(0.0, 6000.0, 1800.0, 7200.0, 20.0, 60.0)
    assert (soft, kill) == (7120.0, 7180.0)


def test_ledger_status_is_read_only_and_reports_starts(tmp_path):
    p, m = tmp_path / "run" / "ledger", tmp_path / "mirror"
    s = K.ledger_status(p, m, ["A", "B"])
    assert s["starts"] == 0 and s["records"] == [] and not p.exists() and not m.exists()
    K.Ledger(p, m).write("start_A.json", {"arm": "A"})
    s = K.ledger_status(p, m, ["A", "B"])
    assert s["starts"] == 1 and s["arms"] == {"A": "STARTED_UNFINISHED", "B": "NOT_STARTED"}


def test_plan_timing_invariants():
    cfg = json.loads((K.FROZEN_DIR / "config.json").read_text(encoding="utf-8"))
    execution = json.loads((R1 / "execution_r1.json").read_text(encoding="utf-8"))
    plan = K.registered_plan(cfg, execution)
    K.validate_plan(plan)  # the registered plan satisfies them
    with pytest.raises(K.ControlError, match="end_margin_s must exceed kill_wait_s"):
        K.validate_plan({**plan, "kill_wait_s": plan["end_margin_s"]})
    short = {**plan, "arms": [{"arm": "X", "max_seconds": plan["end_margin_s"] + plan["eval_reserve_s"]}]}
    with pytest.raises(K.ControlError, match="leaves no time to train"):
        K.validate_plan(short)


def test_module_hashes_must_all_be_listed():
    manifest = {"files": [{"path": p, "sha256": "0" * 64} for p in K.MODULE_PATHS.values()]}
    assert set(K.expected_module_hashes(manifest)) == set(K.MODULE_PATHS)
    manifest["files"] = manifest["files"][1:]
    with pytest.raises(K.ControlError, match="does not list runtime module"):
        K.expected_module_hashes(manifest)


def test_registered_plan_reads_the_frozen_limits():
    cfg = json.loads((K.FROZEN_DIR / "config.json").read_text(encoding="utf-8"))
    execution = json.loads((R1 / "execution_r1.json").read_text(encoding="utf-8"))
    plan = K.registered_plan(cfg, execution)
    assert [(a["arm"], a["max_updates"], a["max_seconds"]) for a in plan["arms"]] == [
        ("U-CHECK", 500, 300.0),
        ("U-FIT", 3000, 1800.0),
    ]
    assert plan["aggregate_seconds"] == 7200.0 and plan["max_training_starts"] == 2
    assert plan["rss_cap_bytes"] == 4 * 2**30 and plan["evidence_cap_bytes"] == 512 * 2**20
    assert plan["prerequisite"] == {"U-FIT": "U-CHECK"} and plan["curve_every"] == 100
    assert plan["cpu"] == {"intra_op_threads": 24, "inter_op_threads": 24}
