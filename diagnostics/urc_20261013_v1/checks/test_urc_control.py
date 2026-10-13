"""ENGINEERING CHECK: URC-v1 resolution, the one-arm plan, package identity and state-file integrity.

No registered model is trained and no registered input is optimized: the state-file test uses a fixture model.
Ledgers live under pytest's tmp_path only.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

URC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URC))

import urc_control as U  # noqa: E402

K = U.K


def resolved():
    return U.resolve(U.load_urc_config())


def test_resolution_reads_the_frozen_uid_settings():
    r = resolved()
    uid = K.load_json(U.FROZEN_DIR / "config.json")
    assert r["optimizer"] == uid["u_fit"]["optimizer"] and r["model_init_seed"] == uid["model_init_seed"] == 1
    assert r["max_updates"] == uid["u_fit"]["max_updates"] == 3000
    assert r["max_seconds"] == uid["u_fit"]["max_seconds"] == 1800
    assert r["cpu"] == {k: uid["cpu_config"][k] for k in ("intra_op_threads", "inter_op_threads")}
    assert r["input"] == uid["input"] and r["criteria"] == uid["criteria"]
    assert r["limits"]["rss_cap_bytes"] == uid["global_limits"]["max_rss_bytes"] == 4 * 2**30
    assert r["limits"]["evidence_cap_bytes"] == 512 * 2**20 and r["limits"]["rss_available_fraction"] == 0.5
    assert r["limits"]["aggregate_seconds"] == uid["global_limits"]["max_scientific_seconds"] == 7200


def test_frozen_resolved_config_equals_a_fresh_resolution():
    assert K.load_json(URC / U.RESOLVED_NAME) == resolved() == U.load_resolved()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("shared_expected", "max_updates"), 2999),
        (("shared_expected", "optimizer", "lr"), 1e-3),
        (("shared_expected", "cpu_config", "intra_op_threads"), 2),
        (("model", "input_hw"), [16, 16]),
        (("model", "down_channels"), [2, 4, 8, 16, 32, 32]),
        (("reference", "frozen_config_raw_sha256"), "0" * 64),
        (("reference", "execution_sha256"), "0" * 64),
    ],
)
def test_any_shared_setting_difference_refuses(path, value):
    cfg = copy.deepcopy(U.load_urc_config())
    target = cfg
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    with pytest.raises(K.ControlError):
        U.resolve(cfg)


def test_scientific_plan_is_exactly_one_r_fit_start():
    plan = U.registered_plan(resolved())
    assert [a["arm"] for a in plan["arms"]] == ["R-FIT"] and plan["max_training_starts"] == 1
    assert plan["max_invocations"] == 1 and plan["prerequisite"] == {}
    assert plan["arms"][0] == {"arm": "R-FIT", "max_updates": 3000, "max_seconds": 1800.0}
    assert plan["run_root"] == "D:/cgfed-runs/urc_20261013_v1"
    assert plan["cpu"] == {"intra_op_threads": 24, "inter_op_threads": 24}


@pytest.mark.parametrize("arms", [["U-CHECK"], ["U-FIT"], ["R-FIT", "U-FIT"], ["R-FIT", "R-FIT"], []])
def test_uid_arms_and_other_plans_are_refused(arms):
    plan = U.registered_plan(resolved())
    bad = {**plan, "arms": [{"arm": a, "max_updates": 3000, "max_seconds": 1800.0} for a in arms]}
    with pytest.raises(K.ControlError):
        U.check_plan(bad)


def test_uid_model_arm_is_refused_even_in_engineering_mode():
    plan = {**U.registered_plan(resolved()), "mode": "engineering"}
    plan["arms"] = [{"arm": "FX", "model_arm": "U-FIT", "max_updates": 6, "max_seconds": 60.0}]
    with pytest.raises(K.ControlError, match="UID-20261012-v2 arm"):
        U.check_plan(plan)


def test_wrong_manifest_hash_refuses_before_anything():
    with pytest.raises(K.ControlError, match="package identity check failed"):
        U.verify_package("0" * 64)


@pytest.mark.parametrize(
    "argv",
    [
        ["--scientific", "--arm", "U-FIT"],
        ["--scientific", "--approved-r1-manifest-sha256", "9d3aba9a"],
        ["--status", "--scientific"],
        [],
    ],
)
def test_supervisor_cli_offers_no_arm_selection(argv):
    import subprocess

    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    p = subprocess.run(
        [sys.executable, "-B", str(URC / "urc_supervise.py"), *argv],
        cwd=URC,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and ("unrecognized arguments" in p.stderr or "error:" in p.stderr)


def test_scientific_without_approved_hash_is_refused():
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run(
        [sys.executable, "-B", str(URC / "urc_supervise.py"), "--scientific"],
        cwd=URC,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and "requires --approved-urc-manifest-sha256" in p.stderr


def test_one_start_accounting_refuses_a_second_start(tmp_path):
    plan = U.registered_plan(resolved())
    lg = K.Ledger(tmp_path / "run" / "ledger", tmp_path / "mirror")
    K.check_can_open(lg.read(), plan)
    K.check_can_start(lg.read(), plan, "R-FIT")
    lg.write("invocation_001.json", {"n": 1})
    lg.write("start_R-FIT.json", {"arm": "R-FIT"})
    with pytest.raises(K.ControlError, match="INCOMPLETE prior start"):
        K.check_can_open(lg.read(), plan)  # a start without an end record blocks every later invocation
    lg.write("end_R-FIT.json", {"arm": "R-FIT", "status": "COMPLETED_FIT_FAIL"})
    with pytest.raises(K.ControlError, match="already has a start record"):
        K.check_can_start(lg.read(), plan, "R-FIT")
    with pytest.raises(K.ControlError, match="1 invocation"):
        K.check_can_open(lg.read(), plan)


def test_state_files_save_and_load_bit_for_bit(tmp_path):
    import torch

    fixture = K.load_json(URC / "checks" / "fixture_config.json")
    m = U.build_model(fixture).train()
    with torch.no_grad():  # move the BN buffers away from their initial values (fixture input, no optimizer)
        m(torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(4245)))
    info = U.save_state(tmp_path / "s.safetensors", m, {"point": "fixture"})
    back, meta = U.load_state(tmp_path / "s.safetensors", fixture)
    sd_a, sd_b = m.state_dict(), back.state_dict()
    assert set(sd_a) == set(sd_b) and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    assert any(k.endswith("num_batches_tracked") and int(v) == 1 for k, v in sd_b.items())
    assert info["buffer_elements"] == sum(b.numel() for b in m.buffers()) and meta["point"] == "fixture"
    assert json.loads(meta["model_config"])["variant"] == "resnet3_stride1_diagnostic"
    with pytest.raises(FileExistsError):
        U.save_state(tmp_path / "s.safetensors", m, {})  # never overwritten
    raw = bytearray((tmp_path / "s.safetensors").read_bytes())
    raw[-1] ^= 0xFF  # corrupt one data byte
    (tmp_path / "t.safetensors").write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="hash differs"):
        U.load_state(tmp_path / "t.safetensors", fixture)


def test_pinned_files_are_checked():
    U.check_pinned()
    with pytest.raises(K.ControlError, match="pinned file"):
        U.check_pinned({"../uid_20261012_v2/uid_common.py": "0" * 64})
