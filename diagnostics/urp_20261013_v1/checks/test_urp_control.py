"""ENGINEERING CHECK: URP-v1 resolution, the registered input, the plan, normalization and state-file integrity.

No registered model is trained and no optimizer touches the registered input: the state-file test uses the fixture
model. The registered patch is only read and hashed. Ledgers live under pytest's tmp_path only.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP))

import urp_control as U  # noqa: E402

K = U.K
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def resolved():
    return U.resolve(U.load_urp_config())


def test_resolution_reads_the_frozen_uid_settings():
    r = resolved()
    uid = K.load_json(U.FROZEN_DIR / "config.json")
    assert r["optimizer"] == uid["u_fit"]["optimizer"] and r["model_init_seed"] == uid["model_init_seed"] == 1
    assert r["optimizer"] == {
        "name": "Adam",
        "lr": 0.0002,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "weight_decay": 0,
    }
    assert r["u_fit"]["model"] == uid["u_fit"]["model"]  # the frozen U-FIT architecture, unchanged
    assert r["max_updates"] == 3000 and r["max_seconds"] == 1800 and r["criteria"] == uid["criteria"]
    assert r["cpu"] == {"intra_op_threads": 24, "inter_op_threads": 24}
    lim = r["limits"]
    assert (lim["max_training_starts"], lim["max_invocations"], lim["aggregate_seconds"]) == (1, 1, 7200)
    assert lim["rss_cap_bytes"] == 4 * 2**30 and lim["rss_available_fraction"] == 0.5
    assert lim["evidence_cap_bytes"] == 512 * 2**20 and lim["curve_every"] == 100


def test_frozen_resolved_config_equals_a_fresh_resolution():
    assert K.load_json(URP / U.RESOLVED_NAME) == resolved() == U.load_resolved()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("shared_expected", "max_updates"), 2999),
        (("shared_expected", "optimizer", "lr"), 1e-3),
        (("shared_expected", "cpu_config", "intra_op_threads"), 2),
        (("reference", "frozen_config_raw_sha256"), "0" * 64),
        (("reference", "child_record_sha256"), "0" * 64),
        (("reference", "model_state_sha256_init"), "0" * 64),
        (("input", "patch_file_sha256"), "0" * 64),
        (("input", "selection_rule_sha256"), "0" * 64),
        (("input", "normalization", "scale_global"), 0.0157),
        (("input_expected", "original_sha256"), "0" * 64),
        (("input_expected", "normalized_max_abs"), 0.85),
        (("input_expected", "shape"), [1, 1, 16, 16]),
    ],
)
def test_any_setting_or_input_difference_refuses(path, value):
    cfg = copy.deepcopy(U.load_urp_config())
    target = cfg
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    with pytest.raises(K.ControlError):
        U.resolve(cfg)


def test_scientific_plan_is_exactly_one_u_patch_start():
    plan = U.registered_plan(resolved())
    assert plan["arms"] == [{"arm": "U-PATCH", "max_updates": 3000, "max_seconds": 1800.0}]
    assert plan["max_training_starts"] == 1 and plan["max_invocations"] == 1 and plan["prerequisite"] == {}
    assert plan["run_root"] == "D:/cgfed-runs/urp_20261013_v1" and plan["aggregate_seconds"] == 7200.0


@pytest.mark.parametrize("arms", [["U-FIT"], ["U-CHECK"], ["R-FIT"], ["U-PATCH", "U-PATCH"], []])
def test_closed_arms_and_other_plans_are_refused(arms):
    plan = U.registered_plan(resolved())
    bad = {**plan, "arms": [{"arm": a, "max_updates": 3000, "max_seconds": 1800.0} for a in arms]}
    with pytest.raises(K.ControlError):
        U.check_plan(bad)


def test_wrong_manifest_hash_refuses_before_anything():
    with pytest.raises(K.ControlError, match="package identity check failed"):
        U.verify_package("0" * 64)


@pytest.mark.parametrize(
    "argv",
    [["--scientific", "--arm", "U-FIT"], ["--scientific", "--approved-urc-manifest-sha256", "b058"], []],
)
def test_supervisor_cli_offers_no_arm_selection(argv):
    p = subprocess.run(
        [sys.executable, "-B", str(URP / "urp_supervise.py"), *argv],
        cwd=URP,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and ("unrecognized arguments" in p.stderr or "error:" in p.stderr)


def test_scientific_without_approved_hash_is_refused():
    p = subprocess.run(
        [sys.executable, "-B", str(URP / "urp_supervise.py"), "--scientific"],
        cwd=URP,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and "requires --approved-urp-manifest-sha256" in p.stderr


def test_registered_input_loads_with_its_recorded_values():
    import torch
    import uid_common as C

    r = resolved()
    x, x_n, digest = U.load_input(r)
    exp = r["input_expected"]
    assert (
        digest == r["input"]["patch_file_sha256"]
        and x.dtype == torch.float32
        and list(x.shape) == [1, 1, 32, 32]
    )
    assert C.describe_input(x)["sha256"] == exp["original_sha256"]
    assert C.describe_input(x_n)["sha256"] == exp["normalized_sha256"]
    rec = {
        "input": {"original": C.describe_input(x), "normalized": C.describe_input(x_n)},
        "baselines": {"original": C.baselines(x), "normalized": C.baselines(x_n)},
    }
    checks = U.input_checks(rec, r, float(x_n.abs().max()))
    assert checks["all_equal"] is True and checks["normalized_max_abs_below_1"] is True
    rec["baselines"]["original"]["shared_scalar_mean"]["mse"] *= 1.0000001
    assert U.input_checks(rec, r, float(x_n.abs().max()))["all_equal"] is False


def test_normalization_equals_the_production_normalizer_bit_for_bit():
    import torch

    src = URP.parents[1] / "src"
    sys.path.insert(0, str(src))
    import cg_fedllm.compression.normalization as N  # this repository's src, never the stale editable install

    assert Path(N.__file__).resolve() == (src / "cg_fedllm" / "compression" / "normalization.py").resolve()
    r = resolved()
    s = r["input"]["normalization"]["scale_global"]
    prod = N.Normalizer("global_exact_maxabs_train", scale_global=s)
    x, x_n, _ = U.load_input(r)
    y = torch.tanh(torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(11)))
    assert torch.equal(x_n, prod.normalize(x)) and torch.equal(U.denormalize(y, s), prod.denormalize(y))
    assert x_n.dtype == torch.float32 and U.denormalize(y, s).dtype == torch.float32


def test_patch_file_is_the_rule_tile_of_the_record():
    rec = K.load_json(URP / "input" / "input_record.json")
    assert rec["coordinates"]["rows"] == [1024, 1056] and rec["coordinates"]["cols"] == [768, 800]
    assert rec["source"]["phi_sha256_observed"] == rec["source"]["selected"]["phi_sha256"]
    assert rec["selection_rule"]["sha256"] == U.load_urp_config()["input"]["selection_rule_sha256"]


def test_state_files_save_and_load_bit_for_bit(tmp_path):
    import torch

    fixture = K.load_json(URP / "checks" / "fixture_config.json")
    m = U.build_model(fixture).train()
    with torch.no_grad():  # move the BatchNorm buffers away from their initial values (no optimizer)
        m.decode(m.encode(torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(4245))))
    info = U.save_state(tmp_path / "s.safetensors", m, {"point": "fixture"})
    back, meta = U.load_state(tmp_path / "s.safetensors", fixture)
    sd_a, sd_b = m.state_dict(), back.state_dict()
    assert set(sd_a) == set(sd_b) and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    assert any(k.endswith("num_batches_tracked") and int(v) == 1 for k, v in sd_b.items())
    assert any(k.endswith("relative_position_index") for k in sd_b)  # non-trainable buffers are saved too
    assert info["buffer_elements"] == sum(b.numel() for b in m.buffers()) and meta["point"] == "fixture"
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
        U.check_pinned({"../uid_20261012_v2/uformer_ae.py": "0" * 64})


def test_one_start_accounting_refuses_a_second_start(tmp_path):
    plan = U.registered_plan(resolved())
    lg = K.Ledger(tmp_path / "run" / "ledger", tmp_path / "mirror")
    K.check_can_open(lg.read(), plan)
    lg.write("invocation_001.json", {"n": 1})
    lg.write("start_U-PATCH.json", {"arm": "U-PATCH"})
    with pytest.raises(K.ControlError, match="INCOMPLETE prior start"):
        K.check_can_open(lg.read(), plan)
    lg.write("end_U-PATCH.json", {"arm": "U-PATCH", "status": "COMPLETED_FIT_FAIL"})
    with pytest.raises(K.ControlError, match="already has a start record"):
        K.check_can_start(lg.read(), plan, "U-PATCH")
    with pytest.raises(K.ControlError, match="1 invocation"):
        K.check_can_open(lg.read(), plan)


def test_resolved_config_is_not_json_text_with_crlf():
    assert b"\r\n" not in (URP / U.RESOLVED_NAME).read_bytes()
    assert json.loads((URP / U.RESOLVED_NAME).read_text(encoding="utf-8"))["protocol_id"] == U.PROTOCOL_ID
