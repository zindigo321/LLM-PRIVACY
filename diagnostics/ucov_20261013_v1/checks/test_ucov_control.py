"""ENGINEERING CHECK: UCOV-v1 resolution, the primary policy binding, the registered tiles, the plan, normalization.

No registered model is trained and no optimizer touches a registered tile. The registered tiles are only read and
hashed. Ledgers live under pytest's tmp_path only.
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

UCOV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UCOV))

import ucov_control as U  # noqa: E402

K = U.K
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def resolved():
    return U.resolve(U.load_ucov_config())


def test_resolution_reads_the_frozen_uid_settings_and_the_new_primary_policy():
    r = resolved()
    uid = K.load_json(U.FROZEN_DIR / "config.json")
    assert r["optimizer"] == uid["u_fit"]["optimizer"] and r["model_init_seed"] == uid["model_init_seed"] == 1
    assert (
        r["u_fit"]["model"] == uid["u_fit"]["model"] and r["max_updates"] == 3000 and r["max_seconds"] == 1800
    )
    assert r["cpu"] == {"intra_op_threads": 24, "inter_op_threads": 24}
    pol = r["primary_policy"]
    assert pol["policy_id"] == "UPP-20261013-v1"
    assert pol["thresholds"] == {"max_rse": 0.5, "min_cosine": 0.65, "baseline_factor": 0.8}
    assert pol["loaded_policy_sha256"] == pol["policy_sha256"] == K.sha256_file(UCOV / pol["policy_file"])
    # the historical criteria are carried only as a secondary reference, never as the primary decision
    assert "criteria" not in r and r["secondary_references"]["historical_strict_criteria"] == uid["criteria"]
    lim = r["limits"]
    assert (lim["max_training_starts"], lim["max_invocations"], lim["aggregate_seconds"]) == (4, 1, 9000)
    assert lim["rss_cap_bytes"] == 4 * 2**30 and lim["rss_available_fraction"] == 0.5
    assert lim["evidence_cap_bytes"] == 512 * 2**20 and lim["curve_every"] == 100


def test_frozen_resolved_config_equals_a_fresh_resolution():
    assert K.load_json(UCOV / U.RESOLVED_NAME) == resolved() == U.load_resolved()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("shared_expected", "max_updates"), 2999),
        (("shared_expected", "optimizer", "lr"), 1e-3),
        (("shared_expected", "cpu_config", "intra_op_threads"), 2),
        (("reference", "frozen_config_raw_sha256"), "0" * 64),
        (("reference", "urp_manifest_sha256"), "0" * 64),
        (("reference", "model_state_sha256_init"), "0" * 64),
        (("inputs", "inputs_record_sha256"), "0" * 64),
        (("inputs", "selection_rule_sha256"), "0" * 64),
        (("inputs", "normalization", "scale_global"), 0.0157),
        (("inputs", "phi_sha256"), "0" * 64),
        (("inputs", "tiles", "U-TILE-3", "patch_file_sha256"), "0" * 64),
        (("inputs", "tiles", "U-TILE-2", "slice"), "Phi[0, 1024:1056, 768:800]"),
        (("primary_policy", "policy_sha256"), "0" * 64),
        (("primary_policy", "classifier_sha256"), "0" * 64),
        (("primary_policy", "thresholds_expected", "min_cosine"), 0.8),
    ],
)
def test_any_setting_input_or_policy_difference_refuses(path, value):
    cfg = copy.deepcopy(U.load_ucov_config())
    target = cfg
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = value
    with pytest.raises(K.ControlError):
        U.resolve(cfg)


def test_scientific_plan_is_the_four_arms_in_registered_order():
    plan = U.registered_plan(resolved())
    assert [a["arm"] for a in plan["arms"]] == list(U.ARMS)
    assert all(a["max_updates"] == 3000 and a["max_seconds"] == 1800.0 and a["usable"] for a in plan["arms"])
    assert plan["max_training_starts"] == 4 and plan["max_invocations"] == 1 and plan["prerequisite"] == {}
    assert plan["run_root"] == "D:/cgfed-runs/ucov_20261013_v1" and plan["aggregate_seconds"] == 9000.0
    assert plan["primary_policy_id"] == "UPP-20261013-v1"


@pytest.mark.parametrize(
    "arms",
    [
        ["U-PATCH"],
        ["U-FIT"],
        ["U-TILE-1", "U-TILE-1"],
        ["U-TILE-2", "U-TILE-1", "U-TILE-3", "U-TILE-4"],
        ["U-TILE-1", "U-TILE-2", "U-TILE-3"],
        [],
    ],
)
def test_closed_reordered_duplicate_or_partial_plans_are_refused(arms):
    plan = U.registered_plan(resolved())
    bad = {**plan, "arms": [{"arm": a, "max_updates": 3000, "max_seconds": 1800.0} for a in arms]}
    with pytest.raises(K.ControlError):
        U.check_plan(bad)


def test_wrong_manifest_hash_refuses_before_anything():
    with pytest.raises(K.ControlError, match="package identity check failed"):
        U.verify_package("0" * 64)


@pytest.mark.parametrize("argv", [["--scientific", "--arm", "U-TILE-1"], ["--scientific", "--skip", "2"], []])
def test_supervisor_cli_offers_no_arm_selection(argv):
    p = subprocess.run(
        [sys.executable, "-B", str(UCOV / "ucov_supervise.py"), *argv],
        cwd=UCOV,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and ("unrecognized arguments" in p.stderr or "error:" in p.stderr)


def test_scientific_without_the_frozen_hash_is_refused():
    p = subprocess.run(
        [sys.executable, "-B", str(UCOV / "ucov_supervise.py"), "--scientific"],
        cwd=UCOV,
        env=ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and "requires --frozen-ucov-manifest-sha256" in p.stderr


def test_scientific_with_pythonpath_set_is_refused():
    env = {**ENV, "PYTHONPATH": str(UCOV.parents[1] / "src")}
    p = subprocess.run(
        [
            sys.executable,
            "-B",
            str(UCOV / "ucov_supervise.py"),
            "--scientific",
            "--frozen-ucov-manifest-sha256",
            "0" * 64,
        ],
        cwd=UCOV,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 2 and "PYTHONPATH is set" in p.stderr


@pytest.mark.parametrize("arm", list(U.ARMS))
def test_registered_tiles_load_with_their_recorded_values(arm):
    import torch
    import uid_common as C

    r = resolved()
    x, x_n, digest = U.load_tile(r, arm)
    exp = r["inputs_expected"][arm]
    assert digest == r["inputs"]["tiles"][arm]["patch_file_sha256"] and x.dtype == torch.float32
    assert list(x.shape) == [1, 1, 32, 32]
    rec = {
        "input": {"original": C.describe_input(x), "normalized": C.describe_input(x_n)},
        "baselines": {"original": C.baselines(x), "normalized": C.baselines(x_n)},
    }
    checks = U.input_checks(rec, exp, float(x_n.abs().max()))
    assert checks["all_equal"] is True and checks["normalized_max_abs_below_1"] is True
    rec["baselines"]["original"]["shared_scalar_mean"]["mse"] *= 1.0000001
    assert U.input_checks(rec, exp, float(x_n.abs().max()))["all_equal"] is False


def test_tiles_are_the_registered_slices_and_differ_from_each_other_and_from_urp():
    r = resolved()
    shas = {r["inputs_expected"][a]["original_sha256"] for a in U.ARMS}
    urp = K.load_json(U.URP_DIR / "input" / "input_record.json")["patch"]["original"]["sha256"]
    assert len(shas) == 4 and urp not in shas
    rec = K.load_json(UCOV / "input" / "inputs_record.json")
    assert [t["coordinates"]["slice"] for t in rec["tiles"]] == [
        "Phi[0, 512:544, 384:416]",
        "Phi[0, 512:544, 1152:1184]",
        "Phi[0, 1536:1568, 384:416]",
        "Phi[0, 1536:1568, 1152:1184]",
    ]
    assert (
        rec["source"]["phi_sha256_observed"]
        == "485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546"
    )


def test_normalization_equals_the_production_normalizer_bit_for_bit():
    import torch

    src = UCOV.parents[1] / "src"
    sys.path.insert(0, str(src))
    import cg_fedllm.compression.normalization as N  # this repository's src, never the stale editable install

    assert Path(N.__file__).resolve() == (src / "cg_fedllm" / "compression" / "normalization.py").resolve()
    r = resolved()
    s = r["inputs"]["normalization"]["scale_global"]
    prod = N.Normalizer("global_exact_maxabs_train", scale_global=s)
    for arm in U.ARMS:
        x, x_n, _ = U.load_tile(r, arm)
        assert torch.equal(x_n, prod.normalize(x))
    y = torch.tanh(torch.randn(1, 1, 32, 32, generator=torch.Generator().manual_seed(11)))
    assert torch.equal(U.urp().denormalize(y, s), prod.denormalize(y))


def test_pinned_files_are_checked():
    U.check_pinned()
    with pytest.raises(K.ControlError, match="pinned file"):
        U.check_pinned({"../upp_20261013_v1/upp_policy.json": "0" * 64})


def test_four_start_accounting_refuses_a_duplicate_and_a_fifth_start(tmp_path):
    plan = U.registered_plan(resolved())
    lg = K.Ledger(tmp_path / "run" / "ledger", tmp_path / "mirror")
    K.check_can_open(lg.read(), plan)
    lg.write("invocation_001.json", {"n": 1})
    for i, arm in enumerate(U.ARMS):
        K.check_can_start(lg.read(), plan, arm)
        lg.write(f"start_{arm}.json", {"arm": arm})
        if i < 3:  # while an arm is unfinished, no other arm may start
            with pytest.raises(K.ControlError, match="unfinished start"):
                K.check_can_start(lg.read(), plan, U.ARMS[i + 1])
        lg.write(f"end_{arm}.json", {"arm": arm, "status": "COMPLETED_FIT_FAIL"})
        with pytest.raises(K.ControlError, match="already has a start record"):
            K.check_can_start(lg.read(), plan, arm)
    with pytest.raises(K.ControlError, match="4 start"):
        K.check_can_start(lg.read(), plan, "U-TILE-5")  # a fifth start is refused by the start count
    with pytest.raises(K.ControlError, match="1 invocation"):
        K.check_can_open(lg.read(), plan)


def test_config_and_resolved_config_are_lf_json():
    for name in (U.CONFIG_NAME, U.RESOLVED_NAME):
        assert b"\r\n" not in (UCOV / name).read_bytes()
    assert json.loads((UCOV / U.RESOLVED_NAME).read_text(encoding="utf-8"))["protocol_id"] == U.PROTOCOL_ID
