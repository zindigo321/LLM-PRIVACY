"""ENGINEERING FIXTURES: real-subprocess lifecycle tests of the URP supervisor and child (fixture config only).

Each test runs ``checks/fixture_run.py`` as a real process with the fixture's own seed, toy 16x16 patch, fixture scale
and small Uformer, in pytest's ``tmp_path``. Optimizer updates happen only on these fixture models and the toy patch.
The r1 lifecycle suite (``uid_20261012_v2_r1/evidence/final4_20261012``) and the URC suite
(``urc_20261013_v1/evidence/final_20261013``) cover the supervision code that is copied unchanged (guards,
deadlines, interrupts, aggregate budget); these tests cover the changed data path, model binding, measurements and
state persistence.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

URP = Path(__file__).resolve().parents[1]
PY = sys.executable
RUN = URP / "checks" / "fixture_run.py"
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | {
    "CUDA_VISIBLE_DEVICES": "-1",
    "HF_HUB_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONIOENCODING": "utf-8",
}
sys.path.insert(0, str(URP))

import urp_control as U  # noqa: E402  (no torch; puts the frozen UID and r1 directories on sys.path)

FIT = {"arm": "FX-UPATCH", "max_updates": 6, "max_seconds": 120.0}
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")
SUMS = {"sse", "target_energy", "pred_energy", "dot", "n"}


def run(tmp, arms, over=None, hooks=None, timeout=240):
    args = [PY, "-B", str(RUN), str(tmp), json.dumps(arms), json.dumps(over or {})]
    if hooks is not None:
        args.append(json.dumps(hooks))
    p = subprocess.run(args, cwd=URP, env=ENV, capture_output=True, text=True, timeout=timeout)
    summary = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    return p, summary


def arm_dir(tmp, arm="FX-UPATCH"):
    return tmp / "run" / "arms" / arm


def child_record(tmp):
    return json.loads((arm_dir(tmp) / "child_record.json").read_text(encoding="utf-8"))


def test_completed_arm_records_both_spaces_states_and_measurements(tmp_path):
    import uid_common as C  # frozen; importable through urp_control's sys.path entries

    C.configure_cpu(2, 1)  # the fixture plan's threads, for the bit-exact replay; before any tensor work here
    import torch

    p, s = run(tmp_path, [FIT])
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL") and a["updates_done"] == 6
    rec = child_record(tmp_path)
    assert rec["cpu"]["applied_as_requested"] is True and rec["step0"]["isolation_verified"] is True
    assert rec["input_checks"]["all_equal"] is True and rec["verdict_basis"] == "eval_mode, original space"
    for mode in ("eval", "train"):
        assert set(rec["step0"]["post_init"][mode]) == {"original", "normalized"}
        assert SUMS <= set(rec["final"][f"{mode}_mode"]) and SUMS <= set(
            rec["final"][f"{mode}_mode_normalized"]
        )
    curve = [json.loads(x) for x in (arm_dir(tmp_path) / "curve.jsonl").read_text().splitlines()]
    assert [c["update"] for c in curve] == [1, 2, 4, 6]
    for c in curve:
        for g in GROUPS:
            assert c["grad"][g]["grad_l2"] > 0 and c["grad"][g]["nonfinite"] == 0
            assert c["param_change_this_update"][g]["delta_l2"] > 0
        assert SUMS <= set(c["pre_update_train"])
    # the data path: original- and normalized-space sums are of the same output, so their RSE agree up to fp32
    # rounding of the scale, and the curve sums are in normalized space (loss = curve MSE)
    fe, fn = rec["final"]["eval_mode"], rec["final"]["eval_mode_normalized"]
    assert fe["target_energy"] == rec["input"]["original"]["energy_sum_sq"]
    assert fn["target_energy"] == rec["input"]["normalized"]["energy_sum_sq"]
    assert abs(fe["rse"] - fn["rse"]) <= 1e-6 * fn["rse"]
    assert (
        abs(curve[0]["pre_update_train_loss"] - curve[0]["pre_update_train"]["mse"])
        <= 1e-6 * curve[0]["pre_update_train"]["mse"]
    )
    # states: saved, hash-equal to the recorded state hashes, loadable bit for bit, BN counters as expected
    fixture = json.loads((URP / "checks" / "fixture_config.json").read_text(encoding="utf-8"))
    for point, key in (("init", "model_state_sha256_init"), ("final", "model_state_sha256_final")):
        assert rec[f"{point}_state_file"]["state_sha256"] == rec[key]
        model, meta = U.load_state(arm_dir(tmp_path) / f"{point}_state.safetensors", fixture)
        assert meta["point"] == point and meta["state_sha256"] == rec[key] == C.state_sha256(model)
    final_model, _ = U.load_state(arm_dir(tmp_path) / "final_state.safetensors", fixture)
    init_model, _ = U.load_state(arm_dir(tmp_path) / "init_state.safetensors", fixture)
    counters = {k: int(v) for k, v in final_model.state_dict().items() if k.endswith("num_batches_tracked")}
    assert counters and set(counters.values()) == {6}
    assert {int(v) for k, v in init_model.state_dict().items() if k.endswith("num_batches_tracked")} == {0}
    assert any(
        v.is_floating_point() and not torch.equal(v, init_model.state_dict()[k])
        for k, v in final_model.state_dict().items()
    )
    # replay: the saved final state reproduces the recorded eval sums exactly (forward only, toy patch)
    x, x_n, _ = U.load_input(fixture)
    final_model.eval()
    with torch.no_grad():
        y_n = C.reconstruct(final_model, x_n)
    replay = C.metrics(U.denormalize(y_n, fixture["input"]["normalization"]["scale_global"]), x)
    assert {k: replay[k] for k in SUMS} == {k: fe[k] for k in SUMS}
    # runtime-file coverage: only the listed modules were imported; cg_fedllm and the URC model were not
    assert rec["modules"]["local"] == [
        "uformer_ae",
        "uformer_blocks",
        "uid_common",
        "uid_r1_control",
        "uid_r1_measure",
        "urp_control",
    ]
    assert rec["modules_at_end"]["local"] == rec["modules"]["local"]
    names = sorted(q.name for q in (tmp_path / "run" / "ledger").iterdir())
    assert names == [
        "end_FX-UPATCH.json",
        "invocation_001.json",
        "invocation_001_end.json",
        "start_FX-UPATCH.json",
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


def test_closed_arm_names_are_refused_by_the_supervisor(tmp_path):
    for arm in ("U-FIT", "U-CHECK", "R-FIT"):
        p, _ = run(tmp_path / arm, [{**FIT, "arm": arm}])
        assert p.returncode == 2 and "closed protocol" in p.stderr
        assert not (tmp_path / arm / "run").exists()  # refused before any record


def test_child_refuses_a_module_with_an_unapproved_hash(tmp_path):
    names = ("uid_common", "uformer_ae", "uformer_blocks", "uid_r1_control", "uid_r1_measure", "urp_control")
    p, s = run(tmp_path, [FIT], over={"expected_module_sha256": {n: "0" * 64 for n in (*names, "urp_child")}})
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
    p, s = run(tmp_path, [{**FIT, "max_seconds": 8.0}], hooks={"sleep_each_update_s": 2.0})
    assert p.returncode == 0, p.stderr
    a = s["arms"][0]
    assert a["status"] == "RESOURCE_STOP" and a["arm_wall_s"] <= 8.0
