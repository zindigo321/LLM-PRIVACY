"""ENGINEERING FIXTURES: direct tests of uid_r1_measure on the fixture Uformer (fixture config, its own seeds).

The isolation claims are tested against a mutating reference: the same forward on the training model itself must
change its BatchNorm buffers, so that "unchanged" below is a real observation, not a vacuous one.
"""

import sys
from pathlib import Path

import pytest
import torch

R1 = Path(__file__).resolve().parents[1]
FROZEN = R1.parent / "uid_20261012_v2"
sys.path[:0] = [str(FROZEN), str(R1)]

import uid_common as C  # noqa: E402
import uid_r1_control as K  # noqa: E402
import uid_r1_measure as M  # noqa: E402

FIXTURE = K.load_json(R1 / "checks" / "fixture_config.json")


@pytest.fixture()
def fx():
    torch.manual_seed(12345)  # a fixed global RNG state for the test, unrelated to any registered seed
    model = C.build_model(FIXTURE, "U-FIT")
    x = C.make_input(FIXTURE)
    return model, x


def bn_buffers(model):
    return {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches" in k}


def test_a_train_mode_forward_on_the_model_itself_mutates_bn(fx):
    model, x = fx
    before = bn_buffers(model)
    model.train()
    with torch.no_grad():
        C.reconstruct(model, x)
    after = bn_buffers(model)
    assert any(not torch.equal(before[k], after[k]) for k in before)  # the reference effect exists


def test_step0_on_isolated_copies_changes_nothing(fx):
    model, x = fx
    model.train()
    fp0 = M.state_fingerprint(model)
    rng0 = torch.get_rng_state().clone()
    outs = {}
    for mode in ("eval", "train"):
        with M.isolated(model, mode) as twin, torch.no_grad():
            outs[mode] = C.reconstruct(twin, x)
    assert M.state_fingerprint(model) == fp0
    assert torch.equal(torch.get_rng_state(), rng0)
    assert model.training is True  # the training model's mode is untouched as well
    assert not torch.equal(outs["eval"], outs["train"])  # BN batch vs running statistics differ at init


def test_isolated_copy_does_not_share_storage(fx):
    model, _ = fx
    with M.isolated(model, "train") as twin:
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), twin.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()


def test_training_after_isolated_step0_matches_training_without_it(fx):
    """Step-0 measurement must not change the subsequent training trajectory (parameters, BN, RNG)."""

    def train(measure_first):
        torch.manual_seed(999)
        m = C.build_model(FIXTURE, "U-FIT")
        x = C.make_input(FIXTURE)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        if measure_first:
            for mode in ("eval", "train"):
                with M.isolated(m, mode) as twin, torch.no_grad():
                    C.reconstruct(twin, x)
        m.train()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(C.reconstruct(m, x), x).backward()
            opt.step()
        return M.state_fingerprint(m)

    torch.use_deterministic_algorithms(True)
    try:
        assert train(True) == train(False)
    finally:
        torch.use_deterministic_algorithms(False)


def test_groups_partition_every_parameter_and_grads_come_from_the_training_backward(fx):
    model, x = fx
    g = M.groups(model)
    assert set(g) == {"encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling"}
    names = [n for ps in g.values() for n, _ in ps]
    assert sorted(names) == sorted(n for n, _ in model.named_parameters()) and len(names) == len(set(names))
    assert all(v["grad_l2"] == 0 and v["numel_with_grad"] == 0 for v in M.grad_stats(model).values())
    model.train()
    torch.nn.functional.mse_loss(C.reconstruct(model, x), x).backward()
    gs = M.grad_stats(model)
    assert all(v["grad_l2"] > 0 and v["numel_with_grad"] == v["numel"] for v in gs.values())


def test_param_change_is_exact(fx):
    model, _ = fx
    ref = M.snapshot(model)
    assert all(v["delta_l2"] == 0 and v["numel_changed"] == 0 for v in M.param_change(model, ref).values())
    p = next(model.decoder.parameters())
    with torch.no_grad():
        p.view(-1)[0] += 0.5
    ch = M.param_change(model, ref)
    assert ch["decoder.core"]["numel_changed"] + ch["decoder.sampling"]["numel_changed"] == 1
    assert max(v["delta_l2"] for v in ch.values()) == pytest.approx(0.5)


def test_rss_sampler_reports_a_labelled_sampled_maximum():
    s = M.RSSSampler(0.05).start()
    ballast = torch.ones(64 * 2**18)  # 64 MiB, touched
    import time

    time.sleep(0.3)
    r = s.stop()
    del ballast
    assert r["samples"] >= 3 and r["interval_s"] == 0.05 and "sampled" in r["label"]
    assert r["sampled_max_rss_bytes"] >= 64 * 2**20
