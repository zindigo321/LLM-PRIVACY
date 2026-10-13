"""ENGINEERING CHECK: the stride-1 ResNet-3 variant vs the production ResNet-3 at HEAD, geometry and gradients.

The production class is loaded read-only from the committed HEAD blob (``git show``), never from the working tree
(which holds paused SSBD edits) and never through ``cg_fedllm`` imports. Fixture seeds and tensors only; no
optimizer; the registered seed (1) and input (data seed 20261012) are not used.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

URC = Path(__file__).resolve().parents[1]
REPO = URC.parents[1]
sys.path.insert(0, str(URC))

import resnet3_s1 as R  # noqa: E402

HEAD_BLOB = "cf38fd3f7749a2cdd45864e41f22d3ab0dc9e3db"
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")


@pytest.fixture(scope="module")
def head():
    """The production ResNet-3 module namespace from the HEAD blob (the io import is unused by the classes)."""
    blob = subprocess.run(
        ["git", "rev-parse", "HEAD:src/cg_fedllm/compression/autoencoder.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert blob == HEAD_BLOB
    src = subprocess.run(
        ["git", "cat-file", "-p", blob], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    import types

    mod = types.ModuleType("head_autoencoder")  # a real module: dataclasses resolve annotations through it
    sys.modules[mod.__name__] = mod
    try:
        code = src.replace("from cg_fedllm.utils.io import load_tensors, save_tensors_atomic", "")
        exec(compile(code, f"git:{blob}", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    assert "cg_fedllm" not in sys.modules
    return mod.__dict__


class Stride2(R.ResNet3S1Config):
    """Test-only: the same layer stack with the production strides, to compare against HEAD."""

    @property
    def strides(self):
        return (2,) * len(self.down_channels)

    @property
    def latent_hw(self):
        f = 2 ** len(self.down_channels)
        return (self.input_hw[0] // f, self.input_hw[1] // f)


def test_production_model_rejects_the_diagnostic_input(head):
    ae = head["ResNetAutoEncoder"]()
    with pytest.raises(ValueError, match="divisible by 64"):
        ae.encode(torch.zeros(1, 1, 32, 32))
    strided = [
        m for m in ae.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.stride == (2, 2)
    ]
    assert len(strided) == 12


def test_variant_has_no_stride_2_and_keeps_the_grid():
    m = R.ResNet3S1AutoEncoder()
    convs = [x for x in m.modules() if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d))]
    assert all(c.stride == (1, 1) for c in convs)
    assert all(c.output_padding == (0, 0) for c in convs if isinstance(c, nn.ConvTranspose2d))
    y, z = m(torch.zeros(1, 1, 32, 32))
    assert tuple(z.shape) == (1, 64, 32, 32) and tuple(y.shape) == (1, 1, 32, 32)
    assert m.cfg.latent_element_ratio == 64.0
    assert m.num_parameters() == {
        "encoder.sampling": 24833,
        "encoder.core": 221952,
        "decoder.sampling": 24746,
        "decoder.core": 221952,
        "total": 493483,
    }
    assert sum(b.numel() for b in m.buffers()) == 1941


@pytest.mark.parametrize("shape", [(1, 1, 32, 16), (1, 2, 32, 32), (1, 1, 64, 64), (32, 32)])
def test_wrong_input_shape_is_refused(shape):
    with pytest.raises(ValueError, match="input shape"):
        R.ResNet3S1AutoEncoder().encode(torch.zeros(shape))


@pytest.mark.parametrize("hw", [(3, 3), (2, 32), (1, 1)])
def test_geometry_too_small_for_reflection_padding_is_refused(hw):
    with pytest.raises(ValueError, match="reflection padding"):
        R.ResNet3S1Config(input_hw=hw)


def test_state_dict_and_forward_equal_head_under_the_name_mapping(head):
    torch.manual_seed(31337)  # engineering seed
    ref = head["ResNetAutoEncoder"]().double()
    var = R.ResNet3S1AutoEncoder(Stride2(input_hw=(128, 128))).double()  # HEAD needs a latent grid >= 2 x 2
    ref_sd, var_sd = ref.state_dict(), var.state_dict()
    mapped = {R.production_name(k): k for k in var_sd}
    assert set(mapped) == set(ref_sd) and len(mapped) == len(var_sd) == 152
    assert all(
        ref_sd[p].shape == var_sd[k].shape and ref_sd[p].dtype == var_sd[k].dtype for p, k in mapped.items()
    )
    var.load_state_dict({k: ref_sd[p] for p, k in mapped.items()}, strict=True)
    x = torch.randn(2, 1, 128, 128, generator=torch.Generator().manual_seed(4243), dtype=torch.float64)
    for mode in (False, True):  # eval, then train (each on fresh copies of the same weights)
        a, b = ref.train(mode), var.train(mode)
        assert torch.equal(a(x)[0], b(x)[0])


def test_structure_matches_head_except_strides(head):
    ref = head["ResNetAutoEncoder"]()
    var = R.ResNet3S1AutoEncoder()
    enc_ref, dec_ref = list(ref.encoder.net), list(ref.decoder.net)
    pairs = [
        (nn.Sequential(*enc_ref[:22]), var.encoder.sampling),
        (nn.Sequential(*enc_ref[22:]), var.encoder.core),
        (nn.Sequential(*dec_ref[:3]), var.decoder.core),
        (nn.Sequential(*dec_ref[3:]), var.decoder.sampling),
    ]
    for a, b in pairs:
        sa, sb = R.structure(a), R.structure(b)
        for d in (*sa, *sb):
            d.pop("stride", None), d.pop("output_padding", None)
        assert sa == sb


def test_gradients_reach_all_four_groups_on_a_fixture_input():
    torch.manual_seed(31338)  # engineering seed
    m = R.ResNet3S1AutoEncoder().train()
    x = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(4244)) - 0.5  # fixture input
    loss = nn.functional.mse_loss(m.decode(m.encode(x)), x)
    loss.backward()
    for g in GROUPS:
        side, part = g.split(".")
        grads = [p.grad for p in getattr(getattr(m, side), part).parameters()]
        assert all(t is not None and torch.isfinite(t).all() for t in grads)
        assert sum(float(t.double().square().sum()) for t in grads) > 0


def test_config_record_names_the_variant():
    d = R.ResNet3S1AutoEncoder().cfg.to_dict()
    assert (
        d["variant"] == "resnet3_stride1_diagnostic"
        and d["strides"] == [1] * 6
        and d["output_paddings"] == [0] * 6
    )
    json.dumps(d)
