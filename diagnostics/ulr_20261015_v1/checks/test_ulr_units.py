"""ULR-20261015-v1 targeted unit tests (synthetic data only): the shape-aware wire format and the old fixed-shape
codec trap; bridge dimensions and the channel-PCA initialization (identity at c = 16); shared initialization without
RNG consumption; optimizer coverage; the payload-only standalone receiver; and evaluation immutability."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "checks"))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

torch = pytest.importorskip("torch")
import ulr_init_child as I  # noqa: E402
import ulr_model as Mo  # noqa: E402
import uoa_model as UM  # noqa: E402
from make_fixture import FIXTURE_MODEL  # noqa: E402


# ---- wire format ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("c", [16, 4, 1])
def test_wire_round_trip_lengths_and_order(c):
    z = torch.randn((1, c, 4, 4), generator=torch.Generator().manual_seed(c)).numpy()
    b = Mo.wire_to_bytes(z, c)
    assert len(b) == 64 * c == S.wire_bytes(c) and S.wire_numel(c) == 16 * c
    assert np.array_equal(
        np.frombuffer(b, dtype="<f4"), z.reshape(-1)
    )  # explicit little-endian, row-major [c, 4, 4]
    back = Mo.wire_from_bytes(b, c)
    assert back.shape == (1, c, 4, 4) and back.dtype == np.float32 and np.array_equal(back, z)


def test_wire_rejects_malformed_and_passes_nonfinite():
    with pytest.raises(ValueError):
        Mo.wire_from_bytes(b"\x00" * 255, 4)
    with pytest.raises(ValueError):
        Mo.wire_from_bytes(b"\x00" * 1024, 4)  # a c = 16 payload is not a c = 4 payload
    with pytest.raises(ValueError):
        Mo.wire_to_bytes(np.zeros((1, 16, 4, 4), np.float32), 4)
    z = np.full((1, 1, 4, 4), np.nan, np.float32)
    assert np.isnan(
        Mo.wire_from_bytes(Mo.wire_to_bytes(z, 1), 1)
    ).all()  # transmitted unchanged; invalid downstream


def test_old_fixed_shape_codec_is_a_trap_for_reduced_widths():
    import ucx_codecs as X  # the UCX / UOA 256-value latent codec

    with pytest.raises(ValueError):
        X.latent_to_bytes(np.zeros((1, 4, 4, 4), np.float32))  # 64 values: refused, never usable for c = 4
    with pytest.raises(ValueError):
        X.latent_from_bytes(b"\x00" * 256)
    src = (PKG / "ulr_eval_child.py").read_text("utf-8") + (PKG / "ulr_train_child.py").read_text("utf-8")
    assert "latent_to_bytes" not in src and "latent_from_bytes" not in src.replace("UM.", "")


# ---- channel PCA and the bridge -------------------------------------------------------------------------------
def fake_latents(n: int = 40):
    g = torch.Generator().manual_seed(5)
    mix = torch.randn((16, 16), generator=g)
    h = torch.einsum("ij,njhw->nihw", mix, torch.randn((n, 16, 4, 4), generator=g)) + 0.3
    return h.contiguous()


def init_tensors(h):
    pca = I.channel_pca(h)
    V64, mu64 = pca["V"], pca["mu"]
    init = {}
    for c in S.ALL_C:
        Vc = V64[:c]
        init[f"reduce_weight_c{c}"] = Vc.astype(np.float32).reshape(c, 16, 1, 1)
        init[f"reduce_bias_c{c}"] = (-(Vc @ mu64)).astype(np.float32)
        init[f"expand_weight_c{c}"] = Vc.T.astype(np.float32).reshape(16, c, 1, 1).copy()
        init[f"expand_bias_c{c}"] = mu64.astype(np.float32)
    return pca, init


def test_channel_pca_is_sorted_signed_orthonormal():
    pca, _ = init_tensors(fake_latents())
    sv = pca["singular_values"]
    assert sv == sorted(sv, reverse=True) and pca["samples"] == 40 * 16 and pca["dimension"] == 16
    V = pca["V"]
    assert np.abs(V @ V.T - np.eye(16)).max() < 1e-12
    assert all(V[i, int(np.argmax(np.abs(V[i])))] > 0 for i in range(16))


def test_bridge_dimensions_and_c16_identity_in_fp32():
    h = fake_latents()
    _, init = init_tensors(h)
    base = UM.build(FIXTURE_MODEL, 7, "leaky_relu").eval()
    for c in (16, 4, 1):
        m = Mo.bridged(copy.deepcopy(base), c)
        Mo.set_bridge(m, init, c)
        assert tuple(m.reduce.weight.shape) == (c, 16, 1, 1) and tuple(m.expand.weight.shape) == (16, c, 1, 1)
        assert Mo.bridge_equals(m, init, c)
        with torch.no_grad():
            z = m.reduce(h)
        assert tuple(z.shape) == (40, c, 4, 4)
    m = Mo.bridged(copy.deepcopy(base), 16)
    Mo.set_bridge(m, init, 16)
    with torch.no_grad():
        err = float((m.expand(m.reduce(h)) - h).abs().max())
    assert err <= S.C16_IDENTITY_TOL["latent_max_abs"] * 0.1


def test_shared_initialization_is_the_source_backbone_and_consumes_no_rng(tmp_path):
    base = UM.build(FIXTURE_MODEL, 7, "leaky_relu")
    f = UM.save_state(tmp_path / "src.safetensors", base, {"point": "test"})
    src = {k: f[k] for k in ("path", "file_sha256", "state_sha256")}
    _, init = init_tensors(fake_latents())
    rng = Mo.rng_sha256()
    arms = {c: Mo.build_arm(src, init, c, FIXTURE_MODEL) for c in (16, 4, 1)}
    assert Mo.rng_sha256() == rng
    assert {Mo.backbone_sha256(m) for m in arms.values()} == {src["state_sha256"]}
    with pytest.raises(K.ULRError):
        Mo.build_arm({**src, "state_sha256": "0" * 64}, init, 4, FIXTURE_MODEL)


def test_optimizer_coverage_and_training_passes_through_the_bottleneck():
    import uid_r1_measure as M

    _, init = init_tensors(fake_latents())
    m = Mo.bridged(UM.build(FIXTURE_MODEL, 7, "leaky_relu"), 4)
    Mo.set_bridge(m, init, 4)
    opt = torch.optim.Adam(m.parameters(), lr=2e-4)
    cov = M.trainable_coverage(m, opt)
    assert cov["in_optimizer"] == cov["parameters"] == sum(p.numel() for p in m.parameters())
    assert cov["tensors_missing_from_optimizer"] == 0
    calls = []
    m.reduce.register_forward_hook(lambda mod, i, o: calls.append(tuple(o.shape)))
    m.train()
    x = torch.rand((8, 1, 32, 32), generator=torch.Generator().manual_seed(1)) * 1.6 - 0.8
    Mo.train_update(m, opt, x, torch.nn.functional)
    assert calls == [(8, 4, 4, 4)]
    g = Mo.grad_stats(m)
    assert g["reduce"]["numel_with_grad"] == 4 * 16 + 4 and g["expand"]["numel_with_grad"] == 16 * 4 + 16
    assert all(v["numel_with_grad"] == v["numel"] and v["nonfinite"] == 0 for v in g.values())


def test_standalone_receiver_decodes_from_bytes_only_and_matches_the_model(tmp_path):
    _, init = init_tensors(fake_latents())
    m = Mo.bridged(UM.build(FIXTURE_MODEL, 7, "leaky_relu"), 4)
    Mo.set_bridge(m, init, 4)
    m.eval()
    f = Mo.save_state(tmp_path / "arm.safetensors", m, {"point": "test"})
    rec, info = Mo.receiver_from_state(f, 4, FIXTURE_MODEL)
    keys = list(rec.state_dict())
    assert keys and all(k.startswith(("expand.", "decoder.")) for k in keys)
    assert not any("encoder" in k or "reduce" in k for k in keys)
    x = (torch.randn((5, 1, 32, 32), generator=torch.Generator().manual_seed(2)) * 0.003).contiguous()
    fp_m, fp_r, rng = Mo.fingerprint(m)["state_sha256"], Mo.fingerprint(rec)["state_sha256"], Mo.rng_sha256()
    with torch.no_grad():
        s = Mo.send(m, x, 0.0136, 4)
        assert all(len(b) == 256 for b in s["payloads"]) and tuple(s["z"].shape) == (5, 4, 4, 4)
        y_n, _ = Mo.receive(rec, s["payloads"], 4, 0.0136)
        ref = torch.cat([m.decode(torch.from_numpy(Mo.wire_from_bytes(b, 4))) for b in s["payloads"]])
        assert torch.equal(y_n, ref)
        x[:] = 0  # the receiver output depends on the bytes only
        y2, _ = Mo.receive(rec, s["payloads"], 4, 0.0136)
        assert torch.equal(y2, y_n)
        with pytest.raises(ValueError):
            Mo.receive(rec, [b"\x00" * 1024], 4, 0.0136)
    assert (
        Mo.fingerprint(m)["state_sha256"] == fp_m
        and Mo.fingerprint(rec)["state_sha256"] == fp_r
        and Mo.rng_sha256() == rng
    )
    with pytest.raises(K.ULRError):
        Mo.send(m.train(), x, 0.0136, 4)
    loaded = Mo.load_arm_state(f, 4, FIXTURE_MODEL)
    assert Mo.state_sha256(loaded) == f["state_sha256"] and not loaded.training


def test_roundtrip_copy_leaves_training_model_unchanged():
    _, init = init_tensors(fake_latents())
    m = Mo.bridged(UM.build(FIXTURE_MODEL, 7, "leaky_relu"), 1)
    Mo.set_bridge(m, init, 1)
    m.train()
    x = (torch.randn((3, 1, 32, 32), generator=torch.Generator().manual_seed(4)) * 0.003).contiguous()
    fp, rng = Mo.fingerprint(m)["state_sha256"], Mo.rng_sha256()
    r = Mo.roundtrip_copy(m, x, 0.0136, 1)
    assert r["payload_bytes"] == [64] and tuple(r["z"].shape) == (3, 1, 4, 4)
    assert m.training and Mo.fingerprint(m)["state_sha256"] == fp and Mo.rng_sha256() == rng
