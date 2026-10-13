"""ENGINEERING CHECK: the positional rule and the archived identities of the input builder, without any payload.

``tile_rule`` and ``extract`` are tested on synthetic tensors only; ``archived`` reads only committed P5-A records and
the committed ``p5a_inputs.py`` source. No snapshot file is opened, no model is built and no optimizer is created.
"""

import sys
from pathlib import Path

import pytest
import torch

URP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(URP / "input"))

import prepare_input as P  # noqa: E402


def test_rule_gives_the_registered_tile_for_the_phi_grid():
    c = P.tile_rule(2048, 1536)
    assert c == {
        "tile": 32,
        "grid": [64, 48],
        "tile_index": [32, 24],
        "rows": [1024, 1056],
        "cols": [768, 800],
    }
    assert c["cols"][0] // 64 == 12 and c["cols"][0] % 64 == 0  # layer 12, starting at its q_proj A^T block


@pytest.mark.parametrize(
    ("h", "w", "rows", "cols"), [(64, 96, [32, 64], [32, 64]), (96, 32, [32, 64], [0, 32])]
)
def test_rule_is_the_floor_centre_tile(h, w, rows, cols):
    c = P.tile_rule(h, w)
    assert (c["rows"], c["cols"]) == (rows, cols)


@pytest.mark.parametrize(("h", "w"), [(2040, 1536), (2048, 1530), (31, 32)])
def test_rule_refuses_a_grid_not_divisible_by_the_tile(h, w):
    with pytest.raises(P.InputError):
        P.tile_rule(h, w)


def test_extract_copies_the_exact_values_without_reordering():
    phi = torch.arange(2048 * 1536, dtype=torch.float32).reshape(1, 2048, 1536) * 0.5
    c = P.tile_rule(2048, 1536)
    x = P.extract(phi, c)
    assert tuple(x.shape) == (1, 1, 32, 32) and x.dtype == torch.float32 and x.is_contiguous()
    assert torch.equal(x[0, 0], phi[0, 1024:1056, 768:800])
    assert (
        float(x[0, 0, 0, 0]) == (1024 * 1536 + 768) * 0.5
        and float(x[0, 0, 31, 31]) == (1055 * 1536 + 799) * 0.5
    )
    x[0, 0, 0, 0] = -1.0  # a copy: the source is untouched
    assert float(phi[0, 1024, 768]) == (1024 * 1536 + 768) * 0.5


def test_block_energies_follow_the_layout_blocks():
    c = P.tile_rule(2048, 1536)
    x = torch.zeros(1, 1, 32, 32)
    x[0, 0, :, 8:16] = 2.0  # q_proj B only
    rows = P.block_energies(x, c)
    assert [r["block"] for r in rows] == ["q_proj A^T", "q_proj B", "k_proj A^T", "k_proj B"]
    assert [r["phi_cols"] for r in rows] == [[768, 776], [776, 784], [784, 792], [792, 800]]
    assert [r["sum_sq"] for r in rows] == [0.0, 32 * 8 * 4.0, 0.0, 0.0] and {r["layer"] for r in rows} == {12}


def test_archived_identities_agree_across_committed_records():
    a = P.archived()
    s = a["selected"]
    assert (s["position"], s["time_index"], s["client_id"]) == (39, 7, 91)
    assert s["phi_sha256"] == "485cbe5f3a450b43d7780fe9cd97afe5169ea81ce0bdaf21166938adfcac2546"
    assert a["index_sha256"] == "b6834e476f21f3d6b07f6489b5113576e566ab3987a2f4593405664476d07240"
    assert set(a["scale_sources_equal"].values()) == {0.015751632224572334}
    assert a["frozen_max_abs"] == 0.014964050613343716 and a["n_train"] == 80
    assert a["archived_normalized_max_abs"] == 0.8891991972923279


def test_rule_file_is_the_pinned_one():
    assert P.sha256_file(URP / "input" / "SELECTION_RULE.md") == P.RULE_SHA256
