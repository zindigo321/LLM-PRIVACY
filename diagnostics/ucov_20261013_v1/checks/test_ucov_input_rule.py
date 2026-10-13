"""ENGINEERING CHECK: the UCOV-v1 positional rule and the layout interpretation, without reading any payload.

The tests use only grid arithmetic, the committed layout source (``block_order`` on the registered geometry; no
tensor data) and the registered rule file. No snapshot or tile value is opened.
"""

import hashlib
import sys
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]
REPO = UCOV.parents[1]
sys.path.insert(0, str(UCOV / "input"))

import prepare_inputs as B  # noqa: E402  (imports nothing heavy at module level)

RULE = UCOV / "input" / "SELECTION_RULE.md"
SLICES = {
    "U-TILE-1": ((512, 544), (384, 416)),
    "U-TILE-2": ((512, 544), (1152, 1184)),
    "U-TILE-3": ((1536, 1568), (384, 416)),
    "U-TILE-4": ((1536, 1568), (1152, 1184)),
}


def test_rule_file_is_the_pinned_registered_rule():
    data = RULE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == B.RULE_SHA256
    assert b"\r\n" not in data
    reg = (UCOV / "input" / "SELECTION_RULE.registered.txt").read_text(encoding="utf-8")
    assert B.RULE_SHA256 in reg and "2026-10-13T20:23:39.610+08:00" in reg


def test_quarter_rule_gives_the_four_registered_half_open_slices_in_order():
    tiles = B.quarter_rule(2048, 1536)
    assert [t["arm"] for t in tiles] == list(SLICES)
    for t in tiles:
        assert (tuple(t["rows"]), tuple(t["cols"])) == SLICES[t["arm"]]
        assert t["grid"] == [64, 48] and t["rows"][1] - t["rows"][0] == 32 == t["cols"][1] - t["cols"][0]


def test_the_centre_tile_of_urp_v1_is_not_registered():
    tiles = B.quarter_rule(2048, 1536)
    assert all((t["rows"], t["cols"]) != ([1024, 1056], [768, 800]) for t in tiles)
    assert all(t["tile_index"] != [32, 24] for t in tiles)


def test_tiles_do_not_overlap_and_lie_inside_phi():
    tiles = B.quarter_rule(2048, 1536)
    cells = [(r, c) for t in tiles for r in range(*t["rows"]) for c in range(*t["cols"])]
    assert len(cells) == len(set(cells)) == 4 * 1024
    assert all(0 <= r < 2048 and 0 <= c < 1536 for r, c in cells)


def test_layout_interpretation_from_the_committed_block_order():
    sys.path.insert(0, str(REPO / "src"))
    import cg_fedllm.compression.layout as L  # this repository's src; not the stale editable install

    assert Path(L.__file__).resolve() == (REPO / "src/cg_fedllm/compression/layout.py").resolve()
    geom = L.LoRAGeometry(24, ("q_proj", "k_proj", "v_proj", "o_proj"), 8, 2048)
    order = L.get_layout("layer_major_qkvo_AtB").block_order(geom)
    assert geom.phi_shape == (1, 2048, 1536) and len(order) * 8 == 1536
    for t in B.quarter_rule(2048, 1536):
        c0, c1 = t["cols"]
        blocks = [order[c // 8] for c in range(c0, c1, 8)]
        layer = 6 if c0 == 384 else 18
        assert blocks == [
            (layer, "q_proj", "A"),
            (layer, "q_proj", "B"),
            (layer, "k_proj", "A"),
            (layer, "k_proj", "B"),
        ]
        assert c0 // 64 == layer and c0 % 64 == 0


def test_a_grid_that_is_not_divisible_is_refused():
    import pytest

    with pytest.raises(ValueError):
        B.quarter_rule(2048, 1530)
