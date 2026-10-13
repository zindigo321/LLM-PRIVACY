"""ENGINEERING CHECK (added after check_upstream_correspondence.py reported exact zeros): negative control.

Shows that the correspondence comparison is not vacuous: with the same mapping, a 1e-9 change of one vendored
weight, or an independently initialised vendored block, must be flagged by the same declared tolerances. Fixture
tensors only; no optimizer; nothing registered.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_upstream_correspondence as U  # noqa: E402


def main() -> int:
    up, _ = U.load_upstream()
    sys.path[:0] = [str(U.FROZEN)]
    import uformer_blocks as V

    g = torch.Generator().manual_seed(20261014)  # engineering-only seed
    dim, heads, win, grid = 64, 2, 8, 32

    def pair():
        u = up.LeWinTransformerBlock(
            dim, (grid, grid), heads, win_size=win, shift_size=win // 2, mlp_ratio=4.0, token_mlp="leff"
        ).double()
        v = V.LeWinTransformerBlock(
            dim, (grid, grid), heads, win_size=win, shift_size=win // 2, mlp_ratio=4.0
        ).double()
        U.randomize(u, g)
        return u, v

    x = torch.randn((2, grid * grid, dim), generator=g, dtype=torch.float64)
    u, v = pair()
    U.map_weights(v, u)
    base = U.compare(v(x).detach(), u(x).detach())
    with torch.no_grad():
        v.attn.proj.weight[0, 0] += 1e-9
    perturbed = U.compare(v(x).detach(), u(x).detach())
    u2, v2 = pair()  # v2 keeps its own initialisation: no mapping
    unmapped = U.compare(v2(x).detach(), u2(x).detach())
    out_norm = float(u(x).detach().norm())
    report = {
        "tolerances": U.TOL,
        "mapped_identical": base,
        "one_weight_plus_1e-9": perturbed,
        "independently_initialised": unmapped,
        "upstream_output_l2": out_norm,
        "control_ok": base["ok"] and not perturbed["ok"] and not unmapped["ok"] and out_norm > 0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["control_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
