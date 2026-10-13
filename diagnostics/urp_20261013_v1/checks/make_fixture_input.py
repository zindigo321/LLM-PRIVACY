"""ENGINEERING FIXTURE: write the unregistered 16x16 toy patch used by every optimizer-based URP check, once.

``python -B checks/make_fixture_input.py`` writes ``checks/fixture_patch.safetensors`` (refuses to overwrite) and
prints its SHA-256 for ``checks/fixture_config.json``. The values come from a seeded generator (seed 4242), scaled to
the order of magnitude of a real Phi patch; they are not real data and are never the registered input.
"""

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save

OUT = Path(__file__).resolve().parent / "fixture_patch.safetensors"
SEED, SHAPE, SCALE = 4242, (1, 1, 16, 16), 0.01


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; never overwritten")
    g = torch.Generator(device="cpu").manual_seed(SEED)
    x = (torch.rand(SHAPE, generator=g, dtype=torch.float32) * 2.0 - 1.0) * SCALE
    data = save({"x": x}, metadata={"format": "urp_20261013_v1.fixture_input/v1", "seed": str(SEED)})
    with OUT.open("xb") as fh:
        fh.write(data)
    print(f"wrote {OUT} ({len(data)} bytes) sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
