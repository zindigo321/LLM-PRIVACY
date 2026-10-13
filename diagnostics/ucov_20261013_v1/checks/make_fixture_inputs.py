"""ENGINEERING FIXTURE: write the four unregistered 16x16 toy tiles used by every optimizer-based UCOV check, once.

``python -B checks/make_fixture_inputs.py`` writes ``checks/fixture_tile_{1..4}.safetensors`` (refuses to overwrite)
and prints their SHA-256 for ``checks/fixture_config.json``. The values come from seeded generators (seeds 5101-5104),
scaled to the order of magnitude of a real Phi tile. They are not real data and are never a registered input.
"""

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save

HERE = Path(__file__).resolve().parent
SHAPE, SCALE = (1, 1, 16, 16), 0.01


def main() -> int:
    outs = [HERE / f"fixture_tile_{k}.safetensors" for k in (1, 2, 3, 4)]
    for p in outs:
        if p.exists():
            raise SystemExit(f"{p} exists; never overwritten")
    for k, path in zip((1, 2, 3, 4), outs):
        g = torch.Generator(device="cpu").manual_seed(5100 + k)
        x = (torch.rand(SHAPE, generator=g, dtype=torch.float32) * 2.0 - 1.0) * SCALE
        data = save({"x": x}, metadata={"format": "ucov_20261013_v1.fixture_input/v1", "seed": str(5100 + k)})
        with path.open("xb") as fh:
            fh.write(data)
        print(f"{path.name} {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
