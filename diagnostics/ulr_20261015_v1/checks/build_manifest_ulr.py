"""ULR-20261015-v1: build ``manifest_ulr.json`` LAST, after every package file is final. Lists every package file
(except the manifest itself and the scientific ledger mirror) and every external file the run binds, each with its
SHA-256 and size. ``python -B checks/build_manifest_ulr.py``; written exclusively; prints the manifest SHA-256.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402


def main() -> int:
    import numpy
    import psutil
    import safetensors
    import torch

    out = PKG / "manifest_ulr.json"
    if out.exists():
        raise SystemExit("manifest_ulr.json exists: the package is already frozen")
    files = []
    for p in sorted(PKG.rglob("*")):
        rel = p.relative_to(PKG).as_posix()
        if not p.is_file() or rel == "manifest_ulr.json" or rel.startswith("scientific_ledger_mirror/"):
            continue
        if "__pycache__" in rel or rel.endswith(".pyc"):
            raise SystemExit(f"bytecode in the package: {rel}")
        files.append({"path": rel, "sha256": K.sha256_file(p), "bytes": p.stat().st_size})
    cfg = K.load_json(PKG / "ulr_config.json")
    ext = []
    refs = {
        "UOA source state (U-LEAKY-LONG, 12,000 updates)": cfg["uoa"]["source_state"]["path"],
        "UOA source latent scales": cfg["uoa"]["source_scales"]["path"],
        "UOA SCALAR-4 levels": cfg["uoa"]["scalar4_levels"]["path"],
        "UOA eval sums (post-run reproduction check only)": cfg["uoa"]["eval_sums"]["path"],
        "UOA eval latents (post-run reproduction check only)": cfg["uoa"]["eval_latents"]["path"],
        "UCX TRAIN tiles": cfg["ucx"]["train_tiles"]["path"],
        "UCX held-out tiles (EVAL only)": cfg["ucx"]["heldout_tiles"]["path"],
        "UCX TRAIN mean tile": cfg["ucx"]["train_mean_tile"]["path"],
        "UCX PCA-256 decoder (input-PCA comparators)": cfg["ucx"]["pca_decoder"]["path"],
        "UOA frozen manifest": "D:/projects-cd/cg-fedllm-uoa/diagnostics/uoa_20261014_v1/manifest_uoa.json",
        "UOA configuration (split keys, wrong map)": "D:/projects-cd/cg-fedllm-uoa/diagnostics/uoa_20261014_v1/uoa_config.json",
        "UOA U-LEAKY-LONG training record": "D:/cgfed-runs/uoa_20261014_v1/steps/U-LEAKY-LONG/train_record.json",
    }
    for role, path in refs.items():
        p = Path(path)
        ext.append(
            {"path": p.as_posix(), "sha256": K.sha256_file(p), "bytes": p.stat().st_size, "role": role}
        )
    m = {
        "protocol_id": S.PROTOCOL_ID,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_base": "0238e3b774e0f97f171bc5cbc53f1ec06aa49c7e",
        "primary_policy": S.POLICY_ID,
        "python": sys.executable,
        "dependencies": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "psutil": psutil.__version__,
            "safetensors": safetensors.__version__,
        },
        "files": files,
        "external_bound_files": ext,
        "external_bound_files_note": "verified by SHA-256 at launch; UOA / UCX artifacts are read, never copied or written",
    }
    print(
        K.write_exclusive(out, (json.dumps(m, indent=2, sort_keys=True) + "\n").encode("utf-8")),
        len(files),
        "files",
        len(ext),
        "external",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
