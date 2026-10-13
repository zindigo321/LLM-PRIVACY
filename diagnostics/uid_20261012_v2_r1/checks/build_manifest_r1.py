"""Build manifest_r1.json: every runtime dependency of the scientific launch (r1 and frozen modules, config,
execution file), the checks and the review documents, with full SHA-256 and sizes. Refuses to overwrite.

``python -B checks/build_manifest_r1.py``. Paths are relative to the r1 directory.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]

RUNTIME = [  # imported or read by `uid_r1_supervise.py --scientific` and its child
    ("uid_r1_supervise.py", "runtime: scientific entry point (supervisor)"),
    ("uid_r1_child.py", "runtime: per-arm child process"),
    ("uid_r1_control.py", "runtime: ledger, limits, plan, package identity"),
    ("uid_r1_measure.py", "runtime: isolated measurements, gradient / parameter evidence, RSS sampler"),
    ("execution_r1.json", "runtime: execution mechanics only (no scientific field)"),
    ("../uid_20261012_v2/config.json", "runtime: frozen scientific config (unchanged)"),
    ("../uid_20261012_v2/uid_common.py", "runtime: frozen helpers (unchanged)"),
    ("../uid_20261012_v2/uformer_ae.py", "runtime: frozen model (unchanged)"),
    ("../uid_20261012_v2/uformer_blocks.py", "runtime: frozen vendored blocks (unchanged)"),
    ("../uid_20261012_v2/manifest.json", "runtime: frozen parent manifest (verified at launch)"),
    ("../uid_20261012_v2/protocol.md", "parent protocol (unchanged)"),
]
CHECKS = sorted(p.relative_to(R1).as_posix() for p in (R1 / "checks").iterdir() if p.is_file())
DOCS = [
    "review_package/REVISION_NOTE.md",
    "review_package/CHANGELOG.md",
    "review_package/protocol_addendum_r1.md",
    "review_package/week_plan.md",
    "review_package/runner_diff.patch",
]


def entry(rel: str, role: str) -> dict:
    p = (R1 / rel).resolve()
    data = p.read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}


def main() -> int:
    out = R1 / "manifest_r1.json"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    files = [entry(r, role) for r, role in RUNTIME]
    files += [
        entry(r, "engineering check (fixtures only)")
        for r in CHECKS
        if not r.endswith("build_manifest_r1.py")
    ]
    files += [entry("checks/build_manifest_r1.py", "manifest builder")]
    files += [entry(r, "review document") for r in DOCS]
    manifest = {
        "revision_id": "UID-20261012-v2-r1",
        "parent_protocol_id": "UID-20261012-v2",
        "parent_manifest_sha256": hashlib.sha256(
            (R1.parent / "uid_20261012_v2/manifest.json").read_bytes()
        ).hexdigest(),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "dependencies": {
            "torch": "2.14.0+cu130",
            "psutil": "7.2.2",
            "numpy": "2.4.6",
            "check-only": {"pytest": "9.1.1", "ruff": "0.16.9", "einops-wheel (zipimport)": "0.8.1"},
        },
        "files": files,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with out.open("xb") as fh:
        fh.write(data)
    print(f"wrote {out} ({len(data)} bytes, {len(files)} files) sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
