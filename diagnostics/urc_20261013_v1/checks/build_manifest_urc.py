"""Build ``manifest_urc.json`` LAST: every runtime file of the scientific launch (URC modules, imported r1 and frozen
UID modules, configs, the post-run analysis and the EP classifier), the reference identities, the checks and the
review documents, with full SHA-256 and sizes. Refuses to overwrite. ``python -B checks/build_manifest_urc.py``.
Paths are relative to the URC directory.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

URC = Path(__file__).resolve().parents[1]

RUNTIME = [  # imported or read by `urc_supervise.py --scientific`, its child, or `urc_analyze.py`
    (
        "urc_supervise.py",
        "runtime: scientific entry point (supervisor; from r1 with review/supervise_diff.patch)",
    ),
    ("urc_child.py", "runtime: R-FIT child process (from r1 with review/child_diff.patch)"),
    ("urc_control.py", "runtime: identity, resolution, one-arm plan, model build, state files"),
    ("resnet3_s1.py", "runtime: the stride-1 ResNet-3 diagnostic variant"),
    ("urc_config.json", "runtime: URC configuration (expected shared values, limits, paths)"),
    (
        "resolved_config.json",
        "runtime: resolved scientific configuration (re-resolved and compared at launch)",
    ),
    ("protocol.md", "protocol"),
    ("urc_analyze.py", "post-run: seal, strict, EP, matching, signed differences"),
    (
        "../uid_20261012_v2/uid_common.py",
        "runtime: frozen UID helpers (unchanged; input, metrics, baselines, CPU)",
    ),
    ("../uid_20261012_v2/config.json", "runtime: frozen UID config (shared settings are read from it)"),
    ("../uid_20261012_v2/manifest.json", "reference identity: frozen UID manifest"),
    (
        "../uid_20261012_v2_r1/uid_r1_control.py",
        "runtime: r1 ledger / start accounting / deadlines (unchanged)",
    ),
    ("../uid_20261012_v2_r1/uid_r1_measure.py", "runtime: r1 measurements and RSS sampler (unchanged)"),
    (
        "../uid_20261012_v2_r1/execution_r1.json",
        "runtime: r1 execution mechanics (read for margins and versions)",
    ),
    ("../uid_20261012_v2_r1/manifest_r1.json", "reference identity: r1 manifest"),
    ("../exploratory_progression_20261012/ep_classify.py", "post-run: frozen EP-20261012-v1 classifier"),
    ("../exploratory_progression_20261012/POLICY.md", "post-run: frozen EP-20261012-v1 policy"),
]
REVIEW = ["review/MATCHING.md", "review/child_diff.patch", "review/supervise_diff.patch"]


def entry(rel: str, role: str) -> dict:
    data = (URC / rel).resolve().read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}


def main() -> int:
    out = URC / "manifest_urc.json"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    checks = sorted(
        p.relative_to(URC).as_posix()
        for p in (URC / "checks").iterdir()
        if p.is_file() and p.name != "build_manifest_urc.py"
    )
    files = [entry(r, role) for r, role in RUNTIME]
    files += [
        entry(r, "engineering check (fixtures, registered input/baselines only, fresh models)")
        for r in checks
    ]
    files += [entry("checks/build_manifest_urc.py", "manifest builder")]
    files += [entry(r, "review document") for r in REVIEW]
    evidence = sorted(p.relative_to(URC).as_posix() for p in (URC / "evidence").rglob("*") if p.is_file())
    files += [entry(r, "engineering evidence (pre-freeze check logs)") for r in evidence]
    reference_record = "D:/cgfed-runs/uid_20261012_v2_r1/arms/U-FIT/child_record.json"
    manifest = {
        "protocol_id": "URC-20261013-v1",
        "arm": "R-FIT",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "dependencies": {
            "torch": "2.14.0+cu130",
            "safetensors": "0.8.0",
            "psutil": "7.2.2",
            "numpy": "2.4.6",
            "check-only": {"pytest": "9.1.1", "ruff": "0.16.9"},
        },
        "reference": {
            "u_fit_child_record": reference_record,
            "u_fit_child_record_sha256": hashlib.sha256(Path(reference_record).read_bytes()).hexdigest(),
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
