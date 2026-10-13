"""Build ``manifest_ucov.json`` LAST. It lists every file of the scientific launch, with full SHA-256 and sizes:
- the UCOV modules and the imported frozen URP, UPP, UID and r1 modules;
- the configs, the policy, the registered inputs and their provenance;
- the analyzer and the EP classifier;
- the reference identities, checks, review documents and pre-freeze check evidence.
External files bound by hash are listed too. Refuses to overwrite and writes LF bytes.
``python -B checks/build_manifest_ucov.py``. Paths are relative to the UCOV directory.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

UCOV = Path(__file__).resolve().parents[1]

RUNTIME = [
    (
        "ucov_supervise.py",
        "runtime: scientific entry point (supervisor; from URP with review/supervise_diff.patch)",
    ),
    ("ucov_child.py", "runtime: tile-arm child process (from URP with review/child_diff.patch)"),
    ("ucov_control.py", "runtime: identity, resolution, inputs, four-arm plan, policy binding"),
    ("ucov_config.json", "runtime: UCOV configuration (expected values, policy, inputs, limits, paths)"),
    (
        "resolved_config.json",
        "runtime: resolved scientific configuration (re-resolved and compared at launch)",
    ),
    ("input/tile_1.safetensors", "runtime: registered tile U-TILE-1 (original space)"),
    ("input/tile_2.safetensors", "runtime: registered tile U-TILE-2 (original space)"),
    ("input/tile_3.safetensors", "runtime: registered tile U-TILE-3 (original space)"),
    ("input/tile_4.safetensors", "runtime: registered tile U-TILE-4 (original space)"),
    (
        "input/inputs_record.json",
        "runtime: provenance, coordinates, normalization and baselines of the tiles",
    ),
    ("input/SELECTION_RULE.md", "input: positional quarter-grid rule, registered before any payload read"),
    ("input/SELECTION_RULE.registered.txt", "input: registration time and hash of the rule"),
    ("input/prepare_inputs.py", "input: the builder that produced the tiles and their record (run once)"),
    ("input/prepare_inputs.out.txt", "input: console output of that single run"),
    ("protocol.md", "protocol"),
    ("ucov_analyze.py", "post-run: seal, evidence, primary policy, aggregate, secondary references"),
    ("../upp_20261013_v1/upp_policy.json", "runtime: primary policy UPP-20261013-v1 (authoritative values)"),
    ("../upp_20261013_v1/upp_classify.py", "runtime: primary policy classifier"),
    ("../upp_20261013_v1/POLICY.md", "primary policy text"),
    ("../upp_20261013_v1/test_upp_classify.py", "primary policy tests"),
    (
        "../urp_20261013_v1/urp_control.py",
        "runtime: frozen URP-v1 model binding, normalization and state files",
    ),
    ("../urp_20261013_v1/manifest_urp.json", "reference identity: URP-v1 manifest"),
    (
        "../urp_20261013_v1/input/prepare_input.py",
        "input: URP-v1 builder whose provenance functions are imported",
    ),
    (
        "../uid_20261012_v2/uid_common.py",
        "runtime: frozen UID helpers (unchanged; build_model, metrics, baselines)",
    ),
    ("../uid_20261012_v2/uformer_ae.py", "runtime: frozen Uformer AE (unchanged)"),
    ("../uid_20261012_v2/uformer_blocks.py", "runtime: frozen LeWin / BasicUformer blocks (unchanged)"),
    ("../uid_20261012_v2/config.json", "runtime: frozen UID config (shared settings and the U-FIT model)"),
    ("../uid_20261012_v2/manifest.json", "reference identity: frozen UID manifest"),
    (
        "../uid_20261012_v2_r1/uid_r1_control.py",
        "runtime: r1 ledger / start accounting / deadlines (unchanged)",
    ),
    ("../uid_20261012_v2_r1/uid_r1_measure.py", "runtime: r1 measurements and RSS sampler (unchanged)"),
    ("../uid_20261012_v2_r1/execution_r1.json", "runtime: r1 execution mechanics (margins and versions)"),
    ("../uid_20261012_v2_r1/manifest_r1.json", "reference identity: r1 manifest"),
    (
        "../exploratory_progression_20261012/ep_classify.py",
        "post-run: frozen EP-20261012-v1 classifier (secondary)",
    ),
    ("../exploratory_progression_20261012/POLICY.md", "post-run: frozen EP-20261012-v1 policy (secondary)"),
]
REVIEW = [
    "review/analyze_diff.patch",
    "review/child_diff.patch",
    "review/control_diff.patch",
    "review/supervise_diff.patch",
]
EXTERNAL = [
    (
        "D:/cgfed-runs/uid_20261012_v2_r1/arms/U-FIT/child_record.json",
        "reference: U-FIT record (initial state hash)",
    ),
    ("D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH/child_record.json", "context: URP-v1 centre-tile record"),
    ("D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/index.jsonl", "input source: index"),
    (
        "D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/snapshots/t0007_c0091.safetensors",
        "input source: end state (read once by prepare_inputs.py)",
    ),
    (
        "D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/starts/"
        "aa3ab51c854c3480985381da05543beb73cb0ea624fe000866b69d630093d580.safetensors",
        "input source: start state (read once by prepare_inputs.py)",
    ),
]


def entry(rel: str, role: str) -> dict:
    data = (UCOV / rel).resolve().read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}


def main() -> int:
    out = UCOV / "manifest_ucov.json"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    checks = sorted(
        p.relative_to(UCOV).as_posix()
        for p in (UCOV / "checks").iterdir()
        if p.is_file() and p.name != "build_manifest_ucov.py"
    )
    files = [entry(r, role) for r, role in RUNTIME]
    files += [
        entry(r, "engineering check (toy fixtures; registered model/tiles never optimized)") for r in checks
    ]
    files += [entry("checks/build_manifest_ucov.py", "manifest builder")]
    files += [entry(r, "review document") for r in REVIEW]
    evidence = sorted(p.relative_to(UCOV).as_posix() for p in (UCOV / "evidence").rglob("*") if p.is_file())
    files += [entry(r, "engineering evidence (pre-freeze check logs)") for r in evidence]
    external = []
    for path, role in EXTERNAL:
        data = Path(path).read_bytes()
        external.append(
            {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}
        )
    manifest = {
        "protocol_id": "UCOV-20261013-v1",
        "arms": ["U-TILE-1", "U-TILE-2", "U-TILE-3", "U-TILE-4"],
        "primary_policy": "UPP-20261013-v1",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "dependencies": {
            "torch": "2.14.0+cu130",
            "safetensors": "0.8.0",
            "psutil": "7.2.2",
            "numpy": "2.4.6",
            "python": "3.11.16",
            "check-only": {"pytest": "9.1.1", "ruff": "0.16.9"},
            "input-build-only": "cg_fedllm from this repository's src at HEAD 0ad2795 (blobs in inputs_record.json)",
        },
        "external_bound_files": external,
        "files": files,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with out.open("xb") as fh:
        fh.write(data)
    print(
        f"wrote {out} ({len(data)} bytes, {len(files)} files, {len(external)} external) "
        f"sha256={hashlib.sha256(data).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
