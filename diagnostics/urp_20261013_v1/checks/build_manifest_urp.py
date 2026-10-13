"""Build ``manifest_urp.json`` LAST: every runtime file of the scientific launch (URP modules, imported r1 and frozen
UID modules, configs, the registered input and its provenance, the post-run analysis and the EP classifier), the
reference identities, the checks, the review documents and the pre-freeze check evidence, with full SHA-256 and sizes.
Refuses to overwrite and writes LF bytes. ``python -B checks/build_manifest_urp.py``. Paths are relative to the URP
directory.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

URP = Path(__file__).resolve().parents[1]

RUNTIME = [  # imported or read by `urp_supervise.py --scientific`, its child, or `urp_analyze.py`
    (
        "urp_supervise.py",
        "runtime: scientific entry point (supervisor; from URC with review/supervise_diff.patch)",
    ),
    ("urp_child.py", "runtime: U-PATCH child process (from URC with review/child_diff.patch)"),
    (
        "urp_control.py",
        "runtime: identity, resolution, input, one-arm plan, model binding, normalization, states",
    ),
    (
        "urp_config.json",
        "runtime: URP configuration (expected shared values, input identities, limits, paths)",
    ),
    (
        "resolved_config.json",
        "runtime: resolved scientific configuration (re-resolved and compared at launch)",
    ),
    ("input/patch_x.safetensors", "runtime: the registered input patch (original space)"),
    ("input/input_record.json", "runtime: provenance, coordinates, normalization and baselines of the patch"),
    ("input/SELECTION_RULE.md", "input: positional rule, fixed before any payload read"),
    ("input/prepare_input.py", "input: the builder that produced the patch and its record (run once)"),
    ("input/prepare_input.out.txt", "input: console output of that single run"),
    ("protocol.md", "protocol"),
    ("urp_analyze.py", "post-run: seal, evidence, strict, EP, reported-only"),
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
    ("../exploratory_progression_20261012/ep_classify.py", "post-run: frozen EP-20261012-v1 classifier"),
    ("../exploratory_progression_20261012/POLICY.md", "post-run: frozen EP-20261012-v1 policy"),
    ("../synthetic_comparison_summary_20261013/STAGE_SUMMARY.md", "background: derived stage summary"),
    ("../synthetic_comparison_summary_20261013/stage_summary.json", "background: its verified numbers"),
    ("../synthetic_comparison_summary_20261013/extract_summary.py", "background: its read-only extraction"),
    ("../synthetic_comparison_summary_20261013/extract_summary.out.txt", "background: its console output"),
]
REVIEW = ["review/child_diff.patch", "review/supervise_diff.patch"]
EXTERNAL = [  # files outside the repository that the package binds by hash
    (
        "D:/cgfed-runs/uid_20261012_v2_r1/arms/U-FIT/child_record.json",
        "reference: U-FIT record (initial state hash)",
    ),
    ("D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/index.jsonl", "input source: index"),
    (
        "D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/snapshots/t0007_c0091.safetensors",
        "input source: end state (read once by prepare_input.py)",
    ),
    (
        "D:/cgfed-runs/phase3_tierb_qwen15_1p8b_seed1/tgap_federated_state/starts/"
        "aa3ab51c854c3480985381da05543beb73cb0ea624fe000866b69d630093d580.safetensors",
        "input source: start state (read once by prepare_input.py)",
    ),
]


def entry(rel: str, role: str) -> dict:
    data = (URP / rel).resolve().read_bytes()
    return {"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}


def main() -> int:
    out = URP / "manifest_urp.json"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    checks = sorted(
        p.relative_to(URP).as_posix()
        for p in (URP / "checks").iterdir()
        if p.is_file() and p.name != "build_manifest_urp.py"
    )
    files = [entry(r, role) for r, role in RUNTIME]
    files += [
        entry(r, "engineering check (toy fixtures; registered model/input never optimized)") for r in checks
    ]
    files += [entry("checks/build_manifest_urp.py", "manifest builder")]
    files += [entry(r, "review document") for r in REVIEW]
    evidence = sorted(p.relative_to(URP).as_posix() for p in (URP / "evidence").rglob("*") if p.is_file())
    files += [entry(r, "engineering evidence (pre-freeze check logs)") for r in evidence]
    external = []
    for path, role in EXTERNAL:
        data = Path(path).read_bytes()
        external.append(
            {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "role": role}
        )
    manifest = {
        "protocol_id": "URP-20261013-v1",
        "arm": "U-PATCH",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "dependencies": {
            "torch": "2.14.0+cu130",
            "safetensors": "0.8.0",
            "psutil": "7.2.2",
            "numpy": "2.4.6",
            "python": "3.11.16",
            "check-only": {"pytest": "9.1.1", "ruff": "0.16.9"},
            "input-build-only": "cg_fedllm from this repository's src at HEAD 0ad2795 (blobs in input_record.json)",
        },
        "external_bound_files": external,
        "files": files,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with out.open("xb") as fh:
        fh.write(data)
    print(f"wrote {out} ({len(data)} bytes, {len(files)} files) sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
