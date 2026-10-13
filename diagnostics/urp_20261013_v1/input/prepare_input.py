"""URP-20261013-v1: build the one registered input patch from the archived R3 training snapshot (CPU, no model).

``python -B input/prepare_input.py`` (from the URP directory). Runs once; refuses to overwrite either output.

Order (fixed in ``SELECTION_RULE.md``, whose SHA-256 is pinned below):
1. archived identities: the committed P5-A R3 single-snapshot record and preflight 1 (pinned SHA-256), cross-checked;
   the frozen R3 constants in the committed ``p5a_inputs.py`` source (read with ``ast``, not imported);
2. ``index.jsonl`` (pinned SHA-256) and the (7, 91) record, which must carry the archived identity fields;
3. ``load_states(verify=True)`` (file-byte and adapter-content hashes), ``balanced_effective_delta(s=2, rank=8)``,
   ``layer_major_qkvo_AtB``: the Phi tensor must have the archived ``phi_sha256``, otherwise no patch is written;
4. the frozen R3 normalizer (production ``Normalizer``) and the archived normalized maximum of the whole snapshot;
5. the floor-centre 32 x 32 tile, its descriptions and baselines in both spaces, range feasibility and degeneracy.

``cg_fedllm`` is imported from this repository's ``src`` (never the stale editable install); every imported
``cg_fedllm`` module must be unmodified against HEAD, and the three modified package files must not be imported.
Writes ``input/patch_x.safetensors`` and ``input/input_record.json`` (LF bytes).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent
URP_DIR = INPUT_DIR.parent
REPO = URP_DIR.parents[1]
FROZEN_DIR = REPO / "diagnostics" / "uid_20261012_v2"
sys.path.insert(0, str(REPO / "src"))  # before any cg_fedllm import: the editable install points elsewhere
sys.path.insert(1, str(FROZEN_DIR))

RULE_SHA256 = "e0b1df83a9ca79ac8e4682d25b6b56d1a8c76b17bd7d373684f48f2a31d7362d"
UID_COMMON_SHA256 = "f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230"
RECORD = REPO / "results/phase5/p5a_v2/R3_balanced_effective_delta_r8_single_snapshot/p5a_record.json"
RECORD_SHA256 = "f60d3d15788b1945953d16394c39762b3a7a304e440d5bd63b96d87a23e97b2d"
PREFLIGHT = REPO / "results/phase5/p5a_v2/preflight/preflight_1.json"
PREFLIGHT_SHA256 = "b058eb1bf01d101e1800bdf05d7701b5ac3de515ea71cd542f70403ee5d48cee"
ARTIFACT_MANIFEST = REPO / "results/phase5/p5a_v2/artifact_manifest.json"
P5A_INPUTS_SRC = "src/cg_fedllm/phase5/p5a_inputs.py"
RUNS_ROOT = Path("D:/cgfed-runs")
P5A_EXECUTION_COMMIT = "6603bd8d9156831562dd76c0bf9dba5a071e867f"
MODIFIED_PACKAGE_FILES = (
    "cg_fedllm/cli.py",
    "cg_fedllm/compression/autoencoder.py",
    "cg_fedllm/phase5/__init__.py",
)
LORA_SCALE, LORA_RANK = 16 / 8, 8  # s = alpha / r, as p5a_inputs (verified end to end by the Phi hash)
LAYOUT_ID = "layer_major_qkvo_AtB"
TILE = 32
POSITION, TIME_INDEX, CLIENT_ID = 39, 7, 91
IDENTITY_FIELDS = (
    "file",
    "file_sha256",
    "start_file",
    "start_file_sha256",
    "end_adapter_hash",
    "start_adapter_hash",
)
OUT_PATCH = INPUT_DIR / "patch_x.safetensors"
OUT_RECORD = INPUT_DIR / "input_record.json"
INTRA, INTER = 24, 24  # the P5-A CPU environment (preflight cpu_environment) and the UID cpu_config


class InputError(RuntimeError):
    """Provenance or rule check failed: no patch is produced and nothing is substituted."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def tile_rule(height: int, width: int, tile: int = TILE) -> dict:
    """The positional rule of SELECTION_RULE.md: the floor-centre tile of the aligned non-overlapping tile grid."""
    if height % tile or width % tile:
        raise InputError(f"{height} x {width} is not divisible by the tile {tile}")
    rows, cols = height // tile, width // tile
    ti, tj = rows // 2, cols // 2
    return {
        "tile": tile,
        "grid": [rows, cols],
        "tile_index": [ti, tj],
        "rows": [ti * tile, ti * tile + tile],
        "cols": [tj * tile, tj * tile + tile],
    }


def extract(phi, coords: dict):
    """``Phi[0, r0:r1, c0:c1]`` as a contiguous fp32 copy shaped [1, 1, tile, tile]; values unchanged."""
    (r0, r1), (c0, c1) = coords["rows"], coords["cols"]
    return phi[0, r0:r1, c0:c1].contiguous().clone().reshape(1, 1, r1 - r0, c1 - c0)


def frozen_source_constants() -> dict:
    """FROZEN_MAX / FROZEN_SCALE of R3 from the committed p5a_inputs.py (HEAD blob), read without importing it."""
    tree = ast.parse(git("show", f"HEAD:{P5A_INPUTS_SRC}"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("FROZEN_MAX", "FROZEN_SCALE"):
                out[node.targets[0].id] = ast.literal_eval(node.value)["R3"]
    return out


def archived() -> dict:
    """Archived identities of the source snapshot, from two committed records that must agree field by field."""
    if sha256_file(RECORD) != RECORD_SHA256 or sha256_file(PREFLIGHT) != PREFLIGHT_SHA256:
        raise InputError("archived P5-A record or preflight differs from its pinned SHA-256")
    rec, pre = (
        json.loads(RECORD.read_text(encoding="utf-8")),
        json.loads(PREFLIGHT.read_text(encoding="utf-8")),
    )
    sel = rec["selected_records"][0]
    r3 = next(r for r in pre["representations"] if r["representation"] == "R3")["evidence"]
    pre_row = r3["records"][POSITION]
    manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    listed = {e["path"]: e["sha256"] for e in manifest["run_root_artifacts"]}
    norm = rec["normalization"]["normalizer"]
    src = frozen_source_constants()
    problems = []
    if listed.get("phase5_p5a_v2_seed1/" + RECORD.relative_to(REPO / "results/phase5/p5a_v2").as_posix()) != (
        RECORD_SHA256
    ):
        problems.append("artifact_manifest.json does not list the record with its pinned SHA-256")
    if (sel["position"], sel["time_index"], sel["client_id"]) != (POSITION, TIME_INDEX, CLIENT_ID):
        problems.append(f"record selects {sel['position'], sel['time_index'], sel['client_id']}")
    for k in (*IDENTITY_FIELDS, "phi_sha256", "position", "time_index", "client_id"):
        if pre_row.get(k) != sel.get(k):
            problems.append(f"record and preflight disagree on {k}")
    if (rec["representation"], rec["kind"], rec["control"]) != (
        "R3",
        "balanced_effective_delta_r8",
        "single_snapshot",
    ):
        problems.append("record is not the R3 balanced_effective_delta_r8 single_snapshot control")
    if rec["input"]["index_sha256"] != r3["index_sha256"]:
        problems.append("record and preflight disagree on index_sha256")
    if (
        norm["mode"] != "global_exact_maxabs_train"
        or not rec["normalization"]["scale_check"]["bitwise_equal"]
    ):
        problems.append("record normalization is not the bitwise-checked global_exact_maxabs_train")
    scales = {
        "record": norm["scale_global"],
        "preflight": r3["frozen_scale"],
        "p5a_inputs_source": src["FROZEN_SCALE"],
    }
    maxima = {
        "record": norm["fit"]["frozen_max_abs"],
        "preflight": r3["frozen_max_abs"],
        "p5a_inputs_source": src["FROZEN_MAX"],
    }
    if len(set(scales.values())) != 1 or len(set(maxima.values())) != 1:
        problems.append(f"frozen R3 scale or maximum disagree: {scales} {maxima}")
    if problems:
        raise InputError(f"archived identities do not agree: {problems}")
    return {
        "record": {"path": RECORD.relative_to(REPO).as_posix(), "sha256": RECORD_SHA256},
        "preflight": {
            "path": PREFLIGHT.relative_to(REPO).as_posix(),
            "sha256": PREFLIGHT_SHA256,
            "entry": f"representations[R3].evidence.records[{POSITION}]",
        },
        "selected": {
            k: sel[k] for k in (*IDENTITY_FIELDS, "phi_sha256", "position", "time_index", "client_id")
        },
        "input_root": rec["input"]["root"],
        "index_sha256": rec["input"]["index_sha256"],
        "n_train": rec["input"]["n_train"],
        "scale_global": norm["scale_global"],
        "frozen_max_abs": norm["fit"]["frozen_max_abs"],
        "scale_sources_equal": scales,
        "archived_normalized_max_abs": rec["normalized_range"]["all"]["max"],
        "archived_sig": rec["evaluation"]["predictors"]["autoencoder"]["per_snapshot"][0]["sig"],
    }


def loaded_package_modules() -> list[dict]:
    """Every imported cg_fedllm module: path, SHA-256, and its blob at HEAD and at the P5-A execution commit."""
    rows = []
    for name, mod in sorted(sys.modules.items()):
        if not (name == "cg_fedllm" or name.startswith("cg_fedllm.")):
            continue
        path = Path(mod.__file__).resolve()
        if not path.is_relative_to(REPO / "src"):
            raise InputError(f"{name} imported from {path}, not from {REPO / 'src'}")
        rel = path.relative_to(REPO / "src").as_posix()
        if rel in MODIFIED_PACKAGE_FILES:
            raise InputError(f"{rel} has uncommitted changes and must not be imported")
        repo_rel = f"src/{rel}"
        row = {
            "module": name,
            "path": repo_rel,
            "sha256": sha256_file(path),
            "worktree_blob": git("hash-object", repo_rel),
            "head_blob": git("rev-parse", f"HEAD:{repo_rel}"),
            "p5a_execution_blob": git("rev-parse", f"{P5A_EXECUTION_COMMIT}:{repo_rel}"),
        }
        if not (row["worktree_blob"] == row["head_blob"] == row["p5a_execution_blob"]):
            raise InputError(f"{repo_rel} differs from HEAD or from the P5-A execution commit: {row}")
        rows.append(row)
    return rows


def block_energies(x, coords: dict) -> list[dict]:
    """Per 8-column factor block of the patch (layout meaning from SELECTION_RULE.md): sum of squares (float64)."""
    names = ("q_proj A^T", "q_proj B", "k_proj A^T", "k_proj B")
    c0 = coords["cols"][0]
    out = []
    for i, name in enumerate(names):
        block = x[0, 0, :, i * LORA_RANK : (i + 1) * LORA_RANK].double()
        out.append(
            {
                "phi_cols": [c0 + i * LORA_RANK, c0 + (i + 1) * LORA_RANK],
                "layer": c0 // 64,
                "block": name,
                "sum_sq": float(block.square().sum()),
                "max_abs": float(block.abs().max()),
            }
        )
    return out


def main() -> int:
    for p in (OUT_PATCH, OUT_RECORD):
        if p.exists():
            raise SystemExit(f"{p} exists; never overwritten")
    if sha256_file(INPUT_DIR / "SELECTION_RULE.md") != RULE_SHA256:
        raise SystemExit("SELECTION_RULE.md differs from the pinned rule; nothing is produced")
    if sha256_file(FROZEN_DIR / "uid_common.py") != UID_COMMON_SHA256:
        raise SystemExit("frozen uid_common.py differs from its pinned SHA-256")
    import uid_common as C

    cpu = C.configure_cpu(INTRA, INTER)  # 24/24 threads, deterministic algorithms, before any tensor work
    import torch
    from cg_fedllm.compression.layout import geometry_from_dict, get_layout
    from cg_fedllm.compression.normalization import Normalizer
    from cg_fedllm.forensics.representations import balanced_effective_delta
    from cg_fedllm.tgap.snapshots import load_states, read_index
    from cg_fedllm.utils.hashing import tensor_sha256
    from safetensors.torch import load, save

    arch = archived()
    root = RUNS_ROOT / arch["input_root"]
    index_sha = sha256_file(root / "index.jsonl")
    if index_sha != arch["index_sha256"]:
        raise InputError(f"index.jsonl SHA-256 {index_sha} != archived {arch['index_sha256']}")
    records = read_index(root)
    keys = [(int(r["time_index"]), int(r["client_id"])) for r in records]
    if len(set(keys)) != len(keys):
        raise InputError("(time_index, client_id) is not unique in index.jsonl")
    rec = next(
        (r for r in records if (int(r["time_index"]), int(r["client_id"])) == (TIME_INDEX, CLIENT_ID)), None
    )
    if rec is None:
        raise InputError(f"index.jsonl has no record ({TIME_INDEX}, {CLIENT_ID})")
    bad = [k for k in IDENTITY_FIELDS if rec[k] != arch["selected"][k]]
    if bad or rec["layout_id"] != LAYOUT_ID:
        raise InputError(f"index record (7, 91) differs from the archived identities in {bad} / layout")
    train_keys = sorted(k for k in keys if k[0] in range(16))
    if len(train_keys) != arch["n_train"] or train_keys[POSITION] != (TIME_INDEX, CLIENT_ID):
        raise InputError(
            f"(7, 91) is not position {POSITION} of the {arch['n_train']}-record training population"
        )
    start, end = load_states(root, rec, verify=True)  # raises on any file-byte or adapter-content mismatch
    geom = geometry_from_dict(rec["geometry"])
    phi = get_layout(LAYOUT_ID).forward(
        balanced_effective_delta(start, end, LORA_SCALE, rank=LORA_RANK)[0], geom
    )
    phi_sha = tensor_sha256(phi)
    if (
        tuple(phi.shape) != (1, 2048, 1536)
        or phi.dtype != torch.float32
        or not bool(torch.isfinite(phi).all())
    ):
        raise InputError(f"Phi {tuple(phi.shape)} {phi.dtype} is not a finite fp32 [1, 2048, 1536] tensor")
    if phi_sha != arch["selected"]["phi_sha256"]:
        raise InputError(f"Phi tensor_sha256 {phi_sha} != archived {arch['selected']['phi_sha256']}")
    norm = Normalizer("global_exact_maxabs_train", scale_global=arch["scale_global"])
    phi_n = norm.normalize(phi)
    phi_n_max = float(phi_n.abs().max())
    if phi_n_max != arch["archived_normalized_max_abs"]:
        raise InputError(f"normalized max {phi_n_max!r} != archived {arch['archived_normalized_max_abs']!r}")
    coords = tile_rule(phi.shape[1], phi.shape[2])
    x = extract(phi, coords)
    x_n = norm.normalize(x)
    from_full = extract(phi_n, coords)
    x64 = x.double()
    tanh_ceiling = norm.denormalize(norm.normalize(x).clamp(-1.0, 1.0))
    desc, desc_n = C.describe_input(x), C.describe_input(x_n)
    base, base_n = C.baselines(x), C.baselines(x_n)
    max_n = float(x_n.abs().max())
    degeneracy = {
        "all_finite": bool(torch.isfinite(x).all()),
        "target_energy_positive": desc["energy_sum_sq"] > 0,
        "scalar_mean_mse_positive": base["shared_scalar_mean"]["mse"] > 0,
        "normalized_max_abs_below_1": max_n < 1.0,
    }
    data = save(
        {"x": x},
        metadata={
            "format": "urp_20261013_v1.input/v1",
            "protocol_id": "URP-20261013-v1",
            "space": "original (Phi values, unnormalized)",
            "source": json.dumps(
                {"t": TIME_INDEX, "client": CLIENT_ID, "phi_sha256": phi_sha}, sort_keys=True
            ),
            "coordinates": json.dumps(coords, sort_keys=True),
        },
    )
    back = load(data)["x"]
    if not (torch.equal(back, x) and back.dtype == x.dtype and tuple(back.shape) == (1, 1, TILE, TILE)):
        raise InputError("the patch does not round-trip through safetensors bit for bit")
    record = {
        "schema": "urp-input-record-v1",
        "protocol_id": "URP-20261013-v1",
        "label": "DERIVED: the registered input of URP-20261013-v1, built before any model evaluation",
        "usable": all(degeneracy.values()),
        "selection_rule": {"path": "input/SELECTION_RULE.md", "sha256": RULE_SHA256},
        "environment": {
            "cpu": cpu,
            "python": sys.version.split()[0],
            "interpreter": sys.executable,
            "cg_fedllm_modules": loaded_package_modules(),
        },
        "source": {
            **arch,
            "index_sha256_observed": index_sha,
            "index_record_identity_equal": True,
            "files_and_adapter_hashes_verified": "load_states(verify=True)",
            "construction": "balanced_effective_delta(start, end, s=2.0, rank=8)[0] -> layout layer_major_qkvo_AtB",
            "geometry": rec["geometry"],
            "phi_shape": list(phi.shape),
            "phi_dtype": str(phi.dtype),
            "phi_sha256_observed": phi_sha,
            "phi_sha256_equal_archived": True,
            "phi_sig_float64": float(phi.double().square().sum()),
            "phi_sig_equal_archived_reported_only": float(phi.double().square().sum())
            == arch["archived_sig"],
            "phi_normalized_max_abs_observed": phi_n_max,
        },
        "coordinates": {
            **coords,
            "slice": f"Phi[0, {coords['rows'][0]}:{coords['rows'][1]}, {coords['cols'][0]}:{coords['cols'][1]}]",
            "layer": coords["cols"][0] // 64,
            "blocks": "q_proj A^T, q_proj B, k_proj A^T, k_proj B (8 columns each)",
        },
        "patch": {
            "file": OUT_PATCH.relative_to(URP_DIR).as_posix(),
            "file_bytes": len(data),
            "file_sha256": sha256_bytes(data),
            "tensor_name": "x",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "file_round_trip_bit_identical": bool(torch.equal(back, x)) and back.dtype == x.dtype,
            "original": desc,
            "normalized": desc_n,
            "normalized_equals_patch_of_normalized_phi": bool(torch.equal(from_full, x_n)),
            "zero_elements": int((x == 0).sum()),
            "std_float64": float(x64.std(unbiased=False)),
            "block_energies_original": block_energies(x, coords),
        },
        "normalization": {
            "mode": "global_exact_maxabs_train",
            "scale_global": arch["scale_global"],
            "scale_float32": float(torch.tensor(arch["scale_global"], dtype=torch.float32)),
            "arithmetic": "production Normalizer: x_n = x / float32(s); denormalize(y) = y * float32(s), fp32",
            "refitted_on_patch": False,
        },
        "baselines": {"original": base, "normalized": base_n},
        "range_feasibility": {
            "normalized_max_abs": max_n,
            "margin_to_tanh_bound": 1.0 - max_n,
            "atanh_of_max": math.atanh(max_n) if max_n < 1 else None,
            "tanh_range_ceiling_original": C.metrics(tanh_ceiling, x),
        },
        "degeneracy_checks": degeneracy,
    }
    text = (json.dumps(C.json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if record["usable"]:
        with OUT_PATCH.open("xb") as fh:
            fh.write(data)
    with OUT_RECORD.open("xb") as fh:
        fh.write(text)
    print(f"input_record.json {len(text)} bytes sha256={sha256_bytes(text)}")
    if record["usable"]:
        print(f"patch_x.safetensors {len(data)} bytes sha256={sha256_bytes(data)}")
    print(json.dumps({"usable": record["usable"], "degeneracy_checks": degeneracy, "coordinates": coords}))
    return 0 if record["usable"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InputError as exc:
        print(f"INPUT NOT PRODUCED: {exc}", file=sys.stderr)
        sys.exit(3)
