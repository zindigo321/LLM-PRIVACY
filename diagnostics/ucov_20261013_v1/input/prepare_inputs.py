"""UCOV-20261013-v1: build the four registered tiles from the archived R3 training snapshot (CPU, no model).

``python -B input/prepare_inputs.py`` (from the UCOV directory). Runs once; refuses to overwrite any output.

The procedure is that of the closed URP-20261013-v1 builder (``diagnostics/urp_20261013_v1/input/prepare_input.py``,
hash-pinned below). Its provenance functions are imported from that file unchanged: ``archived``,
``loaded_package_modules``, ``block_energies`` and ``extract``. Only the positional rule differs: the four
quarter-grid tiles of ``SELECTION_RULE.md``, whose SHA-256 is pinned here. For each tile: its descriptions and
baselines in both spaces, range feasibility and degeneracy. A degenerate tile gets a record and no patch file. A
provenance failure produces nothing.

Writes ``input/tile_<k>.safetensors`` (usable tiles) and ``input/inputs_record.json`` (LF bytes).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent
UCOV_DIR = INPUT_DIR.parent
REPO = UCOV_DIR.parents[1]
URP_INPUT = REPO / "diagnostics" / "urp_20261013_v1" / "input" / "prepare_input.py"
URP_INPUT_SHA256 = "4ccef2ca9dc7c871eb7b0d9046460d93d958f66986426a7ee8a3dd93b315604b"
RULE_SHA256 = "ddd58be9ec8ecae473a2ed48be749e8826f685470abf5c78d15d64f58bfe36f2"
PROTOCOL_ID = "UCOV-20261013-v1"
TILE = 32
# the quarter-grid rule of SELECTION_RULE.md: (arm, tile row, tile column), in the registered order
ARMS = (("U-TILE-1", 16, 12), ("U-TILE-2", 16, 36), ("U-TILE-3", 48, 12), ("U-TILE-4", 48, 36))
OUT_RECORD = INPUT_DIR / "inputs_record.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_urp_builder():
    """The closed URP-v1 builder module, after its bytes match the pinned hash (it is imported, never edited)."""
    if sha256_bytes(URP_INPUT.read_bytes()) != URP_INPUT_SHA256:
        raise SystemExit("the URP-v1 prepare_input.py differs from its pinned SHA-256")
    spec = importlib.util.spec_from_file_location("urp_prepare_input", URP_INPUT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # puts the repository src and the frozen UID directory on sys.path
    return mod


def quarter_rule(height: int, width: int, tile: int = TILE) -> list[dict]:
    """The positional rule: the four quarter points of the aligned non-overlapping tile grid, in registered order."""
    if height % tile or width % tile:
        raise ValueError(f"{height} x {width} is not divisible by the tile {tile}")
    rows, cols = height // tile, width // tile
    points = [(r, c) for r in (rows // 4, 3 * rows // 4) for c in (cols // 4, 3 * cols // 4)]
    if [(a[1], a[2]) for a in ARMS] != points:
        raise ValueError(f"registered arms {ARMS} are not the quarter points {points}")
    return [
        {
            "arm": arm,
            "tile": tile,
            "grid": [rows, cols],
            "tile_index": [ti, tj],
            "rows": [ti * tile, ti * tile + tile],
            "cols": [tj * tile, tj * tile + tile],
        }
        for arm, ti, tj in ARMS
    ]


def describe_tile(x, norm, coords: dict, R, C, torch) -> tuple[dict, bytes | None]:
    """One tile's record and, if it is usable, its safetensors bytes (round-trip checked)."""
    from safetensors.torch import load, save

    x_n = norm.normalize(x)
    desc, desc_n = C.describe_input(x), C.describe_input(x_n)
    base, base_n = C.baselines(x), C.baselines(x_n)
    max_n = float(x_n.abs().max())
    finite = bool(torch.isfinite(x).all())
    degeneracy = {
        "all_finite": finite,
        "target_energy_positive": desc["energy_sum_sq"] > 0,
        "scalar_mean_mse_positive": base["shared_scalar_mean"]["mse"] > 0,
        "normalized_max_abs_below_1": max_n < 1.0,
    }
    usable = all(degeneracy.values())
    tanh_ceiling = norm.denormalize(norm.normalize(x).clamp(-1.0, 1.0))
    rec = {
        "arm": coords["arm"],
        "usable": usable,
        "coordinates": {
            **{k: coords[k] for k in ("tile", "grid", "tile_index", "rows", "cols")},
            "slice": f"Phi[0, {coords['rows'][0]}:{coords['rows'][1]}, {coords['cols'][0]}:{coords['cols'][1]}]",
            "layer": coords["cols"][0] // 64,
            "blocks": "q_proj A^T, q_proj B, k_proj A^T, k_proj B (8 columns each)",
        },
        "patch": {
            "tensor_name": "x",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "original": desc,
            "normalized": desc_n,
            "zero_elements": int((x == 0).sum()),
            "std_float64": float(x.double().std(unbiased=False)),
            "block_energies_original": R.block_energies(x, coords),
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
    if not usable:
        return rec, None
    data = save(
        {"x": x},
        metadata={
            "format": "ucov_20261013_v1.input/v1",
            "protocol_id": PROTOCOL_ID,
            "arm": coords["arm"],
            "space": "original (Phi values, unnormalized)",
            "coordinates": json.dumps({k: coords[k] for k in ("tile_index", "rows", "cols")}, sort_keys=True),
        },
    )
    back = load(data)["x"]
    if not (torch.equal(back, x) and back.dtype == x.dtype and tuple(back.shape) == (1, 1, TILE, TILE)):
        raise RuntimeError(f"{coords['arm']}: the tile does not round-trip through safetensors bit for bit")
    rec["patch"].update(
        file=f"input/tile_{coords['arm'][-1]}.safetensors",
        file_bytes=len(data),
        file_sha256=sha256_bytes(data),
        file_round_trip_bit_identical=True,
    )
    return rec, data


def main() -> int:
    outs = [OUT_RECORD] + [INPUT_DIR / f"tile_{a[0][-1]}.safetensors" for a in ARMS]
    for p in outs:
        if p.exists():
            raise SystemExit(f"{p} exists; never overwritten")
    if sha256_bytes((INPUT_DIR / "SELECTION_RULE.md").read_bytes()) != RULE_SHA256:
        raise SystemExit("SELECTION_RULE.md differs from the pinned rule; nothing is produced")
    R = load_urp_builder()  # pins and checks frozen uid_common.py itself before use below
    if sha256_bytes((R.FROZEN_DIR / "uid_common.py").read_bytes()) != R.UID_COMMON_SHA256:
        raise SystemExit("frozen uid_common.py differs from its pinned SHA-256")
    import uid_common as C

    cpu = C.configure_cpu(R.INTRA, R.INTER)  # 24/24 threads, deterministic algorithms, before any tensor work
    import torch
    from cg_fedllm.compression.layout import geometry_from_dict, get_layout
    from cg_fedllm.compression.normalization import Normalizer
    from cg_fedllm.forensics.representations import balanced_effective_delta
    from cg_fedllm.tgap.snapshots import load_states, read_index
    from cg_fedllm.utils.hashing import tensor_sha256

    arch = (
        R.archived()
    )  # the archived identities of the source snapshot (two committed records, cross-checked)
    root = R.RUNS_ROOT / arch["input_root"]
    index_sha = sha256_bytes((root / "index.jsonl").read_bytes())
    if index_sha != arch["index_sha256"]:
        raise R.InputError(f"index.jsonl SHA-256 {index_sha} != archived {arch['index_sha256']}")
    records = read_index(root)
    keys = [(int(r["time_index"]), int(r["client_id"])) for r in records]
    if len(set(keys)) != len(keys):
        raise R.InputError("(time_index, client_id) is not unique in index.jsonl")
    rec = next(
        (r for r in records if (int(r["time_index"]), int(r["client_id"])) == (R.TIME_INDEX, R.CLIENT_ID)),
        None,
    )
    if rec is None:
        raise R.InputError(f"index.jsonl has no record ({R.TIME_INDEX}, {R.CLIENT_ID})")
    bad = [k for k in R.IDENTITY_FIELDS if rec[k] != arch["selected"][k]]
    if bad or rec["layout_id"] != R.LAYOUT_ID:
        raise R.InputError(f"index record (7, 91) differs from the archived identities in {bad} / layout")
    train_keys = sorted(k for k in keys if k[0] in range(16))
    if len(train_keys) != arch["n_train"] or train_keys[R.POSITION] != (R.TIME_INDEX, R.CLIENT_ID):
        raise R.InputError(f"(7, 91) is not position {R.POSITION} of the {arch['n_train']}-record population")
    start, end = load_states(root, rec, verify=True)  # raises on any file-byte or adapter-content mismatch
    geom = geometry_from_dict(rec["geometry"])
    phi = get_layout(R.LAYOUT_ID).forward(
        balanced_effective_delta(start, end, R.LORA_SCALE, rank=R.LORA_RANK)[0], geom
    )
    phi_sha = tensor_sha256(phi)
    if (
        tuple(phi.shape) != (1, 2048, 1536)
        or phi.dtype != torch.float32
        or not bool(torch.isfinite(phi).all())
    ):
        raise R.InputError(f"Phi {tuple(phi.shape)} {phi.dtype} is not a finite fp32 [1, 2048, 1536] tensor")
    if phi_sha != arch["selected"]["phi_sha256"]:
        raise R.InputError(f"Phi tensor_sha256 {phi_sha} != archived {arch['selected']['phi_sha256']}")
    norm = Normalizer("global_exact_maxabs_train", scale_global=arch["scale_global"])
    phi_n = norm.normalize(phi)
    phi_n_max = float(phi_n.abs().max())
    if phi_n_max != arch["archived_normalized_max_abs"]:
        raise R.InputError(
            f"normalized max {phi_n_max!r} != archived {arch['archived_normalized_max_abs']!r}"
        )
    tiles, payloads = [], {}
    for coords in quarter_rule(phi.shape[1], phi.shape[2]):
        x = R.extract(phi, coords)
        t_rec, data = describe_tile(x, norm, coords, R, C, torch)
        t_rec["normalized_equals_tile_of_normalized_phi"] = bool(
            torch.equal(R.extract(phi_n, coords), norm.normalize(x))
        )
        tiles.append(t_rec)
        if data is not None:
            payloads[coords["arm"]] = data
    record = {
        "schema": "ucov-inputs-record-v1",
        "protocol_id": PROTOCOL_ID,
        "label": "DERIVED: the registered inputs of UCOV-20261013-v1, built before any model evaluation",
        "selection_rule": {"path": "input/SELECTION_RULE.md", "sha256": RULE_SHA256},
        "builder_reuses": {
            "path": "diagnostics/urp_20261013_v1/input/prepare_input.py",
            "sha256": URP_INPUT_SHA256,
        },
        "environment": {
            "cpu": cpu,
            "python": sys.version.split()[0],
            "interpreter": sys.executable,
            "cg_fedllm_modules": R.loaded_package_modules(),
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
        "normalization": {
            "mode": "global_exact_maxabs_train",
            "scale_global": arch["scale_global"],
            "scale_float32": float(torch.tensor(arch["scale_global"], dtype=torch.float32)),
            "arithmetic": "production Normalizer: x_n = x / float32(s); denormalize(y) = y * float32(s), fp32",
            "refitted_on_tiles": False,
        },
        "tiles": tiles,
        "usable_arms": [t["arm"] for t in tiles if t["usable"]],
        "degenerate_arms": [t["arm"] for t in tiles if not t["usable"]],
    }
    text = (json.dumps(C.json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    for arm, data in payloads.items():
        with (INPUT_DIR / f"tile_{arm[-1]}.safetensors").open("xb") as fh:
            fh.write(data)
        print(f"tile_{arm[-1]}.safetensors {len(data)} bytes sha256={sha256_bytes(data)}")
    with OUT_RECORD.open("xb") as fh:
        fh.write(text)
    print(f"inputs_record.json {len(text)} bytes sha256={sha256_bytes(text)}")
    print(
        json.dumps(
            {t["arm"]: {"usable": t["usable"], **t["degeneracy_checks"]} for t in tiles}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:  # InputError of the URP builder is a RuntimeError
        print(f"INPUTS NOT PRODUCED: {exc}", file=sys.stderr)
        sys.exit(3)
