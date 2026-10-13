"""ENGINEERING FIXTURE: write ``checks/fixture_config.json`` once, from the four toy tiles (never a registered tile).

The fixture config has the same structure as ``resolved_config.json``: four arms, per-arm tiles with their recorded
values, the UPP-20261013-v1 primary policy block (the real pinned policy, so verdicts in the fixture runs use the
production policy code), a small Uformer, seed 7 and a fixture scale. ``python -B checks/build_fixture_config.py``.
"""

import hashlib
import json
import sys
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
UCOV = CHECKS.parent
sys.path.insert(0, str(UCOV))

import ucov_control as U  # noqa: E402

OUT = CHECKS / "fixture_config.json"
SCALE = 0.0125
MODEL = {
    "input_hw": [16, 16],
    "in_channels": 1,
    "stem_kernel": 3,
    "stem_channels": 1,
    "down_channels": [2, 4, 8],
    "down_strides": [1, 1, 2],
    "final_kernel": 7,
    "core_layers": 1,
    "core_depth": 2,
    "num_heads": 2,
    "win_size": 4,
    "mlp_ratio": 2.0,
    "qkv_bias": True,
    "shift_flag": True,
}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; never overwritten")
    import torch
    import uid_common as C
    from safetensors.torch import load

    tiles, expected = {}, {}
    for k, arm in enumerate(U.ARMS, start=1):
        rel = f"checks/fixture_tile_{k}.safetensors"
        data = (UCOV / rel).read_bytes()
        x = load(data)["x"]
        x_n = U.urp().normalize(x, SCALE)
        d, dn = C.describe_input(x), C.describe_input(x_n)
        b, bn = C.baselines(x), C.baselines(x_n)
        tiles[arm] = {
            "patch_file": rel,
            "patch_file_sha256": hashlib.sha256(data).hexdigest(),
            "slice": f"fixture {k}",
        }
        checks = {  # the same four conditions as the registered inputs record, computed on the toy tile
            "all_finite": bool(torch.isfinite(x).all()),
            "target_energy_positive": d["energy_sum_sq"] > 0,
            "scalar_mean_mse_positive": b["shared_scalar_mean"]["mse"] > 0,
            "normalized_max_abs_below_1": float(x_n.abs().max()) < 1.0,
        }
        expected[arm] = {
            "usable": all(checks.values()),
            "degeneracy_checks": checks,
            "slice": f"fixture {k}",
            "original_sha256": d["sha256"],
            "normalized_sha256": dn["sha256"],
            "shape": d["shape"],
            "dtype": d["dtype"],
            "original_target_energy": d["energy_sum_sq"],
            "original_zero_prediction_mse": b["zero_prediction"]["mse"],
            "original_scalar_mean_mse": b["shared_scalar_mean"]["mse"],
            "normalized_target_energy": dn["energy_sum_sq"],
            "normalized_zero_prediction_mse": bn["zero_prediction"]["mse"],
            "normalized_scalar_mean_mse": bn["shared_scalar_mean"]["mse"],
            "normalized_max_abs": float(x_n.abs().max()),
        }
    pol_spec = U.load_ucov_config()["primary_policy"]
    pol = U.load_primary_policy(pol_spec)
    cfg = {
        "protocol_id": "UCOV-V1-ENGINEERING-FIXTURE",
        "note": "ENGINEERING FIXTURE ONLY: model seed 7, four unregistered seeded 16x16 toy tiles, a fixture scale and "
        "a small Uformer AE. Optimizer updates are made only on these fixtures. The primary policy is the real pinned "
        "UPP-20261013-v1.",
        "arms": list(U.ARMS),
        "variant": "uformer_uid_20261012_v2_u_fit (fixture geometry)",
        "model_init_seed": 7,
        "reference": {"model_state_sha256_init": "fixture: not compared"},
        "inputs": {
            "tensor_name": "x",
            "normalization": {"mode": "global_exact_maxabs_train", "scale_global": SCALE},
            "tiles": tiles,
        },
        "inputs_expected": expected,
        "primary_policy": {
            **pol_spec,
            "thresholds": pol["thresholds"],
            "basis": pol["basis"],
            "loaded_policy_sha256": pol["policy_sha256"],
        },
        "u_fit": {"model": MODEL},
        "optimizer": {"name": "Adam", "lr": 0.001, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0},
        "max_updates": 6,
        "max_seconds": 120.0,
    }
    data = (json.dumps(cfg, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with OUT.open("xb") as fh:
        fh.write(data)
    print(f"wrote {OUT} ({len(data)} bytes) sha256={hashlib.sha256(data).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
