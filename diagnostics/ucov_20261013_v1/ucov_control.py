"""UCOV-20261013-v1: identity, resolution, the four registered tiles, the four-arm plan, the primary policy, states.

Imports no torch at module level, so the supervisor stays torch-free. Shared scientific settings are read from the
frozen UID-20261012-v2 config (hash-pinned in ``ucov_config.json``) and must equal the expected values listed there.
The frozen UID ``criteria`` (0.01 / 0.99) are deliberately NOT read: the primary verdict is UPP-20261013-v1, read
from its pinned policy file. The model, normalization arithmetic and state files are the frozen URP-20261013-v1
functions (``urp_control``, imported unchanged and hash-checked); ledgers, start accounting and deadlines are the r1
implementation (``uid_r1_control``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

UCOV_DIR = Path(__file__).resolve().parent
DIAG_ROOT = UCOV_DIR.parent
FROZEN_DIR = DIAG_ROOT / "uid_20261012_v2"
R1_DIR = DIAG_ROOT / "uid_20261012_v2_r1"
URP_DIR = DIAG_ROOT / "urp_20261013_v1"
UPP_DIR = DIAG_ROOT / "upp_20261013_v1"
EP_DIR = DIAG_ROOT / "exploratory_progression_20261012"
for _p in (str(R1_DIR), str(FROZEN_DIR), str(URP_DIR), str(UPP_DIR)):
    if _p not in sys.path:
        sys.path.append(_p)

import uid_r1_control as K  # noqa: E402  (r1, unchanged)

PROTOCOL_ID = "UCOV-20261013-v1"
ARMS = ("U-TILE-1", "U-TILE-2", "U-TILE-3", "U-TILE-4")
REGISTERED_ARMS = ARMS
FORBIDDEN_ARMS = ("U-CHECK", "U-FIT", "R-FIT", "U-PATCH")  # the arms of closed protocols
MANIFEST_NAME = "manifest_ucov.json"
RESOLVED_NAME = "resolved_config.json"
CONFIG_NAME = "ucov_config.json"
NORMALIZATION_MODE = "global_exact_maxabs_train"
ControlError = K.ControlError

# modules the scientific child may import: import name -> path relative to UCOV_DIR (as listed in the manifest)
MODULE_PATHS = {
    "uid_common": "../uid_20261012_v2/uid_common.py",
    "uformer_ae": "../uid_20261012_v2/uformer_ae.py",
    "uformer_blocks": "../uid_20261012_v2/uformer_blocks.py",
    "uid_r1_control": "../uid_20261012_v2_r1/uid_r1_control.py",
    "uid_r1_measure": "../uid_20261012_v2_r1/uid_r1_measure.py",
    "urp_control": "../urp_20261013_v1/urp_control.py",
    "upp_classify": "../upp_20261013_v1/upp_classify.py",
    "ucov_control": "ucov_control.py",
    "ucov_child": "ucov_child.py",
}
# files hash-checked against fixed values here, independently of the manifest
PINNED_SHA256 = {
    "../uid_20261012_v2/uid_common.py": "f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230",
    "../uid_20261012_v2/uformer_ae.py": "a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606",
    "../uid_20261012_v2/uformer_blocks.py": "3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747",
    "../uid_20261012_v2_r1/uid_r1_control.py": "bdd8aff7dbccc5f5b33a6da3b6314e35beef5a53501c5c0de9a3a1a65800c074",
    "../uid_20261012_v2_r1/uid_r1_measure.py": "76b1bba28217812aff3985c19fcd1940f50cfa70e2a27ee5b2f7cab8236fc760",
    "../urp_20261013_v1/urp_control.py": "6d3dba01015dd2edf09717d6b7856f03ef0683083d872ff486f9a0eef1ffd076",
    "../urp_20261013_v1/manifest_urp.json": "1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1",
    "../upp_20261013_v1/upp_policy.json": "74b277b952e0af68dee00692cde6b95f8ed747d641843bbe98517c70d98c5770",
    "../upp_20261013_v1/upp_classify.py": "31904623223cc18b216b97bf2cfaa27734487c38a93ebe57efca5de84abecbb1",
    "../exploratory_progression_20261012/ep_classify.py": "f7efd10029e8e50071b9a5013b23eefb1d2e3707d07d05e80fda4a690e160cd1",
    "../exploratory_progression_20261012/POLICY.md": "fe14a225d16fed0373314eb89291035a2c8175bb17f37a7a03cf21c8e7fca592",
}
MECHANICS_KEYS = (
    "torch_version",
    "end_margin_s",
    "eval_reserve_s",
    "poll_interval_s",
    "interrupt_grace_s",
    "kill_wait_s",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_ucov_config(path: Path | None = None) -> dict[str, Any]:
    cfg = K.load_json(path or UCOV_DIR / CONFIG_NAME)
    if cfg.get("protocol_id") != PROTOCOL_ID or tuple(cfg.get("arms", ())) != ARMS:
        raise ControlError(f"not the UCOV-v1 config: {cfg.get('protocol_id')!r} / {cfg.get('arms')!r}")
    return cfg


def check_pinned(rel_paths: dict[str, str] | None = None) -> None:
    bad = []
    for rel, digest in (rel_paths or PINNED_SHA256).items():
        p = UCOV_DIR / rel
        actual = K.sha256_file(p) if p.is_file() else None
        if actual != digest:
            bad.append({"path": rel, "expected": digest, "actual": actual})
    if bad:
        raise ControlError(f"pinned file(s) changed or missing: {bad}")


def _equal(a: Any, b: Any) -> bool:
    """Exact equality after JSON normalisation (tuples as lists, ints and equal floats compare equal)."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def load_primary_policy(spec: dict[str, Any]) -> dict[str, Any]:
    """The UPP-20261013-v1 policy from its pinned file, through its pinned classifier's own loader."""
    pol_path, cls_path = UCOV_DIR / spec["policy_file"], UCOV_DIR / spec["classifier_file"]
    if (
        K.sha256_file(pol_path) != spec["policy_sha256"]
        or K.sha256_file(cls_path) != spec["classifier_sha256"]
    ):
        raise ControlError("the primary policy file or its classifier differs from the pinned SHA-256")
    import upp_classify as P

    if Path(P.__file__).resolve() != cls_path.resolve():
        raise ControlError(f"upp_classify imported from {P.__file__}, not {cls_path}")
    pol = P.load_policy(pol_path)
    if pol["policy_id"] != spec["policy_id"] or pol["thresholds"] != spec["thresholds_expected"]:
        raise ControlError(f"policy {pol['policy_id']} {pol['thresholds']} differs from the config")
    return pol


def tile_expected(rec: dict[str, Any], arm: str) -> dict[str, Any]:
    """The registered values of one tile, from the (hash-pinned) inputs record."""
    t = next(t for t in rec["tiles"] if t["arm"] == arm)
    p, b = t["patch"], t["baselines"]
    return {
        "usable": t["usable"],
        "degeneracy_checks": t["degeneracy_checks"],
        "slice": t["coordinates"]["slice"],
        "original_sha256": p["original"]["sha256"],
        "normalized_sha256": p["normalized"]["sha256"],
        "shape": p["shape"],
        "dtype": p["dtype"],
        "original_target_energy": p["original"]["energy_sum_sq"],
        "original_zero_prediction_mse": b["original"]["zero_prediction"]["mse"],
        "original_scalar_mean_mse": b["original"]["shared_scalar_mean"]["mse"],
        "normalized_target_energy": p["normalized"]["energy_sum_sq"],
        "normalized_zero_prediction_mse": b["normalized"]["zero_prediction"]["mse"],
        "normalized_scalar_mean_mse": b["normalized"]["shared_scalar_mean"]["mse"],
        "normalized_max_abs": t["range_feasibility"]["normalized_max_abs"],
    }


def input_problems(spec: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """The registered inputs: record, rule and every usable tile file must have their pinned SHA-256, and the record
    must name them, the frozen R3 normalization and the archived Phi. Returns (problems, per-arm expected values)."""
    problems, expected = [], {}
    for key in ("inputs_record", "selection_rule"):
        p = UCOV_DIR / spec[key]
        if not p.is_file() or K.sha256_file(p) != spec[f"{key}_sha256"]:
            problems.append(f"{spec[key]}: SHA-256 differs from the pinned value")
    if problems:
        return problems, expected
    rec = K.load_json(UCOV_DIR / spec["inputs_record"])
    if (
        rec.get("protocol_id") != PROTOCOL_ID
        or rec["selection_rule"]["sha256"] != spec["selection_rule_sha256"]
    ):
        problems.append("the inputs record is not the UCOV-v1 record of the pinned rule")
    if (
        rec["source"]["phi_sha256_observed"] != spec["phi_sha256"]
        or not rec["source"]["phi_sha256_equal_archived"]
    ):
        problems.append("the inputs record's Phi is not the archived Phi")
    norm = rec["normalization"]
    if (norm["mode"], norm["scale_global"]) != (
        spec["normalization"]["mode"],
        spec["normalization"]["scale_global"],
    ):
        problems.append(f"inputs record normalization {norm} differs from {spec['normalization']}")
    if spec["normalization"]["mode"] != NORMALIZATION_MODE or norm.get("refitted_on_tiles") is not False:
        problems.append("normalization is not the frozen, unrefitted global_exact_maxabs_train")
    if [t["arm"] for t in rec["tiles"]] != list(ARMS) or set(spec["tiles"]) != set(ARMS):
        problems.append("the inputs record or config does not list exactly the four registered arms in order")
        return problems, expected
    for arm in ARMS:
        exp = expected[arm] = tile_expected(rec, arm)
        tspec = spec["tiles"][arm]
        if exp["slice"] != tspec["slice"]:
            problems.append(f"{arm}: record slice {exp['slice']} != registered {tspec['slice']}")
        if exp["usable"]:
            p = UCOV_DIR / tspec["patch_file"]
            entry = next(t for t in rec["tiles"] if t["arm"] == arm)["patch"]
            if not p.is_file() or K.sha256_file(p) != tspec["patch_file_sha256"]:
                problems.append(f"{arm}: {tspec['patch_file']} differs from its pinned SHA-256")
            elif entry.get("file_sha256") != tspec["patch_file_sha256"]:
                problems.append(f"{arm}: the inputs record names another tile file")
        elif (UCOV_DIR / tspec["patch_file"]).exists():
            problems.append(f"{arm}: a degenerate tile has a patch file")
    return problems, expected


def resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    """The resolved scientific configuration: shared settings and the U-FIT model read from the frozen UID config and
    the r1 execution file (each hash-checked), compared with ``shared_expected``; the primary policy from its pinned
    file; the registered inputs checked as in ``input_problems``. Any difference refuses. Pure: writes nothing."""
    ref = cfg["reference"]
    frozen_raw = (UCOV_DIR / ref["frozen_config"]).read_bytes()
    uid = json.loads(frozen_raw.decode("utf-8"))
    execution_raw = (UCOV_DIR / ref["execution"]).read_bytes()
    execution = json.loads(execution_raw.decode("utf-8"))
    problems = []
    if sha256_bytes(frozen_raw) != ref["frozen_config_raw_sha256"]:
        problems.append("frozen UID config raw bytes differ from the pinned hash")
    if K.sha256_canonical(uid) != ref["frozen_config_canonical_sha256"]:
        problems.append("frozen UID config canonical content differs from the pinned hash")
    if sha256_bytes(execution_raw) != ref["execution_sha256"]:
        problems.append("execution_r1.json differs from the pinned hash")
    if K.sha256_file(FROZEN_DIR / "manifest.json") != ref["frozen_manifest_sha256"]:
        problems.append("frozen UID manifest.json differs from the pinned hash")
    if K.sha256_file(R1_DIR / "manifest_r1.json") != ref["r1_manifest_sha256"]:
        problems.append("manifest_r1.json differs from the pinned hash")
    if K.sha256_file(URP_DIR / "manifest_urp.json") != ref["urp_manifest_sha256"]:
        problems.append("manifest_urp.json differs from the pinned hash")
    child = Path(ref["child_record"])
    if not child.is_file() or K.sha256_file(child) != ref["child_record_sha256"]:
        problems.append("U-FIT child_record.json is missing or differs from the pinned hash")
    elif K.load_json(child)["model_state_sha256_init"] != ref["model_state_sha256_init"]:
        problems.append("U-FIT recorded initial state hash differs from reference.model_state_sha256_init")
    if problems:
        raise ControlError(f"reference identity check failed: {problems}")
    fit = uid["u_fit"]
    read = {
        "model_init_seed": uid["model_init_seed"],
        "optimizer": fit["optimizer"],
        "max_updates": fit["max_updates"],
        "max_seconds": fit["max_seconds"],
        "cpu_config": {k: uid["cpu_config"][k] for k in ("intra_op_threads", "inter_op_threads")},
        "torch_version": execution["torch_version"],
        "input_shape": uid["input"]["shape"],
    }
    exp = cfg["shared_expected"]
    problems = [f"{k}: frozen {v!r} != expected {exp[k]!r}" for k, v in read.items() if not _equal(v, exp[k])]
    if any(int(s) != 1 for s in fit["model"]["down_strides"]):
        problems.append(f"U-FIT strides {fit['model']['down_strides']} are not all 1")
    in_problems, expected = input_problems(cfg["inputs"])
    problems += in_problems
    problems += [
        f"{a}: shape {e['shape']} is not {uid['input']['shape']}"
        for a, e in expected.items()
        if not _equal(e["shape"], uid["input"]["shape"])
    ]
    if problems:
        raise ControlError(f"shared settings or inputs differ from the expected values: {problems}")
    pol = load_primary_policy(cfg["primary_policy"])
    return {
        "protocol_id": PROTOCOL_ID,
        "arms": list(ARMS),
        "variant": cfg["variant"],
        "reference": ref,
        "inputs": cfg["inputs"],
        "inputs_expected": expected,
        "model_init_seed": int(uid["model_init_seed"]),
        "u_fit": {"model": fit["model"]},  # read unchanged by the frozen uid_common.build_model(cfg, "U-FIT")
        "optimizer": fit["optimizer"],
        "max_updates": int(fit["max_updates"]),
        "max_seconds": float(fit["max_seconds"]),
        "cpu": read["cpu_config"],
        "primary_policy": {
            **cfg["primary_policy"],
            "thresholds": pol["thresholds"],
            "basis": pol["basis"],
            "loaded_policy_sha256": pol["policy_sha256"],
        },
        "secondary_references": {
            "historical_strict_criteria": uid["criteria"],  # reported only; never the primary verdict
            "ep_policy_id": "EP-20261012-v1",
        },
        "limits": {
            "max_training_starts": int(cfg["limits"]["max_training_starts"]),
            "max_invocations": int(cfg["limits"]["max_invocations"]),
            "aggregate_seconds": float(cfg["limits"]["aggregate_seconds"]),
            "rss_cap_bytes": int(uid["global_limits"]["max_rss_bytes"]),
            "rss_available_fraction": K.REGISTERED_RSS_AVAILABLE_FRACTION,
            "evidence_cap_bytes": K.REGISTERED_EVIDENCE_CAP_BYTES,
            "curve_every": K.REGISTERED_CURVE_EVERY,
        },
        "mechanics": {k: execution[k] for k in MECHANICS_KEYS},
        "execution": cfg["execution"],
    }


def load_resolved() -> dict[str, Any]:
    """The frozen ``resolved_config.json``; it must equal a fresh resolution of the pinned sources."""
    resolved = K.load_json(UCOV_DIR / RESOLVED_NAME)
    if not _equal(resolved, resolve(load_ucov_config())):
        raise ControlError("resolved_config.json differs from a fresh resolution of the pinned sources")
    if resolved.get("protocol_id") != PROTOCOL_ID:
        raise ControlError("resolved_config.json is not the UCOV-v1 configuration")
    return resolved


def registered_plan(resolved: dict[str, Any]) -> dict[str, Any]:
    """The only scientific plan: the four tile arms in registered order, one start each, no prerequisite."""
    lim, mech, ex = resolved["limits"], resolved["mechanics"], resolved["execution"]
    plan = {
        "mode": "scientific",
        "arms": [
            {
                "arm": arm,
                "max_updates": resolved["max_updates"],
                "max_seconds": resolved["max_seconds"],
                "usable": resolved["inputs_expected"][arm]["usable"],
            }
            for arm in ARMS
        ],
        "prerequisite": {},
        "max_training_starts": lim["max_training_starts"],
        "max_invocations": lim["max_invocations"],
        "aggregate_seconds": lim["aggregate_seconds"],
        "rss_cap_bytes": lim["rss_cap_bytes"],
        "rss_available_fraction": lim["rss_available_fraction"],
        "evidence_cap_bytes": lim["evidence_cap_bytes"],
        "curve_every": lim["curve_every"],
        "cpu": dict(resolved["cpu"]),
        "run_root": ex["run_root"],
        "ledger_mirror": ex["ledger_mirror"],
        "python_executable": ex["python_executable"],
        "primary_policy_id": resolved["primary_policy"]["policy_id"],
        "primary_policy_sha256": resolved["primary_policy"]["policy_sha256"],
        **mech,
    }
    check_plan(plan)
    return plan


def check_plan(plan: dict[str, Any]) -> None:
    """Refuses a scientific plan that is not exactly the four registered arms in order (one start each), and any
    plan in any mode that names an arm of a closed protocol."""
    arms = [a["arm"] for a in plan["arms"]]
    names = arms + [a.get("model_arm", a["arm"]) for a in plan["arms"]]
    if any(n in FORBIDDEN_ARMS for n in names):
        raise ControlError(f"plan names an arm of a closed protocol: {arms}")
    if len(set(arms)) != len(arms):
        raise ControlError(f"plan repeats an arm: {arms}")
    if plan["mode"] == "scientific":
        if arms != list(ARMS) or plan["max_training_starts"] != len(ARMS) or plan["max_invocations"] != 1:
            raise ControlError(f"the scientific plan must be the four tile arms in order, got {arms}")
        if plan["prerequisite"]:
            raise ControlError("the scientific plan has no prerequisite arm")
    K.validate_plan(plan)


def expected_module_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    by_path = {e["path"]: e["sha256"] for e in manifest["files"]}
    missing = [p for p in MODULE_PATHS.values() if p not in by_path]
    if missing:
        raise ControlError(f"{MANIFEST_NAME} does not list runtime module(s) {missing}")
    return {name: by_path[path] for name, path in MODULE_PATHS.items()}


def verify_package(actual_manifest_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identity of UCOV-v1 bound to the frozen manifest hash given on the command line: every manifest entry, every
    pinned file and a fresh resolution of the shared settings, the policy and the inputs must agree."""
    manifest_path = UCOV_DIR / MANIFEST_NAME
    actual = K.sha256_file(manifest_path) if manifest_path.is_file() else None
    problems = []
    if actual != actual_manifest_sha256:
        problems.append(f"{MANIFEST_NAME} SHA-256 {actual} is not the frozen {actual_manifest_sha256}")
    else:
        manifest = K.load_json(manifest_path)
        if manifest.get("protocol_id") != PROTOCOL_ID:
            problems.append("manifest protocol_id is not UCOV-20261013-v1")
        problems += [f"file mismatch: {b}" for b in K.verify_entries(manifest, UCOV_DIR)]
        for e in manifest.get("external_bound_files", []):
            p = Path(e["path"])
            if not p.is_file() or K.sha256_file(p) != e["sha256"]:
                problems.append(f"external bound file mismatch: {e['path']}")
    if problems:
        raise ControlError(f"package identity check failed: {problems}")
    check_pinned()
    resolved = load_resolved()
    ident = {
        "ucov_manifest_sha256": actual,
        "resolved_config_sha256": K.sha256_file(UCOV_DIR / RESOLVED_NAME),
        "ucov_config_sha256": K.sha256_file(UCOV_DIR / CONFIG_NAME),
        "primary_policy": {
            k: resolved["primary_policy"][k] for k in ("policy_id", "policy_sha256", "classifier_sha256")
        },
        "inputs": {k: v for k, v in resolved["inputs"].items() if k.endswith("sha256")},
        "tile_files": {a: t["patch_file_sha256"] for a, t in resolved["inputs"]["tiles"].items()},
        "reference": {k: v for k, v in resolved["reference"].items() if k.endswith("sha256")},
    }
    return ident, resolved


# ---- model, input and states: the frozen URP-v1 functions (torch imported lazily by them) ---------------------
def urp():
    """The frozen URP-v1 control module (hash-pinned): build_model, normalize/denormalize, save_state, load_state."""
    import urp_control as R

    if Path(R.__file__).resolve() != (URP_DIR / "urp_control.py").resolve():
        raise ControlError(f"urp_control imported from {R.__file__}")
    return R


def build_model(cfg: dict[str, Any]):
    """The frozen ``uid_common.build_model(cfg, "U-FIT")`` through the frozen URP binding, unchanged."""
    return urp().build_model(cfg)


def load_tile(cfg: dict[str, Any], arm: str):
    """One registered tile (original space) and its normalized copy. The bytes are read once, hashed, and
    deserialized from those same bytes; the file must hold exactly one fp32 [1, 1, 32, 32] tensor ``x``."""
    import torch
    from safetensors.torch import load

    spec = cfg["inputs"]["tiles"][arm]
    data = (UCOV_DIR / spec["patch_file"]).read_bytes()
    digest = sha256_bytes(data)
    if digest != spec["patch_file_sha256"]:
        raise RuntimeError(f"{arm}: tile file SHA-256 {digest} != {spec['patch_file_sha256']}")
    tensors = load(data)
    name = cfg["inputs"]["tensor_name"]
    if set(tensors) != {name}:
        raise RuntimeError(f"{arm}: tile file holds {sorted(tensors)}, expected exactly [{name!r}]")
    x = tensors[name].contiguous()
    exp = cfg["inputs_expected"][arm]
    if x.dtype != torch.float32 or list(x.shape) != list(exp["shape"]):
        raise RuntimeError(f"{arm}: tile tensor {x.dtype} {list(x.shape)} is not fp32 {exp['shape']}")
    scale = cfg["inputs"]["normalization"]["scale_global"]
    return x, urp().normalize(x, scale), digest


def input_checks(rec: dict[str, Any], exp: dict[str, Any], max_abs_n: float) -> dict[str, Any]:
    """Recorded input and baselines in both spaces against the registered values; any difference refuses the arm
    before the optimizer exists."""
    i, b = rec["input"], rec["baselines"]
    checks = {
        "original_sha256": (i["original"]["sha256"], exp["original_sha256"]),
        "normalized_sha256": (i["normalized"]["sha256"], exp["normalized_sha256"]),
        "shape": (i["original"]["shape"], list(exp["shape"])),
        "dtype": (i["original"]["dtype"], exp["dtype"]),
        "original_target_energy": (i["original"]["energy_sum_sq"], exp["original_target_energy"]),
        "original_zero_prediction_mse": (
            b["original"]["zero_prediction"]["mse"],
            exp["original_zero_prediction_mse"],
        ),
        "original_scalar_mean_mse": (
            b["original"]["shared_scalar_mean"]["mse"],
            exp["original_scalar_mean_mse"],
        ),
        "normalized_target_energy": (i["normalized"]["energy_sum_sq"], exp["normalized_target_energy"]),
        "normalized_zero_prediction_mse": (
            b["normalized"]["zero_prediction"]["mse"],
            exp["normalized_zero_prediction_mse"],
        ),
        "normalized_scalar_mean_mse": (
            b["normalized"]["shared_scalar_mean"]["mse"],
            exp["normalized_scalar_mean_mse"],
        ),
        "normalized_max_abs": (max_abs_n, exp["normalized_max_abs"]),
    }
    out: dict[str, Any] = {k: {"observed": a, "expected": e, "equal": a == e} for k, (a, e) in checks.items()}
    out["normalized_max_abs_below_1"] = max_abs_n < 1.0
    out["all_equal"] = all(v["equal"] for v in out.values() if isinstance(v, dict)) and max_abs_n < 1.0
    return out
