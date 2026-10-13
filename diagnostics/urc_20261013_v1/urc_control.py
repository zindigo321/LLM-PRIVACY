"""URC-20261013-v1: identity, resolution of the shared settings, the one-arm plan and the state files.

Imports no torch at module level, so the supervisor stays torch-free. Every shared scientific setting is read from the
frozen UID-20261012-v2 config (hash-pinned in ``urc_config.json``) and must equal the expected value listed there; the
execution mechanics come from the r1 ``execution_r1.json`` (hash-pinned) except the run root and ledger mirror, which
are this package's own. Ledger, start accounting and deadlines are the r1 implementation (``uid_r1_control``), imported
unchanged and hash-checked; only the one-arm plan, the package identity and the model build are new here.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

URC_DIR = Path(__file__).resolve().parent
DIAG_ROOT = URC_DIR.parent
FROZEN_DIR = DIAG_ROOT / "uid_20261012_v2"
R1_DIR = DIAG_ROOT / "uid_20261012_v2_r1"
EP_DIR = DIAG_ROOT / "exploratory_progression_20261012"
for _p in (str(R1_DIR), str(FROZEN_DIR)):  # r1 control/measure, frozen uid_common
    if _p not in sys.path:
        sys.path.append(_p)

import uid_r1_control as K  # noqa: E402  (r1, unchanged)

PROTOCOL_ID = "URC-20261013-v1"
ARM = "R-FIT"
REGISTERED_ARMS = (ARM,)
FORBIDDEN_ARMS = ("U-CHECK", "U-FIT")
MANIFEST_NAME = "manifest_urc.json"
RESOLVED_NAME = "resolved_config.json"
CONFIG_NAME = "urc_config.json"
ControlError = K.ControlError

# modules the scientific child may import: import name -> path relative to URC_DIR (as listed in the manifest)
MODULE_PATHS = {
    "uid_common": "../uid_20261012_v2/uid_common.py",
    "uid_r1_control": "../uid_20261012_v2_r1/uid_r1_control.py",
    "uid_r1_measure": "../uid_20261012_v2_r1/uid_r1_measure.py",
    "urc_control": "urc_control.py",
    "resnet3_s1": "resnet3_s1.py",
    "urc_child": "urc_child.py",
}
# files hash-checked against fixed values here, independently of the manifest
PINNED_SHA256 = {
    "../uid_20261012_v2/uid_common.py": "f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230",
    "../uid_20261012_v2_r1/uid_r1_control.py": "bdd8aff7dbccc5f5b33a6da3b6314e35beef5a53501c5c0de9a3a1a65800c074",
    "../uid_20261012_v2_r1/uid_r1_measure.py": "76b1bba28217812aff3985c19fcd1940f50cfa70e2a27ee5b2f7cab8236fc760",
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


def load_urc_config(path: Path | None = None) -> dict[str, Any]:
    cfg = K.load_json(path or URC_DIR / CONFIG_NAME)
    if cfg.get("protocol_id") != PROTOCOL_ID or cfg.get("arm") != ARM:
        raise ControlError(f"not the URC-v1 config: {cfg.get('protocol_id')!r} / {cfg.get('arm')!r}")
    return cfg


def check_pinned(rel_paths: dict[str, str] | None = None) -> None:
    bad = []
    for rel, digest in (rel_paths or PINNED_SHA256).items():
        p = URC_DIR / rel
        actual = K.sha256_file(p) if p.is_file() else None
        if actual != digest:
            bad.append({"path": rel, "expected": digest, "actual": actual})
    if bad:
        raise ControlError(f"pinned file(s) changed or missing: {bad}")


def _equal(a: Any, b: Any) -> bool:
    """Exact equality after JSON normalisation (tuples as lists, ints and equal floats compare equal)."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def resolve(urc: dict[str, Any]) -> dict[str, Any]:
    """The resolved scientific configuration: shared settings read from the frozen UID config and the r1 execution
    file (each hash-checked), compared with ``shared_expected``; any difference refuses. Pure: writes nothing."""
    ref = urc["reference"]
    frozen_raw = (URC_DIR / ref["frozen_config"]).read_bytes()
    uid = json.loads(frozen_raw.decode("utf-8"))
    execution_raw = (URC_DIR / ref["execution"]).read_bytes()
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
    exp = urc["shared_expected"]
    problems = [f"{k}: frozen {v!r} != expected {exp[k]!r}" for k, v in read.items() if not _equal(v, exp[k])]
    model = urc["model"]
    if [1, model["in_channels"], *model["input_hw"]] != list(uid["input"]["shape"]):
        problems.append(f"model geometry {model} does not take the U-FIT input shape {uid['input']['shape']}")
    for k in ("in_channels", "stem_kernel", "stem_channels", "down_channels", "final_kernel"):
        if not _equal(model[k], fit["model"][k]):
            problems.append(f"sampling-layer field {k}: {model[k]!r} != U-FIT {fit['model'][k]!r}")
    if any(int(s) != 1 for s in fit["model"]["down_strides"]):
        problems.append(f"U-FIT strides {fit['model']['down_strides']} are not all 1")
    if problems:
        raise ControlError(f"shared settings differ from the expected values: {problems}")
    mechanics = {k: execution[k] for k in MECHANICS_KEYS}
    return {
        "protocol_id": PROTOCOL_ID,
        "arm": ARM,
        "variant": urc["variant"],
        "reference": ref,
        "input": uid["input"],  # passed unchanged to the frozen uid_common.make_input
        "model_init_seed": int(uid["model_init_seed"]),
        "optimizer": fit["optimizer"],
        "max_updates": int(fit["max_updates"]),
        "max_seconds": float(fit["max_seconds"]),
        "cpu": read["cpu_config"],
        "criteria": uid["criteria"],  # strict near-identity criteria, unchanged
        "model": model,
        "expected": exp,
        "limits": {
            "max_training_starts": int(urc["limits"]["max_training_starts"]),
            "max_invocations": int(urc["limits"]["max_invocations"]),
            "aggregate_seconds": float(urc["limits"]["aggregate_seconds"]),
            "rss_cap_bytes": int(uid["global_limits"]["max_rss_bytes"]),
            "rss_available_fraction": K.REGISTERED_RSS_AVAILABLE_FRACTION,
            "evidence_cap_bytes": K.REGISTERED_EVIDENCE_CAP_BYTES,
            "curve_every": K.REGISTERED_CURVE_EVERY,
        },
        "mechanics": mechanics,
        "execution": urc["execution"],
    }


def load_resolved() -> dict[str, Any]:
    """The frozen ``resolved_config.json``; it must equal a fresh resolution from the pinned sources."""
    resolved = K.load_json(URC_DIR / RESOLVED_NAME)
    fresh = resolve(load_urc_config())
    if not _equal(resolved, fresh):
        raise ControlError("resolved_config.json differs from a fresh resolution of the pinned sources")
    if resolved.get("protocol_id") != PROTOCOL_ID:
        raise ControlError("resolved_config.json is not the URC-v1 configuration")
    return resolved


def registered_plan(resolved: dict[str, Any]) -> dict[str, Any]:
    """The only scientific plan: the single arm R-FIT. U-CHECK and U-FIT cannot appear in it."""
    lim, mech, ex = resolved["limits"], resolved["mechanics"], resolved["execution"]
    plan = {
        "mode": "scientific",
        "arms": [
            {"arm": ARM, "max_updates": resolved["max_updates"], "max_seconds": resolved["max_seconds"]}
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
        **mech,
    }
    check_plan(plan)
    return plan


def check_plan(plan: dict[str, Any]) -> None:
    """Refuses any plan whose arms are not exactly [R-FIT] (scientific) or that names a UID arm (any mode)."""
    arms = [a["arm"] for a in plan["arms"]]
    names = arms + [a.get("model_arm", a["arm"]) for a in plan["arms"]]
    if any(n in FORBIDDEN_ARMS for n in names):
        raise ControlError(f"plan names a UID-20261012-v2 arm: {arms}")
    if plan["mode"] == "scientific" and (arms != [ARM] or plan["max_training_starts"] != 1):
        raise ControlError(f"the scientific plan must be exactly one R-FIT start, got {arms}")
    if plan["mode"] == "scientific" and plan["prerequisite"]:
        raise ControlError("the scientific plan has no prerequisite arm")
    K.validate_plan(plan)


def expected_module_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    by_path = {e["path"]: e["sha256"] for e in manifest["files"]}
    missing = [p for p in MODULE_PATHS.values() if p not in by_path]
    if missing:
        raise ControlError(f"{MANIFEST_NAME} does not list runtime module(s) {missing}")
    return {name: by_path[path] for name, path in MODULE_PATHS.items()}


def verify_package(approved_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identity of URC-v1 bound to the approved manifest hash: every manifest entry, every pinned file and a fresh
    resolution of the shared settings must agree. Returns (identity, resolved config)."""
    manifest_path = URC_DIR / MANIFEST_NAME
    actual = K.sha256_file(manifest_path) if manifest_path.is_file() else None
    problems = []
    if actual != approved_sha256:
        problems.append(f"{MANIFEST_NAME} SHA-256 {actual} is not the approved {approved_sha256}")
    else:
        manifest = K.load_json(manifest_path)
        if manifest.get("protocol_id") != PROTOCOL_ID:
            problems.append("manifest protocol_id is not URC-20261013-v1")
        problems += [f"file mismatch: {b}" for b in K.verify_entries(manifest, URC_DIR)]
    if problems:
        raise ControlError(f"package identity check failed: {problems}")
    check_pinned()
    resolved = load_resolved()
    ident = {
        "urc_manifest_sha256": actual,
        "resolved_config_sha256": K.sha256_file(URC_DIR / RESOLVED_NAME),
        "urc_config_sha256": K.sha256_file(URC_DIR / CONFIG_NAME),
        "reference": {k: v for k, v in resolved["reference"].items() if k.endswith("sha256")},
    }
    return ident, resolved


# ---- model and state files (torch imported lazily: the supervisor never calls these) ---------------------
def build_model(cfg: dict[str, Any]):
    """The R-FIT model under ``torch.manual_seed(model_init_seed)`` inside a forked RNG, as the frozen
    ``uid_common.build_model`` does for U-FIT. ``cfg`` is the resolved config (or an engineering fixture)."""
    import torch
    from resnet3_s1 import ResNet3S1AutoEncoder, ResNet3S1Config

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(cfg["model_init_seed"]))
        return ResNet3S1AutoEncoder(ResNet3S1Config(**cfg["model"]))


def save_state(path: Path, model, meta: dict[str, Any]) -> dict[str, Any]:
    """All parameters and buffers (BatchNorm running statistics and counters included) as safetensors, created
    exclusively and fsynced; read back and compared bit for bit before returning."""
    import os

    import torch
    import uid_common as C
    from safetensors.torch import load, save

    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in model.state_dict().items()}
    digest = C.state_sha256(model)
    meta = {"model_config": model.cfg.to_dict(), **meta}
    meta = {
        "format": "urc_20261013_v1.state/v1",
        "state_sha256": digest,
        **{k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in meta.items()},
    }
    data = save(state, metadata=meta)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    back = load(Path(path).read_bytes())
    if set(back) != set(state) or not all(
        back[k].dtype == state[k].dtype and torch.equal(back[k], state[k]) for k in state
    ):
        raise RuntimeError(f"state file {path} does not read back bit for bit")
    return {
        "path": str(path),
        "bytes": len(data),
        "file_sha256": sha256_bytes(data),
        "state_sha256": digest,
        "tensors": len(state),
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "buffer_elements": sum(b.numel() for b in model.buffers()),
    }


def load_state(path: Path, cfg: dict[str, Any]):
    """A fresh model with the saved state loaded strictly; its ``state_sha256`` must equal the file metadata."""
    import uid_common as C
    from safetensors import safe_open
    from safetensors.torch import load

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    model = build_model(cfg)
    model.load_state_dict(load(Path(path).read_bytes()), strict=True)
    if C.state_sha256(model) != meta.get("state_sha256"):
        raise RuntimeError(f"{path}: loaded state hash differs from its metadata")
    return model, meta
