"""URP-20261013-v1: identity, resolution, the registered input, the one-arm plan, the model binding and state files.

Imports no torch at module level, so the supervisor stays torch-free. Every shared scientific setting is read from the
frozen UID-20261012-v2 config (hash-pinned in ``urp_config.json``) and must equal the expected value listed there. The
input is the registered patch file: its bytes, its input record and its selection rule are hash-pinned, and the
record's values must equal ``input_expected``. The model is built only by the frozen ``uid_common.build_model(cfg,
"U-FIT")`` from the frozen U-FIT model configuration. Execution mechanics come from the r1 ``execution_r1.json``
(hash-pinned) except the run root and ledger mirror, which are this package's own. Ledger, start accounting and
deadlines are the r1 implementation (``uid_r1_control``), imported unchanged and hash-checked.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

URP_DIR = Path(__file__).resolve().parent
DIAG_ROOT = URP_DIR.parent
FROZEN_DIR = DIAG_ROOT / "uid_20261012_v2"
R1_DIR = DIAG_ROOT / "uid_20261012_v2_r1"
EP_DIR = DIAG_ROOT / "exploratory_progression_20261012"
for _p in (str(R1_DIR), str(FROZEN_DIR)):  # r1 control/measure, frozen uid_common and Uformer modules
    if _p not in sys.path:
        sys.path.append(_p)

import uid_r1_control as K  # noqa: E402  (r1, unchanged)

PROTOCOL_ID = "URP-20261013-v1"
ARM = "U-PATCH"
REGISTERED_ARMS = (ARM,)
FORBIDDEN_ARMS = ("U-CHECK", "U-FIT", "R-FIT")  # the closed UID-20261012-v2 and URC-20261013-v1 arms
MANIFEST_NAME = "manifest_urp.json"
RESOLVED_NAME = "resolved_config.json"
CONFIG_NAME = "urp_config.json"
NORMALIZATION_MODE = "global_exact_maxabs_train"
ControlError = K.ControlError

# modules the scientific child may import: import name -> path relative to URP_DIR (as listed in the manifest)
MODULE_PATHS = {
    "uid_common": "../uid_20261012_v2/uid_common.py",
    "uformer_ae": "../uid_20261012_v2/uformer_ae.py",
    "uformer_blocks": "../uid_20261012_v2/uformer_blocks.py",
    "uid_r1_control": "../uid_20261012_v2_r1/uid_r1_control.py",
    "uid_r1_measure": "../uid_20261012_v2_r1/uid_r1_measure.py",
    "urp_control": "urp_control.py",
    "urp_child": "urp_child.py",
}
# files hash-checked against fixed values here, independently of the manifest
PINNED_SHA256 = {
    "../uid_20261012_v2/uid_common.py": "f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230",
    "../uid_20261012_v2/uformer_ae.py": "a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606",
    "../uid_20261012_v2/uformer_blocks.py": "3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747",
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
# input_expected key -> dotted path in input/input_record.json
INPUT_RECORD_FIELDS = {
    "original_sha256": "patch.original.sha256",
    "normalized_sha256": "patch.normalized.sha256",
    "shape": "patch.shape",
    "dtype": "patch.dtype",
    "original_target_energy": "patch.original.energy_sum_sq",
    "original_zero_prediction_mse": "baselines.original.zero_prediction.mse",
    "original_scalar_mean_mse": "baselines.original.shared_scalar_mean.mse",
    "normalized_target_energy": "patch.normalized.energy_sum_sq",
    "normalized_zero_prediction_mse": "baselines.normalized.zero_prediction.mse",
    "normalized_scalar_mean_mse": "baselines.normalized.shared_scalar_mean.mse",
    "normalized_max_abs": "range_feasibility.normalized_max_abs",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(rec: dict[str, Any], dotted: str) -> Any:
    for part in dotted.split("."):
        rec = rec[part]
    return rec


def load_urp_config(path: Path | None = None) -> dict[str, Any]:
    cfg = K.load_json(path or URP_DIR / CONFIG_NAME)
    if cfg.get("protocol_id") != PROTOCOL_ID or cfg.get("arm") != ARM:
        raise ControlError(f"not the URP-v1 config: {cfg.get('protocol_id')!r} / {cfg.get('arm')!r}")
    return cfg


def check_pinned(rel_paths: dict[str, str] | None = None) -> None:
    bad = []
    for rel, digest in (rel_paths or PINNED_SHA256).items():
        p = URP_DIR / rel
        actual = K.sha256_file(p) if p.is_file() else None
        if actual != digest:
            bad.append({"path": rel, "expected": digest, "actual": actual})
    if bad:
        raise ControlError(f"pinned file(s) changed or missing: {bad}")


def _equal(a: Any, b: Any) -> bool:
    """Exact equality after JSON normalisation (tuples as lists, ints and equal floats compare equal)."""
    return json.loads(json.dumps(a)) == json.loads(json.dumps(b))


def input_problems(spec: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """The registered input files: patch, input record and selection rule must have their pinned SHA-256, and the
    record must state the expected values, a usable patch and the frozen R3 normalization. Pure: reads only."""
    problems = []
    for key in ("patch_file", "input_record", "selection_rule"):
        p = URP_DIR / spec[key]
        actual = K.sha256_file(p) if p.is_file() else None
        if actual != spec[f"{key}_sha256"]:
            problems.append(f"{spec[key]}: SHA-256 {actual} != pinned {spec[f'{key}_sha256']}")
    if problems:
        return problems
    rec = K.load_json(URP_DIR / spec["input_record"])
    if rec.get("protocol_id") != PROTOCOL_ID or rec.get("usable") is not True:
        problems.append("input record is not a usable URP-v1 input")
    if (
        rec["patch"]["file_sha256"] != spec["patch_file_sha256"]
        or rec["patch"]["tensor_name"] != spec["tensor_name"]
    ):
        problems.append("input record names another patch file or tensor")
    if rec["selection_rule"]["sha256"] != spec["selection_rule_sha256"]:
        problems.append("input record names another selection rule")
    norm = rec["normalization"]
    if (norm["mode"], norm["scale_global"]) != (
        spec["normalization"]["mode"],
        spec["normalization"]["scale_global"],
    ):
        problems.append(f"input record normalization {norm} differs from {spec['normalization']}")
    if spec["normalization"]["mode"] != NORMALIZATION_MODE:
        problems.append(f"normalization mode {spec['normalization']['mode']!r} is not {NORMALIZATION_MODE}")
    for key, path in INPUT_RECORD_FIELDS.items():
        if not _equal(get(rec, path), expected[key]):
            problems.append(f"input record {path} = {get(rec, path)!r} != expected {key} {expected[key]!r}")
    if not all(rec["degeneracy_checks"].values()):
        problems.append(f"degeneracy check failed: {rec['degeneracy_checks']}")
    return problems


def resolve(urp: dict[str, Any]) -> dict[str, Any]:
    """The resolved scientific configuration: shared settings and the U-FIT model read from the frozen UID config and
    the r1 execution file (each hash-checked), compared with ``shared_expected``; the registered input checked as in
    ``input_problems``. Any difference refuses. Pure: writes nothing."""
    ref = urp["reference"]
    frozen_raw = (URP_DIR / ref["frozen_config"]).read_bytes()
    uid = json.loads(frozen_raw.decode("utf-8"))
    execution_raw = (URP_DIR / ref["execution"]).read_bytes()
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
    exp = urp["shared_expected"]
    problems = [f"{k}: frozen {v!r} != expected {exp[k]!r}" for k, v in read.items() if not _equal(v, exp[k])]
    if not _equal(urp["input_expected"]["shape"], uid["input"]["shape"]):
        problems.append(f"patch shape {urp['input_expected']['shape']} is not the U-FIT input shape")
    if any(int(s) != 1 for s in fit["model"]["down_strides"]):
        problems.append(f"U-FIT strides {fit['model']['down_strides']} are not all 1")
    problems += input_problems(urp["input"], urp["input_expected"])
    if problems:
        raise ControlError(f"shared settings or input differ from the expected values: {problems}")
    return {
        "protocol_id": PROTOCOL_ID,
        "arm": ARM,
        "variant": urp["variant"],
        "reference": ref,
        "input": urp["input"],
        "input_expected": urp["input_expected"],
        "model_init_seed": int(uid["model_init_seed"]),
        "u_fit": {"model": fit["model"]},  # read unchanged by the frozen uid_common.build_model(cfg, "U-FIT")
        "optimizer": fit["optimizer"],
        "max_updates": int(fit["max_updates"]),
        "max_seconds": float(fit["max_seconds"]),
        "cpu": read["cpu_config"],
        "criteria": uid["criteria"],  # strict near-identity criteria, unchanged
        "limits": {
            "max_training_starts": int(urp["limits"]["max_training_starts"]),
            "max_invocations": int(urp["limits"]["max_invocations"]),
            "aggregate_seconds": float(urp["limits"]["aggregate_seconds"]),
            "rss_cap_bytes": int(uid["global_limits"]["max_rss_bytes"]),
            "rss_available_fraction": K.REGISTERED_RSS_AVAILABLE_FRACTION,
            "evidence_cap_bytes": K.REGISTERED_EVIDENCE_CAP_BYTES,
            "curve_every": K.REGISTERED_CURVE_EVERY,
        },
        "mechanics": {k: execution[k] for k in MECHANICS_KEYS},
        "execution": urp["execution"],
    }


def load_resolved() -> dict[str, Any]:
    """The frozen ``resolved_config.json``; it must equal a fresh resolution of the pinned sources."""
    resolved = K.load_json(URP_DIR / RESOLVED_NAME)
    fresh = resolve(load_urp_config())
    if not _equal(resolved, fresh):
        raise ControlError("resolved_config.json differs from a fresh resolution of the pinned sources")
    if resolved.get("protocol_id") != PROTOCOL_ID:
        raise ControlError("resolved_config.json is not the URP-v1 configuration")
    return resolved


def registered_plan(resolved: dict[str, Any]) -> dict[str, Any]:
    """The only scientific plan: the single arm U-PATCH. No arm of a closed protocol can appear in it."""
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
    """Refuses any plan whose arms are not exactly [U-PATCH] (scientific) or that names a closed arm (any mode)."""
    arms = [a["arm"] for a in plan["arms"]]
    names = arms + [a.get("model_arm", a["arm"]) for a in plan["arms"]]
    if any(n in FORBIDDEN_ARMS for n in names):
        raise ControlError(f"plan names an arm of a closed protocol: {arms}")
    if plan["mode"] == "scientific" and (arms != [ARM] or plan["max_training_starts"] != 1):
        raise ControlError(f"the scientific plan must be exactly one U-PATCH start, got {arms}")
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
    """Identity of URP-v1 bound to the approved manifest hash: every manifest entry, every pinned file and a fresh
    resolution of the shared settings and the registered input must agree. Returns (identity, resolved config)."""
    manifest_path = URP_DIR / MANIFEST_NAME
    actual = K.sha256_file(manifest_path) if manifest_path.is_file() else None
    problems = []
    if actual != approved_sha256:
        problems.append(f"{MANIFEST_NAME} SHA-256 {actual} is not the approved {approved_sha256}")
    else:
        manifest = K.load_json(manifest_path)
        if manifest.get("protocol_id") != PROTOCOL_ID:
            problems.append("manifest protocol_id is not URP-20261013-v1")
        problems += [f"file mismatch: {b}" for b in K.verify_entries(manifest, URP_DIR)]
    if problems:
        raise ControlError(f"package identity check failed: {problems}")
    check_pinned()
    resolved = load_resolved()
    ident = {
        "urp_manifest_sha256": actual,
        "resolved_config_sha256": K.sha256_file(URP_DIR / RESOLVED_NAME),
        "urp_config_sha256": K.sha256_file(URP_DIR / CONFIG_NAME),
        "input": {k: v for k, v in resolved["input"].items() if k.endswith("sha256")},
        "reference": {k: v for k, v in resolved["reference"].items() if k.endswith("sha256")},
    }
    return ident, resolved


# ---- model, input, normalization and state files (torch imported lazily: the supervisor never calls these) ----
def build_model(cfg: dict[str, Any]):
    """The frozen UID builder, unchanged: ``uid_common.build_model(cfg, "U-FIT")`` builds
    ``UformerAutoEncoder(UformerAEConfig(**cfg["u_fit"]["model"]))`` under ``torch.manual_seed(model_init_seed)`` in a
    forked RNG. "U-FIT" is the builder's model key; the arm is U-PATCH. ``cfg`` is the resolved config or a fixture."""
    import uid_common as C

    return C.build_model(cfg, "U-FIT")


def normalize(x, scale_global: float):
    """The production ``Normalizer.normalize`` arithmetic of ``global_exact_maxabs_train``: ``x / float32(s)`` in fp32
    (``_scale_like`` casts the float64 scale once to a 0-dim fp32 tensor). Re-declared so that the run never imports
    ``cg_fedllm``; ``checks/test_urp_control.py`` compares it bit for bit with the production class."""
    import torch

    return x / torch.tensor(float(scale_global), dtype=torch.float32, device=x.device)


def denormalize(y, scale_global: float):
    """The production ``Normalizer.denormalize`` arithmetic: ``y * float32(s)`` in fp32."""
    import torch

    return y * torch.tensor(float(scale_global), dtype=torch.float32, device=y.device)


def load_input(cfg: dict[str, Any]):
    """The registered patch (original space) and its normalized copy. The file's bytes are read once, hashed and
    deserialized from those same bytes; the file must hold exactly one fp32 tensor of the expected shape."""
    import torch
    from safetensors.torch import load

    spec = cfg["input"]
    data = (URP_DIR / spec["patch_file"]).read_bytes()
    digest = sha256_bytes(data)
    if digest != spec["patch_file_sha256"]:
        raise RuntimeError(f"patch file SHA-256 {digest} != {spec['patch_file_sha256']}")
    tensors = load(data)
    if set(tensors) != {spec["tensor_name"]}:
        raise RuntimeError(f"patch file holds {sorted(tensors)}, expected exactly [{spec['tensor_name']!r}]")
    x = tensors[spec["tensor_name"]].contiguous()
    if x.dtype != torch.float32 or list(x.shape) != list(cfg["input_expected"]["shape"]):
        raise RuntimeError(
            f"patch tensor {x.dtype} {list(x.shape)} is not fp32 {cfg['input_expected']['shape']}"
        )
    return x, normalize(x, spec["normalization"]["scale_global"]), digest


def input_checks(rec: dict[str, Any], cfg: dict[str, Any], max_abs_n: float) -> dict[str, Any]:
    """Recorded input and baselines in both spaces against ``input_expected``; any difference refuses the run before
    the optimizer exists. ``rec`` holds the child's ``input`` / ``baselines`` descriptions."""
    exp, i, b = cfg["input_expected"], rec["input"], rec["baselines"]
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


def save_state(path: Path, model, meta: dict[str, Any]) -> dict[str, Any]:
    """All parameters and buffers (BatchNorm statistics and counters, attention index buffers) as safetensors,
    created exclusively and fsynced; read back and compared bit for bit before returning."""
    import os

    import torch
    import uid_common as C
    from safetensors.torch import load, save

    state = {k: v.detach().to("cpu").contiguous().clone() for k, v in model.state_dict().items()}
    digest = C.state_sha256(model)
    meta = {"model_config": model.cfg.to_dict(), **meta}
    meta = {
        "format": "urp_20261013_v1.state/v1",
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
