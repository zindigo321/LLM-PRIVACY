"""Independent verification of the URP-20261013-v1 U-PATCH result (read-only; imports neither torch nor urp_analyze).

``python -B independent_verify.py [--write]`` from this directory, with PYTHONPATH unset. With ``--write`` it creates
``independent_verification.json`` here (exclusive, LF bytes); without it, it only prints.

1. Integrity after the run: the manifest bytes are the approved ones; 61/61 entries and 4/4 external bound files match
   in SHA-256 and size; the 7 files pinned in ``urp_control`` match the values repeated here; no bytecode.
2. Sealed list: every entry of ``analysis/FINAL_ARTIFACTS_SHA256.txt`` re-hashed (SHA-256 and size); the list names
   every file now in the run root and the mirror; the mirror is byte-identical to the primary ledger.
3. Records: identity, ledger counts, and agreement of status, update count, N and T across the child record, the end
   record, the invocation summary, the start record, the job, the plan and ``--status``; job fields and deadlines.
4. Input and baselines from the patch file bytes (struct and numpy; no torch, no safetensors): both tensor hashes by the
   ``uid_common.tensor_sha256`` definition, T, the zero-prediction and scalar-mean MSE, max |x_n|.
5. Metrics from the recorded sums (SSE, T, P, D, N) in float64 with the frozen formulas and as exact rationals.
6. Strict verdict re-derived; EP-20261012-v1 re-implemented here from POLICY.md; the frozen ``ep_classify.classify``
   (hash-checked first) applied to a classifier input built here; both compared with ``analysis/analysis.json``.
7. State files without torch: header, tensor count, dtypes, parameter and buffer elements, metadata, BatchNorm
   counters, finiteness, unchanged index buffers, and the per-group parameter change from init to final.
8. Curve: schedule, finiteness, gradients, monotonicity, timing, and consistency with step 0.
9. Timing scopes, kept distinct: wrapper, scientific command, supervisor aggregate, arm, child process, training loop.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

APPROVED = "1df7af08c8b5897ad23fac88ab94e4ad36212a47cd542bb584e85ca07df91bd1"
HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
URP = DIAG / "urp_20261013_v1"
RUN = Path("D:/cgfed-runs/urp_20261013_v1")
ARM = RUN / "arms" / "U-PATCH"
LEDGER = RUN / "ledger"
MIRROR = URP / "scientific_ledger_mirror"
EP_DIR = DIAG / "exploratory_progression_20261012"
SCALE = 0.015751632224572334
UFIT_INIT = "7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7"
PINNED = {  # urp_control.PINNED_SHA256, repeated here
    "../uid_20261012_v2/uid_common.py": "f1f55dc90faddb0e4ac10be5fca0cc756592cf218dedb4fb90d1c4e82dc41230",
    "../uid_20261012_v2/uformer_ae.py": "a3ad6e3f2b6e6f009158276f5534e2c68bf42781df18a1c1d9fcd0f5867c8606",
    "../uid_20261012_v2/uformer_blocks.py": "3333f1862288e720d83aadbe8f171898f5f6943d2c0f58c9bd0132aba60b0747",
    "../uid_20261012_v2_r1/uid_r1_control.py": "bdd8aff7dbccc5f5b33a6da3b6314e35beef5a53501c5c0de9a3a1a65800c074",
    "../uid_20261012_v2_r1/uid_r1_measure.py": "76b1bba28217812aff3985c19fcd1940f50cfa70e2a27ee5b2f7cab8236fc760",
    "../exploratory_progression_20261012/ep_classify.py": "f7efd10029e8e50071b9a5013b23eefb1d2e3707d07d05e80fda4a690e160cd1",
    "../exploratory_progression_20261012/POLICY.md": "fe14a225d16fed0373314eb89291035a2c8175bb17f37a7a03cf21c8e7fca592",
}
MODULE_PATHS = {  # urp_control.MODULE_PATHS, repeated here
    "uid_common": "../uid_20261012_v2/uid_common.py",
    "uformer_ae": "../uid_20261012_v2/uformer_ae.py",
    "uformer_blocks": "../uid_20261012_v2/uformer_blocks.py",
    "uid_r1_control": "../uid_20261012_v2_r1/uid_r1_control.py",
    "uid_r1_measure": "../uid_20261012_v2_r1/uid_r1_measure.py",
    "urp_control": "urp_control.py",
    "urp_child": "urp_child.py",
}
EXPECTED_UNLISTED = sorted(
    ["CONTINUATION_CHECKPOINT.md", "PROPOSED_LAUNCH.md", "manifest_urp.json"]
    + [f"scientific_ledger_mirror/{n}" for n in
       ("end_U-PATCH.json", "invocation_001.json", "invocation_001_end.json", "start_U-PATCH.json")]
)
SUMS = ("finite", "n", "sse", "target_energy", "pred_energy", "dot")
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked", "relative_position_index")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def match(p: Path, digest: str, size: int) -> bool:
    return p.is_file() and p.stat().st_size == size and sha(p) == digest


def bytecode() -> list:
    roots = [URP, DIAG / "uid_20261012_v2", DIAG / "uid_20261012_v2_r1", EP_DIR, HERE]
    return sorted(str(p) for r in roots for p in r.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc")


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def ulps(a: float, b: float) -> float:
    return abs(a - b) / math.ulp(b)


# ---- 1. integrity after the run ------------------------------------------------------------------------------
def integrity() -> dict:
    raw = (URP / "manifest_urp.json").read_bytes()
    m = json.loads(raw)
    bad = [e["path"] for e in m["files"] if not match((URP / e["path"]).resolve(), e["sha256"], e["bytes"])]
    ext = [e["path"] for e in m["external_bound_files"] if not match(Path(e["path"]), e["sha256"], e["bytes"])]
    listed = {(URP / e["path"]).resolve() for e in m["files"]}
    unlisted = sorted(
        p.relative_to(URP).as_posix() for p in URP.rglob("*") if p.is_file() and p.resolve() not in listed
    )
    by_path = {e["path"]: e["sha256"] for e in m["files"]}
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_bytes": len(raw),
        "manifest_is_approved": hashlib.sha256(raw).hexdigest() == APPROVED,
        "entries": len(m["files"]),
        "entries_mismatching": bad,
        "external_bound_files": len(m["external_bound_files"]),
        "external_mismatching": ext,
        "unlisted_files": unlisted,
        "unlisted_as_expected": unlisted == EXPECTED_UNLISTED,
        "pinned_match": {k: sha((URP / k).resolve()) == v for k, v in PINNED.items()},
        "module_sha256_from_manifest": {n: by_path.get(p) for n, p in MODULE_PATHS.items()},
        "bytecode_files": bytecode(),
    }


# ---- 2. sealed list and mirror -------------------------------------------------------------------------------
def sealed() -> dict:
    listing = HERE / "analysis" / "FINAL_ARTIFACTS_SHA256.txt"
    data = listing.read_bytes()
    lines = data.decode("utf-8").splitlines()
    rows = []
    for ln in lines[1:]:
        if ln.strip():
            digest, size, name = re.split(r"\s+", ln.strip(), maxsplit=2)
            rows.append((digest, int(size), Path(name)))
    listed = {os.path.normcase(str(p.resolve())) for _, _, p in rows}
    present = sorted(str(p) for root in (RUN, MIRROR) for p in root.rglob("*") if p.is_file())
    names = sorted(p.name for p in LEDGER.iterdir())
    return {
        "listing_sha256": hashlib.sha256(data).hexdigest(),
        "listing_bytes": len(data),
        "listing_has_crlf": b"\r\n" in data,
        "recorded_line": lines[0],
        "entries": len(rows),
        "entries_mismatching": [str(p) for d, s, p in rows if not match(p, d, s)],
        "present_not_listed": [p for p in present if os.path.normcase(str(Path(p).resolve())) not in listed],
        "ledger_records": names,
        "mirror_byte_identical": names == sorted(p.name for p in MIRROR.iterdir())
        and all((LEDGER / n).read_bytes() == (MIRROR / n).read_bytes() for n in names),
    }


# ---- 3. records and their agreement --------------------------------------------------------------------------
def disagreements(values: dict) -> list:
    return [f"{k}: {v!r}" for k, v in values.items() if any(x != v[0] for x in v[1:])]


def records(r: dict, module_sha: dict) -> dict:
    child, end, inv, inv_end, start, job = (r[k] for k in ("child", "end", "inv", "inv_end", "start", "job"))
    sci, after = r["scientific_stdout"], r["status_after"]
    fe, fen = child["final"]["eval_mode"], child["final"]["eval_mode_normalized"]
    b = child["baselines"]
    names = r["ledger_names"]
    plan = inv["plan"]
    t_start, t0 = start["monotonic_s"], inv["monotonic_t0"]
    kill_at = min(t_start + plan["arms"][0]["max_seconds"], t0 + plan["aggregate_seconds"]) - plan["end_margin_s"]
    problems = disagreements({
        "status": (child["status"], end["status"], end["child_status"], inv_end["arms"][0]["status"],
                   sci["arms"][0]["status"], after["arms"]["U-PATCH"]),
        "updates_done": (child["updates_done"], end["updates_done"], inv_end["arms"][0]["updates_done"],
                         sci["arms"][0]["updates_done"]),
        "updates_planned": (3000, start["max_updates"], job["max_updates"], plan["arms"][0]["max_updates"]),
        "N": (fe["n"], fen["n"], child["input"]["original"]["numel"], b["original"]["shared_scalar_mean"]["n"]),
        "T_original": (fe["target_energy"], child["final"]["train_mode"]["target_energy"],
                       child["input"]["original"]["energy_sum_sq"], b["original"]["zero_prediction"]["sse"]),
        "T_normalized": (fen["target_energy"], child["input"]["normalized"]["energy_sum_sq"],
                         b["normalized"]["zero_prediction"]["sse"]),
        "supervisor_pid": (inv["supervisor_pid"], job["supervisor_pid"]),
        "end_record_equals_summary_arm": (end, inv_end["arms"][0]),
    })
    checks = {
        "invocation_names_approved_manifest": inv["identity"]["urp_manifest_sha256"] == APPROVED,
        "ledger_invocations_1": sum(1 for n in names if re.match(r"^invocation_\d{3}\.json$", n)) == 1,
        "ledger_starts_1": sum(1 for n in names if n.startswith("start_")) == 1,
        "status_after_counts": (after["invocations"], after["starts"]) == (1, 1),
        "plan_one_upatch_start": [a["arm"] for a in plan["arms"]] == ["U-PATCH"]
        and (plan["max_training_starts"], plan["max_invocations"]) == (1, 1),
        "job_scientific_no_hooks": (job["mode"], job["fixture_hooks"], job["arm"], job["model_arm"])
        == ("scientific", None, "U-PATCH", "U-PATCH"),
        "job_module_hashes_equal_manifest": job["expected_module_sha256"] == module_sha,
        "child_modules_equal_manifest": child["modules"]["sha256"] == module_sha
        and child["modules_at_end"]["sha256"] == module_sha,
        "child_modules_checked": child["modules"]["checked_against_approved"] is True
        and child["modules_at_end"]["checked_against_approved"] is True,
        "child_local_modules_listed_only": sorted(child["modules_at_end"]["local"]) == sorted(
            m for m in module_sha if m != "urp_child"),
        "child_config_is_resolved_config": child["config_sha256"]["raw"] == job["expected_config_sha256"]
        == inv["identity"]["resolved_config_sha256"],
        "deadlines_as_registered": abs(end["monotonic_s"]["kill_at"] - kill_at) < 1e-6
        and abs(job["soft_stop_monotonic"] - (kill_at - plan["eval_reserve_s"])) < 1e-6
        and abs(job["hard_stop_monotonic"] - (kill_at + plan["kill_wait_s"])) < 1e-6,
        "no_stop_request_file": not (ARM / "STOP_REQUEST").exists(),
    }
    return {"problems": problems, "checks": checks, "kill_at_rederived": kill_at}


# ---- 4. input and baselines from the patch file bytes (no torch, no safetensors) -----------------------------
def read_safetensors(path: Path) -> tuple:
    data = path.read_bytes()
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + n])
    meta = header.pop("__metadata__", {})
    return header, meta, data[8 + n :]


def tensor_sha(name: str, dtype: str, shape, raw: bytes) -> str:
    """The ``uid_common.tensor_sha256`` definition: name, NUL, torch dtype string, repr(shape tuple), raw bytes."""
    h = hashlib.sha256()
    for part in (name.encode("utf-8"), b"\x00", dtype.encode("utf-8"), repr(tuple(shape)).encode("utf-8"), raw):
        h.update(part)
    return h.hexdigest()


def sums(pred: np.ndarray, x: np.ndarray) -> dict:
    a, b = pred.astype(np.float64).ravel(), x.astype(np.float64).ravel()
    sse, t = math.fsum(((a - b) ** 2).tolist()), math.fsum((b * b).tolist())
    return {"sse": sse, "target_energy": t, "mse": sse / b.size}


def input_check(child: dict) -> dict:
    header, _, body = read_safetensors(URP / "input" / "patch_x.safetensors")
    lo, hi = header["x"]["data_offsets"]
    x = np.frombuffer(body[lo:hi], dtype="<f4").reshape(header["x"]["shape"])
    xn = (x / np.float32(SCALE)).astype(np.float32)  # x / float32(s), fp32, as the production Normalizer
    out = {"file_keys": sorted(header), "dtype": header["x"]["dtype"], "shape": header["x"]["shape"]}
    out["tensor_sha256_original"] = tensor_sha("x", "torch.float32", x.shape, x.tobytes())
    out["tensor_sha256_normalized"] = tensor_sha("x", "torch.float32", xn.shape, xn.tobytes())
    out["hashes_equal_recorded"] = (out["tensor_sha256_original"], out["tensor_sha256_normalized"]) == (
        child["input"]["original"]["sha256"], child["input"]["normalized"]["sha256"])
    out["all_finite"], out["zero_elements"] = bool(np.isfinite(x).all()), int((x == 0).sum())
    out["normalized_max_abs"] = float(np.abs(xn).max())
    out["normalized_max_abs_equal_recorded"] = out["normalized_max_abs"] == child["input_checks"][
        "normalized_max_abs"]["observed"]
    for space, arr in (("original", x), ("normalized", xn)):
        mean32 = np.float32(math.fsum(arr.astype(np.float64).ravel().tolist()) / arr.size)  # full_like(x, mean)
        zero, mean = sums(np.zeros_like(arr), arr), sums(np.full_like(arr, mean32), arr)
        rec = child["baselines"][space]
        out[space] = {
            "target_energy_fsum": zero["target_energy"],
            "zero_mse_fsum": zero["mse"],
            "scalar_mean_value_fp32": float(mean32),
            "scalar_mean_mse_fsum": mean["mse"],
            "recorded_zero_mse": rec["zero_prediction"]["mse"],
            "recorded_scalar_mean_mse": rec["shared_scalar_mean"]["mse"],
            "ulps_target_energy": ulps(zero["target_energy"], rec["zero_prediction"]["target_energy"]),
            "ulps_zero_mse": ulps(zero["mse"], rec["zero_prediction"]["mse"]),
            "ulps_scalar_mean_mse": ulps(mean["mse"], rec["shared_scalar_mean"]["mse"]),
            "best_constant": "scalar_mean" if rec["shared_scalar_mean"]["mse"] < rec["zero_prediction"]["mse"]
            else "zero",
        }
    return out


# ---- 5. metrics from the recorded sums -----------------------------------------------------------------------
def metrics(m: dict) -> dict:
    """The frozen ``uid_common.metrics`` formulas in float64, and exact rationals of the recorded binary64 sums."""
    t, p, sse, d, n = m["target_energy"], m["pred_energy"], m["sse"], m["dot"], m["n"]
    f = {
        "rse": sse / t,
        "cosine": d / math.sqrt(p * t),
        "mse": sse / n,
        "norm_ratio": math.sqrt(p / t),
        "q_n": (sse / n) / (t / n),
    }
    f["equal_recorded"] = {k: f[k] == m[k] for k in ("rse", "cosine", "mse", "norm_ratio", "q_n")}
    fr = {k: Fraction(m[k]) for k in ("sse", "target_energy", "pred_energy", "dot")}
    rse_exact = fr["sse"] / fr["target_energy"]
    cos2_exact = fr["dot"] ** 2 / (fr["pred_energy"] * fr["target_energy"])
    f["rse_exact_rounded"] = float(rse_exact)
    f["cosine_from_exact_cos2"] = math.copysign(math.sqrt(float(cos2_exact)), m["dot"])
    f["ulps_rse_float64_vs_exact"] = ulps(f["rse"], float(rse_exact))
    f["sse_identity_rel_err"] = rel(p + t - 2 * d, sse)  # SSE = P + T - 2D, rounding only
    f["finite_recorded"] = m["finite"]
    return f


# ---- 6. strict and EP-20261012-v1 ----------------------------------------------------------------------------
def strict(m: dict) -> str:
    """The frozen ``uid_common.passes`` rule (RSE <= 0.01 and cosine >= 0.99, finite and defined), eval mode."""
    t, p = m["target_energy"], m["pred_energy"]
    if m["finite"] is not True or t <= 0 or p <= 0:
        return "COMPLETED_FIT_FAIL"
    rse, cos = m["sse"] / t, m["dot"] / math.sqrt(p * t)
    if not (math.isfinite(rse) and math.isfinite(cos)):
        return "COMPLETED_FIT_FAIL"
    return "COMPLETED_PASS" if rse <= 0.01 and cos >= 0.99 else "COMPLETED_FIT_FAIL"


def ep_reimplemented(fe: dict, mean_mse: float, verified: bool, compliance: list) -> dict:
    """EP-20261012-v1 from POLICY.md, written here without importing ep_classify (inclusive, float64, no epsilon)."""
    if not verified:
        return {"outcome": "UNVERIFIED"}
    if compliance:
        return {"outcome": "NOT_ELIGIBLE", "reasons": compliance}
    sse, t, p, d, n = (fe[k] for k in ("sse", "target_energy", "pred_energy", "dot", "n"))
    vals = (sse, t, p, d, mean_mse)
    if fe["finite"] is not True or not all(math.isfinite(v) for v in vals) or t <= 0 or p <= 0 or sse < 0:
        return {"outcome": "FAIL", "why": "M0"}
    rse, cos, mse = sse / t, d / math.sqrt(p * t), sse / n
    best = min(t / n, mean_mse)
    m1, m2 = rse <= 0.50, cos >= 0.80
    if best == 0:
        return {"outcome": "UNINFORMATIVE" if m1 and m2 else "FAIL"}
    m3 = mse <= 0.80 * best
    return {"outcome": "PASS" if m1 and m2 and m3 else "FAIL", "M1": [rse, 0.50, m1], "M2": [cos, 0.80, m2],
            "M3": [mse, 0.80 * best, m3], "best": best, "mse_over_best": mse / best}


# ---- 7. state files without torch ----------------------------------------------------------------------------
DTYPES = {"F32": ("torch.float32", "<f4"), "I64": ("torch.int64", "<i8")}


def state_file(point: str, rec: dict, recorded_state: str) -> tuple:
    path = ARM / f"{point}_state.safetensors"
    data_sha = sha(path)
    header, meta, body = read_safetensors(path)
    h, arrays, params, buffers, dtypes = hashlib.sha256(), {}, 0, 0, {}
    for key in sorted(header):  # uid_common.state_sha256: sorted keys, tensor_sha256(value, key) of each
        info = header[key]
        tdt, npdt = DTYPES[info["dtype"]]
        lo, hi = info["data_offsets"]
        raw = body[lo:hi]
        h.update(tensor_sha(key, tdt, info["shape"], raw).encode("ascii"))
        arrays[key] = np.frombuffer(raw, dtype=npdt).reshape(info["shape"])
        n = math.prod(info["shape"])
        buffers, params = (buffers + n, params) if key.endswith(BUFFER_SUFFIXES) else (buffers, params + n)
        dtypes[info["dtype"]] = dtypes.get(info["dtype"], 0) + 1
    floats = [a for a in arrays.values() if a.dtype.kind == "f"]
    out = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "file_sha256": data_sha,
        "file_equals_record": (data_sha, path.stat().st_size) == (rec["file_sha256"], rec["bytes"]),
        "tensors": len(header),
        "dtype_counts": dtypes,
        "parameter_elements": params,
        "buffer_elements": buffers,
        "counts_equal_record": (len(header), params, buffers)
        == (rec["tensors"], rec["parameter_elements"], rec["buffer_elements"]),
        "metadata": {k: meta.get(k) for k in ("format", "protocol_id", "arm", "variant", "point", "state_sha256")},
        "state_sha256_recomputed_from_file": h.hexdigest(),
        "state_hash_agrees": h.hexdigest() == meta.get("state_sha256") == rec["state_sha256"] == recorded_state,
        "num_batches_tracked_values": sorted({int(a) for k, a in arrays.items() if k.endswith("num_batches_tracked")}),
        "all_float_finite": all(bool(np.isfinite(a).all()) for a in floats),
        "running_var_nonnegative": all(bool((a >= 0).all()) for k, a in arrays.items() if k.endswith("running_var")),
    }
    return out, arrays


def group_change(init: dict, final: dict, recorded: dict) -> dict:
    acc = {g: {"d2": [], "r2": [], "changed": 0, "numel": 0} for g in GROUPS}
    for key, a in final.items():
        if key.endswith(BUFFER_SUFFIXES):
            continue
        g = ".".join(key.split(".")[:2])
        d = a.astype(np.float64) - init[key].astype(np.float64)
        acc[g]["d2"].append(float(np.square(d).sum()))
        acc[g]["r2"].append(float(np.square(init[key].astype(np.float64)).sum()))
        acc[g]["changed"] += int((d != 0).sum())
        acc[g]["numel"] += a.size
    out = {}
    for g, v in acc.items():
        dl2, rl2 = math.sqrt(math.fsum(v["d2"])), math.sqrt(math.fsum(v["r2"]))
        r = recorded[g]
        out[g] = {
            "delta_l2": dl2,
            "ref_l2": rl2,
            "relative_change": dl2 / rl2,
            "numel_changed": v["changed"],
            "numel": v["numel"],
            "rel_diff_delta_l2_vs_record": rel(dl2, r["delta_l2"]),
            "rel_diff_ref_l2_vs_record": rel(rl2, r["ref_l2"]),
            "counts_equal_record": (v["changed"], v["numel"]) == (r["numel_changed"], r["numel"]),
        }
    return out


# ---- 8. curve (point k: train-mode forward BEFORE update k; grad of update k; changes and time AFTER it) -------
def curve(child: dict) -> tuple:
    pts = [json.loads(x) for x in (ARM / "curve.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ups = [p["update"] for p in pts]
    loss = {p["update"]: p["pre_update_train_loss"] for p in pts}
    s0 = child["step0"]["post_init"]["train"]["normalized"]
    fin_tr = child["final"]["train_mode_normalized"]
    t = [p["monotonic_s"] for p in pts]
    gaps = [b - a for a, b in zip(t[1:], t[2:])]  # seconds per 100 updates after the first full interval

    def slope(a: int, b: int) -> float:  # log-log slope of the loss against completed updates (k - 1)
        return math.log(loss[b] / loss[a]) / math.log((b - 1) / (a - 1))

    out = {
        "points": len(pts),
        "schedule_as_registered": ups == [1, *range(100, 3001, 100)],
        "all_losses_finite": all(math.isfinite(v) for v in loss.values()),
        "grad_nonfinite_total": sum(g["nonfinite"] for p in pts for g in p["grad"].values()),
        "every_parameter_had_a_gradient": all(
            g["numel_with_grad"] == g["numel"] for p in pts for g in p["grad"].values()),
        "zero_gradient_groups": sorted({n for p in pts for n, g in p["grad"].items() if g["grad_l2"] == 0}),
        "loss_strictly_decreasing_at_recorded_points": all(
            loss[b] < loss[a] for a, b in zip(ups, ups[1:])),
        "loss_fp32_vs_metrics_mse_max_rel_diff": max(
            rel(p["pre_update_train_loss"], p["pre_update_train"]["mse"]) for p in pts),
        "update1_metrics_equal_step0_train_normalized": all(
            pts[0]["pre_update_train"][k] == s0[k] for k in ("sse", "pred_energy", "dot", "rse", "cosine")),
        "pre_update_rse_normalized": {str(k): pts[i]["pre_update_train"]["rse"] for i, k in enumerate(ups)
                                      if k in (1, 100, 500, 1000, 1500, 2000, 2500, 2900, 3000)},
        "loss_ratio_3000_over_2000": loss[3000] / loss[2000],
        "loss_ratio_3000_over_2500": loss[3000] / loss[2500],
        "loglog_slope_1000_to_3000": slope(1000, 3000),
        "loglog_slope_2000_to_3000": slope(2000, 3000),
        "final_train_mode_mse_after_3000_over_pre_update_loss_3000": fin_tr["mse"] / loss[3000],
        "final_param_change_equals_point_3000": pts[-1]["param_change_from_init"] == child["param_change_total"],
        "relative_change_from_init_at_3000": {
            g: pts[-1]["param_change_from_init"][g]["delta_l2"] / pts[-1]["param_change_from_init"][g]["ref_l2"]
            for g in GROUPS},
        "grad_l2_first_last": {g: [pts[0]["grad"][g]["grad_l2"], pts[-1]["grad"][g]["grad_l2"]] for g in GROUPS},
        "train_loop_s_point1_to_point3000": t[-1] - t[0],
        "mean_s_per_update": (t[-1] - t[0]) / (ups[-1] - ups[0]),
        "s_per_100_updates_min_max": [min(gaps), max(gaps)],
        "rss_at_points_min_max": [min(p["rss_bytes_at_point"] for p in pts),
                                  max(p["rss_bytes_at_point"] for p in pts)],
    }
    return out, pts


# ---- 9. timing scopes (kept distinct) ------------------------------------------------------------------------
def timing(child: dict, end: dict, inv: dict, inv_end: dict, pts: list) -> dict:
    log = (HERE / "launch" / "COMMANDS_AND_EXIT_CODES.txt").read_text(encoding="utf-8")
    ts = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{2}:\d{2})"
    wrapper0 = datetime.fromisoformat(re.search(r"launch wrapper: .* started " + ts, log).group(1))
    wrapper1 = datetime.fromisoformat(re.search(r"LAUNCH_SCRIPT_END .* at " + ts, log).group(1))
    sci = re.search(r"command scientific: .*\nstart: " + ts + r"\nend: " + ts + r"\nexit_code: (\d+)", log)
    s0, s1 = datetime.fromisoformat(sci.group(1)), datetime.fromisoformat(sci.group(2))
    cm = child["child_monotonic_s"]
    p1, p3000 = pts[0]["monotonic_s"], pts[-1]["monotonic_s"]
    return {
        "wrapper_elapsed_s": (wrapper1 - wrapper0).total_seconds(),
        "wrapper_scope": "launch_scientific.ps1: status before, scientific command, status after (wall clock)",
        "scientific_command_wall_s": (s1 - s0).total_seconds(),
        "scientific_command_scope": "Start-Process wall clock: interpreter start to process exit",
        "scientific_exit_code": int(sci.group(3)),
        "supervisor_aggregate_s": inv_end["aggregate_wall_s"],
        "supervisor_aggregate_scope": "monotonic, from the import of urp_supervise to the invocation summary",
        "arm_wall_s": end["arm_wall_s"],
        "arm_scope": "monotonic, from just before the start record to the child's exit",
        "child_process_s": cm["end"] - cm["begin"],
        "child_scope": "child monotonic, after its imports, to just before child_record.json was written",
        "child_spawn_and_imports_s": cm["begin"] - end["monotonic_s"]["start"],
        "setup_to_point1_s": p1 - cm["begin"],
        "train_loop_point1_to_point3000_s": p3000 - p1,
        "final_phase_after_point3000_s": cm["end"] - p3000,
        "final_phase_scope": "final state save and read-back, two final evaluations, module check",
        "invocation_wall_clock_utc": inv["wall_clock_utc"],
        "invocation_record_after_import_s": inv["monotonic_at_record"] - inv["monotonic_t0"],
        "arm_limit_s": inv["plan"]["arms"][0]["max_seconds"],
        "aggregate_limit_s": inv["plan"]["aggregate_seconds"],
        "soft_stop_margin_s": end["monotonic_s"]["soft_stop"] - cm["end"],
    }


# ---- main ----------------------------------------------------------------------------------------------------
def classifier_input(child: dict, end: dict, inv: dict, inv_end: dict, starts: int, invs: int, ev: dict) -> dict:
    """Built here from the records (not from urp_analyze): original-space final eval sums, the patch's own
    original-space scalar-mean baseline, every applicable limit."""
    rss = max(end["guard"]["sampled_max_rss_bytes"], child["rss"]["sampled_max_rss_bytes"],
              child["rss"]["os_peak_wset_bytes"])
    plan = inv["plan"]
    return {
        "evidence": ev,
        "run": {
            "status": child["status"],
            "updates_done": child["updates_done"],
            "updates_planned": plan["arms"][0]["max_updates"],
            "prerequisite": None,
            "interrupted": inv_end["interrupted"],
            "guard_actions": [] if end["guard"]["action"] is None else [end["guard"]["action"]],
            "limits": {
                "arm_wall_s": {"used": end["arm_wall_s"], "limit": plan["arms"][0]["max_seconds"]},
                "aggregate_wall_s": {"used": inv_end["aggregate_wall_s"], "limit": plan["aggregate_seconds"]},
                "training_starts": {"used": starts, "limit": plan["max_training_starts"]},
                "invocations": {"used": invs, "limit": plan["max_invocations"]},
                "rss_bytes": {"used": rss, "limit": inv["rss_limit_bytes"]},
                "evidence_bytes": {"used": inv_end["evidence_bytes_run_root"], "limit": plan["evidence_cap_bytes"]},
            },
        },
        "final_eval": {k: child["final"]["eval_mode"][k] for k in SUMS},
        "baselines": {"scalar_mean_mse": child["baselines"]["original"]["shared_scalar_mean"]["mse"]},
    }


def context() -> dict:
    """U-FIT and U-CHECK figures for orientation only (other input, no normalizer), read only after their files
    match the sealed r1 list."""
    listing = DIAG / "uid_20261012_v2_r1_execution" / "FINAL_ARTIFACTS_SHA256.txt"
    rows = {}
    for ln in listing.read_text(encoding="utf-8").splitlines()[1:]:
        if ln.strip():
            d, s, name = re.split(r"\s+", ln.strip(), maxsplit=2)
            rows[os.path.normcase(name)] = (d, int(s))
    r1 = Path("D:/cgfed-runs/uid_20261012_v2_r1")
    files = {"child": r1 / "arms/U-FIT/child_record.json", "end": r1 / "ledger/end_U-FIT.json",
             "end_check": r1 / "ledger/end_U-CHECK.json", "curve": r1 / "arms/U-FIT/curve.jsonl"}
    ok = {k: match(p, *rows.get(os.path.normcase(str(p.resolve())), ("", -1))) for k, p in files.items()}
    if not all(ok.values()):
        return {"verified": False, "files_match": ok}
    c, e, ec = load(files["child"]), load(files["end"]), load(files["end_check"])
    last = json.loads(files["curve"].read_text(encoding="utf-8").splitlines()[-1])
    return {
        "verified": True,
        "label": "context only: U-FIT used a synthetic uniform input with no normalizer; not a matched comparison",
        "u_fit_final_eval": {k: c["final"]["eval_mode"][k] for k in ("rse", "cosine")},
        "u_fit_final_train": {k: c["final"]["train_mode"][k] for k in ("rse", "cosine")},
        "u_fit_step0_eval_rse": c["step0"]["post_init"]["eval"]["rse"],
        "u_fit_pre_update_loss_3000": last["pre_update_train_loss"],
        "u_fit_target_mean_square": c["input"]["z_n"],
        "u_fit_arm_wall_s": e["arm_wall_s"],
        "u_fit_child_process_s": c["child_monotonic_s"]["end"] - c["child_monotonic_s"]["begin"],
        "u_fit_rss": {"guard_sampled_max": e["guard"]["sampled_max_rss_bytes"],
                      "os_peak_wset": c["rss"]["os_peak_wset_bytes"]},
        "u_check_rss_guard_sampled_max": ec["guard"]["sampled_max_rss_bytes"],
        "u_check_note": "1x1 Conv2d on the same 32x32 input, same process setup: an upper bound on fixed overhead",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out_path = HERE / "independent_verification.json"
    if args.write and out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    child = load(ARM / "child_record.json")
    r = {
        "child": child, "end": load(LEDGER / "end_U-PATCH.json"), "inv": load(LEDGER / "invocation_001.json"),
        "inv_end": load(LEDGER / "invocation_001_end.json"), "start": load(LEDGER / "start_U-PATCH.json"),
        "job": load(ARM / "job.json"), "scientific_stdout": load(HERE / "launch" / "scientific.stdout.txt"),
        "status_after": load(HERE / "launch" / "status_after.stdout.txt"),
        "ledger_names": sorted(p.name for p in LEDGER.iterdir()),
    }
    out = {"recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch_imported": None}
    out["integrity"] = integrity()
    out["sealed"] = sealed()
    out["records"] = records(r, out["integrity"]["module_sha256_from_manifest"])
    out["input"] = input_check(child)
    reg = load(URP / "input" / "input_record.json")["baselines"]["original"]
    exp = load(URP / "urp_config.json")["input_expected"]
    b = child["baselines"]["original"]
    out["registered_baselines_original"] = {
        "zero_prediction_mse": [4.132131066150689e-06, reg["zero_prediction"]["mse"],
                                exp["original_zero_prediction_mse"], b["zero_prediction"]["mse"]],
        "scalar_mean_mse": [4.131472976536517e-06, reg["shared_scalar_mean"]["mse"],
                            exp["original_scalar_mean_mse"], b["shared_scalar_mean"]["mse"]],
        "sources": "task statement, input_record.json, urp_config.json input_expected, child record",
    }
    out["registered_baselines_original"]["all_equal"] = all(
        len(set(v)) == 1 for k, v in out["registered_baselines_original"].items() if isinstance(v, list))
    fin = child["final"]
    out["metrics"] = {k: metrics(fin[k]) for k in
                      ("eval_mode", "eval_mode_normalized", "train_mode", "train_mode_normalized")}
    out["metrics"]["step0_eval_original"] = metrics(child["step0"]["post_init"]["eval"]["original"])
    out["metrics"]["normalized_vs_original_rel_diff"] = {
        mode: {k: rel(fin[f"{mode}_normalized"][k], fin[mode][k]) for k in ("rse", "cosine")}
        for mode in ("eval_mode", "train_mode")}
    fe = fin["eval_mode"]
    out["derived"] = {
        "eval_sse_over_scalar_mean_sse": fe["sse"] / b["shared_scalar_mean"]["sse"],
        "eval_sse_over_zero_sse": fe["sse"] / b["zero_prediction"]["sse"],
        "train_minus_eval_output_sse_over_eval_sse_normalized":
            fin["train_minus_eval_output_sse_normalized"] / fin["eval_mode_normalized"]["sse"],
        "eval_rse_minus_train_rse": fe["rse"] - fin["train_mode"]["rse"],
        "step0_eval_rse_minus_final_eval_rse": child["step0"]["post_init"]["eval"]["original"]["rse"] - fe["rse"],
    }
    integ, seal, recs, inp = out["integrity"], out["sealed"], out["records"], out["input"]
    ev_problems = [f"manifest entry mismatch: {p}" for p in integ["entries_mismatching"]]
    ev_problems += [f"external mismatch: {p}" for p in integ["external_mismatching"]]
    ev_problems += [f"pinned mismatch: {k}" for k, v in integ["pinned_match"].items() if not v]
    ev_problems += [f"sealed mismatch: {p}" for p in seal["entries_mismatching"]]
    ev_problems += [f"present but not sealed: {p}" for p in seal["present_not_listed"]]
    ev_problems += [f"record disagreement: {p}" for p in recs["problems"]]
    ev_problems += [f"record check failed: {k}" for k, v in recs["checks"].items() if not v]
    if not integ["manifest_is_approved"]:
        ev_problems.append("manifest is not the approved one")
    if not seal["mirror_byte_identical"]:
        ev_problems.append("mirror differs from the primary ledger")
    if not inp["hashes_equal_recorded"]:
        ev_problems.append("patch tensor hashes differ from the record")
    ev = {"verified": not ev_problems, "problems": ev_problems}
    names = r["ledger_names"]
    starts = sum(1 for n in names if n.startswith("start_"))
    invs = sum(1 for n in names if re.match(r"^invocation_\d{3}\.json$", n))
    ci = classifier_input(child, r["end"], r["inv"], r["inv_end"], starts, invs, ev)
    lim = ci["run"]["limits"]
    compliance = [f"{k}: used {v['used']} > limit {v['limit']}" for k, v in lim.items() if not v["used"] <= v["limit"]]
    run = ci["run"]
    if run["status"] not in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL"):
        compliance.append(f"status {run['status']}")
    if run["updates_done"] != run["updates_planned"]:
        compliance.append("updates done != planned")
    if run["interrupted"] is not False or run["guard_actions"]:
        compliance.append("interrupted or guard action")
    out["strict"] = {
        "criteria": {"max_rse": 0.01, "min_cosine": 0.99, "mode": "eval", "space": "original"},
        "rederived_eval": strict(fe),
        "rederived_train_reported_only": strict(fin["train_mode"]),
        "recorded": child["status"],
        "valid_completion": ev["verified"] and not compliance,
    }
    out["ep_reimplemented"] = ep_reimplemented(fe, ci["baselines"]["scalar_mean_mse"], ev["verified"], compliance)
    out["ep_reimplemented_train_reported_only"] = ep_reimplemented(
        fin["train_mode"], ci["baselines"]["scalar_mean_mse"], ev["verified"], compliance)
    ep_path = EP_DIR / "ep_classify.py"
    if sha(ep_path) != PINNED["../exploratory_progression_20261012/ep_classify.py"]:
        raise SystemExit("ep_classify.py differs from its pinned hash")
    sys.path.insert(0, str(EP_DIR))
    import ep_classify as E  # frozen classifier, hash-checked just above; stdlib only

    frozen = E.classify(ci)
    analysis = load(HERE / "analysis" / "analysis.json")
    out["ep_frozen_classifier"] = frozen
    out["agreement_with_urp_analyze"] = {
        "evidence_verified": analysis["evidence"]["verified"] == ev["verified"],
        "ep_outcome": analysis["exploratory_ep"]["outcome"] == frozen["outcome"] == out["ep_reimplemented"]["outcome"],
        "ep_quantities_equal": analysis["exploratory_ep"]["quantities"] == frozen.get("quantities"),
        "ep_checks_equal": analysis["exploratory_ep"]["checks"] == frozen["checks"],
        "strict": analysis["strict_near_identity"]["rederived_status"] == out["strict"]["rederived_eval"]
        == child["status"],
        "classifier_input_rss_used": lim["rss_bytes"]["used"],
    }
    init_info, init_arr = state_file("init", child["init_state_file"], child["model_state_sha256_init"])
    final_info, final_arr = state_file("final", child["final_state_file"], child["model_state_sha256_final"])
    out["states"] = {
        "init": init_info,
        "final": final_info,
        "init_equals_u_fit_record": child["model_state_sha256_init"] == UFIT_INIT
        == init_info["state_sha256_recomputed_from_file"],
        "same_keys_and_shapes": sorted(init_arr) == sorted(final_arr)
        and all(init_arr[k].shape == final_arr[k].shape for k in init_arr),
        "index_buffers_unchanged": all(np.array_equal(init_arr[k], final_arr[k]) for k in init_arr
                                       if k.endswith("relative_position_index")),
        "group_change_init_to_final": group_change(init_arr, final_arr, child["param_change_total"]),
        "read_back_note": "the child compared the written bytes with the in-memory state bit for bit before "
                          "returning (urp_control.save_state); the run completed, so that comparison passed",
    }
    out["curve"], pts = curve(child)
    out["timing"] = timing(child, r["end"], r["inv"], r["inv_end"], pts)
    out["resources"] = {
        "guard_sampled_max_rss_bytes": r["end"]["guard"]["sampled_max_rss_bytes"],
        "guard_samples": r["end"]["guard"]["samples"],
        "child_sampled_max_rss_bytes": child["rss"]["sampled_max_rss_bytes"],
        "child_os_peak_wset_bytes": child["rss"]["os_peak_wset_bytes"],
        "rss_limit_bytes": r["inv"]["rss_limit_bytes"],
        "available_ram_bytes_at_launch": r["inv"]["available_ram_bytes_at_launch"],
        "evidence_bytes_run_root": r["inv_end"]["evidence_bytes_run_root"],
        "evidence_cap_bytes": r["inv"]["plan"]["evidence_cap_bytes"],
        "guard_action": r["end"]["guard"]["action"],
        "interrupted": r["inv_end"]["interrupted"],
        "child_stdout_stderr_bytes": [(ARM / "child.stdout.txt").stat().st_size,
                                      (ARM / "child.stderr.txt").stat().st_size],
        "supervisor_stderr_bytes": (HERE / "launch" / "scientific.stderr.txt").stat().st_size,
    }
    out["context_u_fit"] = context()
    sys_ok = {
        "evidence_verified": ev["verified"],
        "no_bytecode": not integ["bytecode_files"],
        "unlisted_as_expected": integ["unlisted_as_expected"],
        "sealed_no_crlf": not seal["listing_has_crlf"],
        "baselines_registered_equal": out["registered_baselines_original"]["all_equal"],
        "metrics_equal_recorded": all(all(v["equal_recorded"].values()) for k, v in out["metrics"].items()
                                      if isinstance(v, dict) and "equal_recorded" in v),
        "frozen_and_reimplemented_ep_agree": out["agreement_with_urp_analyze"]["ep_outcome"],
        "analyze_agrees": all(v for k, v in out["agreement_with_urp_analyze"].items() if isinstance(v, bool)),
        "states_ok": all(out["states"][p][k] for p in ("init", "final") for k in
                         ("file_equals_record", "counts_equal_record", "state_hash_agrees", "all_float_finite",
                          "running_var_nonnegative"))
        and out["states"]["init_equals_u_fit_record"] and out["states"]["index_buffers_unchanged"],
        "curve_ok": out["curve"]["schedule_as_registered"] and out["curve"]["all_losses_finite"]
        and out["curve"]["grad_nonfinite_total"] == 0 and out["curve"]["update1_metrics_equal_step0_train_normalized"],
        "context_verified": out["context_u_fit"]["verified"],
    }
    out["summary_checks"] = sys_ok
    out["all_ok"] = all(sys_ok.values())
    out["torch_imported"] = "torch" in sys.modules
    text = json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        with out_path.open("xb") as fh:
            fh.write(text.encode("utf-8"))
        print(f"independent_verification.json {len(text.encode())} bytes "
              f"sha256={hashlib.sha256(text.encode()).hexdigest()}")
    print(text, end="")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
