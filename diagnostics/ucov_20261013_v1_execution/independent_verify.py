"""Independent verification of the UCOV-20261013-v1 result. Read-only; imports neither torch nor ucov_analyze.

``python -B independent_verify.py [--write]`` from this directory, with PYTHONPATH unset. With ``--write`` it creates
``independent_verification.json`` here (exclusive, LF bytes).

1. **Integrity after the run.**
   - Manifest bytes, 65/65 entries and 5/5 externals; the pinned policy files.
   - Every entry of the sealed list re-hashed, and every run-root and mirror file on it.
   - The mirror is byte-identical to the primary ledger.
2. **Records, per arm.** Status, update count, N and T agree across the child record, the end record, the summary
   and ``--status``. Identity, policy and module hashes; the job; deadlines.
3. **Inputs from the tile bytes** (struct and numpy; no torch, no safetensors): both tensor hashes, T, and the zero and
   scalar-mean baselines, against the record.
4. **Metrics.** Recomputed from the recorded SSE, T, P, D and N in float64 and as exact rationals.
5. **Primary policy UPP-20261013-v1.**
   - Re-implemented here from POLICY.md, without importing ``upp_classify``.
   - The pinned ``upp_classify.classify`` (hash-checked) applied to an input built here.
   - Both compared with the child's verdict and with ``analysis/analysis.json``.
   - The aggregate recomputed both ways.
6. **Secondary references.** The historical strict criterion re-derived; the frozen EP-v1 classifier.
7. **State files without torch.** Header, counts, hashes, BatchNorm counters, finiteness, index buffers, and the
   change from init to final.
8. **Curves and timing scopes**, per arm and for the invocation.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

FROZEN = "3bc2ab302f6cd77b63b9bb8b77894041fd97cbdd63d5fa7f380a4dae080fe824"
HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
UCOV = DIAG / "ucov_20261013_v1"
UPP = DIAG / "upp_20261013_v1"
EP_DIR = DIAG / "exploratory_progression_20261012"
RUN = Path("D:/cgfed-runs/ucov_20261013_v1")
LEDGER, MIRROR = RUN / "ledger", UCOV / "scientific_ledger_mirror"
ARMS = ("U-TILE-1", "U-TILE-2", "U-TILE-3", "U-TILE-4")
SCALE = 0.015751632224572334
UFIT_INIT = "7a36411bbfb83197a433ca23b099e9429dc5699fb5af765a1d6f1c2f5a35e6f7"
POLICY_SHA = "74b277b952e0af68dee00692cde6b95f8ed747d641843bbe98517c70d98c5770"
CLASSIFIER_SHA = "31904623223cc18b216b97bf2cfaa27734487c38a93ebe57efca5de84abecbb1"
EP_SHA = "f7efd10029e8e50071b9a5013b23eefb1d2e3707d07d05e80fda4a690e160cd1"
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked", "relative_position_index")
SUMS = ("finite", "n", "sse", "target_energy", "pred_energy", "dot")
CLASS = {
    "PASS": "P",
    "FIT_FAIL": "F",
    "NONINFORMATIVE": "F",
    "DEGENERATE_INPUT": "F",
    "UNVERIFIED": "U",
    "NOT_ELIGIBLE": "U",
    "NOT_RUN": "U",
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def match(p: Path, digest: str, size: int) -> bool:
    return p.is_file() and p.stat().st_size == size and sha(p) == digest


def ulps(a: float, b: float) -> float:
    return abs(a - b) / math.ulp(b) if b else abs(a - b)


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def read_safetensors(path: Path) -> tuple:
    data = path.read_bytes()
    n = int.from_bytes(data[:8], "little")
    header = json.loads(data[8 : 8 + n])
    meta = header.pop("__metadata__", {})
    return header, meta, data[8 + n :]


def tensor_sha(name: str, dtype: str, shape, raw: bytes) -> str:
    """The ``uid_common.tensor_sha256`` definition: name, NUL, torch dtype string, repr(shape tuple), raw bytes."""
    h = hashlib.sha256()
    for part in (
        name.encode("utf-8"),
        b"\x00",
        dtype.encode("utf-8"),
        repr(tuple(shape)).encode("utf-8"),
        raw,
    ):
        h.update(part)
    return h.hexdigest()


# ---- 1. integrity ---------------------------------------------------------------------------------------------
def integrity() -> dict:
    raw = (UCOV / "manifest_ucov.json").read_bytes()
    m = json.loads(raw)
    bad = [e["path"] for e in m["files"] if not match((UCOV / e["path"]).resolve(), e["sha256"], e["bytes"])]
    ext = [
        e["path"] for e in m["external_bound_files"] if not match(Path(e["path"]), e["sha256"], e["bytes"])
    ]
    listed = {(UCOV / e["path"]).resolve() for e in m["files"]}
    unlisted = sorted(
        p.relative_to(UCOV).as_posix() for p in UCOV.rglob("*") if p.is_file() and p.resolve() not in listed
    )
    by_path = {e["path"]: e["sha256"] for e in m["files"]}
    return {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_is_frozen": hashlib.sha256(raw).hexdigest() == FROZEN,
        "entries": len(m["files"]),
        "entries_mismatching": bad,
        "external": len(m["external_bound_files"]),
        "external_mismatching": ext,
        "unlisted": unlisted,
        "policy_files": {
            "upp_policy.json": sha(UPP / "upp_policy.json") == POLICY_SHA,
            "upp_classify.py": sha(UPP / "upp_classify.py") == CLASSIFIER_SHA,
            "ep_classify.py": sha(EP_DIR / "ep_classify.py") == EP_SHA,
        },
        "module_sha256_from_manifest": {
            n: by_path.get(p)
            for n, p in (
                ("uid_common", "../uid_20261012_v2/uid_common.py"),
                ("uformer_ae", "../uid_20261012_v2/uformer_ae.py"),
                ("uformer_blocks", "../uid_20261012_v2/uformer_blocks.py"),
                ("uid_r1_control", "../uid_20261012_v2_r1/uid_r1_control.py"),
                ("uid_r1_measure", "../uid_20261012_v2_r1/uid_r1_measure.py"),
                ("urp_control", "../urp_20261013_v1/urp_control.py"),
                ("upp_classify", "../upp_20261013_v1/upp_classify.py"),
                ("ucov_control", "ucov_control.py"),
                ("ucov_child", "ucov_child.py"),
            )
        },
    }


def sealed() -> dict:
    listing = HERE / "analysis" / "FINAL_ARTIFACTS_SHA256.txt"
    data = listing.read_bytes()
    rows = []
    for ln in data.decode("utf-8").splitlines()[1:]:
        if ln.strip():
            d, s, n = re.split(r"\s+", ln.strip(), maxsplit=2)
            rows.append((d, int(s), Path(n)))
    listed = {os.path.normcase(str(p.resolve())) for _, _, p in rows}
    present = sorted(str(p) for root in (RUN, MIRROR) for p in root.rglob("*") if p.is_file())
    names = sorted(p.name for p in LEDGER.iterdir())
    return {
        "listing_sha256": hashlib.sha256(data).hexdigest(),
        "listing_has_crlf": b"\r\n" in data,
        "entries": len(rows),
        "entries_mismatching": [str(p) for d, s, p in rows if not match(p, d, s)],
        "present_not_listed": [p for p in present if os.path.normcase(str(Path(p).resolve())) not in listed],
        "ledger_records": names,
        "mirror_byte_identical": names == sorted(p.name for p in MIRROR.iterdir())
        and all((LEDGER / n).read_bytes() == (MIRROR / n).read_bytes() for n in names),
    }


# ---- 3. inputs from the tile bytes ---------------------------------------------------------------------------
def fsum_stats(pred: np.ndarray, x: np.ndarray) -> dict:
    a, b = pred.astype(np.float64).ravel(), x.astype(np.float64).ravel()
    sse = math.fsum(((a - b) ** 2).tolist())
    return {"sse": sse, "target_energy": math.fsum((b * b).tolist()), "mse": sse / b.size}


def tile_check(arm: str, child: dict | None, exp: dict, tile_spec: dict) -> dict:
    path = UCOV / tile_spec["patch_file"]
    header, _, body = read_safetensors(path)
    lo, hi = header["x"]["data_offsets"]
    x = np.frombuffer(body[lo:hi], dtype="<f4").reshape(header["x"]["shape"])
    xn = (x / np.float32(SCALE)).astype(np.float32)
    out = {
        "file_sha256": sha(path),
        "file_equals_pinned": sha(path) == tile_spec["patch_file_sha256"],
        "keys": sorted(header),
        "dtype": header["x"]["dtype"],
        "shape": header["x"]["shape"],
        "tensor_sha256_original": tensor_sha("x", "torch.float32", x.shape, x.tobytes()),
        "tensor_sha256_normalized": tensor_sha("x", "torch.float32", xn.shape, xn.tobytes()),
        "all_finite": bool(np.isfinite(x).all()),
        "zero_elements": int((x == 0).sum()),
        "normalized_max_abs": float(np.abs(xn).max()),
    }
    out["hashes_equal_registered"] = (out["tensor_sha256_original"], out["tensor_sha256_normalized"]) == (
        exp["original_sha256"],
        exp["normalized_sha256"],
    )
    mean32 = np.float32(math.fsum(x.astype(np.float64).ravel().tolist()) / x.size)
    zero, mean = fsum_stats(np.zeros_like(x), x), fsum_stats(np.full_like(x, mean32), x)
    out["original"] = {
        "target_energy": zero["target_energy"],
        "zero_mse": zero["mse"],
        "scalar_mean_mse": mean["mse"],
        "ulps_vs_registered": {
            "T": ulps(zero["target_energy"], exp["original_target_energy"]),
            "zero": ulps(zero["mse"], exp["original_zero_prediction_mse"]),
            "mean": ulps(mean["mse"], exp["original_scalar_mean_mse"]),
        },
    }
    out["normalized_max_abs_equal_registered"] = out["normalized_max_abs"] == exp["normalized_max_abs"]
    if child is not None:
        b = child["baselines"]["original"]
        out["child_baselines_equal_registered"] = (
            b["zero_prediction"]["mse"],
            b["shared_scalar_mean"]["mse"],
        ) == (exp["original_zero_prediction_mse"], exp["original_scalar_mean_mse"])
    return out


# ---- 4. metrics ------------------------------------------------------------------------------------------------
def metrics(m: dict) -> dict:
    t, p, sse, d, n = m["target_energy"], m["pred_energy"], m["sse"], m["dot"], m["n"]
    f = {"rse": sse / t, "cosine": d / math.sqrt(p * t), "mse": sse / n, "norm_ratio": math.sqrt(p / t)}
    f["equal_recorded"] = {k: f[k] == m[k] for k in ("rse", "cosine", "mse", "norm_ratio")}
    fr = {k: Fraction(m[k]) for k in ("sse", "target_energy")}
    f["ulps_rse_float64_vs_exact"] = ulps(f["rse"], float(fr["sse"] / fr["target_energy"]))
    f["sse_identity_rel_err"] = rel(p + t - 2 * d, sse)
    return f


# ---- 5. primary policy, re-implemented from POLICY.md ----------------------------------------------------------
def upp_reimplemented(
    fe: dict | None, zero_mse: float, mean_mse: float, valid: bool, started: bool, degenerate: bool
) -> dict:
    """UPP-20261013-v1 from POLICY.md: order UNVERIFIED, DEGENERATE_INPUT, NOT_RUN, NOT_ELIGIBLE, then metrics."""
    if degenerate:
        return {"outcome": "DEGENERATE_INPUT"}
    if not started:
        return {"outcome": "NOT_RUN"}
    if not valid:
        return {"outcome": "NOT_ELIGIBLE"}
    vals = (fe["sse"], fe["target_energy"], fe["pred_energy"], fe["dot"], zero_mse, mean_mse)
    if fe["finite"] is not True or not all(isinstance(v, float) and math.isfinite(v) for v in vals):
        return {"outcome": "FIT_FAIL", "why": "undefined"}
    if fe["target_energy"] <= 0 or fe["pred_energy"] <= 0 or fe["sse"] < 0:
        return {"outcome": "FIT_FAIL", "why": "undefined"}
    rse = fe["sse"] / fe["target_energy"]
    cos = fe["dot"] / math.sqrt(fe["pred_energy"] * fe["target_energy"])
    mse, best = fe["sse"] / fe["n"], min(zero_mse, mean_mse)
    c1, c2 = rse <= 0.50, cos >= 0.65
    if best == 0:
        return {"outcome": "NONINFORMATIVE" if c1 and c2 else "FIT_FAIL"}
    c3 = mse <= 0.80 * best
    return {
        "outcome": "PASS" if c1 and c2 and c3 else "FIT_FAIL",
        "rse": [rse, 0.50, c1],
        "cosine": [cos, 0.65, c2],
        "mse": [mse, 0.80 * best, c3],
        "mse_over_best": mse / best,
        "baseline_improvement": 1 - mse / best,
    }


def aggregate_reimplemented(outcomes: dict) -> dict:
    counts = {c: sum(1 for o in outcomes.values() if CLASS[o] == c) for c in ("P", "F", "U")}
    counts["U"] += 4 - len(outcomes)
    d = (
        "COVERAGE_CRITERION_MET"
        if counts["P"] >= 3
        else "COVERAGE_CRITERION_NOT_MET"
        if counts["F"] >= 2
        else "INCONCLUSIVE"
    )
    return {"counts": counts, "decision": d}


# ---- 7. state files without torch --------------------------------------------------------------------------------
DTYPES = {"F32": ("torch.float32", "<f4"), "I64": ("torch.int64", "<i8")}


def state_file(path: Path, rec: dict, recorded_state: str) -> tuple:
    header, meta, body = read_safetensors(path)
    h, arrays, params, buffers = hashlib.sha256(), {}, 0, 0
    for key in sorted(header):
        info = header[key]
        tdt, npdt = DTYPES[info["dtype"]]
        lo, hi = info["data_offsets"]
        raw = body[lo:hi]
        h.update(tensor_sha(key, tdt, info["shape"], raw).encode("ascii"))
        arrays[key] = np.frombuffer(raw, dtype=npdt).reshape(info["shape"])
        n = math.prod(info["shape"])
        if key.endswith(BUFFER_SUFFIXES):
            buffers += n
        else:
            params += n
    floats = [a for a in arrays.values() if a.dtype.kind == "f"]
    out = {
        "file_sha256": sha(path),
        "bytes": path.stat().st_size,
        "file_equals_record": (sha(path), path.stat().st_size) == (rec["file_sha256"], rec["bytes"]),
        "tensors": len(header),
        "parameter_elements": params,
        "buffer_elements": buffers,
        "counts_equal_record": (len(header), params, buffers)
        == (rec["tensors"], rec["parameter_elements"], rec["buffer_elements"]),
        "metadata_point": meta.get("point"),
        "metadata_arm": meta.get("arm"),
        "state_sha256_recomputed": h.hexdigest(),
        "state_hash_agrees": h.hexdigest()
        == meta.get("state_sha256")
        == rec["state_sha256"]
        == recorded_state,
        "num_batches_tracked": sorted(
            {int(a) for k, a in arrays.items() if k.endswith("num_batches_tracked")}
        ),
        "all_float_finite": all(bool(np.isfinite(a).all()) for a in floats),
        "running_var_nonnegative": all(
            bool((a >= 0).all()) for k, a in arrays.items() if k.endswith("running_var")
        ),
    }
    return out, arrays


def group_change(init: dict, final: dict) -> dict:
    acc = {g: [[], [], 0, 0] for g in GROUPS}
    for key, a in final.items():
        if key.endswith(BUFFER_SUFFIXES):
            continue
        g = ".".join(key.split(".")[:2])
        d = a.astype(np.float64) - init[key].astype(np.float64)
        acc[g][0].append(float(np.square(d).sum()))
        acc[g][1].append(float(np.square(init[key].astype(np.float64)).sum()))
        acc[g][2] += int((d != 0).sum())
        acc[g][3] += a.size
    return {
        g: {
            "relative_change": math.sqrt(math.fsum(v[0])) / math.sqrt(math.fsum(v[1])),
            "numel_changed": v[2],
            "numel": v[3],
        }
        for g, v in acc.items()
    }


# ---- 8. curve ----------------------------------------------------------------------------------------------------
def curve(arm_dir: Path) -> dict:
    pts = [
        json.loads(x) for x in (arm_dir / "curve.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    ups = [p["update"] for p in pts]
    loss = {p["update"]: p["pre_update_train_loss"] for p in pts}
    t = [p["monotonic_s"] for p in pts]
    last = [k for k in ups if k >= 2000]
    drops = [1 - loss[b] / loss[a] for a, b in zip(last, last[1:])]
    return {
        "points": len(pts),
        "schedule_as_registered": ups == [1, *range(100, 3001, 100)],
        "all_losses_finite": all(math.isfinite(v) for v in loss.values()),
        "grad_nonfinite_total": sum(g["nonfinite"] for p in pts for g in p["grad"].values()),
        "every_parameter_had_a_gradient": all(
            g["numel_with_grad"] == g["numel"] for p in pts for g in p["grad"].values()
        ),
        "loss_strictly_decreasing": all(loss[b] < loss[a] for a, b in zip(ups, ups[1:])),
        "increases": [[a, b] for a, b in zip(ups, ups[1:]) if loss[b] >= loss[a]],
        "loss": {str(k): loss[k] for k in (1, 100, 500, 1000, 2000, 2500, 3000) if k in loss},
        "pre_update_rse": {
            str(p["update"]): p["pre_update_train"]["rse"]
            for p in pts
            if p["update"] in (1, 100, 500, 1000, 2000, 2500, 3000)
        },
        "loss_ratio_3000_over_2000": loss[3000] / loss[2000] if 3000 in loss else None,
        "loss_ratio_3000_over_2500": loss[3000] / loss[2500] if 3000 in loss else None,
        "drop_per_100_last_1000_min_max": [min(drops), max(drops)] if drops else None,
        "relative_change_from_init_at_end": {
            g: pts[-1]["param_change_from_init"][g]["delta_l2"]
            / pts[-1]["param_change_from_init"][g]["ref_l2"]
            for g in GROUPS
        },
        "train_loop_s": t[-1] - t[0],
        "mean_s_per_update": (t[-1] - t[0]) / (ups[-1] - ups[0]),
        "rss_at_points_min_max": [
            min(p["rss_bytes_at_point"] for p in pts),
            max(p["rss_bytes_at_point"] for p in pts),
        ],
    }


# ---- main ------------------------------------------------------------------------------------------------------
def strict(m: dict) -> str:
    t, p = m["target_energy"], m["pred_energy"]
    if m["finite"] is not True or t <= 0 or p <= 0:
        return "FAIL"
    return "PASS" if m["sse"] / t <= 0.01 and m["dot"] / math.sqrt(p * t) >= 0.99 else "FAIL"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out_path = HERE / "independent_verification.json"
    if args.write and out_path.exists():
        raise SystemExit(f"{out_path} exists; never overwritten")
    resolved = load(UCOV / "resolved_config.json")
    inv, inv_end = load(LEDGER / "invocation_001.json"), load(LEDGER / "invocation_001_end.json")
    status_after = load(HERE / "launch" / "status_after.stdout.txt")
    sci = load(HERE / "launch" / "scientific.stdout.txt")
    analysis = load(HERE / "analysis" / "analysis.json")
    names = sorted(p.name for p in LEDGER.iterdir())
    out: dict = {
        "recorded_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "integrity": integrity(),
        "sealed": sealed(),
    }
    integ, seal = out["integrity"], out["sealed"]
    global_problems = []
    if not integ["manifest_is_frozen"] or integ["entries_mismatching"] or integ["external_mismatching"]:
        global_problems.append("package integrity after the run")
    if not all(integ["policy_files"].values()):
        global_problems.append("policy file hashes")
    if seal["entries_mismatching"] or seal["present_not_listed"] or not seal["mirror_byte_identical"]:
        global_problems.append("sealed list or mirror")
    if inv["identity"]["ucov_manifest_sha256"] != FROZEN:
        global_problems.append("invocation names another manifest")
    if (
        inv["identity"]["primary_policy"]["policy_sha256"] != POLICY_SHA
        or inv["plan"]["primary_policy_sha256"] != POLICY_SHA
    ):
        global_problems.append("invocation names another policy")
    if [a["arm"] for a in inv_end["arms"]] != list(ARMS) or sci != inv_end:
        global_problems.append("invocation summary arms, or stdout differs from the ledger summary")
    starts = sum(1 for n in names if n.startswith("start_"))
    invs = sum(1 for n in names if re.match(r"^invocation_\d{3}\.json$", n))
    out["ledger"] = {
        "records": names,
        "invocations": invs,
        "starts": starts,
        "status_after": {
            "invocations": status_after["invocations"],
            "starts": status_after["starts"],
            "arms": status_after["arms"],
        },
        "global_stop": inv_end.get("global_stop"),
    }
    sys.path.insert(0, str(UPP))
    import upp_classify as P  # pinned; hash checked in integrity()

    if not integ["policy_files"]["upp_classify.py"]:
        raise SystemExit("upp_classify.py differs from its pinned hash")
    policy = P.load_policy(UPP / "upp_policy.json")
    sys.path.insert(0, str(EP_DIR))
    import ep_classify as E  # frozen; hash checked in integrity()

    by_arm = {a["arm"]: a for a in inv_end["arms"]}
    out["arms"], re_out, fr_out = {}, {}, {}
    for arm in ARMS:
        arm_dir = RUN / "arms" / arm
        started = f"start_{arm}.json" in names
        end = load(LEDGER / f"end_{arm}.json") if started else None
        child = load(arm_dir / "child_record.json") if (arm_dir / "child_record.json").is_file() else None
        exp = resolved["inputs_expected"][arm]
        row: dict = {"started": started, "summary_status": by_arm[arm]["status"]}
        row["input"] = tile_check(arm, child, exp, resolved["inputs"]["tiles"][arm])
        problems = list(global_problems)
        if not row["input"]["file_equals_pinned"] or not row["input"]["hashes_equal_registered"]:
            problems.append("tile bytes or tensor hashes")
        valid = False
        if started:
            st = [
                child["status"] if child else None,
                end["status"],
                by_arm[arm]["status"],
                status_after["arms"][arm],
            ]
            row["status_agreement"] = st
            if len(set(st)) != 1:
                problems.append(f"status disagreement {st}")
            if child is None:
                problems.append("no child record")
            else:
                if child["model_state_sha256_init"] != UFIT_INIT:
                    problems.append("initial state")
                if (
                    child["modules"]["sha256"] != integ["module_sha256_from_manifest"]
                    or child["modules_at_end"]["sha256"] != integ["module_sha256_from_manifest"]
                ):
                    problems.append("module hashes")
                if child.get("primary_policy", {}).get("policy_sha256") not in (None, POLICY_SHA):
                    problems.append("child policy hash")
                if not row["input"].get("child_baselines_equal_registered", True):
                    problems.append("child baselines")
                if (
                    child["input"]["patch_file_sha256"]
                    != resolved["inputs"]["tiles"][arm]["patch_file_sha256"]
                ):
                    problems.append("child read another tile")
            job = load(arm_dir / "job.json")
            if (job["mode"], job["fixture_hooks"], job["arm"], job["primary_policy_sha256"]) != (
                "scientific",
                None,
                arm,
                POLICY_SHA,
            ):
                problems.append("job fields")
            rss = max(
                end["guard"]["sampled_max_rss_bytes"],
                *(
                    [child["rss"]["sampled_max_rss_bytes"], child["rss"]["os_peak_wset_bytes"]]
                    if child
                    else []
                ),
            )
            limits_ok = (
                end["arm_wall_s"] <= 1800.0
                and inv_end["aggregate_wall_s"] <= 9000.0
                and starts <= 4
                and invs <= 1
                and rss <= inv["rss_limit_bytes"]
                and inv_end["evidence_bytes_run_root"] <= 512 * 2**20
            )
            valid = (
                child is not None
                and child["status"] in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL")
                and child["updates_done"] == 3000
                and end["guard"]["action"] is None
                and inv_end["interrupted"] is False
                and limits_ok
            )
            row["resources"] = {
                "arm_wall_s": end["arm_wall_s"],
                "guard_action": end["guard"]["action"],
                "guard_sampled_max_rss_bytes": end["guard"]["sampled_max_rss_bytes"],
                "rss_used_max": rss,
                "limits_ok": limits_ok,
                "child_process_s": None
                if child is None
                else child["child_monotonic_s"]["end"] - child["child_monotonic_s"]["begin"],
                "updates_done": None if child is None else child.get("updates_done"),
            }
        verified = not problems
        row["evidence"] = {"verified": verified, "problems": problems}
        fe = child["final"]["eval_mode"] if child and child.get("final") else None
        degenerate = not exp["usable"]
        if verified:
            ri = upp_reimplemented(
                fe,
                exp["original_zero_prediction_mse"],
                exp["original_scalar_mean_mse"],
                valid,
                started,
                degenerate,
            )
        else:
            ri = {"outcome": "UNVERIFIED"}
        row["upp_reimplemented"] = ri
        re_out[arm] = ri["outcome"]
        # the pinned classifier on an input built here
        ci = {
            "evidence": {"verified": verified, "problems": problems},
            "input": {
                "all_finite": exp["degeneracy_checks"]["all_finite"],
                "target_energy": exp["original_target_energy"],
                "scalar_mean_mse": exp["original_scalar_mean_mse"],
                "normalized_max_abs": exp["normalized_max_abs"],
            },
            "run": {
                "started": started,
                "status": by_arm[arm]["status"],
                "updates_done": None if child is None else child.get("updates_done"),
                "updates_planned": 3000,
                "interrupted": inv_end["interrupted"],
                "guard_actions": []
                if end is None or end["guard"]["action"] is None
                else [end["guard"]["action"]],
                "limits": {
                    "arm_wall_s": {"used": None if end is None else end["arm_wall_s"], "limit": 1800.0},
                    "aggregate_wall_s": {"used": inv_end["aggregate_wall_s"], "limit": 9000.0},
                    "training_starts": {"used": starts, "limit": 4},
                    "invocations": {"used": invs, "limit": 1},
                    "rss_bytes": {
                        "used": row.get("resources", {}).get("rss_used_max"),
                        "limit": inv["rss_limit_bytes"],
                    },
                    "evidence_bytes": {"used": inv_end["evidence_bytes_run_root"], "limit": 512 * 2**20},
                },
            },
            "baselines": {
                "zero_prediction_mse": exp["original_zero_prediction_mse"],
                "scalar_mean_mse": exp["original_scalar_mean_mse"],
            },
        }
        if fe is not None:
            ci["final_eval"] = {k: fe[k] for k in SUMS}
        fr = P.classify(ci, policy)
        row["upp_frozen_classifier"] = {
            k: fr.get(k) for k in ("outcome", "class", "reasons", "quantities", "checks")
        }
        fr_out[arm] = fr["outcome"]
        an = analysis["arms"][arm]["primary"]
        row["agreement"] = {
            "reimplemented_eq_classifier": ri["outcome"] == fr["outcome"],
            "classifier_eq_analysis": fr["outcome"] == an["outcome"]
            and fr.get("quantities") == an.get("quantities"),
            "child_verdict": None if child is None else (child.get("primary_verdict") or {}).get("outcome"),
            "child_eq_classifier": child is None
            or not child.get("primary_verdict")
            or fr["class"] == "U"
            or child["primary_verdict"]["outcome"] == fr["outcome"],
            "child_status_matches_verdict": child is None
            or child["status"] not in ("COMPLETED_PASS", "COMPLETED_FIT_FAIL")
            or (child["status"] == "COMPLETED_PASS") == (child["primary_verdict"]["outcome"] == "PASS"),
        }
        if fe is not None:
            row["metrics"] = {
                k: metrics(child["final"][k])
                for k in ("eval_mode", "eval_mode_normalized", "train_mode", "train_mode_normalized")
            }
            row["secondary"] = {
                "historical_strict": strict(fe),
                "ep_v1": E.classify(
                    {
                        "evidence": ci["evidence"],
                        "final_eval": ci["final_eval"],
                        "baselines": {"scalar_mean_mse": exp["original_scalar_mean_mse"]},
                        "run": {
                            **{
                                k: ci["run"][k]
                                for k in (
                                    "status",
                                    "updates_done",
                                    "updates_planned",
                                    "interrupted",
                                    "guard_actions",
                                    "limits",
                                )
                            },
                            "prerequisite": None,
                        },
                    }
                )["outcome"],
            }
            row["secondary"]["analysis_agrees"] = (
                row["secondary"]["historical_strict"]
                == analysis["arms"][arm]["secondary"]["historical_strict"]["outcome"]
                and row["secondary"]["ep_v1"] == analysis["arms"][arm]["secondary"]["ep_v1"]["outcome"]
            )
            si, ia = state_file(
                arm_dir / "init_state.safetensors", child["init_state_file"], child["model_state_sha256_init"]
            )
            sf, fa = state_file(
                arm_dir / "final_state.safetensors",
                child["final_state_file"],
                child["model_state_sha256_final"],
            )
            row["states"] = {
                "init": si,
                "final": sf,
                "change": group_change(ia, fa),
                "index_buffers_unchanged": all(
                    np.array_equal(ia[k], fa[k]) for k in ia if k.endswith("relative_position_index")
                ),
            }
            row["curve"] = curve(arm_dir)
        out["arms"][arm] = row
    out["aggregate"] = {
        "reimplemented": aggregate_reimplemented(re_out),
        "classifier": P.aggregate(fr_out, 4, policy),
        "analysis": analysis["aggregate"],
    }
    out["aggregate"]["all_agree"] = (
        out["aggregate"]["reimplemented"]["decision"]
        == out["aggregate"]["classifier"]["decision"]
        == analysis["aggregate"]["decision"]
        and out["aggregate"]["reimplemented"]["counts"]
        == out["aggregate"]["classifier"]["counts"]
        == analysis["aggregate"]["counts"]
    )
    out["init_states_identical"] = (
        len({r["states"]["init"]["state_sha256_recomputed"] for r in out["arms"].values() if "states" in r})
        <= 1
    )
    out["global_problems"] = global_problems
    out["all_ok"] = (
        not global_problems
        and out["aggregate"]["all_agree"]
        and out["init_states_identical"]
        and all(
            all(v for k, v in r["agreement"].items() if isinstance(v, bool)) for r in out["arms"].values()
        )
    )
    out["torch_imported"] = "torch" in sys.modules
    text = json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        with out_path.open("xb") as fh:
            fh.write(text.encode("utf-8"))
        print(
            f"independent_verification.json {len(text.encode())} bytes sha256={hashlib.sha256(text.encode()).hexdigest()}"
        )
    print(
        json.dumps(
            {
                "all_ok": out["all_ok"],
                "aggregate": out["aggregate"]["classifier"],
                "arms": {a: r["upp_frozen_classifier"]["outcome"] for a, r in out["arms"].items()},
                "global_problems": global_problems,
            },
            indent=1,
        )
    )
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
