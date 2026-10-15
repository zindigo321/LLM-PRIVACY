"""ULR-20261015-v1 independent verification (written before the run; executed once after it).

``python -B independent_verify.py <out.json> [config.json] [analysis.json]`` with PYTHONPATH unset and
OPENBLAS_NUM_THREADS=1. It imports neither torch nor any ULR / UOA / UCX package module; it reads files with
``safetensors.numpy`` and computes with numpy indexing and ``math.fsum`` only. Checks:
1. both ledger roots byte-identical;
2. every wire payload of every arm state has exactly 64 c bytes and equals the saved wire latent as little-endian fp32;
3. every per-tile sum (SSE, T, P, D) of every neural prediction (arm states and the source), raw and normalized,
   recomputed with math.fsum from the saved predictions and the targets;
4. input PCA-256 / 64 / 16, SCALAR-4 and INT8-SHARED re-encoded from the targets with an independent implementation
   (bytes compared), re-decoded from the saved bytes, and their correct and wrong-payload sums recomputed;
5. the source INT8 latent bytes re-derived from the saved source latents and the UOA scales;
6. the UPP-20261014-v1 bounds (per-cohort pooled RSE <= 1.0, cosine >= 0.1, finite) and the two-cohort rule,
   the secondary normalized-MSE distances and the element counts, re-derived and compared with ``analysis.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load

EXE = Path(__file__).resolve().parent
PKG = EXE.parent / "ulr_20261015_v1"
TOL = 1e-12
ARMS = {"C16-CONTROL": 16, "C4-REDUCED": 4, "C1-REDUCED": 1}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read(path: str, digest: str, problems: list) -> dict:
    data = Path(path).read_bytes()
    if sha(data) != digest:
        problems.append(f"{Path(path).name}: SHA-256 differs")
    return load(data)


def tile_sums(pred, target):
    out = []
    for a, b in zip(pred.astype(np.float64), target.astype(np.float64)):
        out.append(
            (
                math.fsum(((a - b) ** 2).tolist()),
                math.fsum((b * b).tolist()),
                math.fsum((a * a).tolist()),
                math.fsum((a * b).tolist()),
            )
        )
    return out


def compare(ours, arr, label: str, worst: dict) -> None:
    for i, row in enumerate(ours):
        sc = sum(abs(v) for v in row) or 1.0
        dev = max(abs(row[j] - float(arr[i, j])) for j in range(4)) / sc
        if dev > worst.get(label, 0.0):
            worst[label] = dev
    worst.setdefault(label, 0.0)


def assign4(x, lv32):
    lv = lv32.astype(np.float64)
    t = [(lv[i] + lv[i + 1]) / 2 for i in range(3)]
    x64 = x.astype(np.float64)
    return sum((x64 >= ti).astype(np.int64) for ti in t)


def main(out_path: str, cfg_path: str | None = None, analysis_path: str | None = None) -> int:
    problems: list[str] = []
    cfg = json.loads(Path(cfg_path or PKG / "ulr_config.json").read_text("utf-8"))
    root = Path(cfg["execution"]["run_root"])
    la, lb = root / "ledger", Path(cfg["execution"]["ledger_mirror"])
    names = sorted(p.name for p in la.iterdir())
    roots_ok = names == sorted(p.name for p in lb.iterdir()) and all(
        (la / n).read_bytes() == (lb / n).read_bytes() for n in names
    )
    ev = json.loads((root / "steps/EVAL/eval_record.json").read_text("utf-8"))
    job = json.loads((root / "steps/EVAL/eval_job.json").read_text("utf-8"))
    f = ev["files"]
    sums = read(f["sums"]["path"], f["sums"]["sha256"], problems)
    preds = read(f["predictions"]["path"], f["predictions"]["sha256"], problems)
    lats = read(f["latents"]["path"], f["latents"]["sha256"], problems)
    pays = read(f["payloads"]["path"], f["payloads"]["sha256"], problems)
    u, o = cfg["ucx"], cfg["uoa"]
    d256 = read(u["pca_decoder"]["path"], u["pca_decoder"]["sha256"], problems)
    lv4 = read(o["scalar4_levels"]["path"], o["scalar4_levels"]["sha256"], problems)["levels"]
    ssc = read(o["source_scales"]["path"], o["source_scales"]["sha256"], problems)["scales"].astype(
        np.float64
    )
    x = {"TRAIN": read(u["train_tiles"]["path"], u["train_tiles"]["sha256"], problems)["x"]}
    held = read(u["heldout_tiles"]["path"], u["heldout_tiles"]["sha256"], problems)
    x["EARLY"], x["LATE"] = held["x_EARLY"], held["x_LATE"]
    s32 = struct.unpack("<f", struct.pack("<f", float(u["scale_global"])))[0]
    q8 = np.float32(u["int8_shared_step"])
    worst, bytes_ok, pooled = {}, {}, {}
    arm_keys = [f"{a}:{p}" for a in job["arms"] for p in ("STEP0", "FINAL")]
    for g in ("TRAIN", "EARLY", "LATE"):
        xr = x[g].reshape(x[g].shape[0], -1)
        xn = (xr / np.float32(s32)).astype(np.float32)
        n_t = xr.shape[0]
        perm = ev["extra"][g]["wrong_source_tile_index"]
        for key in arm_keys:
            c = ARMS[key.split(":")[0]]
            p, z = pays[f"{g}.{key}"], lats[f"{g}.{key}.z"]
            bytes_ok[f"{g}.{key}.width_64c"] = p.shape == (n_t, 64 * c) and z.shape == (n_t, c, 4, 4)
            bytes_ok[f"{g}.{key}.bytes_equal_wire_latent"] = bool(
                (p == np.frombuffer(z.astype("<f4").tobytes(), np.uint8).reshape(n_t, -1)).all()
            )
            compare(tile_sums(preds[f"{g}.{key}.y"], xr), sums[f"{g}.{key}.raw"], f"{g}.{key}.raw", worst)
            compare(tile_sums(preds[f"{g}.{key}.y_n"], xn), sums[f"{g}.{key}.norm"], f"{g}.{key}.norm", worst)
        for name in ("SOURCE-LEAKY:FLOAT", "SOURCE-LEAKY:INT8LAT"):
            compare(tile_sums(preds[f"{g}.{name}.y"], xr), sums[f"{g}.{name}.raw"], f"{g}.{name}.raw", worst)
            compare(
                tile_sums(preds[f"{g}.{name}.y_n"], xn), sums[f"{g}.{name}.norm"], f"{g}.{name}.norm", worst
            )
        zs = lats[f"{g}.SOURCE-LEAKY.z"].astype(np.float64).reshape(n_t, 16, 16)
        codes = np.zeros((n_t, 16, 16), np.int64)
        for ch in range(16):
            if ssc[ch] != 0:
                codes[:, ch] = np.clip(np.rint(zs[:, ch] / ssc[ch]), -127, 127)
        bytes_ok[f"{g}.SOURCE-LEAKY:INT8LAT"] = bool(
            (pays[f"{g}.SOURCE-LEAKY:INT8LAT"].view(np.int8).astype(np.int64) == codes.reshape(n_t, -1)).all()
        )
        bytes_ok[f"{g}.SOURCE-LEAKY:FLOAT"] = bool(
            (
                pays[f"{g}.SOURCE-LEAKY:FLOAT"]
                == np.frombuffer(lats[f"{g}.SOURCE-LEAKY.z"].astype("<f4").tobytes(), np.uint8).reshape(
                    n_t, -1
                )
            ).all()
        )
        mean, basis = d256["mean"].astype(np.float64), d256["basis"].astype(np.float64)
        decs = {}
        for name, k in (("PCA-256", 256), ("PCA-64", 64), ("PCA-16", 16)):
            cf = ((xr.astype(np.float64) - mean) @ basis[:k].T).astype(np.float32)
            bytes_ok[f"{g}.{name}"] = bool(
                (
                    np.frombuffer(cf.astype("<f4").tobytes(), np.uint8).reshape(n_t, -1)
                    == pays[f"{g}.{name}"]
                ).all()
            )
            decs[name] = lambda rows, k=k, mean=mean, basis=basis: (
                mean
                + np.stack([np.frombuffer(r.tobytes(), "<f4").astype(np.float64) for r in rows]) @ basis[:k]
            ).astype(np.float32)
        idx = assign4(xr, lv4)
        mine = np.stack(
            [
                sum((idx[i].reshape(-1, 4)[:, k] << (2 * k)) for k in range(4)).astype(np.uint8)
                for i in range(n_t)
            ]
        )
        bytes_ok[f"{g}.SCALAR-4"] = bool((mine == pays[f"{g}.SCALAR-4"]).all())

        def dec4(rows):
            out = []
            for r in rows:
                b = r.astype(np.int64)
                ix = np.empty(1024, np.int64)
                for k in range(4):
                    ix[k::4] = (b >> (2 * k)) & 3
                out.append(lv4[ix])
            return np.stack(out).astype(np.float32)

        decs["SCALAR-4"] = dec4
        k8 = np.clip(np.rint(xr.astype(np.float64) / float(q8)), -127, 127).astype(np.int8)
        bytes_ok[f"{g}.INT8-SHARED"] = bool((k8.view(np.uint8) == pays[f"{g}.INT8-SHARED"]).all())
        decs["INT8-SHARED"] = lambda rows: np.stack([r.view(np.int8).astype(np.float32) * q8 for r in rows])
        for name, fn in decs.items():
            p = pays[f"{g}.{name}"]
            compare(tile_sums(fn(p), xr), sums[f"{g}.{name}.raw"], f"{g}.{name}.raw", worst)
            compare(
                tile_sums(fn(p[perm]), xr), sums[f"{g}.C-WRONG:{name}.raw"], f"{g}.C-WRONG:{name}.raw", worst
            )
        compare(tile_sums(np.zeros_like(xr), xr), sums[f"{g}.C-ZERO.raw"], f"{g}.C-ZERO.raw", worst)
        for key, arr in sums.items():
            if (
                key.startswith(f"{g}.")
                and key.endswith((".raw", ".norm"))
                and arr.ndim == 2
                and arr.shape[1] == 4
            ):
                t = [math.fsum(arr[:, j].tolist()) for j in range(4)]
                pooled[key] = {"mse": t[0] / (arr.shape[0] * 1024), "rse": t[0] / t[1] if t[1] > 0 else None,
                               "cosine": t[3] / math.sqrt(t[1] * t[2]) if t[1] > 0 and t[2] > 0 else None}  # fmt: skip
    agree = {}
    an_p = Path(analysis_path) if analysis_path else EXE / "analysis" / "analysis.json"
    if an_p.is_file():
        an = json.loads(an_p.read_text("utf-8"))
        for arm, c in ARMS.items():
            unit = f"{arm}:FINAL"
            outs = {}
            for g in ("EARLY", "LATE"):
                p = pooled[f"{g}.{unit}.raw"]
                ok = (
                    p["rse"] is not None
                    and p["cosine"] is not None
                    and p["rse"] <= 1.0
                    and p["cosine"] >= 0.1
                    and bool(sums[f"{g}.{unit}.finite"].all())
                )
                outs[g] = "PASS" if ok else "FIT_FAIL"
                agree[f"upp.{unit}.{g}"] = outs[g] == an["B_upp"][unit]["cohorts"][g]["pooled_outcome"]
            agree[f"upp.{unit}.study"] = {2: "PASS", 1: "PARTIAL", 0: "FAIL"}[
                sum(v == "PASS" for v in outs.values())
            ] == an["B_upp"][unit]["study"]
            for g in ("TRAIN", "EARLY", "LATE"):
                mine = pooled[f"{g}.{unit}.norm"]["mse"]
                theirs = an["D_secondary_target"][unit][g]
                agree[f"target.{unit}.{g}"] = (
                    mine == theirs["normalized_mse"] and (mine <= 1e-4) == theirs["met"]
                )
            ar = an["A_element_reduction"]["arms"][unit]
            agree[f"elements.{unit}"] = (
                ar["wire_latent_numel_observed"] == 16 * c
                and ar["payload_bytes_observed"] == 64 * c
                and ar["verified"]
            )
    else:
        problems.append("analysis.json not found")
    bad = {k: v for k, v in worst.items() if v > TOL}
    res = {
        "schema": "ulr-independent-verification-v1",
        "imports": sorted(
            m for m in sys.modules if m.startswith(("torch", "ulr_", "uoa_", "ucx_", "uid_", "upp_"))
        ),
        "ledger_roots_identical": roots_ok,
        "payload_byte_checks": bytes_ok,
        "sum_max_rel_dev": worst,
        "sums_over_tolerance": bad,
        "analysis_agreement": agree,
        "problems": problems,
    }
    res["all_ok"] = (
        roots_ok
        and all(bytes_ok.values())
        and not bad
        and bool(agree)
        and all(agree.values())
        and not problems
        and not res["imports"]
    )
    data = (json.dumps(res, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with open(out_path, "xb") as fh:
        fh.write(data)
    print(
        json.dumps(
            {
                "all_ok": res["all_ok"],
                "problems": problems,
                "bad_sums": len(bad),
                "bytes_ok": all(bytes_ok.values()),
                "agree": f"{sum(agree.values())}/{len(agree)}",
                "sha256": sha(data),
            }
        )
    )
    return 0 if res["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
