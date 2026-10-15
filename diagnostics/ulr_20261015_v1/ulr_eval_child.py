"""ULR-20261015-v1 EVAL child: the forward-only evaluation of every arm state, comparator and control on TRAIN, EARLY
and LATE, in one fresh process.

``python -B ulr_eval_child.py <job.json>``. Started only after the three arms completed. Its ledger start record
carries the SHA-256 of every state and shared file, and this is the only ULR process that opens the held-out tiles.
Gradients are disabled, optimizer construction is refused, and every model is in eval mode. Fingerprints, files and
the RNG are compared before and after.

Every arm state (step zero and final) runs through two separate objects:
- the sender, the full saved model: x -> encoder -> reduce -> 64 c bytes;
- the standalone receiver, holding only the expand and decoder tensors of the same file: bytes -> parse -> expand ->
  decoder -> y.

Comparators are decoded from their parsed payload bytes plus the persisted shared state:
- input PCA-256 / 64 / 16 (fp32 coefficients, from the UCX TRAIN basis);
- SCALAR-4 (the UOA levels) and INT8-SHARED (the UCX step);
- the inherited source model (UOA U-LEAKY-LONG FLOAT 256 fp32 latents and INT8 256 int8 latents, with the UOA
  scales).

Controls are C-ZERO, C-MEAN, C-WRONG (next-snapshot same-position payloads crossing clients), C-ZWIRE (an all-zero
wire payload) and C-ZLAT (the source's zero latent).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ulr_common as K  # noqa: E402


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    attempts: list[str] = []
    K.install_open_guard(job["forbidden_paths"], attempts)
    out = Path(job["out_dir"])
    rec: dict = {"phase": "eval", "mode": job["mode"], "pid": os.getpid(), "status": "EXECUTION_ERROR"}
    t0 = time.monotonic()
    try:
        K.start_watchdog(job, "ulr_eval_child")
        K.check_env(rec)
        import torch
        import uid_common as C

        rec["cpu"] = C.configure_cpu(job["cpu"]["intra_op_threads"], job["cpu"]["inter_op_threads"])

        def refuse(self, *args, **kwargs):
            raise RuntimeError("the evaluation child must not construct an optimizer")

        torch.optim.Optimizer.__init__ = refuse
        rec["optimizer_construction_refused"] = True
        torch.set_grad_enabled(False)
        run(job, rec, out)
        K.check_imports(rec)
        rec["status"] = "COMPLETED"
    except KeyboardInterrupt:
        rec["status"] = "INCOMPLETE"
    except Exception as exc:
        rec["status"] = "EXECUTION_ERROR"
        rec["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        rec["split_enforcement"] = {"forbidden_open_attempts": attempts, "audit_hook_installed": True}
        rec["memory"] = K.peak_memory()
        rec["child_wall_s"] = time.monotonic() - t0
        K.write_json(out / "eval_record.json", rec)
    return K.EXIT[rec["status"]]


def as_u8(payloads: list[bytes]):
    import numpy as np

    return np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(len(payloads), -1).copy()


def pca_encode(xr, mean32, basis32, k: int) -> list[bytes]:
    import numpy as np

    c = ((xr.astype(np.float64) - mean32.astype(np.float64)) @ basis32[:k].astype(np.float64).T).astype(
        np.float32
    )
    return [r.astype("<f4").tobytes() for r in c]


def pca_decode(payloads: list[bytes], mean32, basis32, k: int):
    import numpy as np

    if any(len(b) != 4 * k for b in payloads):
        raise ValueError(f"PCA-{k} payloads must have {4 * k} bytes")
    c = np.stack([np.frombuffer(b, dtype="<f4") for b in payloads]).astype(np.float64)
    return (mean32.astype(np.float64) + c @ basis32[:k].astype(np.float64)).astype(np.float32)


def run(job: dict, rec: dict, out: Path) -> None:
    import numpy as np
    import torch
    import ucx_codecs as X  # vendored (INT8-SHARED)
    import uid_common as C
    import ulr_model as Mo
    import ulr_spec as S
    import uoa_codecs as Q  # vendored (SCALAR-4)
    import uoa_model as UM  # vendored (source model FLOAT / INT8 historical paths)
    import uoa_sums as U

    cfg, scale = job["model"], float(job["scale_global"])
    s32 = float(np.float32(scale))
    rng0 = Mo.rng_sha256()
    train = K.load_checked(job["train_tiles"], job["train_tiles_sha256"], "torch")["x"]
    held = K.load_checked(job["heldout_tiles"], job["heldout_tiles_sha256"], "torch")
    mean_tile = K.load_checked(job["train_mean_tile"], job["train_mean_tile_sha256"], "torch")["x"]
    if not torch.equal(train.double().mean(dim=0).to(torch.float32), mean_tile):
        raise K.ULRError("the TRAIN mean tile differs from its recomputation")
    sets = {"TRAIN": train, "EARLY": held["x_EARLY"], "LATE": held["x_LATE"]}
    rec["sets"] = {g: list(v.shape) for g, v in sets.items()}
    d256 = K.load_checked(job["pca_decoder"], job["pca_decoder_sha256"])
    lv4 = K.load_checked(job["scalar4_levels"], job["scalar4_levels_sha256"])["levels"]
    q8 = np.float32(job["int8_shared_step"])
    src = job["source"]
    source = Mo.load_source(src, cfg)
    src_scales = K.load_checked(job["source_scales"], job["source_scales_sha256"])["scales"]
    senders, receivers, recv_info, fp0 = {}, {}, {}, {}
    for arm, a in job["arms"].items():
        for point, f in a["states"].items():
            key = S.arm_method(arm, point)
            senders[key] = Mo.load_arm_state(f, int(a["c"]), cfg)
            receivers[key], recv_info[key] = Mo.receiver_from_state(f, int(a["c"]), cfg)
            fp0[key] = Mo.fingerprint(senders[key])["state_sha256"]
            fp0[key + ".receiver"] = Mo.fingerprint(receivers[key])["state_sha256"]
    fp0[S.SOURCE] = Mo.fingerprint(source)["state_sha256"]
    rec["receivers"] = recv_info
    sums, preds, lats, pays, extra = {}, {}, {}, {}, {}

    def put(g: str, name: str, y, xr, y_n=None, xn=None):
        yr = np.asarray(y, dtype=np.float32).reshape(len(xr), -1)
        sums[f"{g}.{name}.raw"] = U.sums4(yr, xr)
        sums[f"{g}.{name}.raw_blocks"] = U.block_sums(yr, xr)
        sums[f"{g}.{name}.finite"] = U.finite_rows(yr).astype(np.uint8)
        if y_n is not None:
            sums[f"{g}.{name}.norm"] = U.sums4(np.asarray(y_n, dtype=np.float32).reshape(len(xr), -1), xn)

    t_eval = time.perf_counter()
    for g, x in sets.items():
        n_t = x.shape[0]
        xr = np.ascontiguousarray(x.reshape(n_t, -1).numpy())
        xn = Mo.normalize(x, scale).reshape(n_t, -1).numpy()
        keys = job["cohort_keys"][g]
        nxt = {tuple(a): tuple(b) for a, b in job["wrong_map"][g]}
        srcix = [keys.index(list(nxt[tuple(k)])) for k in keys]
        perm = [
            srcix[i // S.TILES_PER_SNAPSHOT] * S.TILES_PER_SNAPSHOT + i % S.TILES_PER_SNAPSHOT
            for i in range(n_t)
        ]
        extra[g] = {"wrong_source_tile_index": perm}
        t = time.perf_counter()
        # ---- input PCA (fp32 coefficients), SCALAR-4, INT8-SHARED --------------------------------------------------
        for name, k in S.INPUT_PCA.items():
            b = pca_encode(xr, d256["mean"], d256["basis"], k)
            pays[f"{g}.{name}"] = as_u8(b)
            put(g, name, pca_decode(b, d256["mean"], d256["basis"], k), xr)
            put(g, f"C-WRONG:{name}", pca_decode([b[j] for j in perm], d256["mean"], d256["basis"], k), xr)
        b = Q.sq_encode(xr, lv4)
        pays[f"{g}.SCALAR-4"] = as_u8(b)
        put(g, "SCALAR-4", Q.sq_decode([bytes(p) for p in b], lv4), xr)
        put(g, "C-WRONG:SCALAR-4", Q.sq_decode([bytes(b[j]) for j in perm], lv4), xr)
        b = [X.int8_shared_to_bytes(X.int8_encode(xr[i], q8)[0]) for i in range(n_t)]
        pays[f"{g}.INT8-SHARED"] = as_u8(b)

        def dec8(bb):
            return np.stack([X.int8_decode(X.int8_shared_from_bytes(p), q8) for p in bb])

        put(g, "INT8-SHARED", dec8(b), xr)
        put(g, "C-WRONG:INT8-SHARED", dec8([b[j] for j in perm]), xr)
        extra[g]["codecs_s"] = time.perf_counter() - t
        # ---- the inherited source model (historical UOA paths, 256 latent values) ---------------------------------
        t = time.perf_counter()
        fo = UM.float_outputs(source, x, scale)
        qo = UM.int8_outputs(source, fo["z"], src_scales, scale)
        sf, sq = f"{S.SOURCE}:FLOAT", f"{S.SOURCE}:INT8LAT"
        pays[f"{g}.{sf}"], pays[f"{g}.{sq}"] = as_u8(fo["payloads"]), as_u8(qo["payloads"])
        put(g, sf, fo["y"], xr, fo["y_n"], xn)
        put(g, sq, qo["y"], xr, qo["y_n"], xn)
        wy_n, wy = UM.decode_float_payloads(source, [fo["payloads"][j] for j in perm], scale)
        put(g, f"C-WRONG:{sf}", wy, xr, wy_n, xn)
        wy_n, wy = UM.decode_int8_payloads(source, [qo["payloads"][j] for j in perm], src_scales, scale)
        put(g, f"C-WRONG:{sq}", wy, xr, wy_n, xn)
        zy_n = source.decode(torch.zeros((1, *S.BACKBONE_INTERNAL_SHAPE)))
        put(
            g,
            f"C-ZLAT:{S.SOURCE}",
            np.broadcast_to(Mo.denormalize(zy_n, scale).reshape(1, -1).numpy(), xr.shape).copy(),
            xr,
            np.broadcast_to(zy_n.reshape(1, -1).numpy(), xr.shape).copy(),
            xn,
        )
        for name, o in ((sf, fo), (sq, qo)):
            preds[f"{g}.{name}.y"] = o["y"].reshape(n_t, -1).numpy()
            preds[f"{g}.{name}.y_n"] = o["y_n"].reshape(n_t, -1).numpy()
        lats[f"{g}.{S.SOURCE}.z"] = fo["z"].numpy()
        extra[g]["source_s"] = time.perf_counter() - t
        # ---- the arms: sender -> bytes -> standalone receiver ----------------------------------------------------
        for key, snd in senders.items():
            t = time.perf_counter()
            c = int(snd.c)
            s = Mo.send(snd, x, scale, c)
            lens = sorted({len(p) for p in s["payloads"]})
            if lens != [S.wire_bytes(c)]:
                raise K.ULRError(f"{key}: payload lengths {lens} are not {S.wire_bytes(c)}")
            y_n, y = Mo.receive(receivers[key], s["payloads"], c, scale)
            pays[f"{g}.{key}"] = as_u8(s["payloads"])
            put(g, key, y, xr, y_n, xn)
            wy_n, wy = Mo.receive(receivers[key], [s["payloads"][j] for j in perm], c, scale)
            put(g, f"C-WRONG:{key}", wy, xr, wy_n, xn)
            zy_n, zy = Mo.receive(receivers[key], [bytes(S.wire_bytes(c))], c, scale)
            put(
                g,
                f"C-ZWIRE:{key}",
                np.broadcast_to(zy.reshape(1, -1).numpy(), xr.shape).copy(),
                xr,
                np.broadcast_to(zy_n.reshape(1, -1).numpy(), xr.shape).copy(),
                xn,
            )
            preds[f"{g}.{key}.y"] = y.reshape(n_t, -1).numpy()
            preds[f"{g}.{key}.y_n"] = y_n.reshape(n_t, -1).numpy()
            lats[f"{g}.{key}.z"] = s["z"].numpy()
            extra[g][f"{key}_s"] = time.perf_counter() - t
        # ---- constant controls ------------------------------------------------------------------------------------
        put(g, "C-ZERO", np.zeros_like(xr), xr)
        put(g, "C-MEAN", np.broadcast_to(mean_tile.reshape(1, -1).numpy(), xr.shape).copy(), xr)
    rec["eval_compute_s"] = time.perf_counter() - t_eval
    devs = []
    for g, x in sets.items():
        flat = x.reshape(x.shape[0], -1)
        for name in (*S.ARM_FINAL, *S.ARM_STEP0, f"{S.SOURCE}:FLOAT"):
            for i in range(0, flat.shape[0], 37):
                ref = C.metrics(torch.from_numpy(preds[f"{g}.{name}.y"][i]), flat[i])
                ours = sums[f"{g}.{name}.raw"][i]
                sc = abs(ours[0]) + ours[1] + ours[2]
                devs.append(
                    max(
                        abs(ref[k] - float(ours[j]))
                        for j, k in enumerate(("sse", "target_energy", "pred_energy", "dot"))
                    )
                    / sc
                )
    rec["torch_reference_max_rel_dev"] = max(devs)
    rec["pooled"] = {}
    for key, arr in sums.items():
        if key.endswith(".raw"):
            name = key[: -len(".raw")]
            p = U.pooled(arr, sums[f"{name}.finite"])
            row = {
                q: p[q]
                for q in (
                    "sse",
                    "target_energy",
                    "pred_energy",
                    "dot",
                    "n",
                    "mse",
                    "rse",
                    "cosine",
                    "norm_ratio",
                    "finite",
                )
            }
            row["normalized_mse_crosswalk"] = U.normalized_from_raw(p, s32)
            if f"{name}.norm" in sums:
                row["normalized_mse_direct"] = U.pooled(sums[f"{name}.norm"])["mse"]
            rec["pooled"][name] = row
    rec["extra"] = extra
    fp_after = {k: Mo.fingerprint(m)["state_sha256"] for k, m in senders.items()}
    fp_after.update({k + ".receiver": Mo.fingerprint(r)["state_sha256"] for k, r in receivers.items()})
    fp_after[S.SOURCE] = Mo.fingerprint(source)["state_sha256"]
    files = {
        S.arm_method(arm, p): K.sha256_file(f["path"]) == f["file_sha256"]
        for arm, a in job["arms"].items()
        for p, f in a["states"].items()
    }
    files[S.SOURCE] = K.sha256_file(src["path"]) == src["file_sha256"]
    models = [*senders.values(), *receivers.values(), source]
    rec["isolation"] = {
        "fingerprints_unchanged": {k: fp_after[k] == fp0[k] for k in fp0},
        "state_files_unchanged": files,
        "global_rng_unchanged": Mo.rng_sha256() == rng0,
        "grad_enabled": torch.is_grad_enabled(),
        "any_model_in_training_mode": any(m.training for m in models),
        "train_mode_forwards": 0,
    }
    iso = rec["isolation"]
    if (
        not (
            all(iso["fingerprints_unchanged"].values())
            and all(files.values())
            and iso["global_rng_unchanged"]
        )
        or iso["grad_enabled"]
        or iso["any_model_in_training_mode"]
    ):
        raise K.ULRError(f"evaluation changed a state, a file, the RNG or a mode: {iso}")
    meta = {"format": "ulr_20261015_v1.eval/v1", "protocol_id": job["protocol_id"]}
    rec["files"] = {
        "sums": K.save_tensors(
            out / "eval_sums.safetensors",
            {k: np.ascontiguousarray(v) for k, v in sums.items()},
            {**meta, "content": "float64 sums"},
        ),
        "predictions": K.save_tensors(
            out / "eval_predictions.safetensors",
            {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in preds.items()},
            {**meta, "content": "neural predictions; .y original, .y_n normalized"},
        ),
        "latents": K.save_tensors(
            out / "eval_latents.safetensors",
            {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in lats.items()},
            {**meta, "content": "wire latents [n, c, 4, 4]; source latents [n, 16, 4, 4]"},
        ),
        "payloads": K.save_tensors(
            out / "eval_payloads.safetensors", pays, {**meta, "content": "payload bytes, one row per tile"}
        ),
    }
    rec["payload_bytes_per_tile"] = {
        k.split(".", 1)[1]: int(v.shape[1]) for k, v in pays.items() if k.startswith("EARLY.")
    }
    rec["wire_latent_shapes"] = {
        k.split(".", 1)[1].rsplit(".", 1)[0]: list(v.shape[1:])
        for k, v in lats.items()
        if k.startswith("EARLY.")
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
