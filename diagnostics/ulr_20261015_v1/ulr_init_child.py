"""ULR-20261015-v1 INIT child: the pre-training diagnostic, the ONE channel-PCA fit, the bridge initializations,
the C16 identity check and the step-zero TRAIN evaluation, in one fresh process (TRAIN only).

``python -B ulr_init_child.py <job.json>``. Installs the split-enforcing open guard first: no held-out file, and no
UCX or UOA evaluation output. Steps:
1. The verified source, UOA U-LEAKY-LONG at 12,000 updates (both hashes), loaded in eval mode.
2. Train-statistics versus eval-statistics on the four registered UCX / UOA TRAIN batches, on disposable deep copies
   only. Recorded, never acted on.
3. TRAIN internal latents h of the source (eval mode, every tile alone) -> 23,040 samples x 16 channels.
   - One centered float64 channel PCA: mu is the TRAIN mean, and V holds the principal directions as rows, sorted by
     singular value, with the registered sign rule.
4. Bridge initializations for c in (16, 8, 4, 2, 1):
   - W_reduce = V_c, b_reduce = -V_c mu, W_expand = V_c^T, b_expand = mu;
   - computed in float64 and stored as fp32.
5. The C16 identity check against the preregistered tolerances (``ulr_spec.C16_IDENTITY_TOL``). It fails closed.
6. The step-zero TRAIN evaluation of every c through the real wire path (64 c bytes per tile).
   - c = 8 and c = 2 are forward-only mappings. They are never trained and are not adapted models.

Gradients are disabled except inside the copies of step 2, and no optimizer is constructed. The source fingerprint,
its state file and the global RNG are checked unchanged.
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
    rec: dict = {"phase": "init", "mode": job["mode"], "pid": os.getpid(), "status": "EXECUTION_ERROR"}
    t0 = time.monotonic()
    try:
        K.start_watchdog(job, "ulr_init_child")
        K.check_env(rec)
        import torch
        import uid_common as C

        rec["cpu"] = C.configure_cpu(job["cpu"]["intra_op_threads"], job["cpu"]["inter_op_threads"])

        def refuse(self, *args, **kwargs):
            raise RuntimeError("the INIT child must not construct an optimizer")

        torch.optim.Optimizer.__init__ = refuse
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
        K.write_json(out / "init_record.json", rec)
    return K.EXIT[rec["status"]]


def bn_train_vs_eval(source, x_n, job: dict) -> list[dict]:
    """Disposable deep copies only: eval-mode (running statistics) versus train-mode (batch statistics) outputs."""
    import copy

    import torch
    import ulr_model as Mo
    from torch.nn import functional as F

    sched = Mo.schedule(
        x_n.shape[0], int(job["diag_batches"]), int(job["batch_size"]), int(job["diag_schedule_seed"])
    )
    rows = []
    for bi, idx in enumerate(sched):
        xb = x_n[idx]
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            ev = copy.deepcopy(source).eval()
            y_eval = ev.decode(ev.encode(xb))
            tr = copy.deepcopy(source).train()
            bn_in: dict = {}
            hooks = [
                mod.register_forward_hook(
                    lambda m, i, o, name=name, store=bn_in: store.__setitem__(name, i[0].detach().double())
                )
                for name, mod in tr.named_modules()
                if isinstance(mod, torch.nn.BatchNorm2d)
            ]
            y_train = tr.decode(tr.encode(xb))
            for h in hooks:
                h.remove()
        bn = {}
        for name, mod in source.named_modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                v = bn_in[name]
                bm, bv = v.mean(dim=(0, 2, 3)), v.var(dim=(0, 2, 3), unbiased=False)
                rm, rv = mod.running_mean.double(), mod.running_var.double()
                bn[name] = {
                    "batch_mean_minus_running_mean_max_abs": float((bm - rm).abs().max()),
                    "batch_var_over_running_var_min": float((bv / rv).min()),
                    "batch_var_over_running_var_max": float((bv / rv).max()),
                }
        rows.append(
            {
                "batch": bi,
                "indices": idx,
                "normalized_mse_eval_mode": float(F.mse_loss(y_eval, xb).double()),
                "normalized_mse_train_mode": float(F.mse_loss(y_train, xb).double()),
                "max_abs_train_minus_eval_output": float((y_train - y_eval).abs().max()),
                "batchnorm": bn,
            }
        )
    return rows


def channel_pca(h) -> dict:
    """float64 centered PCA of [N, 16, 4, 4] latents as N * 16 samples of 16 channels; registered sign rule."""
    import numpy as np

    H = h.double().permute(0, 2, 3, 1).reshape(-1, h.shape[1]).numpy()
    mu = H.mean(axis=0)
    t = time.perf_counter()
    _, sv, vt = np.linalg.svd(H - mu, full_matrices=False)
    fit_s = time.perf_counter() - t
    V = vt.copy()
    flips = []
    for i in range(V.shape[0]):
        j = int(np.argmax(np.abs(V[i])))
        if V[i, j] < 0:
            V[i] = -V[i]
            flips.append(i)
    e = np.square(sv)
    rel_gaps = [float((sv[i] - sv[i + 1]) / sv[0]) for i in range(len(sv) - 1)]
    return {
        "samples": int(H.shape[0]),
        "dimension": int(H.shape[1]),
        "mu": mu,
        "V": V,
        "singular_values": [float(v) for v in sv],
        "explained_variance_fraction": [float(v) for v in e / e.sum()],
        "cumulative_explained_variance": [float(v) for v in np.cumsum(e) / e.sum()],
        "min_relative_gap_between_adjacent_singular_values": min(rel_gaps),
        "near_degenerate_pairs_1e-12": [i for i, g in enumerate(rel_gaps) if g < 1e-12],
        "sign_flips": flips,
        "orthonormality_max_abs_error_fp64": float(np.abs(V @ V.T - np.eye(V.shape[0])).max()),
        "orthonormality_max_abs_error_fp32": float(
            np.abs(
                V.astype(np.float32).astype(np.float64) @ V.astype(np.float32).astype(np.float64).T
                - np.eye(V.shape[0])
            ).max()
        ),
        "fit_s": fit_s,
        "solver": "numpy.linalg.svd(H - mu, full_matrices=False) on float64; OPENBLAS_NUM_THREADS=1",
    }


def run(job: dict, rec: dict, out: Path) -> None:
    import copy

    import numpy as np
    import torch
    import ulr_model as Mo
    import ulr_spec as S
    import uoa_model as UM
    import uoa_sums as U

    cfg, scale = job["model"], float(job["scale_global"])
    x = K.load_checked(job["train_tiles"], job["train_tiles_sha256"], "torch")["x"].contiguous()
    if tuple(x.shape) != (job["n_train_tiles"], 1, 32, 32):
        raise K.ULRError(f"TRAIN tiles {tuple(x.shape)} are not [{job['n_train_tiles']}, 1, 32, 32]")
    x_n = Mo.normalize(x, scale).contiguous()
    xr, x_n_flat = x.reshape(x.shape[0], -1).numpy(), x_n.reshape(x.shape[0], -1).numpy()
    src = job["source"]
    source = Mo.load_source(src, cfg)
    fp0, rng0 = Mo.fingerprint(source)["state_sha256"], Mo.rng_sha256()
    rec["source"] = {**src, "activation_scope": UM.check_activations(source, S.SOURCE_ACTIVATION, cfg)}
    # 2. train-statistics versus eval-statistics on copies
    t = time.perf_counter()
    rec["bn_train_vs_eval"] = bn_train_vs_eval(source, x_n, job)
    rec["bn_train_vs_eval_s"] = time.perf_counter() - t
    torch.set_grad_enabled(False)
    # 3. TRAIN latents (eval mode, every tile alone) and the one channel PCA
    t = time.perf_counter()
    h = torch.cat([source.encode(x_n[i : i + 1]) for i in range(x_n.shape[0])])
    rec["latents_s"] = time.perf_counter() - t
    if tuple(h.shape[1:]) != S.BACKBONE_INTERNAL_SHAPE:
        raise K.ULRError(f"source latent {tuple(h.shape)} is not [N, 16, 4, 4]")
    pca = channel_pca(h)
    rec["channel_pca"] = {k: v for k, v in pca.items() if k not in ("mu", "V")}
    mu64, V64 = pca["mu"], pca["V"]
    init = {"V64": V64, "mu64": mu64, "singular_values64": np.asarray(pca["singular_values"])}
    for c in S.ALL_C:
        Vc = V64[:c]
        init[f"reduce_weight_c{c}"] = Vc.astype(np.float32).reshape(c, 16, 1, 1)
        init[f"reduce_bias_c{c}"] = (-(Vc @ mu64)).astype(np.float32)
        init[f"expand_weight_c{c}"] = Vc.T.astype(np.float32).reshape(16, c, 1, 1).copy()
        init[f"expand_bias_c{c}"] = mu64.astype(np.float32)
    init = {k: np.ascontiguousarray(v) for k, v in init.items()}
    rec["bridge_init_file"] = K.save_tensors(
        out / "bridge_init.safetensors",
        init,
        {
            "format": "ulr_20261015_v1.bridge_init/v1",
            "sign_rule": S.SIGN_RULE,
            "source_state_sha256": src["state_sha256"],
        },
    )
    # 4-6. bridged models at step zero (fresh deep copies of the source), the identity check, the wire-path TRAIN eval
    sums, steps0 = {}, {}
    y_src = UM.float_outputs(source, x, scale)
    src_norm = U.pooled(U.sums4(y_src["y_n"].reshape(x.shape[0], -1).numpy(), x_n_flat))
    for c in S.ALL_C:
        with torch.random.fork_rng(devices=[]):
            m = Mo.bridged(copy.deepcopy(source), c)
        Mo.set_bridge(m, init, c)
        m.eval()
        if Mo.backbone_sha256(m) != src["state_sha256"] or not Mo.bridge_equals(m, init, c):
            raise K.ULRError(
                f"c={c}: the step-zero model does not hold the source backbone and the PCA bridge"
            )
        t = time.perf_counter()
        s = Mo.send(m, x, scale, c)
        lens = sorted({len(b) for b in s["payloads"]})
        if lens != [S.wire_bytes(c)]:
            raise K.ULRError(f"c={c}: payload lengths {lens} are not {S.wire_bytes(c)}")
        y_n = torch.cat([m.decode(torch.from_numpy(Mo.wire_from_bytes(b, c))) for b in s["payloads"]])
        y = Mo.denormalize(y_n, scale)
        yr, ynr = y.reshape(x.shape[0], -1).numpy(), y_n.reshape(x.shape[0], -1).numpy()
        sums[f"TRAIN.c{c}.raw"] = U.sums4(yr, xr)
        sums[f"TRAIN.c{c}.norm"] = U.sums4(ynr, x_n_flat)
        sums[f"TRAIN.c{c}.raw_blocks"] = U.block_sums(yr, xr)
        sums[f"TRAIN.c{c}.finite"] = U.finite_rows(yr).astype(np.uint8)
        pr, pn = U.pooled(sums[f"TRAIN.c{c}.raw"]), U.pooled(sums[f"TRAIN.c{c}.norm"])
        h_rec = m.expand(m.reduce(h))
        steps0[str(c)] = {
            "wire_latent_shape": list(S.wire_shape(c)),
            "wire_latent_numel": S.wire_numel(c),
            "payload_bytes": lens[0],
            "role": "registered arm"
            if c in S.ARMS.values()
            else "forward-only step-zero mapping (never trained)",
            "train_rse": pr["rse"],
            "train_cosine": pr["cosine"],
            "train_norm_ratio": pr["norm_ratio"],
            "train_normalized_mse_direct": pn["mse"],
            "latent_relative_reconstruction_error": float(
                (h_rec - h).double().square().sum() / h.double().square().sum()
            ),
            "pca_expected_latent_residual_fraction": float(1 - pca["cumulative_explained_variance"][c - 1]),
            "eval_s": time.perf_counter() - t,
        }
        if c == 16:
            d_lat = float((h_rec - h).abs().max())
            d_out = float((y_n - y_src["y_n"]).abs().max())
            d_mse = abs(pn["mse"] - src_norm["mse"]) / src_norm["mse"]
            tol = S.C16_IDENTITY_TOL
            steps0["16"]["identity_check"] = {
                "latent_max_abs_error": d_lat,
                "output_max_abs_difference_vs_source": d_out,
                "pooled_normalized_mse_relative_difference": d_mse,
                "source_normalized_mse": src_norm["mse"],
                "tolerances": tol,
                "within_tolerance": d_lat <= tol["latent_max_abs"]
                and d_out <= tol["output_max_abs"]
                and d_mse <= tol["pooled_nmse_rel"],
                "bit_identical_outputs": bool(torch.equal(y_n, y_src["y_n"])),
            }
            if not steps0["16"]["identity_check"]["within_tolerance"]:
                raise K.ULRError(
                    f"C16 identity check outside the preregistered tolerance: {steps0['16']['identity_check']}"
                )
        del m
    rec["step_zero_train"] = steps0
    rec["source_train_normalized_mse_direct"] = src_norm["mse"]
    rec["train_sums_file"] = K.save_tensors(
        out / "init_train_sums.safetensors",
        {k: np.ascontiguousarray(v) for k, v in sums.items()},
        {"format": "ulr_20261015_v1.sums/v1"},
    )
    rec["isolation"] = {
        "source_fingerprint_unchanged": Mo.fingerprint(source)["state_sha256"] == fp0,
        "global_rng_unchanged": Mo.rng_sha256() == rng0,
        "source_file_unchanged": K.sha256_file(src["path"]) == src["file_sha256"],
        "source_eval_mode": not source.training,
        "grad_enabled": torch.is_grad_enabled(),
    }
    iso = rec["isolation"]
    if (
        not (
            iso["source_fingerprint_unchanged"]
            and iso["global_rng_unchanged"]
            and iso["source_file_unchanged"]
            and iso["source_eval_mode"]
        )
        or iso["grad_enabled"]
    ):
        raise K.ULRError(f"INIT changed the source, the RNG or a file: {iso}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
