"""ULR-20261015-v1 engineering fixture: small synthetic stand-ins for the UCX and UOA artifacts plus a fixture
config, so the supervisor and all five children can be exercised end to end. Never the registered data or run area.

``make_fixture(root, hooks=None) -> Path`` writes, under ``root``:
- synthetic TRAIN / EARLY / LATE tiles (288 / 48 / 48), the TRAIN mean tile, a 256-row PCA decoder of the TRAIN tiles
  and dummy "evaluation outputs" for the open guard;
- a small LeakyReLU source model (seed 7, three Adam steps), its latent scales and SCALAR-4 levels;
- ``fixture_config.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

FIXTURE_MODEL = {**S.MODEL, "down_channels": [2, 2, 4, 8, 8, 16], "core_layers": 1, "core_depth": 1}


def make_fixture(root: Path, hooks: dict | None = None, max_updates: int = 30) -> Path:
    import numpy as np
    import torch
    import ucx_codecs as X
    import uid_common as C
    import uoa_codecs as Q
    import uoa_model as UM
    from safetensors.numpy import save as np_save
    from safetensors.torch import save

    root = Path(root)
    ucx, uoa = root / "fake_ucx", root / "fake_uoa"
    ucx.mkdir(parents=True)
    uoa.mkdir(parents=True)
    keys = {
        "TRAIN": [[t, 100 + t] for t in range(12)],
        "EARLY": [[12, 200], [13, 201]],
        "LATE": [[16, 300], [17, 301]],
    }
    g = torch.Generator().manual_seed(20261112)
    n = {k: len(v) * S.TILES_PER_SNAPSHOT for k, v in keys.items()}
    x = {k: (torch.randn((n[k], 1, 32, 32), generator=g) * 0.003).contiguous() for k in keys}
    (ucx / "train_tiles.safetensors").write_bytes(save({"x": x["TRAIN"]}))
    (ucx / "heldout_tiles.safetensors").write_bytes(save({"x_EARLY": x["EARLY"], "x_LATE": x["LATE"]}))
    (ucx / "train_mean_tile.safetensors").write_bytes(
        save({"x": x["TRAIN"].double().mean(dim=0).to(torch.float32).contiguous()})
    )
    fit = X.pca_fit(x["TRAIN"].reshape(n["TRAIN"], -1).numpy(), 256)
    (ucx / "pca_decoder.safetensors").write_bytes(
        np_save({"mean": fit["mean32"].reshape(-1).copy(), "basis": fit["basis32"].copy()})
    )
    m_abs = float(x["TRAIN"].abs().max())
    scale = m_abs / 0.95
    model = UM.build(FIXTURE_MODEL, 7, "leaky_relu")
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    x_n = UM.normalize(x["TRAIN"], scale).contiguous()
    model.train()
    for k in range(3):
        UM.train_update(model, opt, x_n[k * 8 : (k + 1) * 8], torch.nn.functional)
        opt.step()
    model.eval()
    src = UM.save_state(uoa / "state_source.safetensors", model, {"point": "fixture source"})
    with torch.no_grad():
        z = torch.cat([model.encode(x_n[i : i + 1]) for i in range(n["TRAIN"])])
    sc = Q.latent_scales([float(v) for v in z.abs().amax(dim=(0, 2, 3))])
    (uoa / "latent_scales.safetensors").write_bytes(
        np_save({"scales": sc, "max_abs": z.abs().amax(dim=(0, 2, 3)).numpy()})
    )
    lv = Q.uniform_levels(0.004)
    (uoa / "scalar4_levels.safetensors").write_bytes(np_save({"levels": lv}))
    outs_u = [ucx / f"fake_eval_{s}.bin" for s in ("predictions", "payloads", "record")]
    outs_o = [uoa / f"fake_eval_{s}.bin" for s in ("predictions", "payloads", "latents", "sums", "record")]
    for p in (*outs_u, *outs_o):
        p.write_bytes(b"fake evaluation output: must never be opened by a ULR child before EVAL\n")
    sched = UM.schedule(n["TRAIN"], max_updates, S.BATCH_SIZE, 993)

    def ref(p: Path) -> dict:
        return {"path": str(p), "sha256": K.sha256_file(p)}

    vdir = PKG / "vendor" / "uoa_20261014_v1" / "vendor" / "upp_20261014_v1"
    cfg = {
        "protocol_id": S.FIXTURE_PROTOCOL_ID,
        "model": FIXTURE_MODEL,
        "optimizer": S.OPTIMIZER,
        "batch_size": S.BATCH_SIZE,
        "max_updates": max_updates,
        "checkpoints": [0, 10, 20, max_updates],
        "schedule_seed": 993,
        "schedule_sha256": UM.schedule_sha256(sched),
        "curve_every": 5,
        "diag_schedule_seed": 994,
        "diag_batches": 2,
        "cpu": {"intra_op_threads": 2, "inter_op_threads": 1},
        "torch_version": torch.__version__,
        "limits": S.LIMITS,
        "mechanics": S.MECHANICS,
        "arms": S.ARMS,
        "ucx": {
            "train_tiles": ref(ucx / "train_tiles.safetensors"),
            "heldout_tiles": ref(ucx / "heldout_tiles.safetensors"),
            "train_mean_tile": ref(ucx / "train_mean_tile.safetensors"),
            "pca_decoder": ref(ucx / "pca_decoder.safetensors"),
            "eval_outputs": [str(p) for p in outs_u],
            "scale_global": scale,
            "int8_shared_step": float(X.int8_step(m_abs)),
            "normalized_sha256": C.tensor_sha256(x_n, "x_n"),
            "split_finite": {"TRAIN": True, "EARLY": True, "LATE": True},
        },
        "uoa": {
            "source_state": {k: src[k] for k in ("path", "file_sha256", "state_sha256")},
            "source_updates": 3,
            "source_scales": ref(uoa / "latent_scales.safetensors"),
            "scalar4_levels": ref(uoa / "scalar4_levels.safetensors"),
            "eval_outputs": [str(p) for p in outs_o],
            "eval_sums": None,
            "eval_latents": None,
        },
        "split": {"keys": keys},
        "wrong_map": {
            gname: [[v[i], v[(i + 1) % len(v)]] for i in range(len(v))] for gname, v in keys.items()
        },
        "execution": {
            "run_root": str(root / "run"),
            "ledger_mirror": str(root / "ledger_mirror"),
            "python_executable": sys.executable,
        },
        "preflight": {"pilots": {}, "pilot_wall_s_total": 0.0, "min_rss_limit_bytes": 50_000_000},
        "primary_policy": {
            "policy_id": S.POLICY_ID,
            "policy_sha256": K.sha256_file(vdir / "upp_policy.json"),
            "classifier_sha256": K.sha256_file(vdir / "upp_20261014_classify.py"),
        },
        "fixture_hooks": hooks or {},
    }
    path = root / "fixture_config.json"
    path.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    del np
    return path


if __name__ == "__main__":
    print(make_fixture(Path(sys.argv[1])))
