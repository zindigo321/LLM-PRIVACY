"""ULR-20261015-v1 forward-only replay of the actual payload decoding (written before the run; executed once after).

``python -B replay_final_states.py <out.json> [config.json]`` with PYTHONPATH unset, in a fresh process. Gradients are
disabled and optimizer construction is refused. For every arm state of the EVAL job (step zero and final):
- the sender (the full saved model) re-encodes TRAIN / EARLY / LATE, and its wire payload bytes must equal the saved
  ones bit for bit;
- a NEW standalone receiver, built from only the decoder-side tensors of the saved state, decodes the SAVED payload
  bytes; its predictions must equal the saved ones bit for bit;
- the wrong-payload and zero-wire control sums are recomputed from the saved bytes and compared with the EVAL sums.

The source model's FLOAT and INT8 predictions are replayed and compared as well. State files and the RNG must be
unchanged.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EXE = Path(__file__).resolve().parent
PKG = EXE.parent / "ulr_20261015_v1"
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402


def main(out_path: str, cfg_path: str | None = None) -> int:
    t0 = time.monotonic()
    import uid_common as C

    cfg = K.load_json(cfg_path or PKG / "ulr_config.json")
    C.configure_cpu(cfg["cpu"]["intra_op_threads"], cfg["cpu"]["inter_op_threads"])
    import numpy as np
    import torch
    import ulr_model as Mo
    import ulr_spec as S
    import uoa_model as UM
    import uoa_sums as U

    def refuse(self, *a, **k):
        raise RuntimeError("the replay must not construct an optimizer")

    torch.optim.Optimizer.__init__ = refuse
    torch.set_grad_enabled(False)
    st = Path(cfg["execution"]["run_root"]) / "steps"
    ev, job = K.load_json(st / "EVAL/eval_record.json"), K.load_json(st / "EVAL/eval_job.json")
    f = ev["files"]
    sums = K.load_checked(f["sums"]["path"], f["sums"]["sha256"])
    preds = K.load_checked(f["predictions"]["path"], f["predictions"]["sha256"])
    pays = K.load_checked(f["payloads"]["path"], f["payloads"]["sha256"])
    u = cfg["ucx"]
    x = {"TRAIN": K.load_checked(u["train_tiles"]["path"], u["train_tiles"]["sha256"], "torch")["x"]}
    held = K.load_checked(u["heldout_tiles"]["path"], u["heldout_tiles"]["sha256"], "torch")
    x["EARLY"], x["LATE"] = held["x_EARLY"], held["x_LATE"]
    scale, mcfg = float(u["scale_global"]), cfg["model"]
    rng0 = Mo.rng_sha256()
    res: dict = {"checks": {}}
    for arm, a in job["arms"].items():
        c = int(a["c"])
        for point, sf in a["states"].items():
            key = S.arm_method(arm, point)
            snd = Mo.load_arm_state(sf, c, mcfg)
            rcv, info = Mo.receiver_from_state(sf, c, mcfg)
            for g, xg in x.items():
                n_t = xg.shape[0]
                xr = xg.reshape(n_t, -1).numpy()
                saved = [bytes(r.tobytes()) for r in pays[f"{g}.{key}"]]
                s = Mo.send(snd, xg, scale, c)
                y_n, y = Mo.receive(rcv, saved, c, scale)
                perm = ev["extra"][g]["wrong_source_tile_index"]
                _, wy = Mo.receive(rcv, [saved[j] for j in perm], c, scale)
                _, zy = Mo.receive(rcv, [bytes(S.wire_bytes(c))], c, scale)
                res["checks"][f"{g}.{key}"] = {
                    "payload_lengths": sorted({len(b) for b in s["payloads"]}) == [S.wire_bytes(c)],
                    "sender_bytes_equal_saved": [bytes(b) for b in s["payloads"]] == saved,
                    "receiver_y_equal_saved": bool(
                        np.array_equal(y.reshape(n_t, -1).numpy(), preds[f"{g}.{key}.y"])
                    ),
                    "receiver_y_n_equal_saved": bool(
                        np.array_equal(y_n.reshape(n_t, -1).numpy(), preds[f"{g}.{key}.y_n"])
                    ),
                    "wrong_payload_sums_equal": bool(
                        np.array_equal(
                            U.sums4(wy.reshape(n_t, -1).numpy(), xr), sums[f"{g}.C-WRONG:{key}.raw"]
                        )
                    ),
                    "zero_wire_sums_equal": bool(
                        np.array_equal(
                            U.sums4(np.broadcast_to(zy.reshape(1, -1).numpy(), xr.shape).copy(), xr),
                            sums[f"{g}.C-ZWIRE:{key}.raw"],
                        )
                    ),
                }
            res["checks"][f"{key}.receiver_tensors"] = info["tensors"] > 0
            res["checks"][f"{key}.state_file_unchanged"] = K.sha256_file(sf["path"]) == sf["file_sha256"]
    src = Mo.load_source(job["source"], mcfg)
    scl = K.load_checked(job["source_scales"], job["source_scales_sha256"])["scales"]
    for g, xg in x.items():
        n_t = xg.shape[0]
        fo = UM.float_outputs(src, xg, scale)
        qo = UM.int8_outputs(src, fo["z"], scl, scale)
        res["checks"][f"{g}.SOURCE-LEAKY"] = {
            "float_y_equal": bool(
                np.array_equal(fo["y"].reshape(n_t, -1).numpy(), preds[f"{g}.SOURCE-LEAKY:FLOAT.y"])
            ),
            "int8_y_equal": bool(
                np.array_equal(qo["y"].reshape(n_t, -1).numpy(), preds[f"{g}.SOURCE-LEAKY:INT8LAT.y"])
            ),
        }
    res["global_rng_unchanged"] = Mo.rng_sha256() == rng0
    res["grad_enabled"] = torch.is_grad_enabled()
    flat = [v for v in res["checks"].values() if isinstance(v, bool)] + [
        b for v in res["checks"].values() if isinstance(v, dict) for b in v.values()
    ]
    mods = {n: str(Path(m.__file__).resolve()) for n, m in list(sys.modules.items())
            if isinstance(getattr(m, "__file__", None), str) and Path(m.__file__).is_absolute()
            and "projects-cd" in Path(m.__file__).resolve().as_posix() and n not in ("__main__", "__mp_main__")}  # fmt: skip
    res["imports_only_from_package"] = all(Path(p).is_relative_to(PKG) for p in mods.values())
    res["cg_fedllm_imported"] = "cg_fedllm" in sys.modules
    res["all_bit_identical"] = (
        all(flat)
        and res["global_rng_unchanged"]
        and not res["grad_enabled"]
        and res["imports_only_from_package"]
        and not res["cg_fedllm_imported"]
    )
    res["wall_s"] = time.monotonic() - t0
    K.write_json(out_path, res)
    print(json.dumps({"all_bit_identical": res["all_bit_identical"], "wall_s": res["wall_s"]}))
    return 0 if res["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
