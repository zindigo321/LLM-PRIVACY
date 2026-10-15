"""ULR-20261015-v1: build ``ulr_config.json`` from the registered constants, the UOA / UCX records (read-only) and
the three pilot records. It refuses unless:
- the UOA manifest is the frozen one;
- the source checkpoint equals the UOA training record and the UOA EVAL job;
- every reused artifact matches its record;
- the split keys and the wrong-payload map equal the UOA configuration, and every pair crosses clients;
- all three pilots are GO.

``python -B checks/build_ulr_config.py`` (PYTHONPATH unset); writes ``ulr_config.json`` exclusively.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

UOA_PKG = Path("D:/projects-cd/cg-fedllm-uoa/diagnostics/uoa_20261014_v1")
UOA_RUN = Path("D:/cgfed-runs/uoa_20261014_v1")
UOA_MANIFEST_SHA256 = "c8b30ab08abcdae5a3564ece49e77c64d305856b8d0041b1b7da8eda135446ed"


def main() -> int:
    import ulr_model as Mo

    if K.sha256_file(UOA_PKG / "manifest_uoa.json") != UOA_MANIFEST_SHA256:
        raise K.ULRError("the UOA manifest is not the frozen one")
    ucfg = K.load_json(UOA_PKG / "uoa_config.json")
    st = UOA_RUN / "steps"
    tr = K.load_json(st / "U-LEAKY-LONG/train_record.json")
    last = [c for c in tr["checkpoints"] if c["update"] == 12000]
    if tr["status"] != "COMPLETED" or tr["updates_done"] != 12000 or len(last) != 1:
        raise K.ULRError("the UOA U-LEAKY-LONG record has no completed 12,000-update state")
    sf = last[0]["state_file"]
    ev_job = K.load_json(st / "EVAL/eval_job.json")
    m = ev_job["models"]["U-LEAKY-LONG"]
    if (m["path"], m["file_sha256"], m["state_sha256"]) != (
        sf["path"],
        sf["file_sha256"],
        sf["state_sha256"],
    ):
        raise K.ULRError("the UOA EVAL job and the training record disagree on the source state")
    if K.sha256_file(sf["path"]) != sf["file_sha256"] or tr["model_state_sha256_final"] != sf["state_sha256"]:
        raise K.ULRError("the source state file differs from its record")
    a = K.load_json(st / "STAGE-A/stage_a_record.json")
    cal = K.load_json(st / "CALIB/calib_record.json")
    ev = K.load_json(st / "EVAL/eval_record.json")

    def ref(f: dict) -> dict:
        p = Path(f["path"])
        if K.sha256_file(p) != f["sha256"]:
            raise K.ULRError(f"{p} differs from its record")
        return {"path": p.as_posix(), "sha256": f["sha256"], "bytes": p.stat().st_size}

    ucx = {
        k: ucfg["ucx"][k]
        for k in (
            "train_tiles",
            "heldout_tiles",
            "train_mean_tile",
            "pca_decoder",
            "eval_outputs",
            "scale_global",
            "int8_shared_step",
            "normalized_sha256",
        )
    }
    for k in ("train_tiles", "heldout_tiles", "train_mean_tile", "pca_decoder"):
        ucx[k] = ref(ucx[k])
    ub = K.load_json(Path(ucfg["ucx"]["run_root"]) / "steps/BUILD-TRAIN/build_train_record.json")
    uh = K.load_json(Path(ucfg["ucx"]["run_root"]) / "steps/BUILD-HELDOUT/build_heldout_record.json")
    ucx["split_finite"] = {
        "TRAIN": ub["tensors"]["x"]["all_finite"],
        "EARLY": uh["tensors"]["x_EARLY"]["all_finite"],
        "LATE": uh["tensors"]["x_LATE"]["all_finite"],
    }
    uoa = {
        "manifest_sha256": UOA_MANIFEST_SHA256,
        "source_state": {
            "path": Path(sf["path"]).as_posix(),
            "file_sha256": sf["file_sha256"],
            "state_sha256": sf["state_sha256"],
        },
        "source_updates": tr["updates_done"],
        "source_scales": ref(cal["models"]["U-LEAKY-LONG"]["scales_file"]),
        "scalar4_levels": ref(a["fitted_files"]["scalar4_levels"]),
        "eval_sums": ref(ev["files"]["sums"]),
        "eval_latents": ref(ev["files"]["latents"]),
        "eval_outputs": [
            (st / "EVAL" / n).as_posix()
            for n in (
                "eval_predictions.safetensors",
                "eval_payloads.safetensors",
                "eval_latents.safetensors",
                "eval_sums.safetensors",
                "eval_record.json",
            )
        ],
    }
    keys, wrong = ucfg["split"]["keys"], ucfg["wrong_map"]
    for g, pairs in wrong.items():
        if [p[0] for p in pairs] != keys[g] or any(p0[1] == p1[1] for p0, p1 in pairs):
            raise K.ULRError(f"the {g} wrong-payload map is not the registered crossing-client map")
    if [len(keys[g]) * 24 for g in S.COHORTS] != [1440, 168, 336]:
        raise K.ULRError("cohort sizes are not 1,440 / 168 / 336 tiles")
    sched = Mo.schedule(1440, S.MAX_UPDATES, S.BATCH_SIZE, S.SCHEDULE_SEED)
    pilots, walls, floors = {}, [], []
    for arm in S.ARM_ORDER:
        rel = f"preflight/pilot_{arm}_record.json"
        rec = K.load_json(PKG / rel)
        if rec["gate"]["decision"] != "GO" or rec["mode"] != "full":
            raise K.ULRError(f"pilot {arm} is not a full GO pilot")
        pilots[arm] = {"record": rel, "sha256": K.sha256_file(PKG / rel), "wall_s": rec["wall_s"]}
        walls.append(rec["wall_s"])
        floors.append(rec["gate"]["projection"]["memory_floor_bytes"])
    vdir = PKG / "vendor" / "uoa_20261014_v1" / "vendor" / "upp_20261014_v1"
    cfg = {
        "protocol_id": S.PROTOCOL_ID,
        "model": S.MODEL,
        "optimizer": S.OPTIMIZER,
        "batch_size": S.BATCH_SIZE,
        "max_updates": S.MAX_UPDATES,
        "checkpoints": list(S.CHECKPOINTS),
        "schedule_seed": S.SCHEDULE_SEED,
        "schedule_sha256": Mo.schedule_sha256(sched),
        "curve_every": S.CURVE_EVERY,
        "diag_schedule_seed": S.DIAG_SCHEDULE_SEED,
        "diag_batches": S.DIAG_BATCHES,
        "cpu": S.CPU_THREADS,
        "torch_version": S.TORCH_VERSION,
        "limits": S.LIMITS,
        "mechanics": S.MECHANICS,
        "arms": S.ARMS,
        "ucx": ucx,
        "uoa": uoa,
        "split": {"keys": keys},
        "wrong_map": wrong,
        "execution": {
            "run_root": "D:/cgfed-runs/ulr_20261015_v1",
            "ledger_mirror": (PKG / "scientific_ledger_mirror").as_posix(),
            "python_executable": "D:/conda_envs/cgfedllm/python.exe",
        },
        "preflight": {
            "pilots": pilots,
            "pilot_wall_s_total": math.fsum(walls),
            "min_rss_limit_bytes": max(floors),
        },
        "primary_policy": {
            "policy_id": S.POLICY_ID,
            "policy_sha256": K.sha256_file(vdir / "upp_policy.json"),
            "classifier_sha256": K.sha256_file(vdir / "upp_20261014_classify.py"),
            "thresholds": {"max_rse": 1.0, "min_cosine": 0.1},
        },
    }
    if (
        cfg["primary_policy"]["policy_sha256"]
        != "a7e357df98bc79d5541103196b543c9fd7236f2df57679269eb6c8ac1bc692b7"
    ):
        raise K.ULRError("the vendored policy is not the pinned UPP-20261014-v1 file")
    data = (json.dumps(cfg, indent=2, sort_keys=True) + "\n").encode("utf-8")
    print(K.write_exclusive(PKG / "ulr_config.json", data), cfg["schedule_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
