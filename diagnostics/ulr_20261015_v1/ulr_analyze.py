"""ULR-20261015-v1 analyzer (frozen with the package; run once after the invocation).

``python -B ulr_analyze.py <out_dir> [config.json]`` with PYTHONPATH unset. Reads the ledger, every step record and the
persisted sums (re-hashed against their records), and writes ``analysis.json``, ``learning_curves.csv``,
``checkpoints.csv`` and ``per_tile_metrics.csv`` into ``out_dir`` (created exclusively). As registered in
``protocol.md`` section 10, it reports separately:
A. the verified actual element reduction (wire shapes, element counts, payload lengths, receiver tensors);
B. UPP-20261014-v1 outcomes per arm (final states: primary; step zero and comparators: descriptive), through the
   vendored hash-pinned classifier;
C. bottleneck damage (step zero against the source), recovery (final against step zero), final reduced arms against
   the final C16 control, learning curves, and input dependence (correct against wrong / zero payloads);
D. comparator performance at matching element counts and at matching bytes, and the secondary normalized-MSE
   distance to 1e-4.
Plus breakdowns, byte accounting, diagnostics, resources and reproduction checks against the UOA records. Pooled
values are ratios of math.fsum-pooled float64 sums. Nothing here selects a state or changes a rule.
"""

from __future__ import annotations

import csv
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))
import ulr_common as K  # noqa: E402
import ulr_spec as S  # noqa: E402

KEYS4 = ("sse", "target_energy", "pred_energy", "dot")
SIZES = {"64B": ("C1-REDUCED:FINAL", "PCA-16"), "256B": ("C4-REDUCED:FINAL", "PCA-64", "SCALAR-4", f"{S.SOURCE}:INT8LAT"),
         "1024B": ("C16-CONTROL:FINAL", "PCA-256", "INT8-SHARED", f"{S.SOURCE}:FLOAT")}  # fmt: skip
MATCHING_ELEMENTS = {"C1-REDUCED": "PCA-16", "C4-REDUCED": "PCA-64", "C16-CONTROL": "PCA-256"}


def fsum_cols(rows) -> dict[str, float]:
    return {k: math.fsum(float(r[j]) for r in rows) for j, k in enumerate(KEYS4)}


def metrics(s: dict[str, float], n: int) -> dict[str, Any]:
    sse, t, p, d = (s[k] for k in KEYS4)
    return {**s, "n": n, "mse": sse / n, "rse": sse / t if t > 0 else None,
            "cosine": d / math.sqrt(p * t) if p > 0 and t > 0 else None, "norm_ratio": math.sqrt(p / t) if t > 0 else None}  # fmt: skip


class Run:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.root = Path(cfg["execution"]["run_root"])
        self.problems: list[str] = []
        self.ledger = K.Ledger(self.root / "ledger", cfg["execution"]["ledger_mirror"]).read()
        self.status = K.ledger_status(self.ledger)
        st = self.root / "steps"
        self.rec = {}
        for step in S.STEPS:
            name = {"INIT": "init", "EVAL": "eval"}.get(step, "train")
            p = st / step / f"{name}_record.json"
            self.rec[step] = K.load_json(p) if p.is_file() else None
            end = self.ledger.get(f"end_{step}.json")
            if end is None or end.get("status") != "COMPLETED":
                self.problems.append(
                    f"{step}: ledger end status {None if end is None else end.get('status')}"
                )
            elif end.get("stderr_bytes") != 0:
                self.problems.append(f"{step}: stderr has {end.get('stderr_bytes')} bytes")
            if end is not None and (end.get("guard") or {}).get("action") is not None:
                self.problems.append(f"{step}: guard action {end['guard']['action']}")
        ev = self.rec["EVAL"]
        self.sums = self._load(ev["files"]["sums"]) if ev else {}
        ini = self.rec["INIT"]
        self.init_sums = self._load(ini["train_sums_file"]) if ini else {}
        self.ck_sums = {}
        for arm in S.NEURAL_STEPS:
            for ck in (self.rec[arm] or {}).get("checkpoints", []):
                self.ck_sums[(arm, ck["update"])] = self._load(ck["sums_file"])
                if K.sha256_file(ck["state_file"]["path"]) != ck["state_file"]["file_sha256"]:
                    self.problems.append(f"{arm}@{ck['update']}: state file changed")

    def _load(self, f: dict) -> dict:
        from safetensors.numpy import load

        data = Path(f["path"]).read_bytes()
        if K.sha256_bytes(data) != f["sha256"]:
            self.problems.append(f"{Path(f['path']).name}: SHA-256 differs from its record")
        return load(data)

    def pooled(self, g: str, name: str, space: str = "raw") -> dict[str, Any] | None:
        arr = self.sums.get(f"{g}.{name}.{space}")
        return None if arr is None else metrics(fsum_cols(arr), arr.shape[0] * S.N_TILE)


def s32_of(cfg: dict) -> float:
    return struct.unpack("<f", struct.pack("<f", float(cfg["ucx"]["scale_global"])))[0]


def method_row(run: Run, g: str, name: str, s32: float) -> dict[str, Any] | None:
    raw = run.pooled(g, name)
    if raw is None:
        return None
    row = {
        "raw_sse": raw["sse"],
        "raw_mse": raw["mse"],
        "target_energy": raw["target_energy"],
        "rse": raw["rse"],
        "cosine": raw["cosine"],
        "norm_ratio": raw["norm_ratio"],
        "normalized_mse_crosswalk": raw["mse"] / (s32 * s32),
        "finite": bool(run.sums[f"{g}.{name}.finite"].all()),
        "raw_sums": {k: raw[k] for k in KEYS4},
        "n": raw["n"],
    }
    nrm = run.pooled(g, name, "norm")
    if nrm is not None:
        row["normalized_mse_direct"] = nrm["mse"]
    row["normalized_mse"] = row.get("normalized_mse_direct", row["normalized_mse_crosswalk"])
    row["normalized_mse_basis"] = "direct" if "normalized_mse_direct" in row else "crosswalk raw / s32^2"
    row["distance_to_1e-4_factor"] = row["normalized_mse"] / S.TARGET_NORMALIZED_MSE
    m = name.split("C-")[-1] if name.startswith("C-") else name
    if m in S.ELEMENTS:
        row["elements"], row["element_kind"], row["payload_bytes"] = (
            S.ELEMENTS[m],
            S.ELEMENT_KIND[m],
            S.PAYLOAD_BYTES[m],
        )
    return row


def paired(run: Run, g: str, a: str, b: str) -> dict[str, Any] | None:
    x, y = run.sums.get(f"{g}.{a}.raw"), run.sums.get(f"{g}.{b}.raw")
    if x is None or y is None:
        return None
    sa, sb = x[:, 0], y[:, 0]
    return {"a": a, "b": b, "tiles": int(sa.shape[0]), "a_lower": int((sa < sb).sum()), "equal": int((sa == sb).sum()),
            "pooled_sse_ratio_a_over_b": math.fsum(sa.tolist()) / math.fsum(sb.tolist())}  # fmt: skip


def element_reduction(run: Run) -> dict[str, Any]:
    ev, ini = run.rec["EVAL"] or {}, run.rec["INIT"] or {}
    widths = ev.get("payload_bytes_per_tile", {})
    shapes = ev.get("wire_latent_shapes", {})
    out = {"input_elements": S.N_TILE, "input_bytes_fp32": S.INPUT_TILE_BYTES, "arms": {}}
    for arm, c in S.ARMS.items():
        for point in ("STEP0", "FINAL"):
            key = S.arm_method(arm, point)
            sh = shapes.get(key)
            out["arms"][key] = {
                "c": c,
                "backbone_internal_shape": list(S.BACKBONE_INTERNAL_SHAPE),
                "wire_latent_shape_observed": sh,
                "wire_latent_shape_registered": list(S.wire_shape(c)),
                "wire_latent_numel_observed": None if sh is None else int(math.prod(sh)),
                "payload_bytes_observed": widths.get(key),
                "payload_bytes_registered": S.wire_bytes(c),
                "element_ratio": S.wire_numel(c) / S.N_TILE,
                "bare_payload_byte_ratio": S.wire_bytes(c) / S.INPUT_TILE_BYTES,
                "receiver": (ev.get("receivers") or {}).get(key),
            }
            r = out["arms"][key]
            r["verified"] = (
                sh == list(S.wire_shape(c)) and r["payload_bytes_observed"] == S.wire_bytes(c)
                and r["wire_latent_numel_observed"] == S.wire_numel(c) and r["receiver"] is not None
            )  # fmt: skip
        for ck in (run.rec[arm] or {}).get("checkpoints", []):
            out["arms"][S.arm_method(arm, "FINAL")].setdefault("training_checkpoint_payload_bytes", {})[
                str(ck["update"])
            ] = ck["train_eval"]["payload_bytes"]
    out["step_zero_payload_bytes"] = {
        c: v.get("payload_bytes") for c, v in (ini.get("step_zero_train") or {}).items()
    }
    out["all_verified"] = all(v["verified"] for v in out["arms"].values())
    return out


def damage_recovery(run: Run, s32: float) -> dict[str, Any]:
    out = {}
    for arm in S.ARM_ORDER:
        r0, rf = S.arm_method(arm, "STEP0"), S.arm_method(arm, "FINAL")
        rows = {}
        for g in S.COHORTS:
            src = method_row(run, g, f"{S.SOURCE}:FLOAT", s32)
            a0, af = method_row(run, g, r0, s32), method_row(run, g, rf, s32)
            c16 = method_row(run, g, "C16-CONTROL:FINAL", s32)
            if not (src and a0 and af and c16):
                continue
            rows[g] = {
                "source_nmse": src["normalized_mse"], "step0_nmse": a0["normalized_mse"], "final_nmse": af["normalized_mse"],
                "damage_ratio_step0_over_source": a0["normalized_mse"] / src["normalized_mse"],
                "recovery_ratio_final_over_step0": af["normalized_mse"] / a0["normalized_mse"],
                "fraction_of_damage_recovered": (a0["normalized_mse"] - af["normalized_mse"]) / (a0["normalized_mse"] - src["normalized_mse"])
                if a0["normalized_mse"] != src["normalized_mse"] else None,
                "final_over_source": af["normalized_mse"] / src["normalized_mse"],
                "final_over_c16_final": af["normalized_mse"] / c16["normalized_mse"],
                "final_minus_c16_final_rse": af["rse"] - c16["rse"],
                "step0_rse": a0["rse"], "final_rse": af["rse"], "step0_cosine": a0["cosine"], "final_cosine": af["cosine"],
                "paired_final_vs_step0": paired(run, g, rf, r0), "paired_final_vs_c16_final": paired(run, g, rf, "C16-CONTROL:FINAL"),
                "paired_final_vs_source": paired(run, g, rf, f"{S.SOURCE}:FLOAT"),
            }  # fmt: skip
        out[arm] = rows
    return out


def learning_curves(run: Run, s32: float) -> dict[str, Any]:
    out = {}
    for arm in S.ARM_ORDER:
        rows = {}
        for ck in (run.rec[arm] or {}).get("checkpoints", []):
            te = ck["train_eval"]
            rows[str(ck["update"])] = {"train_normalized_mse_direct": te["normalized_mse_direct"], "train_rse": te["raw"]["rse"],
                                       "train_cosine": te["raw"]["cosine"], "train_norm_ratio": te["raw"]["norm_ratio"]}  # fmt: skip
        for upd, point in ((0, "STEP0"), (run.cfg["max_updates"], "FINAL")):
            ck = run.ck_sums.get((arm, upd))
            key = S.arm_method(arm, point)
            if ck is not None and f"TRAIN.{key}.raw" in run.sums:
                rows.setdefault(str(upd), {})["train_child_eval_bit_identical_to_eval_step"] = bool(
                    (ck["TRAIN.raw"] == run.sums[f"TRAIN.{key}.raw"]).all()
                    and (ck["TRAIN.norm"] == run.sums[f"TRAIN.{key}.norm"]).all()
                )
        ini = run.init_sums.get(f"TRAIN.c{S.ARMS[arm]}.raw")
        if ini is not None and f"TRAIN.{S.arm_method(arm, 'STEP0')}.raw" in run.sums:
            rows["0"]["init_step_zero_bit_identical_to_eval_step"] = bool(
                (ini == run.sums[f"TRAIN.{S.arm_method(arm, 'STEP0')}.raw"]).all()
            )
        out[arm] = rows
    return out


def minibatch_curve(run: Run, arm: str) -> list[dict[str, Any]]:
    p = run.root / "steps" / arm / "curve.jsonl"
    rows = [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()] if p.is_file() else []
    return [{"update": r["update"], "pre_update_train_loss": r["pre_update_train_loss"], "batch_rse_normalized": r["pre_update_batch"]["rse"],
             **{f"grad_l2.{gname}": v["grad_l2"] for gname, v in r["grad"].items()},
             **{f"param_change_from_start.{gname}": v["delta_l2"] for gname, v in r["param_change_from_start"].items()}} for r in rows]  # fmt: skip


def input_dependence(run: Run, s32: float) -> dict[str, Any]:
    out = {}
    names = (
        *S.ARM_FINAL,
        *S.ARM_STEP0,
        *S.INPUT_PCA,
        "SCALAR-4",
        "INT8-SHARED",
        f"{S.SOURCE}:FLOAT",
        f"{S.SOURCE}:INT8LAT",
    )
    for g in S.COHORTS:
        rows = {}
        for m in names:
            rows[m] = {
                "vs_wrong": paired(run, g, m, f"C-WRONG:{m}"),
                "vs_zero_output": paired(run, g, m, "C-ZERO"),
                "vs_mean_tile": paired(run, g, m, "C-MEAN"),
            }
            if m in (*S.ARM_FINAL, *S.ARM_STEP0):
                rows[m]["vs_zero_wire"] = paired(run, g, m, f"C-ZWIRE:{m}")
            if m.startswith(S.SOURCE):
                rows[m]["vs_zero_latent"] = paired(run, g, m, f"C-ZLAT:{S.SOURCE}")
        pooled = {}
        for k in [
            *(f"C-WRONG:{m}" for m in names),
            *(f"C-ZWIRE:{m}" for m in (*S.ARM_FINAL, *S.ARM_STEP0)),
            "C-ZERO",
            "C-MEAN",
            f"C-ZLAT:{S.SOURCE}",
        ]:
            r = method_row(run, g, k, s32)
            if r:
                pooled[k] = {q: r[q] for q in ("rse", "cosine", "normalized_mse")}
        out[g] = {"paired": rows, "pooled_controls": pooled}
    return out


def comparators(run: Run, s32: float) -> dict[str, Any]:
    out = {"matching_elements": {}, "matching_bytes": {}}
    for arm, pca in MATCHING_ELEMENTS.items():
        key = S.arm_method(arm, "FINAL")
        out["matching_elements"][arm] = {g: {"arm": method_row(run, g, key, s32), "pca": method_row(run, g, pca, s32), "pca_name": pca,
                                              "paired_arm_vs_pca": paired(run, g, key, pca)} for g in S.COHORTS}  # fmt: skip
        for g in S.COHORTS:
            r = out["matching_elements"][arm][g]
            r["arm"] = {q: r["arm"][q] for q in ("rse", "cosine", "normalized_mse")} if r["arm"] else None
            r["pca"] = {q: r["pca"][q] for q in ("rse", "cosine", "normalized_mse")} if r["pca"] else None
    for budget, ms in SIZES.items():
        out["matching_bytes"][budget] = {}
        for g in S.COHORTS:
            rows = {m: method_row(run, g, m, s32) for m in ms}
            rows = {m: r for m, r in rows.items() if r is not None}
            rank = sorted(rows, key=lambda m: rows[m]["raw_sums"]["sse"])
            arm = ms[0]
            out["matching_bytes"][budget][g] = {
                "methods": {
                    m: {
                        q: r[q]
                        for q in (
                            "rse",
                            "cosine",
                            "normalized_mse",
                            "elements",
                            "element_kind",
                            "payload_bytes",
                        )
                    }
                    for m, r in rows.items()
                },
                "ranking_by_pooled_sse": rank,
                "best": rank[0] if rank else None,
                "paired": {f"{arm} vs {b}": paired(run, g, arm, b) for b in ms[1:]},
            }
    return out


def secondary_target(run: Run, s32: float) -> dict[str, Any]:
    out = {"target_normalized_mse": S.TARGET_NORMALIZED_MSE, "role": "secondary outcome only"}
    for m in (*S.ARM_FINAL, *S.ARM_STEP0, f"{S.SOURCE}:FLOAT", *S.INPUT_PCA, "SCALAR-4", "INT8-SHARED"):
        out[m] = {}
        for g in S.COHORTS:
            r = method_row(run, g, m, s32)
            if r:
                out[m][g] = {"normalized_mse": r["normalized_mse"], "basis": r["normalized_mse_basis"], "met": r["normalized_mse"] <= S.TARGET_NORMALIZED_MSE,
                             "factor_above_target": r["normalized_mse"] / S.TARGET_NORMALIZED_MSE}  # fmt: skip
    return out


def compliance_limits(run: Run) -> dict[str, Any]:
    led, lim = run.ledger, run.cfg["limits"]
    ends = {s: led.get(f"end_{s}.json") or {} for s in S.STEPS}
    inv, inv_end = led.get("invocation_001.json") or {}, led.get("invocation_001_end.json") or {}
    guards = [e.get("guard") or {} for e in ends.values()]
    return {
        "neural_arm_s": {
            "used": max(ends[a].get("wall_s", math.inf) for a in S.NEURAL_STEPS),
            "limit": lim["neural_arm_s"],
        },
        "init_s": {"used": ends["INIT"].get("wall_s", math.inf), "limit": lim["init_s"]},
        "eval_child_s": {"used": ends["EVAL"].get("wall_s", math.inf), "limit": lim["eval_child_s"]},
        "aggregate_s": {
            "used": inv_end.get("aggregate_wall_s", math.inf),
            "limit": (inv.get("plan") or {}).get("aggregate_s"),
        },
        "rss_bytes": {
            "used": max((g.get("sampled_max_rss_bytes", math.inf) for g in guards), default=math.inf),
            "limit": min((g.get("rss_limit_bytes", 0) for g in guards), default=0),
        },  # fmt: skip
        "evidence_bytes": {
            "used": inv_end.get("evidence_bytes_run_root", math.inf),
            "limit": lim["evidence_cap_bytes"],
        },
        "invocations": {"used": run.status["invocations"], "limit": lim["max_invocations"]},
        "neural_starts": {"used": run.status["neural_starts"], "limit": lim["max_neural_starts"]},
        "channel_pca_fits": {"used": run.status["channel_pca_fits"], "limit": lim["max_channel_pca_fits"]},
    }


def upp(run: Run, cfg: dict) -> dict[str, Any]:
    import upp_20261014_classify as P  # vendored, hash-pinned

    pol = P.load_policy()
    out: dict[str, Any] = {
        "policy_id": pol["policy_id"],
        "policy_sha256": pol["policy_sha256"],
        "classifier_sha256": K.sha256_file(Path(P.__file__)),
    }
    if pol["policy_sha256"] != cfg["primary_policy"]["policy_sha256"]:
        run.problems.append("the vendored policy file differs from the pinned policy SHA-256")
    limits = compliance_limits(run)
    out["limits"] = limits
    ev = {"verified": not run.problems, "problems": list(run.problems)}
    finite_in = cfg["ucx"]["split_finite"]
    all_done = all(v == "COMPLETED" for v in run.status["steps"].values())
    units = {
        S.arm_method(a, "FINAL"): ((run.rec[a] or {}).get("updates_done"), cfg["max_updates"], "primary")
        for a in S.ARM_ORDER
    }
    units.update({S.arm_method(a, "STEP0"): (0, 0, "step zero (descriptive)") for a in S.ARM_ORDER})
    units.update({f"{S.SOURCE}:FLOAT": (cfg["uoa"]["source_updates"], cfg["uoa"]["source_updates"], "reference"),
                  f"{S.SOURCE}:INT8LAT": (cfg["uoa"]["source_updates"], cfg["uoa"]["source_updates"], "reference")})  # fmt: skip
    units.update({m: (0, 0, "comparator (report only)") for m in (*S.INPUT_PCA, "SCALAR-4", "INT8-SHARED")})
    guard_actions = [
        e["guard"]["action"]
        for s in S.STEPS
        if (e := run.ledger.get(f"end_{s}.json")) and (e.get("guard") or {}).get("action")
    ]
    for name, (done, planned, role) in units.items():
        run_info = {"started": True, "status": "COMPLETED" if all_done else "INCOMPLETE", "updates_done": done, "updates_planned": planned,
                    "interrupted": bool((run.ledger.get("invocation_001_end.json") or {}).get("interrupted", True)),
                    "guard_actions": guard_actions, "limits": limits}  # fmt: skip
        res: dict[str, Any] = {"role": role, "cohorts": {}}
        for g in S.COHORTS:
            arr, fin = run.sums.get(f"{g}.{name}.raw"), run.sums.get(f"{g}.{name}.finite")
            if arr is None:
                continue
            members, tiles = {}, {"PASS": 0, "FIT_FAIL": 0, "other": 0}
            for i in range(arr.shape[0]):
                fe = {
                    "finite": bool(fin[i]),
                    "n": S.N_TILE,
                    **{k: float(arr[i, j]) for j, k in enumerate(KEYS4)},
                }
                inp = {"all_finite": bool(finite_in[g]), "target_energy": float(arr[i, 1])}
                o = P.classify(
                    {"evidence": ev, "input": inp, "run": run_info, "final_eval": fe}, pol, S.REQUIRED_LIMITS
                )
                tiles[o["outcome"] if o["outcome"] in tiles else "other"] += 1
                members[f"{g}:{i}"] = {"input": inp, "final_eval": fe}
            po = P.classify_pooled(
                {"evidence": ev, "run": run_info, "registered_members": list(members), "members": members},
                pol,
                S.REQUIRED_LIMITS,
            )
            res["cohorts"][g] = {
                "tiles": tiles,
                "pooled_outcome": po["outcome"],
                "pooled_quantities": po.get("quantities"),
                "reasons": po["reasons"],
            }
        if all(g in res["cohorts"] for g in S.HELDOUT):
            res["study"] = P.study_summary(
                {g: res["cohorts"][g]["pooled_outcome"] for g in S.HELDOUT}, list(S.HELDOUT), pol
            )["decision"]
        out[name] = res
    passing = [a for a in S.ARM_ORDER if out[S.arm_method(a, "FINAL")].get("study") == "PASS"]
    out["smallest_tested_adapted_element_count_with_study_pass"] = min(
        (S.wire_numel(S.ARMS[a]) for a in passing), default=None
    )
    return out


def breakdowns(run: Run, s32: float) -> dict[str, Any]:
    keys = run.cfg["split"]["keys"]
    out = {}
    for g in S.COHORTS:
        out[g] = {}
        n_t = len(keys[g]) * S.TILES_PER_SNAPSHOT
        lab = {
            "snapshot": [f"t{keys[g][i // 24][0]}-c{keys[g][i // 24][1]}" for i in range(n_t)],
            "client": [f"c{keys[g][i // 24][1]}" for i in range(n_t)],
            "row_index": [f"row{S.ROWS[(i % 24) // 6]}" for i in range(n_t)],
            "layer": [f"layer{S.LAYERS[(i % 24) % 6]}" for i in range(n_t)],
        }
        for m in (*S.ARM_FINAL, *S.ARM_STEP0, *S.INPUT_PCA, "SCALAR-4", f"{S.SOURCE}:FLOAT"):
            arr = run.sums.get(f"{g}.{m}.raw")
            if arr is None:
                continue
            res = {}
            for kind, labels in lab.items():
                groups: dict[str, list[int]] = {}
                for i, la in enumerate(labels):
                    groups.setdefault(la, []).append(i)
                res[kind] = {la: {"tiles": len(ix), **{q: v for q, v in metrics(fsum_cols(arr[ix]), len(ix) * S.N_TILE).items() if q in ("rse", "cosine")},
                                  "normalized_mse": metrics(fsum_cols(arr[ix]), len(ix) * S.N_TILE)["mse"] / (s32 * s32)} for la, ix in groups.items()}  # fmt: skip
            bl = run.sums.get(f"{g}.{m}.raw_blocks")
            res["lora_block"] = {
                name: {
                    q: v
                    for q, v in metrics(fsum_cols(bl[:, j, :]), arr.shape[0] * 256).items()
                    if q in ("rse", "cosine")
                }
                for j, name in enumerate(S.BLOCKS)
            }
            rse_t = arr[:, 0] / arr[:, 1]
            res["per_tile_rse_range"] = [float(rse_t.min()), float(rse_t.max())]
            out[g][m] = res
    return out


def byte_accounting(run: Run) -> dict[str, Any]:
    import numpy as np
    from safetensors import safe_open

    ev = run.rec["EVAL"] or {}
    out = {"payload_bytes_per_tile_observed": ev.get("payload_bytes_per_tile", {}), "arms": {}}
    for arm, c in S.ARMS.items():
        last = [
            ck for ck in (run.rec[arm] or {}).get("checkpoints", []) if ck["update"] == run.cfg["max_updates"]
        ]
        if not last:
            continue
        sizes = {}
        with safe_open(last[0]["state_file"]["path"], framework="numpy") as f:
            for k in f.keys():
                a = f.get_tensor(k)
                sizes[k] = int(a.size) * np.dtype(a.dtype).itemsize

        def part(prefix: str, sizes: dict = sizes) -> dict:
            ks = [k for k in sizes if k.startswith(prefix)]
            bn = [k for k in ks if k.endswith(("running_mean", "running_var", "num_batches_tracked"))]
            rp = [k for k in ks if k.endswith("relative_position_index")]
            return {"parameter_bytes": sum(sizes[k] for k in ks if k not in bn and k not in rp), "batchnorm_buffer_bytes": sum(sizes[k] for k in bn),
                    "position_index_buffer_bytes_int64": sum(sizes[k] for k in rp), "total_bytes": sum(sizes[k] for k in ks)}  # fmt: skip

        rec = {
            "decoder": part("base.decoder."),
            "expand": part("expand."),
            "encoder": part("base.encoder."),
            "reduce": part("reduce."),
        }
        rec["receiver_one_time_bytes"] = rec["decoder"]["total_bytes"] + rec["expand"]["total_bytes"] + 4
        rec["sender_one_time_bytes"] = rec["encoder"]["total_bytes"] + rec["reduce"]["total_bytes"] + 4
        rec["bridge_parameters"] = {"reduce": 16 * c + c, "expand": c * 16 + 16}
        rec["payload_bytes"] = S.wire_bytes(c)
        rec["amortized_bytes_per_message"] = {
            str(n): (S.wire_bytes(c) * n + rec["receiver_one_time_bytes"]) / n for n in (504, 3072, 64512)
        }
        rec["container_file_bytes"] = Path(last[0]["state_file"]["path"]).stat().st_size
        out["arms"][arm] = rec
    out["comparator_one_time_bytes"] = {"PCA-256": 4 * 1024 + 256 * 4096, "PCA-64": 4 * 1024 + 64 * 4096, "PCA-16": 4 * 1024 + 16 * 4096,
                                         "SCALAR-4": 16, "INT8-SHARED": 4}  # fmt: skip
    out["notes"] = ["receiver one-time = decoder parameters + BN buffers + position-index buffers + expand + the 4-B scale; the sender also needs the encoder + reduce",
                    "transport framing, message headers, a method identifier, codec refresh and codec training are not modeled"]  # fmt: skip
    return out


def reproduction_vs_uoa(run: Run) -> dict[str, Any]:
    o = run.cfg["uoa"]
    if not o.get("eval_sums"):
        return {"available": False}
    from safetensors.numpy import load

    data = Path(o["eval_sums"]["path"]).read_bytes()
    if K.sha256_bytes(data) != o["eval_sums"]["sha256"]:
        return {"available": False, "problem": "UOA eval sums differ from their record"}
    uoa = load(data)
    pairs = {f"{S.SOURCE}:FLOAT": "U-LEAKY-LONG:FLOAT", f"{S.SOURCE}:INT8LAT": "U-LEAKY-LONG:INT8LAT", "PCA-256": "PCA-256", "PCA-64": "PCA-64",
             "SCALAR-4": "SCALAR-4", "INT8-SHARED": "INT8-SHARED"}  # fmt: skip
    out = {"available": True}
    for ours, theirs in pairs.items():
        out[ours] = {
            g: bool((run.sums[f"{g}.{ours}.raw"] == uoa[f"{g}.{theirs}.raw"]).all()) for g in S.COHORTS
        }
    lat = o.get("eval_latents")
    if lat:
        ld = Path(lat["path"]).read_bytes()
        if K.sha256_bytes(ld) == lat["sha256"]:
            evl = run.rec["EVAL"]["files"]["latents"]
            ours = load(Path(evl["path"]).read_bytes())
            theirs = load(ld)
            out["source_latents_vs_uoa"] = {
                g: bool((ours[f"{g}.{S.SOURCE}.z"] == theirs[f"{g}.U-LEAKY-LONG.z"]).all()) for g in S.COHORTS
            }
    return out


def resources(run: Run) -> dict[str, Any]:
    out = {"steps": {}}
    for s in S.STEPS:
        e = run.ledger.get(f"end_{s}.json") or {}
        g = e.get("guard") or {}
        out["steps"][s] = {"status": e.get("status"), "wall_s": e.get("wall_s"), "limit_s": e.get("limit_s"), "sampled_max_rss_bytes": g.get("sampled_max_rss_bytes"),
                           "rss_limit_bytes": g.get("rss_limit_bytes"), "child_os_peak_wset_bytes": ((run.rec[s] or {}).get("memory") or {}).get("os_peak_wset_bytes")}  # fmt: skip
    inv_end = run.ledger.get("invocation_001_end.json") or {}
    out.update(invocation_aggregate_wall_s=inv_end.get("aggregate_wall_s"), pilots_wall_s=run.cfg["preflight"]["pilot_wall_s_total"],
               evidence_bytes_run_root=inv_end.get("evidence_bytes_run_root"))  # fmt: skip
    for arm in S.NEURAL_STEPS:
        r = run.rec[arm] or {}
        out[arm] = {
            "update_time_compute_only": (r.get("update_time_s") or {}).get("compute_only"),
            "checkpoint_time_s": r.get("checkpoint_time_s"),
        }
    return out


def write_csvs(run: Run, out: Path, s32: float) -> dict[str, str]:
    files = {}
    rows = [{"arm": arm, **r} for arm in S.ARM_ORDER for r in minibatch_curve(run, arm)]
    if rows:
        p = out / "learning_curves.csv"
        with p.open("x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        files["learning_curves.csv"] = K.sha256_file(p)
    p = out / "checkpoints.csv"
    with p.open("x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "update", "elements", "train_normalized_mse_direct", "train_rse", "train_cosine"])
        for arm in S.ARM_ORDER:
            for ck in (run.rec[arm] or {}).get("checkpoints", []):
                te = ck["train_eval"]
                w.writerow(
                    [
                        arm,
                        ck["update"],
                        S.wire_numel(S.ARMS[arm]),
                        repr(te["normalized_mse_direct"]),
                        repr(te["raw"]["rse"]),
                        repr(te["raw"]["cosine"]),
                    ]
                )
    files["checkpoints.csv"] = K.sha256_file(p)
    p = out / "per_tile_metrics.csv"
    keys = run.cfg["split"]["keys"]
    meths = (*S.ARM_FINAL, *S.ARM_STEP0, *S.INPUT_PCA, "SCALAR-4", f"{S.SOURCE}:FLOAT")
    with p.open("x", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "cohort",
                "tile",
                "time",
                "client",
                "row_index",
                "layer",
                *[f"{m}|{q}" for m in meths for q in ("rse", "cosine", "nmse")],
            ]
        )
        for g in S.COHORTS:
            arrs = [run.sums.get(f"{g}.{m}.raw") for m in meths]
            for i in range(len(keys[g]) * S.TILES_PER_SNAPSHOT):
                t, c = keys[g][i // 24]
                vals = []
                for a in arrs:
                    sse, te, pe, d = (float(v) for v in a[i])
                    vals += [
                        repr(sse / te),
                        repr(d / math.sqrt(pe * te)) if pe > 0 and te > 0 else "",
                        repr(sse / S.N_TILE / (s32 * s32)),
                    ]
                w.writerow([g, i, t, c, S.ROWS[(i % 24) // 6], S.LAYERS[(i % 24) % 6], *vals])
    files["per_tile_metrics.csv"] = K.sha256_file(p)
    return files


def main(out_dir: str, cfg_path: str | None = None) -> int:
    cfg = K.load_json(cfg_path or PKG / "ulr_config.json")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    run = Run(cfg)
    s32 = s32_of(cfg)
    ini = run.rec["INIT"] or {}
    res: dict[str, Any] = {"schema": "ulr-20261015-v1.analysis/v1", "protocol_id": cfg["protocol_id"], "ledger_status": run.status, "s32": s32,
                           "inherited_source_updates": cfg["uoa"]["source_updates"], "adaptation_updates_per_arm": cfg["max_updates"]}  # fmt: skip
    all_methods = (*S.ARM_FINAL, *S.ARM_STEP0, *S.INPUT_PCA, *S.REFERENCES)
    res["methods"] = {g: {m: method_row(run, g, m, s32) for m in all_methods} for g in S.COHORTS}
    res["A_element_reduction"] = element_reduction(run)
    res["B_upp"] = upp(run, cfg)
    res["C_damage_recovery"] = damage_recovery(run, s32)
    res["C_learning_curves"] = learning_curves(run, s32)
    res["C_input_dependence"] = input_dependence(run, s32)
    res["D_comparators"] = comparators(run, s32)
    res["D_secondary_target"] = secondary_target(run, s32)
    res["step_zero_all_widths_train"] = ini.get("step_zero_train")
    res["channel_pca"] = ini.get("channel_pca")
    res["bn_train_vs_eval_source"] = [{q: r[q] for q in ("batch", "normalized_mse_eval_mode", "normalized_mse_train_mode", "max_abs_train_minus_eval_output")}
                                      for r in ini.get("bn_train_vs_eval", [])]  # fmt: skip
    res["bridge_stats"] = {}
    for arm in S.ARM_ORDER:
        p = run.root / "steps" / arm / "checkpoints.jsonl"
        if p.is_file():
            res["bridge_stats"][arm] = {
                str(json.loads(x)["update"]): json.loads(x)["bridge"]
                for x in p.read_text("utf-8").splitlines()
                if x.strip()
            }
    res["breakdowns"] = breakdowns(run, s32)
    res["bytes"] = byte_accounting(run)
    res["reproduction_vs_uoa"] = reproduction_vs_uoa(run)
    res["resources"] = resources(run)
    res["evidence_problems"] = run.problems
    res["files"] = write_csvs(run, out, s32)
    sha = K.write_exclusive(out / "analysis.json", K.pretty_bytes(res))
    print(json.dumps({"analysis_sha256": sha, "problems": run.problems, "files": res["files"]}, indent=1))
    return 0 if not run.problems else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
