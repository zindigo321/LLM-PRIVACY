"""Learning-curve figure and table of the URP-20261013-v1 U-PATCH run, from the recorded observations only.

``python -B make_curve_figure.py`` from this directory. Reads ``curve.jsonl`` and ``child_record.json`` (both on the
sealed list) and writes, exclusively, ``learning_curve_u_patch.svg`` (standard library only: matplotlib is not
installed and no dependency is added) and ``curve_u_patch.csv``. No model code, no computation beyond reading.
"""

import json
import math
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
ARM = Path("D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH")
GROUPS = ("encoder.sampling", "encoder.core", "decoder.core", "decoder.sampling")
W, LEFT, RIGHT, PH, GAP, TOP = 920, 90, 250, 190, 70, 70


def write_new(path: Path, text: str) -> None:
    with path.open("xb") as fh:
        fh.write(text.encode("utf-8"))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(i, title, ylab, pts, ymin, ymax, log, refs, marks, ticks):
    top, w = TOP + i * (PH + GAP), W - LEFT - RIGHT
    tf = (lambda v: math.log10(v)) if log else (lambda v: v)
    xs = lambda k: LEFT + k / 3000 * w  # noqa: E731
    ys = lambda v: top + (tf(ymax) - tf(v)) / (tf(ymax) - tf(ymin)) * PH  # noqa: E731
    s = [f'<rect x="{LEFT}" y="{top}" width="{w}" height="{PH}" fill="none" stroke="#444"/>',
         f'<text x="{LEFT}" y="{top - 8}" font-size="13" font-weight="bold">{esc(title)}</text>',
         f'<text transform="translate(22,{top + PH / 2}) rotate(-90)" text-anchor="middle" '
         f'font-size="11">{esc(ylab)}</text>']
    for v in ticks:
        s.append(f'<line x1="{LEFT - 4}" x2="{LEFT}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#444"/>'
                 f'<text x="{LEFT - 7}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="10">{v:g}</text>')
    for k in range(0, 3001, 500):
        s.append(f'<line x1="{xs(k):.1f}" x2="{xs(k):.1f}" y1="{top + PH}" y2="{top + PH + 4}" stroke="#444"/>'
                 f'<text x="{xs(k):.1f}" y="{top + PH + 16}" text-anchor="middle" font-size="10">{k}</text>')
    for v, label, color in refs:
        s.append(f'<line x1="{LEFT}" x2="{LEFT + w}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="{color}" '
                 f'stroke-dasharray="6,4"/><text x="{LEFT + w + 6}" y="{ys(v) + 4:.1f}" font-size="10" '
                 f'fill="{color}">{esc(label)}</text>')
    line = " ".join(f"{xs(k):.1f},{ys(v):.1f}" for k, v in pts)
    s.append(f'<polyline points="{line}" fill="none" stroke="#1f5fa8" stroke-width="1.6"/>')
    s += [f'<circle cx="{xs(k):.1f}" cy="{ys(v):.1f}" r="2.2" fill="#1f5fa8"/>' for k, v in pts]
    for k, v, label, color, dy in marks:
        s.append(f'<rect x="{xs(k) - 4:.1f}" y="{ys(v) - 4:.1f}" width="8" height="8" fill="{color}"/>'
                 f'<text x="{LEFT + w + 6}" y="{ys(v) + 4 + dy:.1f}" font-size="10" fill="{color}">'
                 f'{esc(label)}</text>')
    return s


def main() -> int:
    pts = [json.loads(x) for x in (ARM / "curve.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rec = json.loads((ARM / "child_record.json").read_text(encoding="utf-8"))
    fe, ft, s0 = rec["final"]["eval_mode"], rec["final"]["train_mode"], rec["step0"]["post_init"]["eval"]["original"]
    base_n = rec["baselines"]["normalized"]["shared_scalar_mean"]
    loss = [(p["update"], p["pre_update_train_loss"]) for p in pts]
    rse = [(p["update"], p["pre_update_train"]["rse"]) for p in pts]
    cos = [(p["update"], p["pre_update_train"]["cosine"]) for p in pts]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{TOP + 3 * (PH + GAP) + 50}" '
           f'font-family="Arial, Helvetica, sans-serif"><rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="22" font-size="14" font-weight="bold">URP-20261013-v1 U-PATCH: Uformer, one real R3 '
           'patch, seed 1, Adam 2e-4, 3,000 updates</text>',
           '<text x="20" y="38" font-size="11">Blue: curve point k = train-mode forward on the patch BEFORE optimizer '
           'update k (k = 1 uses the initial state), normalized space.</text>',
           '<text x="20" y="52" font-size="11">Squares: evaluation of the initial state (step 0, x = 0) and of the '
           'final state after update 3,000 (x = 3000), original space.</text>']
    svg += panel(0, "A. Pre-update training loss (MSE, normalized space, train mode)", "MSE (log scale)", loss,
                 1e-3, 1.0, True, [(base_n["mse"], f"best constant (scalar mean) {base_n['mse']:.5f}", "#b8860b")],
                 [], [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    svg += panel(1, "B. Relative squared error RSE = SSE / sum(x^2)", "RSE (log scale)", rse, 0.005, 20.0, True,
                 [(0.5, "EP M1 bound 0.50", "#2e7d32"), (0.01, "strict bound 0.01", "#c62828"),
                  (base_n["rse"], "best constant 0.9998", "#b8860b")],
                 [(0, s0["rse"], f"step 0 eval {s0['rse']:.4f}", "#6a1b9a", -12),
                  (3000, fe["rse"], f"final eval {fe['rse']:.4f} (verdict basis)", "#c62828", 10),
                  (3000, ft["rse"], f"final train {ft['rse']:.4f} (reported)", "#ef6c00", 24)],
                 [0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    svg += panel(2, "C. Cosine similarity between reconstruction and patch", "cosine", cos, -0.1, 1.0, False,
                 [(0.99, "strict bound 0.99", "#c62828"), (0.8, "EP M2 bound 0.80", "#2e7d32")],
                 [(0, s0["cosine"], f"step 0 eval {s0['cosine']:.4f}", "#6a1b9a", 0),
                  (3000, fe["cosine"], f"final eval {fe['cosine']:.4f} / train {ft['cosine']:.4f}", "#c62828", 3),
                  (3000, ft["cosine"], "", "#ef6c00", 0)],
                 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    y = TOP + 3 * (PH + GAP) - 30
    svg.append(f'<text x="{LEFT + (W - LEFT - RIGHT) / 2}" y="{y}" text-anchor="middle" font-size="11">'
               'optimizer update k</text>')
    svg.append(f'<text x="20" y="{y + 22}" font-size="10">Verdicts use the final eval-mode reconstruction in the '
               'original space; RSE and cosine are scale-invariant up to fp32 rounding.</text>')
    svg.append(f'<text x="20" y="{y + 36}" font-size="10">Curve values are reported only; no best point is selected. '
               'Not shown: compression, generalization or any other patch.</text>')
    svg.append("</svg>")
    text = "\n".join(svg) + "\n"
    ElementTree.fromstring(text)  # well-formed XML, or this raises
    rows = ["update,monotonic_s_after_update,pre_update_train_loss_normalized,pre_update_train_rse_normalized,"
            "pre_update_train_cosine_normalized," + ",".join(f"grad_l2_{g}" for g in GROUPS) + ","
            + ",".join(f"rel_change_from_init_{g}" for g in GROUPS) + ",rss_bytes_at_point"]
    for p in pts:
        rc = [p["param_change_from_init"][g]["delta_l2"] / p["param_change_from_init"][g]["ref_l2"] for g in GROUPS]
        rows.append(",".join(repr(v) for v in [p["update"], p["monotonic_s"], p["pre_update_train_loss"],
                                               p["pre_update_train"]["rse"], p["pre_update_train"]["cosine"],
                                               *[p["grad"][g]["grad_l2"] for g in GROUPS], *rc,
                                               p["rss_bytes_at_point"]]))
    write_new(HERE / "learning_curve_u_patch.svg", text)
    write_new(HERE / "curve_u_patch.csv", "\n".join(rows) + "\n")
    print(f"learning_curve_u_patch.svg {len(text.encode())} bytes; curve_u_patch.csv {len(rows) - 1} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
