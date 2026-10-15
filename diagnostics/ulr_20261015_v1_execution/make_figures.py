"""ULR-20261015-v1 figures (SVG, standard library only), from ``analysis/analysis.json`` and ``learning_curves.csv``.

``python -B make_figures.py`` writes into this directory:
- ``element_count_vs_quality.svg``: pooled RSE against the number of transmitted elements per tile (log2 axis), for
  the step-zero PCA bridges (TRAIN, c = 16 / 8 / 4 / 2 / 1), the final adapted arms and the input PCA comparators
  (TRAIN / EARLY / LATE), the unmodified source (256 fp32), and SCALAR-4 / INT8-SHARED (1,024 2-bit / int8 values,
  different element types; shown for reference only);
- ``learning_curves_ulr.svg``: pre-update minibatch loss (normalized MSE, train mode) of the three arms every 100
  updates, the eval-mode full-TRAIN checkpoints, and the 1e-4 secondary target line (log scale).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

EXE = Path(__file__).resolve().parent
W, H, L, R, T, B = 780, 440, 70, 210, 30, 50
COL = {"C16-CONTROL": "#1f77b4", "C4-REDUCED": "#2ca02c", "C1-REDUCED": "#d62728", "PCA": "#9467bd", "STEP0": "#7f7f7f", "SRC": "#000000", "REF": "#ff7f0e"}
DASH = {"TRAIN": "", "EARLY": "6,3", "LATE": "2,3"}


def frame(title: str, ylab: str, xlab: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{(L + W - R) / 2}" y="18" text-anchor="middle" font-size="13">{title}</text>',
        f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="black"/><line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="black"/>',
        f'<text x="{(L + W - R) / 2}" y="{H - 12}" text-anchor="middle">{xlab}</text>',
        f'<text x="16" y="{(T + H - B) / 2}" transform="rotate(-90 16 {(T + H - B) / 2})" text-anchor="middle">{ylab}</text>',
    ]


def legend(parts: list, rows: list) -> None:
    for i, (label, color, dash, marker) in enumerate(rows):
        y = T + 12 + 15 * i
        d = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{W - R + 8}" y1="{y}" x2="{W - R + 30}" y2="{y}" stroke="{color}" stroke-width="2"{d}/>')
        if marker:
            parts.append(f'<circle cx="{W - R + 19}" cy="{y}" r="3" fill="{color}"/>')
        parts.append(f'<text x="{W - R + 34}" y="{y + 4}">{label}</text>')


def main() -> int:
    a = json.loads((EXE / "analysis" / "analysis.json").read_text("utf-8"))
    X = lambda e: L + (W - L - R) * (math.log2(e) - 4) / 6  # noqa: E731  (16 .. 1,024 elements)
    Y = lambda v: T + (H - T - B) * (1.4 - min(v, 1.4)) / 1.4  # noqa: E731  (RSE 0 .. 1.4)
    p = frame("Pooled RSE against transmitted elements per tile (1,024-element input)", "pooled RSE (SSE / target energy)", "transmitted elements per tile (log2)")
    for e in (16, 32, 64, 128, 256, 512, 1024):
        p.append(f'<line x1="{X(e):.1f}" y1="{H - B}" x2="{X(e):.1f}" y2="{H - B + 4}" stroke="black"/><text x="{X(e):.1f}" y="{H - B + 16}" text-anchor="middle">{e}</text>')
    for v in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        p.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke="#eee"/><text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    s0 = a["step_zero_all_widths_train"]
    pts = sorted((16 * int(c), v["train_rse"]) for c, v in s0.items())
    p.append('<polyline fill="none" stroke="{}" stroke-width="1.5" stroke-dasharray="3,2" points="{}"/>'.format(COL["STEP0"], " ".join(f"{X(e):.1f},{Y(r):.1f}" for e, r in pts)))
    p += [f'<circle cx="{X(e):.1f}" cy="{Y(r):.1f}" r="3" fill="{COL["STEP0"]}"/>' for e, r in pts]
    m = a["methods"]
    for g in ("TRAIN", "EARLY", "LATE"):
        fin = sorted((m[g][f"{arm}:FINAL"]["elements"], m[g][f"{arm}:FINAL"]["rse"]) for arm in ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED"))
        p.append('<polyline fill="none" stroke="#000" stroke-width="1.8"{} points="{}"/>'.format(f' stroke-dasharray="{DASH[g]}"' if DASH[g] else "", " ".join(f"{X(e):.1f},{Y(r):.1f}" for e, r in fin)))
        for arm in ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED"):
            r = m[g][f"{arm}:FINAL"]
            p.append(f'<circle cx="{X(r["elements"]):.1f}" cy="{Y(r["rse"]):.1f}" r="4" fill="{COL[arm]}"/>')
        pca = sorted((m[g][k]["elements"], m[g][k]["rse"]) for k in ("PCA-16", "PCA-64", "PCA-256"))
        p.append('<polyline fill="none" stroke="{}" stroke-width="1.5"{} points="{}"/>'.format(COL["PCA"], f' stroke-dasharray="{DASH[g]}"' if DASH[g] else "", " ".join(f"{X(e):.1f},{Y(r):.1f}" for e, r in pca)))
        p += [f'<rect x="{X(e) - 3:.1f}" y="{Y(r) - 3:.1f}" width="6" height="6" fill="{COL["PCA"]}"/>' for e, r in pca]
        src = m[g]["SOURCE-LEAKY:FLOAT"]
        p.append(f'<circle cx="{X(256):.1f}" cy="{Y(src["rse"]):.1f}" r="5" fill="none" stroke="{COL["SRC"]}" stroke-width="1.5"/>')
        for k in ("SCALAR-4", "INT8-SHARED"):
            p.append(f'<rect x="{X(1024) - 3:.1f}" y="{Y(m[g][k]["rse"]) - 3:.1f}" width="6" height="6" fill="{COL["REF"]}"/>')
    legend(p, [("final arms (TRAIN)", "#000", "", False), ("final arms (EARLY)", "#000", "6,3", False), ("final arms (LATE)", "#000", "2,3", False),
               ("C16 / C4 / C1 points", COL["C4-REDUCED"], "", True), ("step-zero PCA bridge (TRAIN)", COL["STEP0"], "3,2", True),
               ("input PCA-16/64/256", COL["PCA"], "", False), ("source FLOAT (open circle)", COL["SRC"], "", False),
               ("SCALAR-4 / INT8 (1,024 values)", COL["REF"], "", False)])  # fmt: skip
    p.append("</svg>")
    (EXE / "element_count_vs_quality.svg").write_text("\n".join(p) + "\n", encoding="utf-8")
    rows = list(csv.DictReader((EXE / "analysis" / "learning_curves.csv").open(encoding="utf-8")))
    vals = [float(r["pre_update_train_loss"]) for r in rows] + [1e-4]
    lo, hi = math.log10(min(vals) / 1.5), math.log10(max(vals) * 1.5)
    X2 = lambda u: L + (W - L - R) * u / 6000  # noqa: E731
    Y2 = lambda v: T + (H - T - B) * (hi - math.log10(v)) / (hi - lo)  # noqa: E731
    p = frame("Adaptation: minibatch loss (lines) and full-TRAIN eval (dots), normalized MSE", "normalized MSE (log)", "adaptation updates (after 12,000 inherited source updates)")
    for e in range(math.floor(lo), math.ceil(hi) + 1):
        v = 10.0**e
        if 10**lo <= v <= 10**hi:
            p.append(f'<line x1="{L}" y1="{Y2(v):.1f}" x2="{W - R}" y2="{Y2(v):.1f}" stroke="#eee"/><text x="{L - 6}" y="{Y2(v) + 4:.1f}" text-anchor="end">1e{e}</text>')
    for u in range(0, 6001, 1000):
        p.append(f'<text x="{X2(u):.1f}" y="{H - B + 16}" text-anchor="middle">{u}</text>')
    for arm in ("C16-CONTROL", "C4-REDUCED", "C1-REDUCED"):
        pts = [(int(r["update"]), float(r["pre_update_train_loss"])) for r in rows if r["arm"] == arm]
        p.append('<polyline fill="none" stroke="{}" stroke-width="1.2" points="{}"/>'.format(COL[arm], " ".join(f"{X2(x):.1f},{Y2(y):.1f}" for x, y in pts)))
        for u, v in a["C_learning_curves"][arm].items():
            p.append(f'<circle cx="{X2(int(u)):.1f}" cy="{Y2(v["train_normalized_mse_direct"]):.1f}" r="4" fill="{COL[arm]}" stroke="black"/>')
    p.append(f'<line x1="{L}" y1="{Y2(1e-4):.1f}" x2="{W - R}" y2="{Y2(1e-4):.1f}" stroke="black" stroke-dasharray="8,4"/>')
    legend(p, [(f"{arm} ({16 * c} elements)", COL[arm], "", True) for arm, c in (("C16-CONTROL", 16), ("C4-REDUCED", 4), ("C1-REDUCED", 1))] + [("secondary target 1e-4", "black", "8,4", False)])
    p.append("</svg>")
    (EXE / "learning_curves_ulr.svg").write_text("\n".join(p) + "\n", encoding="utf-8")
    print("written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
