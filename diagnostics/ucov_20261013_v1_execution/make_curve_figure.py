"""Learning curves of the four UCOV-20261013-v1 arms, plus the closed URP-v1 centre tile for context, from the
recorded observations only.

``python -B make_curve_figure.py``. It reads each arm's ``curve.jsonl`` and ``child_record.json``, and the URP-v1
records for context, then writes, exclusively:
- ``learning_curves_ucov.svg``, using the standard library only (matplotlib is not installed);
- ``curves_ucov.csv``.
No model code is involved.
"""

import json
import math
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
RUN = Path("D:/cgfed-runs/ucov_20261013_v1/arms")
URP = Path("D:/cgfed-runs/urp_20261013_v1/arms/U-PATCH")
ARMS = ("U-TILE-1", "U-TILE-2", "U-TILE-3", "U-TILE-4")
COLORS = {
    "U-TILE-1": "#1f5fa8",
    "U-TILE-2": "#2e7d32",
    "U-TILE-3": "#c62828",
    "U-TILE-4": "#6a1b9a",
    "URP-v1 centre (context)": "#888888",
}
W, LEFT, RIGHT, PH, GAP, TOP = 960, 90, 290, 220, 80, 80


def load_arm(d: Path):
    if not (d / "curve.jsonl").is_file():
        return None, None
    pts = [json.loads(x) for x in (d / "curve.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rec = (
        json.loads((d / "child_record.json").read_text(encoding="utf-8"))
        if (d / "child_record.json").is_file()
        else None
    )
    return pts, rec


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def panel(i, title, ylab, series, ymin, ymax, refs, ticks, finals):
    top, w = TOP + i * (PH + GAP), W - LEFT - RIGHT
    lg = lambda v: math.log10(v)  # noqa: E731
    xs = lambda k: LEFT + k / 3000 * w  # noqa: E731
    ys = lambda v: top + (lg(ymax) - lg(max(v, ymin))) / (lg(ymax) - lg(ymin)) * PH  # noqa: E731
    s = [
        f'<rect x="{LEFT}" y="{top}" width="{w}" height="{PH}" fill="none" stroke="#444"/>',
        f'<text x="{LEFT}" y="{top - 10}" font-size="13" font-weight="bold">{esc(title)}</text>',
        f'<text transform="translate(24,{top + PH / 2}) rotate(-90)" text-anchor="middle" font-size="11">'
        f"{esc(ylab)}</text>",
    ]
    for v in ticks:
        s.append(
            f'<line x1="{LEFT - 4}" x2="{LEFT}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="#444"/>'
            f'<text x="{LEFT - 7}" y="{ys(v) + 4:.1f}" text-anchor="end" font-size="10">{v:g}</text>'
        )
    for k in range(0, 3001, 500):
        s.append(
            f'<line x1="{xs(k):.1f}" x2="{xs(k):.1f}" y1="{top + PH}" y2="{top + PH + 4}" stroke="#444"/>'
            f'<text x="{xs(k):.1f}" y="{top + PH + 16}" text-anchor="middle" font-size="10">{k}</text>'
        )
    for v, label, color in refs:
        s.append(
            f'<line x1="{LEFT}" x2="{LEFT + w}" y1="{ys(v):.1f}" y2="{ys(v):.1f}" stroke="{color}" '
            f'stroke-dasharray="6,4"/><text x="{LEFT + 4}" y="{ys(v) - 4:.1f}" font-size="10" fill="{color}">'
            f"{esc(label)}</text>"
        )
    for name, pts in series.items():
        dash = ' stroke-dasharray="3,3"' if "context" in name else ""
        line = " ".join(f"{xs(k):.1f},{ys(v):.1f}" for k, v in pts)
        s.append(f'<polyline points="{line}" fill="none" stroke="{COLORS[name]}" stroke-width="1.5"{dash}/>')
    for j, (name, v, label) in enumerate(finals):
        s.append(
            f'<rect x="{xs(3000) - 4:.1f}" y="{ys(v) - 4:.1f}" width="8" height="8" fill="{COLORS[name]}"/>'
            f'<text x="{LEFT + w + 8}" y="{top + 14 + 15 * j}" font-size="10" fill="{COLORS[name]}">'
            f"{esc(label)}</text>"
        )
    return s


def main() -> int:
    data = {arm: load_arm(RUN / arm) for arm in ARMS}
    data["URP-v1 centre (context)"] = load_arm(URP)
    loss = {
        n: [(p["update"], p["pre_update_train_loss"]) for p in pts] for n, (pts, _) in data.items() if pts
    }
    rse = {
        n: [(p["update"], p["pre_update_train"]["rse"]) for p in pts] for n, (pts, _) in data.items() if pts
    }
    finals_rse, finals_loss = [], []
    for n, (pts, rec) in data.items():
        if rec and rec.get("final"):
            fe = rec["final"]["eval_mode"]
            verdict = (rec.get("primary_verdict") or {}).get("outcome", "n/a (other protocol)")
            finals_rse.append(
                (n, fe["rse"], f"{n}: final eval RSE {fe['rse']:.4f}, cos {fe['cosine']:.4f}, {verdict}")
            )
            mean_n = rec["baselines"]["normalized"]["shared_scalar_mean"]["mse"]
            finals_loss.append(
                (
                    n,
                    pts[-1]["pre_update_train_loss"],
                    f"{n}: loss@3000 {pts[-1]['pre_update_train_loss']:.5f} (const {mean_n:.5f})",
                )
            )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{TOP + 2 * (PH + GAP) + 60}" '
        'font-family="Arial, Helvetica, sans-serif"><rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="22" font-size="14" font-weight="bold">UCOV-20261013-v1: four tile arms (fresh Uformer each, '
        "seed 1, Adam 2e-4, 3,000 updates) and the URP-v1 centre tile for context</text>",
        '<text x="20" y="40" font-size="11">Lines: curve point k = train-mode forward on the arm\'s own tile BEFORE '
        "optimizer update k (k = 1 uses the initial state), normalized space; recorded at 1, 100, 200, ..., 3000.</text>",
        '<text x="20" y="56" font-size="11">Squares at x = 3000 in panel B: final eval-mode RSE of the final state, '
        "original space (the primary verdict basis, UPP-20261013-v1).</text>",
    ]
    svg += panel(
        0,
        "A. Pre-update training loss (MSE, normalized space, train mode)",
        "MSE (log scale)",
        loss,
        1e-3,
        0.5,
        [(0.0166, "about the best constant MSE of each tile (normalized)", "#b8860b")],
        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        finals_loss,
    )
    svg += panel(
        1,
        "B. Relative squared error RSE = SSE / sum(x^2)",
        "RSE (log scale)",
        rse,
        0.05,
        20.0,
        [
            (0.5, "primary bound RSE <= 0.50 (UPP-20261013-v1)", "#2e7d32"),
            (1.0, "RSE of a zero or near-constant prediction (about 1)", "#b8860b"),
        ],
        [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        finals_rse,
    )
    y = TOP + 2 * (PH + GAP) - 30
    svg.append(
        f'<text x="{LEFT + (W - LEFT - RIGHT) / 2}" y="{y}" text-anchor="middle" font-size="11">'
        "optimizer update k</text>"
    )
    svg.append(
        f'<text x="20" y="{y + 24}" font-size="10">Curve values are reported only; no best point is '
        "selected. Each arm fits its own tile: this is input-coverage fitting, not held-out evaluation or "
        "compression.</text>"
    )
    svg.append(
        f'<text x="20" y="{y + 38}" font-size="10">The URP-v1 centre tile is a separate, closed experiment '
        "shown for context only (dashed); its verdicts are its own.</text>"
    )
    svg.append("</svg>")
    text = "\n".join(svg) + "\n"
    ElementTree.fromstring(text)  # well-formed XML, or this raises
    rows = [
        "series,update,monotonic_s_after_update,pre_update_train_loss_normalized,pre_update_train_rse_normalized,"
        "pre_update_train_cosine_normalized,rss_bytes_at_point"
    ]
    for n, (pts, _) in data.items():
        for p in pts or []:
            rows.append(
                ",".join(
                    [
                        n,
                        *(
                            repr(v)
                            for v in (
                                p["update"],
                                p["monotonic_s"],
                                p["pre_update_train_loss"],
                                p["pre_update_train"]["rse"],
                                p["pre_update_train"]["cosine"],
                                p["rss_bytes_at_point"],
                            )
                        ),
                    ]
                )
            )
    for name, content in (("learning_curves_ucov.svg", text), ("curves_ucov.csv", "\n".join(rows) + "\n")):
        with (HERE / name).open("xb") as fh:
            fh.write(content.encode("utf-8"))
    print(f"learning_curves_ucov.svg {len(text.encode())} bytes; curves_ucov.csv {len(rows) - 1} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
