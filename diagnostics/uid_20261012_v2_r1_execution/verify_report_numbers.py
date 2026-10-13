"""Read-only: recompute two derived numbers quoted in REPORT.md from the run records."""

import json
import math
from pathlib import Path

RUN = Path("D:/cgfed-runs/uid_20261012_v2_r1")

c = [json.loads(line) for line in (RUN / "arms/U-FIT/curve.jsonl").read_text().splitlines()]
span = c[-1]["monotonic_s"] - c[0]["monotonic_s"]
print("U-FIT s/update between curve points 1 and 3000:", span / (c[-1]["update"] - c[0]["update"]))
rec = json.loads((RUN / "arms/U-FIT/child_record.json").read_text())
print("U-FIT child wall s / 3000:", (rec["child_monotonic_s"]["end"] - rec["child_monotonic_s"]["begin"]) / 3000)
u = json.loads((RUN / "arms/U-CHECK/child_record.json").read_text())["final"]["eval_mode"]
p, t, d, s = u["pred_energy"], u["target_energy"], u["dot"], u["sse"]
print("U-CHECK P+T-2D:", p + t - 2 * d, "recorded SSE:", s, "ulp(P+T):", math.ulp(p + t))
