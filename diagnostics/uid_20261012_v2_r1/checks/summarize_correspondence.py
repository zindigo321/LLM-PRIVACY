"""Print a one-line-per-case summary of evidence/upstream_correspondence.json (read-only)."""

import json
import sys
from pathlib import Path

r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("all_ok", r["all_ok"], "| einops", r["einops"]["version"], "| tol", r["tolerances_declared_before_run"])
print("vendored_sha256", r["vendored_sha256"])
for c in r["cases"]:
    parts = []
    for k in ("forward", "input_grad", "param_grads"):
        v = c.get(k)
        if isinstance(v, dict) and "max_abs" in v:
            parts.append(f"{k}.max_abs={v['max_abs']:.3e} rel={v['rel_l2']:.3e}")
        elif isinstance(v, dict):
            parts.append(k + "=" + ",".join(f"{q}:{w['max_abs']:.1e}" for q, w in v.items()))
    for k in ("exact", "roundtrip_exact", "state_keys", "upstream", "vendored"):
        if k in c:
            parts.append(f"{k}={c[k]}")
    print(f"{c['case']:<40} ok={c['ok']}  " + "  ".join(parts))
