#!/usr/bin/env python3
"""P3 comparison: rstsr matmul against the raw cblas gemv anchor, before vs after.

Reads two criterion passes (default `results/bench-run5-old.txt` = `062c970`
baseline, `results/bench-run6.txt` = branch head) and emits `results/tables-p3.md`:

- the per-cell table with the P0 and P3 rstsr times, both ratios and the
  speedup `P0 / P3`;
- the extras (batched gemv, batched gemm, `matmul_from` with `beta != 0`).

Usage: python3 results/make_p3_table.py [p0.txt] [p3.txt]
"""

import sys
from collections import OrderedDict

from make_tables import parse


def rows_of(data):
    """-> {(device/dtype, cell, orientation): {impl: ms}}"""
    rows = OrderedDict()
    for bid, ms in data.items():
        parts = bid.split("/")
        if parts[0] not in ("openblas", "faer"):
            continue
        device, dtype, *rest = parts
        if len(rest) < 3:
            continue
        cell, orient, impl = rest[-3], rest[-2], rest[-1]
        rows.setdefault((f"{device}/{dtype}", cell, orient), {})[impl] = ms
    return rows


def main():
    p0_path = sys.argv[1] if len(sys.argv) > 1 else "results/bench-run5-old.txt"
    p3_path = sys.argv[2] if len(sys.argv) > 2 else "results/bench-run6.txt"
    raw0, raw3 = parse(p0_path), parse(p3_path)
    p0, p3 = rows_of(raw0), rows_of(raw3)

    lines = [f"<!-- generated from {p0_path} (before) and {p3_path} (after) -->\n"]
    lines.append("| device/dtype | cell | orientation | before (ms) | before/anchor | after (ms) | after/anchor | speedup |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in p0:
        if key not in p3:
            continue
        a, b = p0[key], p3[key]
        if "rstsr" not in a or "rstsr" not in b or "gemv-anchor" not in b:
            continue
        dev, cell, orient = key
        lines.append(
            f"| {dev} | {cell} | {orient} | {a['rstsr']:.4f} | {a['rstsr'] / a['gemv-anchor']:.2f}× "
            f"| {b['rstsr']:.4f} | {b['rstsr'] / b['gemv-anchor']:.2f}× | {a['rstsr'] / b['rstsr']:.1f}× |"
        )

    lines.append("")
    lines.append("| extra | before (ms) | after (ms) | speedup |")
    lines.append("|---|---|---|---|")
    extras = OrderedDict()
    for data, tag in ((raw0, "p0"), (raw3, "p3")):
        for bid, ms in data.items():
            if "/extra/" in bid:
                extras.setdefault(bid, {})[tag] = ms
    for bid, vals in extras.items():
        if "p0" in vals and "p3" in vals:
            lines.append(f"| {bid} | {vals['p0']:.3f} | {vals['p3']:.3f} | {vals['p0'] / vals['p3']:.2f}× |")
        else:
            keys = ", ".join(f"{k}={v:.3f}" for k, v in vals.items())
            lines.append(f"| {bid} | {keys} | | |")

    text = "\n".join(lines) + "\n"
    print(text)
    with open("results/tables-p3.md", "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
