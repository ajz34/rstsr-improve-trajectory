#!/usr/bin/env python3
"""Build the P0 results table from a criterion pass.

Parses `results/bench-runN.txt` (criterion's `time: [lo mid hi]` lines keyed by
the benchmark id) and emits `results/tables.md`.

Usage: python3 results/make_tables.py [results/bench-run1.txt]
"""

import re
import sys
from collections import OrderedDict

LINE_ID = re.compile(r"^(openblas|faer)/(f64|f32)/(.*)$")
TIME = re.compile(r"time:\s+\[([0-9.]+) (µs|ms|s) ([0-9.]+) (µs|ms|s) ([0-9.]+) (µs|ms|s)\]")
SCALE = {"µs": 1e-3, "ms": 1.0, "s": 1e3}


def parse(path):
    """-> {id: (mid_ms,)} keeping the last occurrence (criterion repeats)."""
    out = OrderedDict()
    last_id = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            m = LINE_ID.match(line)
            if m:
                last_id = line
                continue
            if last_id is not None:
                t = TIME.search(line)
                if t:
                    unit = t.group(4)
                    out[last_id] = float(t.group(3)) * SCALE[unit]
                    last_id = None
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/bench-run1.txt"
    data = parse(path)

    # (device/dtype/cell/orientation) -> {impl: ms}
    rows = OrderedDict()
    for bid, ms in data.items():
        parts = bid.split("/")
        if parts[0] not in ("openblas", "faer"):
            continue
        device, dtype, *rest = parts
        if len(rest) < 3:
            continue
        cell, orient, impl = rest[-3], rest[-2], rest[-1]
        key = (f"{device}/{dtype}", cell, orient)
        rows.setdefault(key, {})[impl] = ms

    lines = []
    lines.append(f"<!-- generated from {path} by results/make_tables.py -->\n")
    lines.append("| device/dtype | cell | orientation | rstsr (ms) | cblas gemv (ms) | ratio |")
    lines.append("|---|---|---|---|---|---|")
    for (dev, cell, orient), impls in rows.items():
        r = impls.get("rstsr")
        g = impls.get("gemv-anchor")
        if r is None or g is None:
            continue
        lines.append(f"| {dev} | {cell} | {orient} | {r:.4f} | {g:.4f} | {r / g:.2f}× |")
    text = "\n".join(lines) + "\n"
    print(text)
    with open("results/tables.md", "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
