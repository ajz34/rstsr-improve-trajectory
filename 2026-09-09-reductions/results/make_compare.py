#!/usr/bin/env python3
"""Emit the phase-2 before/after table from the candidate criterion logs.

The final phase-2 build was benched TWICE per RUSTFLAGS config
(bench_reduce_{portable,native}_candidate_r{1,2}.log, each against the saved
phase-1 baseline via `--baseline <cfg>`); this script parses the per-cell
`time:`/`change:` lines, averages the change mid-estimates across the two
runs, and writes results/tables_candidate.md.

criterion "change" compares the run against the SAVED BASELINE of that
config (phase-1 numbers), so negative = faster than phase 1.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

UNIT = {"ns": 1e-6, "µs": 1e-3, "ms": 1.0, "s": 1e3}


def parse_log(path):
    cells = {}
    if not os.path.isfile(path):
        return cells
    txt = open(path).read()
    pat = re.compile(r"(reduce[^\s]+)\n\s+time:\s+\[([^\]]+)\]\n\s+change:\s+\[([^\]]+)\]")
    for name, t, ch in pat.findall(txt):
        m = re.search(r"[\d.]+ \w+ ([\d.]+) (\w+)", t)
        c = re.search(r"([-+0-9.]+)% ([-+0-9.]+)% ([-+0-9.]+)%", ch)
        if not (m and c):
            continue
        ms = float(m.group(1)) * UNIT.get(m.group(2), 1.0)
        cells[name] = (ms, float(c.group(1)), float(c.group(2)), float(c.group(3)))
    return cells


def main():
    lines = [
        "# T2' phase-2 candidate vs phase-1 baseline",
        "",
        "criterion `change`% vs the saved phase-1 `portable`/`native` baselines;",
        "**candidate = mean of two full-suite runs** (r1/r2 columns show each run's",
        "mid estimate; the build-to-build layout lottery on ~0.5 ms streaming cells",
        "is ±5% per the T4' precedent, so single-run cells in the ±2-5% band are",
        "not conclusive). Negative = faster than phase 1.",
        "",
    ]
    for cfg in ["portable", "native"]:
        r1 = parse_log(os.path.join(HERE, f"bench_reduce_{cfg}_candidate_r1.log"))
        r2 = parse_log(os.path.join(HERE, f"bench_reduce_{cfg}_candidate_r2.log"))
        lines.append(f"\n## config: {cfg}\n")
        lines.append("| case | baseline ms | cand r1 | cand r2 | change r1 | change r2 | mean change |")
        lines.append("|---|---|---|---|---|---|---|")
        for name in sorted(set(r1) | set(r2)):
            e1, e2 = r1.get(name), r2.get(name)
            if not (e1 and e2):
                continue
            base = e1[0] / (1 + e1[2] / 100.0)  # back out baseline from r1
            mean_ch = (e1[2] + e2[2]) / 2.0
            flag = " <-- IMPROVED" if mean_ch <= -10.0 else (" <-- REGRESSION?" if mean_ch >= 2.5 else "")
            lines.append(
                f"| {name} | {base:.4f} | {e1[0]:.4f} | {e2[0]:.4f} | {e1[2]:+.2f}% | {e2[2]:+.2f}% | {mean_ch:+.2f}%{flag} |"
            )
    out = os.path.join(HERE, "tables_candidate.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
