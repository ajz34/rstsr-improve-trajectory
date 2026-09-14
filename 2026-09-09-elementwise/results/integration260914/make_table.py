#!/usr/bin/env python3
"""Parse paired criterion logs (refA vs cand) into a comparison table.

Uses only the criterion point estimate (median of the printed bracket) from
each `time:` line; ignores criterion's `change:` lines (they compare against
whatever ran before in target/criterion, which alternates refA/cand and is
meaningless here). The bench id is the bare line printed just above each
`time:` line (e.g. `add/add_contig_64x64-serial_f64/A`).

Output: markdown table, one row per benchmark id:
  refA medians (3 passes) | cand medians (3 passes) | ratio range
Ratio < 1 = candidate faster.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TIME_RE = re.compile(r"^\s+time:\s+\[([0-9.]+) ([nµm]?s) ([0-9.]+) ([nµm]?s) ([0-9.]+) ([nµm]?s)\]")
ID_RE = re.compile(r"^(\S+/\S+)\s*$")

UNIT_TO_US = {"ns": 1e-3, "µs": 1.0, "ms": 1e3, "s": 1e6}


def parse(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    current = None
    for line in path.read_text(errors="replace").splitlines():
        if m := ID_RE.match(line):
            current = m.group(1)
        elif (m := TIME_RE.match(line)) and current:
            out[current] = float(m.group(3)) * UNIT_TO_US[m.group(4)]
    return out


def fmt(us: float) -> str:
    if us >= 1000:
        return f"{us/1000:.2f} ms"
    if us >= 10:
        return f"{us:.0f} µs"
    return f"{us:.2f} µs"


def main(cfg: str) -> None:
    ref = [parse(HERE / f"refA_{cfg}_pass{i}.log") for i in (1, 2, 3)]
    cand = [parse(HERE / f"cand_{cfg}_pass{i}.log") for i in (1, 2, 3)]
    ids = sorted(set().union(*(r.keys() for r in ref)))
    rows, missing = [], []
    for bid in ids:
        if not all(bid in r for r in ref + cand):
            missing.append(bid)
            continue
        rv = [r[bid] for r in ref]
        cv = [c[bid] for c in cand]
        ratios = [c / r for c, r in zip(cv, rv)]
        rmin, rmax = min(ratios), max(ratios)
        flag = ""
        if rmin > 1.05:
            flag = " **REG**"
        elif rmax < 0.95:
            flag = " **WIN**"
        rows.append(
            f"| `{bid}` | {fmt(min(rv))} … {fmt(max(rv))} | {fmt(min(cv))} … {fmt(max(cv))} | "
            f"{rmin:.2f}–{rmax:.2f}× |{flag}"
        )
    print(f"## {cfg}: cand/refA per-bench ratio (3 paired passes, 1.00× = parity)\n")
    print("| benchmark | refA (master) | cand (patched) | ratio range | verdict |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    if missing:
        print(f"\n(missing in some passes: {missing})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "portable")
