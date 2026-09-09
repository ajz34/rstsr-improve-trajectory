#!/usr/bin/env python3
"""Derive a per-op summary from results/perf/raw.txt (perf stat -d output).

Prints ms/iter (from the profile_ops banner), task-clock GB/s, ins/elem,
cyc/elem, IPC, L1d-miss%, page-faults/iter. 2048x2048 f64 = 4,194,304
elements; iteration counts are the profile_ops ITERS constants.
"""

import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) if (os := __import__("os")) else None
ITERS = {"sum_axis0": 600, "sum_axis1": 1500, "sum_all": 1500, "min_axis0": 600, "min_all_1e7": 300}
ELEMS_BY_OP = {"min_all_1e7": 10_000_000}
ELEMS = 2048 * 2048


def parse(path):
    blocks = open(path).read().split("== ")
    out = []
    for b in blocks:
        m = re.match(r"(portable|native) (\w+) ==", b)
        if not m:
            continue
        cfg, op = m.group(1), m.group(2)
        iters = ITERS.get(op)
        if iters is None:
            # accept custom counts echoed by the banner
            mm = re.search(r"iters=(\d+)", b)
            iters = int(mm.group(1)) if mm else 1

        def num(pat):
            mm = re.search(pat, b)
            return float(mm.group(1).replace(",", "")) if mm else float("nan")

        task_clock_ms = num(r"([\d,\.]+) msec task-clock")
        ins = num(r"([\d,\.]+)      instructions")
        cyc = num(r"([\d,\.]+)      cpu-cycles")
        l1d_pct = num(r"#\s*([\d\.]+) %\s*l1d_miss_rate")
        faults = num(r"([\d,\.]+)      page-faults")
        banner_ms = re.search(r"elapsed=([\d\.]+) s", b)
        ms_iter = float(banner_ms.group(1)) * 1000 / iters if banner_ms else task_clock_ms / iters
        elems = ELEMS_BY_OP.get(op, ELEMS)
        out.append({
            "cfg": cfg, "op": op, "ms_iter": ms_iter,
            "gbs": elems * 8 / (task_clock_ms / iters) / 1e6 if task_clock_ms else 0,
            "ins_elem": ins / iters / elems,
            "cyc_elem": cyc / iters / elems,
            "ipc": ins / cyc if cyc else 0,
            "l1d_pct": l1d_pct,
            "faults_iter": faults / iters,
        })
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/raw.txt"
    rows = parse(path)
    print("| op | config | ms/iter | GB/s | ins/elem | cyc/elem | IPC | L1d-miss% | faults/iter |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['op']} | {r['cfg']} | {r['ms_iter']:.3f} | {r['gbs']:.1f} | "
              f"{r['ins_elem']:.2f} | {r['cyc_elem']:.2f} | {r['ipc']:.2f} | "
              f"{r['l1d_pct']:.1f} | {r['faults_iter']:.1f} |")


if __name__ == "__main__":
    main()
