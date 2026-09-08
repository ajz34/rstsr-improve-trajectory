#!/usr/bin/env python3
"""Regenerate results/tables.md from raw criterion estimates.

Reads results/{portable,native}/criterion/<group>/<id>/<variant>/new/estimates.json
(mean -> median-of-means reported by criterion) and emits the A-vs-B tables.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def fmt_time(ns):
    # criterion estimates are stored in nanoseconds
    if ns is None:
        return "n/a"
    if ns >= 1e6:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.2f} µs"
    return f"{ns:.1f} ns"

def load(tag):
    base = os.path.join(ROOT, tag, "criterion")
    out = {}
    if not os.path.isdir(base):
        return out
    for group in sorted(os.listdir(base)):
        gdir = os.path.join(base, group)
        if not os.path.isdir(gdir) or group == "report":
            continue
        for bench_id in sorted(os.listdir(gdir)):
            bdir = os.path.join(gdir, bench_id)
            if not os.path.isdir(bdir) or bench_id == "report":
                continue
            for variant in sorted(os.listdir(bdir)):
                vdir = os.path.join(bdir, variant)
                est = os.path.join(vdir, "new", "estimates.json")
                if not os.path.isfile(est):
                    continue
                with open(est) as f:
                    data = json.load(f)
                out[(group, bench_id, variant)] = data["mean"]["point_estimate"]
    return out

def main():
    lines = ["# T7 tables (regenerated from criterion raw estimates)", ""]
    for tag in ("native", "portable"):
        if not os.path.isdir(os.path.join(ROOT, tag, "criterion")):
            continue
        data = load(tag)
        lines += [f"## {tag}", "",
                  "| group | case | A_alloc | B_reuse | B_bound | best B | rider (A-bestB)/A |",
                  "|---|---|---|---|---|---|---|"]
        groups = sorted(set(g for g, _, _ in data))
        for group in groups:
            ids = sorted(set(i for g, i, _ in data if g == group))
            for bid in ids:
                a = data.get((group, bid, "A_alloc"))
                variants = [v for (g, i, v) in data if g == group and i == bid and v != "A_alloc"]
                vals = {v: data[(group, bid, v)] for v in variants}
                a_str = fmt_time(a)
                if a is None:
                    continue
                b_strs = " / ".join(f"{v.replace('B_reuse_','').replace('B_','')}={fmt_time(t)}" for v, t in sorted(vals.items()))
                best = min(vals.values()) if vals else a
                rider = (a - best) / a * 100
                lines.append(f"| {group} | {bid} | {a_str} | {b_strs} | - | {fmt_time(best)} | {rider:.0f}% |")
        lines.append("")
    out = os.path.join(ROOT, "tables.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")
    print("\n".join(lines[:40]))

if __name__ == "__main__":
    sys.exit(main())
