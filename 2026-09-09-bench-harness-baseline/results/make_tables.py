#!/usr/bin/env python3
"""Regenerate the markdown baseline tables in README.md from results/.

Usage: python3 results/make_tables.py   (from the experiment dir)

Parses criterion human-readable output under results/{portable,native,native_d8}/,
derives GB/s / %-of-peak columns, and prints markdown tables. The README tables
were produced by this script.

Traffic accounting (stated policy, allocation included):
- transpose copy: 2*mn bytes (read + write)      - add: 3*mn (read 2 + write 1)
- reductions:     mn bytes (read)                - vecdot: 2*n (read both)
- triad:          3*n bytes                      - memcpy: 2*n
- fill (full):    mn written                     - argmax: n read

CAVEAT (L3 residency): the 9950X3D has 128 MiB total L3 (X3D CCD); the large
class (2048x2048 f64 = 32 MiB) and the 1e7-element 2-buffer vecdot set
(160 MB) are partially L3-resident across criterion iterations, so their
derived GB/s can EXCEED the DRAM streaming ceiling set by triad large (240 MB
working set, 3 buffers). Treat GB/s above triad as L3 assistance. The same
effect applies to rstsr/ndarray/numpy alike, so cross-op ratios stay fair.
"""

import re

BENCH_FILES = ["bench_triad.txt", "bench_transpose.txt", "bench_reduce.txt", "bench_vecdot.txt",
               "bench_elementwise.txt", "bench_fill_argmax.txt", "bench_anchors_ndarray.txt"]


def parse(tag):
    results = {}
    for fn in BENCH_FILES:
        try:
            txt = open(f"results/{tag}/{fn}").read()
        except FileNotFoundError:
            continue
        cur = None
        for ln in txt.splitlines():
            m = re.match(r"^(?:Testing|Benchmarking)\s+(\S+?)(?::|$)", ln)
            if m:
                cur = m.group(1)
                continue
            m = re.search(r"time:\s+\[([0-9.]+) ([µmn]?s) ([0-9.]+) ([µmn]?s) ([0-9.]+) ([µmn]?s)\]", ln)
            if m and cur:
                def to_s(v, u):
                    v = float(v)
                    return v if u == "s" else v * 1e-3 if u == "ms" else v * 1e-6 if u == "µs" else v * 1e-9
                results[cur] = to_s(m.group(3), m.group(4))
                cur = None
    return results


def fmt_t(s):
    if s is None:
        return "-"
    if s >= 1e-3:
        return f"{s*1e3:.2f} ms"
    if s >= 1e-6:
        return f"{s*1e6:.2f} µs"
    return f"{s*1e9:.0f} ns"


def gbs(t, nbytes):
    if t is None or t <= 0:
        return None
    return nbytes / t / 1e9


def cell(t, nbytes, with_gbs=True):
    if t is None:
        return "- | -" if with_gbs else "-"
    g = gbs(t, nbytes) if nbytes > 0 else None
    return f"{fmt_t(t)} | {g:6.1f}" if with_gbs and g else f"{fmt_t(t)} | -" if with_gbs else fmt_t(t)


def main():
    port = parse("portable")
    nat = parse("native")
    d8 = parse("native_d8")

    mn = 2048 * 2048
    n7 = 10_000_000

    # ---- ceiling table -----------------------------------------------------
    print("### Bandwidth ceiling (pure Rust, serial, 240 MB / 160 MB working sets)\n")
    print("| op | portable | GB/s | native | GB/s |")
    print("|---|---|---|---|---|")
    for name, k, traffic in [
        ("triad large (3x1e7 f64)", "triad/triad_f64/large", 3 * n7 * 8),
        ("memcpy large (2x1e7 f64)", "memcpy/copy_f64/large", 2 * n7 * 8),
    ]:
        print(f"| {name} | {cell(port.get(k), traffic)} | {cell(nat.get(k), traffic)} |")

    # ---- large-class rstsr table -------------------------------------------
    print("\n### Large class 2048x2048 f64 (32 MiB) — time and derived GB/s (alloc included)\n")
    print("| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = [
        ("transpose copy", "transpose_copy/large_2048x2048-{dev}_f64/to_contig", 2 * mn * 8),
        ("add contiguous", "elementwise_add/large_2048x2048-contig/{dev}_f64", 3 * mn * 8),
        ("add broadcast row", "elementwise_add/large_2048x2048-broadcast/{dev}_f64", 3 * mn * 8),
        ("add strided (b.t)", "elementwise_add/large_2048x2048-strided/{dev}_f64", 3 * mn * 8),
        ("sum axis0", "reduce/large_2048x2048-{dev}_f64/sum_axis0", mn * 8),
        ("sum axislast", "reduce/large_2048x2048-{dev}_f64/sum_axislast", mn * 8),
        ("sum all", "reduce/large_2048x2048-{dev}_f64/sum_all", mn * 8),
        ("fill: full", "fill/large_2048x2048-full/{dev}_f64", mn * 8),
        ("fill: zeros (calloc-lazy)", "fill/large_2048x2048-zeros/{dev}_f64", 0),
        ("argmax 1e7", "argmax/large_10000000-{dev}_f64/argmax", n7 * 8),
        ("vecdot 1e7", "vecdot/dot1d_large_10000000-{dev}_f64/vecdot", 2 * n7 * 8),
        ("vecdot batched 4096x512", "vecdot/batched_4096x512-{dev}_f64/vecdot", 2 * 4096 * 512 * 8),
        ("transpose copy f32", "transpose_copy/large_2048x2048-{dev}_f32/to_contig", 2 * mn * 4),
        ("add contiguous f32", "elementwise_add/large_2048x2048-contig/{dev}_f32", 3 * mn * 4),
        ("sum axis0 f32", "reduce/large_2048x2048-{dev}_f32/sum_axis0", mn * 4),
        ("vecdot 1e7 f32", "vecdot/dot1d_large_10000000-{dev}_f32/vecdot", 2 * n7 * 4),
    ]
    for name, tmpl, traffic in rows:
        cells = []
        for dev in ("serial", "faer16"):
            for tag in (port, nat):
                cells.append(cell(tag.get(tmpl.format(dev=dev)), traffic))
        print("| " + name + " | " + " | ".join(cells) + " |")

    # ---- medium / odd / small classes ---------------------------------------
    for label, idl in [("medium 512x512", "medium_512x512"), ("odd 1000x777", "odd_1000x777"), ("small 64x64", "small_64x64")]:
        mul = 8 if "64x64" not in idl or True else 8
        print(f"\n### Class {label} f64 — time and derived GB/s\n")
        print("| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        mn2 = 512 * 512 if idl == "medium_512x512" else 1000 * 777 if idl == "odd_1000x777" else 64 * 64
        rows2 = [
            ("transpose copy", f"transpose_copy/{idl}-{{dev}}_f64/to_contig", 2 * mn2 * 8),
            ("sum axis0", f"reduce/{idl}-{{dev}}_f64/sum_axis0", mn2 * 8),
            ("sum axislast", f"reduce/{idl}-{{dev}}_f64/sum_axislast", mn2 * 8),
            ("sum all", f"reduce/{idl}-{{dev}}_f64/sum_all", mn2 * 8),
            ("argmax", f"argmax/{'medium_1000000' if idl=='medium_512x512' else idl}-{{dev}}_f64/argmax", 0),
        ]
        if idl == "medium_512x512":
            rows2.insert(1, ("add contiguous", f"elementwise_add/{idl}-contig/{{dev}}_f64", 3 * mn2 * 8))
        for name, tmpl, traffic in rows2:
            cells = []
            for dev in ("serial", "faer16"):
                for tag in (port, nat):
                    cells.append(cell(tag.get(tmpl.format(dev=dev)), traffic))
            print("| " + name + " | " + " | ".join(cells) + " |")

    # ---- vecdot across sizes -------------------------------------------------
    print("\n### vecdot 1-D f64 across sizes — time and GB/s\n")
    print("| size | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s | ndarray dot |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for lbl, n_, idl in [("1e3", 1_000, "small_1000"), ("1e5", 100_000, "medium_100000"), ("1e7", 10_000_000, "large_10000000")]:
        cells = []
        for dev in ("serial", "faer16"):
            for tag in (port, nat):
                cells.append(cell(tag.get(f"vecdot/dot1d_{idl}-{{dev}}_f64/vecdot".replace("{dev}", dev)), 2 * n_ * 8))
        nd = nat.get(f"anchor_ndarray/dot1d_{idl}-f64/dot")
        cells.append(fmt_t(nd) if nd else "-")
        print(f"| {lbl} | " + " | ".join(cells) + " |")

    # ---- numpy anchors -------------------------------------------------------
    try:
        nplines = open("numpy_ref/results.txt").read().splitlines()
    except FileNotFoundError:
        nplines = []
    if nplines:
        print("\n### numpy anchors (numpy 2.5.1, conda torch env, single-threaded unless BLAS)\n")
        print("```")
        for ln in nplines:
            print(ln)
        print("```")

    # ---- vecdot %-of-peak ---------------------------------------------------
    print("\n### vecdot 1e7 f64 vs analytic peak (32 DP FLOP/cycle/core @ ~5.7 GHz => ~182 GFLOP/s/core)\n")
    print("| variant | portable t | GFLOP/s | %peak | native t | GFLOP/s | %peak |")
    print("|---|---|---|---|---|---|---|")
    for name, key in [
        ("rstsr serial", "vecdot/dot1d_large_10000000-serial_f64/vecdot"),
        ("rstsr faer16", "vecdot/dot1d_large_10000000-faer16_f64/vecdot"),
        ("ndarray dot (serial)", "anchor_ndarray/dot1d_large_10000000-f64/dot"),
        ("batched 4096x512 (per-2*4096*512*512 flop not comparable)", None),
    ]:
        if key is None:
            continue
        for tagname, tag in (("portable", port), ("native", nat)):
            pass
        p, n_ = port.get(key), nat.get(key)
        def gf(t):
            return f"{2 * n7 / t / 1e9:.1f}" if t else "-"
        def pct(t):
            return f"{100 * (2 * n7 / t) / 182.4e9:.1f}%" if t else "-"
        print(f"| {name} | {fmt_t(p)} | {gf(p)} | {pct(p)} | {fmt_t(n_)} | {gf(n_)} | {pct(n_)} |")

    # ---- D8 secondary column ------------------------------------------------
    if d8:
        print("\n### D8 secondary column: native + dispatch_dim_layout_iter (large|odd subset)\n")
        print("| bench | native | native+D8 | speedup |")
        print("|---|---|---|---|")
        for k in sorted(d8):
            n_ = nat.get(k)
            d = d8[k]
            r = f"{n_/d:.2f}x" if n_ else "-"
            ns = fmt_t(n_) if n_ else "(not run)"
            print(f"| `{k}` | {ns} | {fmt_t(d)} | {r} |")


if __name__ == "__main__":
    main()
