#!/usr/bin/env python3
"""Regenerate results/tables.md from the raw criterion logs in results/*/."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def parse_txt(path):
    import re
    out, cur = {}, None
    with open(path) as f:
        for line in f:
            m = re.match(r"\s*(\S+)\s*$", line.rstrip())
            if m and ("/" in m.group(1)) and not line.strip().startswith(
                ("time","change","Performance","regress","improve","Warning","Found")):
                cur = m.group(1); continue
            m = re.match(r"\s*time:\s+\[([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\s+([\d.]+)\s+(\w+)\]", line)
            if m and cur:
                mult = {"ps":1e-12,"ns":1e-9,"µs":1e-6,"ms":1e-3,"s":1.0}[m.group(4)]
                out[cur] = float(m.group(3))*mult
                cur = None
    return out

D = {}
for tag in ["portable","native","portable_c","native_c","candidate_portable","candidate_native","candidate_portable_c","candidate_native_c"]:
    D[tag] = {}
    for bench in ["transpose","anchors_ndarray","kernels_probe"]:
        p = os.path.join(ROOT, tag, f"bench_{bench}.txt")
        if os.path.exists(p): D[tag].update(parse_txt(p))

def fmt(v):
    return "n/a" if v is None else (f"{v*1e6:.2f} µs" if v < 1e-4 else f"{v*1e3:.3f} ms")
def g(tag, key): return D[tag].get(key)

L = []
L.append("# tables.md — T1' phase-1 baseline (criterion medians from raw logs)\n")
L.append("Regenerate: `python3 results/make_tables.py`. Configs: `portable` (no RUSTFLAGS),")
L.append("`native` (`-C target-cpu=native`), `portable_c`/`native_c` = A-filter re-run under")
L.append("`MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`.\n")
L.append("GB/s figures use 2x tensor bytes (read + write). Rider = (A-B)/A.\n")

L.append("## transpose_copy f64 — B (reuse `c.assign`, primary judge) / A (allocating idiom)\n")
L.append("| case | serial nat B | faer16 nat B | serial por B | faer16 por B | serial nat A | serial por A | faer16 nat A | faer16 por A |")
L.append("|---|---|---|---|---|---|---|---|---|")
for label in ["small_64x64","medium_512x512","large_2048x2048","odd_1000x777","oddT_777x1000"]:
    row = [f"| {label}"]
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"transpose_copy/{label}-serial_f64/B")))
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"transpose_copy/{label}-faer16_f64/B")))
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"transpose_copy/{label}-serial_f64/A")))
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"transpose_copy/{label}-faer16_f64/A")))
    L.append("|".join(row) + "|")
L.append("")
L.append("### f32 secondary\n")
L.append("| case | serial nat B | serial nat A | faer16 nat B | faer16 nat A |")
L.append("|---|---|---|---|---|")
for label in ["large_2048x2048","odd_1000x777"]:
    L.append(f"| {label} | {fmt(g('native', f'transpose_copy/{label}-serial_f32/B'))} | {fmt(g('native', f'transpose_copy/{label}-serial_f32/A'))} | {fmt(g('native', f'transpose_copy/{label}-faer16_f32/B'))} | {fmt(g('native', f'transpose_copy/{label}-faer16_f32/A'))} |")
L.append("")
L.append("### Variant C (A + MALLOC tunables, A-filter re-run)\n")
L.append("| case | serial nat C | serial por C | faer16 nat C | faer16 por C |")
L.append("|---|---|---|---|---|")
for label in ["small_64x64","medium_512x512","large_2048x2048","odd_1000x777","oddT_777x1000"]:
    L.append(f"| {label} f64 | {fmt(g('native_c', f'transpose_copy/{label}-serial_f64/A'))} | {fmt(g('portable_c', f'transpose_copy/{label}-serial_f64/A'))} | {fmt(g('native_c', f'transpose_copy/{label}-faer16_f64/A'))} | {fmt(g('portable_c', f'transpose_copy/{label}-faer16_f64/A'))} |")
L.append("")
L.append("### Derived (large 2048x2048 f64)\n")
L.append("- Rider (A-B)/A: serial native 26%, serial portable 25%, faer16 native 56%, faer16 portable 57% — T7 reproduced.")
L.append("- B wall-clock GB/s (2x bytes): serial native 17.155 ms -> 3.9; faer16 native 1.450 ms -> 46.2.")
L.append("- C recovery: serial native A 23.159 -> C 17.015 (B = 17.155: full); faer16 A 3.325 -> C 1.564 (B = 1.450: 95%).")
L.append("- portable oddT B wobbled this run (2.642 vs 2.430 ms first portable run; native stable ~2.43): the")
L.append("  known +-5% portable build/run lottery (T4' precedent); all A/B directions unaffected.")
L.append("")
L.append("## assign_gates (phase-2 fall-through canaries)\n")
L.append("| case | serial nat | faer16 nat | serial por | faer16 por |")
L.append("|---|---|---|---|---|")
for key in ["contig_small_64x64","contig_large_2048x2048","sliced_t_large_2048x2048","sliced_large_2048x2048","sliced_t_odd_1000x777","sliced_odd_1000x777"]:
    row = [f"| {key}"]
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"assign_gates/{key}-serial-f64/assign")))
    for tag in ["native","portable"]: row.append(fmt(g(tag, f"assign_gates/{key}-faer16-f64/assign")))
    L.append("|".join(row) + "|")
L.append("")
L.append("## ndarray anchors (`a.t().to_owned()`, serial in-process)\n")
L.append("| case | portable | native |")
L.append("|---|---|---|")
for label in ["small_64x64","medium_512x512","large_2048x2048","odd_1000x777","oddT_777x1000"]:
    L.append(f"| {label} f64 | {fmt(g('portable', f'ndarray_transpose/{label}-serial_f64/t_to_owned'))} | {fmt(g('native', f'ndarray_transpose/{label}-serial_f64/t_to_owned'))} |")
L.append("")
L.append("Caveat (verified in examples/correctness.rs): ndarray `.t().to_owned()` PRESERVES the f-order")
L.append("memory layout (is_standard_layout()==false; content = the logical transpose), so its cost is")
L.append("that of an f-order materialization, not a c-order rewrite. Anchor kept for T0 comparability;")
L.append("treat as context (same status as numpy's anomalous .T.copy()).")
L.append("")
L.append("## kernels_probe (direct raw-kernel calls, f64, preallocated buffers)\n")
L.append("| probe | 2048x2048 nat | 2048x2048 por | 1000x777 nat | 1000x777 por |")
L.append("|---|---|---|---|---|")
for probe in ["P_generic_serial","P_generic_rayon16","P_blocked_c2r_serial","P_blocked_r2c_serial","P_blocked_c2r_rayon16","P_blocked_swap_serial","P_blocked_rawptr_serial"]:
    row = [f"| {probe}"]
    for (mm,nn) in [(2048,2048),(1000,777)]:
        for tag in ["native","portable"]:
            row.append(fmt(g(tag, f"kernels_probe/probe_{mm}x{nn}_f64/{probe}")))
    L.append("|".join(row) + "|")
L.append("")
L.append("Derived ratios (native, 2048x2048): blocked c2r serial / generic serial = 5.70/15.24 = **2.7x**;")
L.append("blocked rayon16 / generic rayon16 = 0.542/1.184 = **2.2x**; odd 1000x777: 0.296/0.446 = 1.5x serial,")
L.append("97.9/212.0 µs = 2.2x rayon16. The swapped-loop probe (contiguous reads, strided writes) is 3.1x")
L.append("SLOWER than the in-tree orientation (17.41 vs 5.70 ms) — the dormant kernel's orientation")
L.append("(strided reads, contiguous write-combined stores) is the right one. rawptr vs bounds-checked")
L.append("serial: ~0-5% — bounds checks are not the lever.\n")


L.append("")
L.append("## PHASE 2 (candidate): before/after on the patched tree\n")
L.append("### transpose_copy f64 — baseline -> candidate (speedup)\n")
L.append("| case | serial nat B | faer16 nat B | serial por B | faer16 por B |")
L.append("|---|---|---|---|---|")
for label in ["small_64x64","medium_512x512","large_2048x2048","odd_1000x777","oddT_777x1000"]:
    row = [f"| {label}"]
    for tag, ctag in [("native","candidate_native"),("portable","candidate_portable")]:
        for dev in ["serial","faer16"]:
            b = g(tag, f"transpose_copy/{label}-{dev}_f64/B")
            a = g(ctag, f"transpose_copy/{label}-{dev}_f64/B")
            if b and a:
                row.append(f"{fmt(b)} -> {fmt(a)} (**{b/a:.2f}x**)")
            else:
                row.append("n/a")
    L.append("|".join(row) + "|")
L.append("")
L.append("### transpose_copy f64 variant A (allocating idiom)\n")
L.append("| case | serial nat A | faer16 nat A | serial por A | faer16 por A |")
L.append("|---|---|---|---|---|")
for label in ["small_64x64","medium_512x512","large_2048x2048","odd_1000x777","oddT_777x1000"]:
    row = [f"| {label}"]
    for tag, ctag in [("native","candidate_native"),("portable","candidate_portable")]:
        for dev in ["serial","faer16"]:
            b = g(tag, f"transpose_copy/{label}-{dev}_f64/A")
            a = g(ctag, f"transpose_copy/{label}-{dev}_f64/A")
            if b and a:
                row.append(f"{fmt(b)} -> {fmt(a)} (**{b/a:.2f}x**)")
            else:
                row.append("n/a")
    L.append("|".join(row) + "|")
L.append("")
L.append("### f32 variant A/B\n")
L.append("| case | serial nat B | faer16 nat B | serial nat A | faer16 nat A |")
L.append("|---|---|---|---|---|")
for label in ["large_2048x2048","odd_1000x777"]:
    row = [f"| {label}"]
    for var in ["B","A"]:
        for dev in ["serial","faer16"]:
            b = g("native", f"transpose_copy/{label}-{dev}_f32/{var}")
            a = g("candidate_native", f"transpose_copy/{label}-{dev}_f32/{var}")
            row.append(f"{fmt(b)} -> {fmt(a)} (**{b/a:.2f}x**)" if b and a else "n/a")
    L.append("|".join(row) + "|")
L.append("")
L.append("### Variant C (A + MALLOC tunables)\n")
L.append("| case | serial nat C | faer16 nat C |")
L.append("|---|---|---|")
for label in ["large_2048x2048","odd_1000x777"]:
    b = g("native_c", f"transpose_copy/{label}-serial_f64/A")
    a = g("candidate_native_c", f"transpose_copy/{label}-serial_f64/A")
    bf = g("native_c", f"transpose_copy/{label}-faer16_f64/A")
    af = g("candidate_native_c", f"transpose_copy/{label}-faer16_f64/A")
    L.append(f"| {label} | {fmt(b)} -> {fmt(a)} (**{b/a:.2f}x**) | {fmt(bf)} -> {fmt(af)} (**{bf/af:.2f}x**) |")
L.append("")
L.append("### orderchange_extra (review rows: r2c orientation + bcast slow-axis-0)\n")
L.append("| case | variant | serial nat | faer16 nat |")
L.append("|---|---|---|---|")
for grp in ["to_fcontig","bcast_t"]:
    for label in ["large_2048x2048","odd_1000x777"]:
        for var in ["A","B"]:
            row = [f"| {grp} {label} | {var} |"]
            for dev in ["serial","faer16"]:
                b = g("native", f"orderchange_extra/{grp}_{label}-{dev}-f64/{var}")
                a = g("candidate_native", f"orderchange_extra/{grp}_{label}-{dev}-f64/{var}")
                row.append(f"{fmt(b)} -> {fmt(a)} (**{b/a:.2f}x**)" if b and a else "n/a")
            L.append("|".join(row) + " |")
L.append("")
L.append("### Gates (assign_gates: must stay within +-3%)\n")
L.append("| case | serial nat | faer16 nat | serial por | faer16 por |")
L.append("|---|---|---|---|---|")
for key in ["contig_small_64x64","contig_large_2048x2048","sliced_t_large_2048x2048","sliced_large_2048x2048","sliced_t_odd_1000x777","sliced_odd_1000x777"]:
    row = [f"| {key}"]
    for tag, ctag in [("native","candidate_native"),("portable","candidate_portable")]:
        for dev in ["serial","faer16"]:
            b = g(tag, f"assign_gates/{key}-{dev}-f64/assign")
            a = g(ctag, f"assign_gates/{key}-{dev}-f64/assign")
            if b and a:
                delta = (a / b - 1) * 100
                row.append(f"{fmt(b)} -> {fmt(a)} ({delta:+.1f}%)")
            else:
                row.append("n/a")
    L.append("|".join(row) + "|")
L.append("")
L.append("### ndarray anchors (unchanged tree-independent)\n")
L.append("| case | native baseline | native candidate |")
L.append("|---|---|---|")
for label in ["large_2048x2048","odd_1000x777"]:
    b = g("native", f"ndarray_transpose/{label}-serial_f64/t_to_owned")
    a = g("candidate_native", f"ndarray_transpose/{label}-serial_f64/t_to_owned")
    L.append(f"| {label} | {fmt(b)} | {fmt(a)} |")
L.append("")

with open(os.path.join(ROOT, "tables.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("tables.md regenerated")
