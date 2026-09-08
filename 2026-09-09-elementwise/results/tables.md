# T4' phase-1 baseline tables (rstsr 386948be, clean tree)

All numbers are criterion median point estimates from `results/<config>/*.log`
(raw criterion JSONs in `results/<config>/criterion/`). Variants:
**A** = idiomatic allocating op (alloc included); **B** = reuse via public
`op_mutc_refa_refb_func` into pre-allocated, pre-warmed output (kernel-only,
PRIMARY judge for large per T7); **C** = A re-run under
`MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`.

Environment: AMD Ryzen 9 9950X3D (Zen 5, AVX-512, 128 MiB X3D L3), rustc
1.97.1 nightly, stock release profile, criterion 2 s + 0.7 s warm-up,
RAYON_NUM_THREADS=16. f64 primary, f32 secondary spots.
"nominal GB/s" = 24 B/elem (2R+1W) for add/mul, 16 B/elem for scale.

## add contig (f64), both configs, both devices

| case | var | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|---|
| small 64x64 | A | 1.462 µs | 1.337 µs | 26.24 µs | 26.15 µs |
| small 64x64 | B | 1.382 µs | 1.289 µs | 26.08 µs | 26.62 µs |
| medium 512x512 | A | 49.54 µs | 41.80 µs | 61.32 µs | 61.95 µs |
| medium 512x512 | B | 49.08 µs | 40.76 µs | 61.39 µs | 61.62 µs |
| large 2048x2048 | A | 6.338 ms | 6.225 ms | 2.802 ms | 2.641 ms |
| large 2048x2048 | B | 2.138 ms (47.1 GB/s) | 2.103 ms (47.9 GB/s) | 565.7 µs | 562.2 µs (179 GB/s) |
| odd 1000x777 | A | 150.1 µs | 118.8 µs | 99.86 µs | 100.4 µs |
| odd 1000x777 | B | 124.6 µs | 119.1 µs | 99.70 µs | 100.5 µs |

## add broadcast / strided (large 2048x2048, f64)

| case | var | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|---|
| bcast ([1,n] row, zero-stride) | A | 6.190 ms | 5.634 ms | 2.402 ms | 2.271 ms |
| bcast | B | 1.599 ms | 1.485 ms (67.8 nominal) | 284.0 µs | 267.5 µs |
| strided (a + bt.t()) | A | 28.30 ms | 28.38 ms | 4.357 ms | 4.099 ms |
| strided | B | 22.84 ms (4.4 GB/s) | 23.20 ms (4.3 GB/s) | 2.154 ms | 2.136 ms (47.1 GB/s) |

## mul / scale spots (large 2048x2048, f64)

| case | var | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|---|
| mul contig | A | 6.260 ms | 6.242 ms | 2.880 ms | 2.667 ms |
| mul contig | B | 2.124 ms | 2.132 ms | 551.4 µs | 596.9 µs |
| scale contig | A | 5.320 ms | 5.138 ms | 2.227 ms | 2.407 ms |
| scale contig | B | 1.619 ms | 1.528 ms | 274.2 µs | 277.7 µs |

(scale B is the documented deviation: reuse expressed via a `[1,n]` row of
exactly 2.0 through the refa-refb driver.)

## f32 secondary spots (add contig)

| case | var | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|---|
| medium 512x512 | A | 23.91 µs | 22.93 µs | 55.27 µs | 55.18 µs |
| medium 512x512 | B | 23.72 µs | 23.10 µs | 55.56 µs | 56.64 µs |
| large 2048x2048 | A | 781.8 µs | 749.2 µs | 233.1 µs | 229.8 µs |
| large 2048x2048 | B | 781.7 µs | 740.0 µs | 236.7 µs | 233.8 µs |

(A≈B at f32 large: 16 MiB outputs sit under the 32 MiB glibc cap — no rider,
matching T7.)

## Variant C — A under MALLOC tunables (secondary column; T7 carry-forward)

| case (large 2048x2048 f64) | A plain | C (tunables) | B (reference) |
|---|---|---|---|
| add contig serial native | 6.225 ms | 2.144 ms | 2.103 ms |
| add contig faer16 native | 2.641 ms | 591.3 µs | 562.2 µs |
| add contig serial portable | 6.338 ms | 2.140 ms | 2.138 ms |
| bcast serial native | 5.634 ms | 1.546 ms | 1.485 ms |
| strided serial native | 28.38 ms | 23.00 ms | 23.20 ms |
| strided faer16 native | 4.099 ms | 2.116 ms | 2.136 ms |
| mul serial native | 6.242 ms | 2.108 ms | 2.132 ms |
| scale serial native | 5.138 ms | 1.342 ms | 1.528 ms |
| small/medium/odd | unchanged vs A | unchanged | — |

Reproduces T7 RQ2a: the env pair recovers the ~4 ms rider (add A 6.23 →
2.14 ms, 97% of the A→B gap). scale C (1.34 ms) edges B (1.53 ms) by ~12% in
both configs — the warm-heap reallocation appears to beat the fixed reuse
buffer slightly for the 1R+1W op; not load-bearing for any conclusion.

## ndarray anchors (f64)

| case | var | portable | native |
|---|---|---|---|
| medium 512x512 | owned (alloc incl.) | 47.56 µs | 39.67 µs |
| medium 512x512 | zip_prealloc (kernel) | 46.57 µs | 39.64 µs |
| large 2048x2048 | owned (alloc incl.) | 6.547 ms | 6.370 ms |
| large 2048x2048 | zip_prealloc (kernel) | 2.117 ms | 2.132 ms |

rstsr B ties ndarray zip_prealloc at large in both configs (2.10 vs 2.13 ms
native); at medium native 40.8 vs 39.6 µs (+3%).

## Kernel-structure probes (raw slices, serial, kernel-only)

Contig (c = a + b; µs medium 262144 / ms large 4194304):

| probe | medium portable | medium native | large portable | large native |
|---|---|---|---|---|
| idx_generic_mu_closure (rstsr branch shape) | 48.61 | 40.00 | 2.117 | 2.146 |
| idx_scalar_bounds | 48.39 | 39.31 | 2.111 | 2.162 |
| idx_unchecked | 48.58 | 40.19 | 2.129 | 2.002 |
| zip_direct | 47.40 | 40.28 | 2.080 | 2.121 |
| zip_generic_mu_closure | 48.55 | 40.89 | 2.134 | 2.120 |
| zip_dyn_mu_closure | 48.51 | 41.10 | 2.114 | 2.119 |
| chunks_exact8 | 48.14 | 39.86 | 1.977 | 2.134 |

All shapes tie within noise at both configs; medium native ≈ 40 µs for every
shape (1.21x vs portable — the loop is vectorized; the difference is ISA, not
structure). The rstsr contig-branch shape (index loop + generic MaybeUninit
closure + bounds checks) compiles to the same speed as a hand-zipped loop.

Strided (c[i,j] = a[i,j] + b[j,i], 2048x2048, native; portable matches):

| probe | ms | vs rstsr B (23.2) |
|---|---|---|
| bound_rowmajor (T7 shape: c,a inner-contiguous, b hopping) | 16.62 | 1.40x |
| colmajor_walk (c,a hopping, b contiguous) | 30.79 | 0.75x (worse) |
| blocked64 | 6.81 | **3.4x** |
| blocked128 | 7.31 | 3.2x |

The rstsr strided branch runs the bound_rowmajor memory pattern (greedy
permutation puts c,a contiguous, b on the hopping axis) but pays ~6.2 ms of
layout-iterator machinery above the raw bound (16.6 -> 23.2). A 64x64 tile
kernel removes both the hop pattern and the machinery: 6.8 ms.

---

# T4' phase-2 candidate tables (blocked 2-D strided kernel; patch applied)

Baseline = `results/pre2_*/` (CLEAN tree, extended matrix, same crate build);
Candidate = `results/candidate_*/` (patch applied). Values are criterion
medians in µs (µs everywhere; 2048² = 4.19M elem). "B" = reuse variant
(primary judge for large).

## Primary — strided large 2048×2048 (the deep case)

| case | pre2 | candidate | speedup |
|---|---|---|---|
| strided B serial native | 23098 | 7507 | **3.08×** |
| strided B serial portable | 22955 | 7618 | **3.01×** |
| strided B faer16 native | 2137 | 957 | **2.23×** |
| strided B faer16 portable | 2174 | 1050 | **2.07×** |
| stridedfirst B serial native | 23005 | 7524 | 3.06× |
| stridedfirst B serial portable | 22748 | 7606 | 2.99× |
| stridedfirst B faer16 native | 2171 | 1128 | 1.93× |
| addasgn (c += bᵀ) B serial native | 17281 | 6173 | 2.80× |
| addasgn B faer16 native | 1488 | 696 | 2.14× |
| strided A serial native | 28832 | 12712 | 2.27× |
| strided A serial portable | 28041 | 12453 | 2.25× |
| strided odd 1000×777 B serial native | 3689 | 426 | **8.66×** |
| strided small 64×64 B serial native | 20.7 | 2.9 | **7.1×** |
| strided small B faer16 native | 89.5 | 5.7 | **15.7×** |
| strided odd B faer16 native | 503 | 66.6 | 7.6× |

D3 (≥10% on large reuse-variant, BOTH configs): **PASS** — 3.0× serial and
2.1-2.2× faer16 in both configs; the strided target "beat the 17.2 ms T7
bound" is met (7.5 ms; blocking removes the hop pattern the bound still
pays).

## Gates (untouched code paths; ±3% contract)

Serial rows — all within the contract:
contig small B 1.34→1.30 µs; contig medium B 42.00→42.13 µs (+0.3%);
contig odd B 115.97→119.07 µs (−2.7%); bcast B 1486→1508 µs (−1.4%);
contig large B 2162.6→2144.3 µs (+0.8%); mul B 2174→2154 µs (+1.0%);
scale B 1513→1498 µs (+1.0%) — all native. Portable: contig small/medium/
large and bcast within ±2.3%; contig odd B 134.7→148.3 µs (−10%) is a
**build-layout lottery cell**, not a code effect (see below).
faer16 sub-ms rows show ±4-6% run-to-run spread; medians across 4 runs
(bcast B 268.8/284.4/277.3; contig-large B 558.8/580.4/571.8/577.2) sit
within ±3% of the pre2 medians; the parallel alloc-A rows swing ±6% (malloc
rider scheduling; judged per T7 on B).
f32 large contig B (secondary spot, portable only): 782.5→806-828 µs
(+3-6%, consistent across 3 runs; native improved −5.9%) — same
monomorphization-layout effect, flagged for G2 with the samples below.

**Layout-lottery evidence (portable contig odd B):** the CLEAN tree itself,
across independent rebuilds, produces 124.6 / 131.8 / 127.8 / 156.5 µs
while A anti-correlates (A+B sum invariant ≈ 278 µs). Patched samples:
147.0 / 152.8 / 154.2 — inside the clean tree's own build-to-build band.
The contig code path is byte-identical by construction (diff adds branches
only), and the native config shows 115.8-119.1 µs throughout.

## perf stat -d, add strided reuse-B, serial native (before → after)

(derived: results/perf/perf_add_strided_b_serial.txt vs
results/perf/perf_add_strided_b_serial_after.txt)

| metric | before | after |
|---|---|---|
| ms/iter (task-clock) | 24.2 | 7.67 |
| GB/s | 4.2 | 13.1 |
| ins/elem | 161.6 | **19.0** |
| cyc/elem | 31.2 | 10.4 |
| IPC | 5.17 | 1.84 |
| L1d miss rate | 2.1% | 23.2% |

The instruction mountain collapsed 8.5×; the kernel is now streaming-bound
(rising L1d miss rate = actually streaming), with the remaining 19 ins/elem
= tile-loop arithmetic + safe indexing + closure call.

## Variant C (MALLOC tunables) on the candidate, native

add strided A: 28.4 ms → **7.06 ms** under
`MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`
(≈ B 7.5 ms) — the rider fix and the kernel fix compose for allocating
users: 28.4 → 7.1 ms = **4.0×** idiomatic.
