# tables.md — T1' phase-1 baseline (criterion medians from raw logs)

Regenerate: `python3 results/make_tables.py`. Configs: `portable` (no RUSTFLAGS),
`native` (`-C target-cpu=native`), `portable_c`/`native_c` = A-filter re-run under
`MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`.

GB/s figures use 2x tensor bytes (read + write). Rider = (A-B)/A.

## transpose_copy f64 — B (reuse `c.assign`, primary judge) / A (allocating idiom)

| case | serial nat B | faer16 nat B | serial por B | faer16 por B | serial nat A | serial por A | faer16 nat A | faer16 por A |
|---|---|---|---|---|---|---|---|---|
| small_64x64|14.09 µs|14.09 µs|14.04 µs|14.18 µs|13.60 µs|13.64 µs|13.58 µs|13.54 µs|
| medium_512x512|0.816 ms|0.817 ms|0.204 ms|0.214 ms|0.813 ms|0.814 ms|0.203 ms|0.211 ms|
| large_2048x2048|17.155 ms|17.546 ms|1.450 ms|1.504 ms|23.159 ms|23.613 ms|3.325 ms|3.461 ms|
| odd_1000x777|2.425 ms|2.447 ms|0.360 ms|0.367 ms|2.422 ms|2.448 ms|0.361 ms|0.367 ms|
| oddT_777x1000|2.429 ms|2.642 ms|0.361 ms|0.370 ms|2.440 ms|2.500 ms|0.360 ms|0.367 ms|

### f32 secondary

| case | serial nat B | serial nat A | faer16 nat B | faer16 nat A |
|---|---|---|---|---|
| large_2048x2048 | 15.543 ms | 15.532 ms | 1.302 ms | 1.304 ms |
| odd_1000x777 | 2.412 ms | 2.406 ms | 0.358 ms | 0.356 ms |

### Variant C (A + MALLOC tunables, A-filter re-run)

| case | serial nat C | serial por C | faer16 nat C | faer16 por C |
|---|---|---|---|---|
| small_64x64 f64 | 13.65 µs | 13.60 µs | 13.61 µs | 13.71 µs |
| medium_512x512 f64 | 0.817 ms | 0.817 ms | 0.204 ms | 0.208 ms |
| large_2048x2048 f64 | 17.066 ms | 17.126 ms | 1.490 ms | 1.546 ms |
| odd_1000x777 f64 | 2.428 ms | 2.439 ms | 0.360 ms | 0.371 ms |
| oddT_777x1000 f64 | 2.445 ms | 2.430 ms | 0.363 ms | 0.367 ms |

### Derived (large 2048x2048 f64)

- Rider (A-B)/A: serial native 26%, serial portable 25%, faer16 native 56%, faer16 portable 57% — T7 reproduced.
- B wall-clock GB/s (2x bytes): serial native 17.155 ms -> 3.9; faer16 native 1.450 ms -> 46.2.
- C recovery: serial native A 23.159 -> C 17.015 (B = 17.155: full); faer16 A 3.325 -> C 1.564 (B = 1.450: 95%).
- portable oddT B wobbled this run (2.642 vs 2.430 ms first portable run; native stable ~2.43): the
  known +-5% portable build/run lottery (T4' precedent); all A/B directions unaffected.

## assign_gates (phase-2 fall-through canaries)

| case | serial nat | faer16 nat | serial por | faer16 por |
|---|---|---|---|---|
| contig_small_64x64|1.04 µs|1.06 µs|1.06 µs|1.08 µs|
| contig_large_2048x2048|1.368 ms|1.389 ms|1.352 ms|1.379 ms|
| sliced_t_large_2048x2048|8.753 ms|8.739 ms|0.916 ms|0.840 ms|
| sliced_large_2048x2048|6.403 ms|6.419 ms|0.709 ms|0.717 ms|
| sliced_t_odd_1000x777|1.214 ms|1.213 ms|0.248 ms|0.250 ms|
| sliced_odd_1000x777|1.182 ms|1.187 ms|0.243 ms|0.247 ms|

## ndarray anchors (`a.t().to_owned()`, serial in-process)

| case | portable | native |
|---|---|---|
| small_64x64 f64 | 0.27 µs | 0.27 µs |
| medium_512x512 f64 | 29.33 µs | 29.66 µs |
| large_2048x2048 f64 | 5.484 ms | 6.138 ms |
| odd_1000x777 f64 | 72.58 µs | 72.89 µs |
| oddT_777x1000 f64 | 73.79 µs | 75.61 µs |

Caveat (verified in examples/correctness.rs): ndarray `.t().to_owned()` PRESERVES the f-order
memory layout (is_standard_layout()==false; content = the logical transpose), so its cost is
that of an f-order materialization, not a c-order rewrite. Anchor kept for T0 comparability;
treat as context (same status as numpy's anomalous .T.copy()).

## kernels_probe (direct raw-kernel calls, f64, preallocated buffers)

| probe | 2048x2048 nat | 2048x2048 por | 1000x777 nat | 1000x777 por |
|---|---|---|---|---|
| P_generic_serial|15.700 ms|15.239 ms|0.446 ms|0.404 ms|
| P_generic_rayon16|1.185 ms|1.110 ms|0.213 ms|0.192 ms|
| P_blocked_c2r_serial|5.810 ms|5.851 ms|0.292 ms|0.248 ms|
| P_blocked_r2c_serial|5.670 ms|5.980 ms|0.290 ms|0.247 ms|
| P_blocked_c2r_rayon16|0.545 ms|0.555 ms|99.08 µs|99.75 µs|
| P_blocked_swap_serial|17.404 ms|17.563 ms|0.302 ms|0.259 ms|
| P_blocked_rawptr_serial|5.625 ms|5.874 ms|0.126 ms|0.131 ms|

Derived ratios (native, 2048x2048): blocked c2r serial / generic serial = 5.70/15.24 = **2.7x**;
blocked rayon16 / generic rayon16 = 0.542/1.184 = **2.2x**; odd 1000x777: 0.296/0.446 = 1.5x serial,
97.9/212.0 µs = 2.2x rayon16. The swapped-loop probe (contiguous reads, strided writes) is 3.1x
SLOWER than the in-tree orientation (17.41 vs 5.70 ms) — the dormant kernel's orientation
(strided reads, contiguous write-combined stores) is the right one. rawptr vs bounds-checked
serial: ~0-5% — bounds checks are not the lever.


## PHASE 2 (candidate): before/after on the patched tree

### transpose_copy f64 — baseline -> candidate (speedup)

| case | serial nat B | faer16 nat B | serial por B | faer16 por B |
|---|---|---|---|---|
| small_64x64|14.09 µs -> 2.18 µs (**6.45x**)|14.04 µs -> 2.26 µs (**6.20x**)|14.09 µs -> 2.19 µs (**6.42x**)|14.18 µs -> 2.22 µs (**6.40x**)|
| medium_512x512|0.816 ms -> 0.315 ms (**2.59x**)|0.204 ms -> 57.72 µs (**3.54x**)|0.817 ms -> 0.313 ms (**2.61x**)|0.214 ms -> 56.83 µs (**3.77x**)|
| large_2048x2048|17.155 ms -> 5.989 ms (**2.86x**)|1.450 ms -> 0.547 ms (**2.65x**)|17.546 ms -> 5.958 ms (**2.94x**)|1.504 ms -> 0.562 ms (**2.68x**)|
| odd_1000x777|2.425 ms -> 0.251 ms (**9.67x**)|0.360 ms -> 0.106 ms (**3.41x**)|2.447 ms -> 0.248 ms (**9.88x**)|0.367 ms -> 0.101 ms (**3.64x**)|
| oddT_777x1000|2.429 ms -> 0.301 ms (**8.07x**)|0.361 ms -> 97.37 µs (**3.71x**)|2.642 ms -> 0.299 ms (**8.83x**)|0.370 ms -> 96.97 µs (**3.81x**)|

### transpose_copy f64 variant A (allocating idiom)

| case | serial nat A | faer16 nat A | serial por A | faer16 por A |
|---|---|---|---|---|
| small_64x64|13.60 µs -> 1.56 µs (**8.71x**)|13.58 µs -> 1.67 µs (**8.12x**)|13.64 µs -> 1.60 µs (**8.54x**)|13.54 µs -> 1.64 µs (**8.26x**)|
| medium_512x512|0.813 ms -> 0.312 ms (**2.61x**)|0.203 ms -> 50.71 µs (**4.01x**)|0.814 ms -> 0.311 ms (**2.61x**)|0.211 ms -> 49.50 µs (**4.26x**)|
| large_2048x2048|23.159 ms -> 10.350 ms (**2.24x**)|3.325 ms -> 3.179 ms (**1.05x**)|23.613 ms -> 10.735 ms (**2.20x**)|3.461 ms -> 3.147 ms (**1.10x**)|
| odd_1000x777|2.422 ms -> 0.249 ms (**9.74x**)|0.361 ms -> 94.96 µs (**3.80x**)|2.448 ms -> 0.249 ms (**9.83x**)|0.367 ms -> 96.07 µs (**3.82x**)|
| oddT_777x1000|2.440 ms -> 0.296 ms (**8.25x**)|0.360 ms -> 93.75 µs (**3.84x**)|2.500 ms -> 0.306 ms (**8.18x**)|0.367 ms -> 98.53 µs (**3.73x**)|

### f32 variant A/B

| case | serial nat B | faer16 nat B | serial nat A | faer16 nat A |
|---|---|---|---|---|
| large_2048x2048|15.543 ms -> 5.459 ms (**2.85x**)|1.302 ms -> 0.414 ms (**3.14x**)|15.532 ms -> 5.399 ms (**2.88x**)|1.304 ms -> 0.407 ms (**3.20x**)|
| odd_1000x777|2.412 ms -> 0.284 ms (**8.49x**)|0.358 ms -> 63.99 µs (**5.59x**)|2.406 ms -> 0.284 ms (**8.48x**)|0.356 ms -> 60.59 µs (**5.87x**)|

### Variant C (A + MALLOC tunables)

| case | serial nat C | faer16 nat C |
|---|---|---|
| large_2048x2048 | 17.066 ms -> 5.846 ms (**2.92x**) | 1.490 ms -> 0.542 ms (**2.75x**) |
| odd_1000x777 | 2.428 ms -> 0.254 ms (**9.57x**) | 0.360 ms -> 91.47 µs (**3.94x**) |

### orderchange_extra (review rows: r2c orientation + bcast slow-axis-0)

| case | variant | serial nat | faer16 nat |
|---|---|---|---|
| to_fcontig large_2048x2048 | A ||21.071 ms -> 10.634 ms (**1.98x**)|10.717 ms -> 3.057 ms (**3.51x**) |
| to_fcontig large_2048x2048 | B ||17.153 ms -> 5.862 ms (**2.93x**)|1.483 ms -> 0.554 ms (**2.68x**) |
| to_fcontig odd_1000x777 | A ||2.483 ms -> 0.261 ms (**9.51x**)|0.378 ms -> 92.08 µs (**4.11x**) |
| to_fcontig odd_1000x777 | B ||2.429 ms -> 0.256 ms (**9.47x**)|0.358 ms -> 92.37 µs (**3.88x**) |
| bcast_t large_2048x2048 | A ||16.813 ms -> 5.440 ms (**3.09x**)|2.704 ms -> 2.464 ms (**1.10x**) |
| bcast_t large_2048x2048 | B ||12.716 ms -> 1.506 ms (**8.44x**)|1.185 ms -> 0.322 ms (**3.67x**) |
| bcast_t odd_1000x777 | A ||2.347 ms -> 0.220 ms (**10.65x**)|0.353 ms -> 88.00 µs (**4.01x**) |
| bcast_t odd_1000x777 | B ||2.345 ms -> 0.217 ms (**10.79x**)|0.355 ms -> 98.66 µs (**3.60x**) |

### Gates (assign_gates: must stay within +-3%)

| case | serial nat | faer16 nat | serial por | faer16 por |
|---|---|---|---|---|
| contig_small_64x64|1.04 µs -> 1.04 µs (+0.4%)|1.06 µs -> 1.05 µs (-1.6%)|1.06 µs -> 1.06 µs (-0.1%)|1.08 µs -> 1.09 µs (+0.2%)|
| contig_large_2048x2048|1.368 ms -> 1.365 ms (-0.2%)|1.352 ms -> 1.342 ms (-0.7%)|1.389 ms -> 1.391 ms (+0.1%)|1.379 ms -> 1.456 ms (+5.5%)|
| sliced_t_large_2048x2048|8.753 ms -> 8.721 ms (-0.4%)|0.916 ms -> 0.972 ms (+6.1%)|8.739 ms -> 8.708 ms (-0.4%)|0.840 ms -> 0.838 ms (-0.2%)|
| sliced_large_2048x2048|6.403 ms -> 6.421 ms (+0.3%)|0.709 ms -> 0.705 ms (-0.5%)|6.419 ms -> 6.418 ms (-0.0%)|0.717 ms -> 0.709 ms (-1.0%)|
| sliced_t_odd_1000x777|1.214 ms -> 1.213 ms (-0.1%)|0.248 ms -> 0.248 ms (+0.3%)|1.213 ms -> 1.209 ms (-0.3%)|0.250 ms -> 0.251 ms (+0.1%)|
| sliced_odd_1000x777|1.182 ms -> 1.182 ms (-0.0%)|0.243 ms -> 0.246 ms (+1.2%)|1.187 ms -> 1.190 ms (+0.2%)|0.247 ms -> 0.249 ms (+0.8%)|

### ndarray anchors (unchanged tree-independent)

| case | native baseline | native candidate |
|---|---|---|
| large_2048x2048 | 6.138 ms | 5.292 ms |
| odd_1000x777 | 72.89 µs | 72.92 µs |

