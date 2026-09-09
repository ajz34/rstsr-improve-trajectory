# T5 fill/creation baseline tables (clean 386948be)

Criterion medians (100 samples, 2 s + 0.7 s warm-up; raw logs
`portable/bench_fill.txt`, `native/bench_fill.txt`). f64 write bytes =
8·m·n; f32 = 4·m·n. "GB/s(w)" = write-only nominal bandwidth from the
native median (L3-assisted where the buffer fits 128 MiB X3D — both 16 MiB
and 32 MiB buffers fit, so these are NOT DRAM numbers; comparisons are
between cells of the same table).

## Creation, alloc-inclusive (A) — `rt::full` / `rt::zeros` / `rt::ones`

| bench (µs) | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|
| full large 2048² | 4673.6 (7.2 GB/s) | 4654.0 (7.2) | 4376.2 | 4437.4 |
| ones large 2048² | 4627.0 (7.3) | 4445.3 (7.5) | 4586.9 | 4597.7 |
| zeros large 2048² | **0.00226 ms** | 0.00242 ms | 0.00233 | 0.00245 |
| full large 2048² f32 | 128.4 (131 GB/s) | 104.3 (161) | 110.0 | 112.9 |
| full medium 512² | 14.16 | 13.38 | 14.07 | 13.59 |
| ones medium 512² | 14.27 | 13.35 | 14.32 | 13.87 |
| zeros medium 512² | 13.53 | 13.30 | 13.41 | 13.46 |
| full odd 1000×777 | 44.51 | 45.36 | 46.32 | 43.13 |
| ones odd 1000×777 | 44.56 | 43.13 | 45.02 | 43.64 |
| zeros odd 1000×777 | **107.45** | **106.92** | 107.09 | 108.27 |
| full small 64² | 0.23 | 0.17 | 0.24 | 0.18 |
| ones small 64² | 0.23 | 0.17 | 0.25 | 0.18 |
| zeros small 64² | 0.14 | 0.14 | 0.14 | 0.15 |

Reading:

- `rt::full`/`ones` large = 4.4–4.7 ms — the T7 fault rider dominates
  (perf: 8193 faults/iter, 90 % sys). faer16 ≈ serial: faer creation
  delegates to `DeviceCpuSerial::full_impl`
  (`feature_rayon/auto_impl/creation.rs:19`) — one allocating thread, no
  pool involvement.
- `rt::zeros` laziness is **size-regime dependent** (new finding, refines
  T0/T7's "calloc-lazy, 2.3 µs"):
  - large 32 MiB (≥ glibc mmap cap): lazy zero pages, 2.3 µs, ~1 fault/iter
    — T0/T7 reproduced;
  - medium 2 MiB: calloc = alloc + explicit memset of a recycled chunk
    (perf: 8935 ins/iter memset-class, 1.2 faults/iter) at ~144–157 GB/s —
    ties `full` (13.3 vs 13.4 µs);
  - odd 6.2 MiB: calloc memset switches to the non-temporal regime
    (19.4k ins/iter, 3.2 faults/iter, ~58 GB/s = DRAM write speed) and is
    **2.4× slower than `full`** (106.9 vs 45.4 µs — `vec![v; n]`'s broadcast
    stays cache-resident). Absolute cost is tiny; "leave zeros alone"
    still stands, but "zeros is calloc-lazy" is only true in the ≥cap regime.

## Reuse fill (B, kernel-only) — `c.fill(v)` into a pre-warmed tensor

This is the ONLY tensor-level route into `fill_promote_cpu_serial/_rayon`
(an owned tensor's layout is always contiguous; strided layouts only reach
the kernel via the device-level API — `eye`'s diagonal, BLAS beta-zeroing).

| bench (µs) | serial portable | serial native (GB/s w) | faer16 portable | faer16 native |
|---|---|---|---|---|
| fill large 2048² | 616.5 (54) | **507.4 (66)** | 607.9 | 529.8 |
| fill large 2048² f32 | 108.5 (156) | 107.2 (157) | — | — |
| fill medium 512² | 14.50 | 13.80 (152) | 17.46 | 16.36 |
| fill odd 1000×777 | 44.70 | 40.58 (153) | 47.79 | 48.63 |
| fill small 64² | 0.39 | 0.33 | 0.40 | 0.34 |

## Raw write-only bound — `v.iter_mut().for_each(|x| *x = v)` on a pre-warmed `Vec`

| bench (µs) | serial portable | serial native (GB/s w) |
|---|---|---|
| slice large 2048² | 611.0 | 543.9 (62) |
| slice medium 512² | 13.94 | 13.52 (155) |
| slice odd 1000×777 | 46.30 | 42.80 (145) |
| slice small 64² | 0.18 | 0.12 |

**Kernel verdict (D3 denominators, T7 rule):** the reuse fill kernel ties or
beats the raw bound at every size class ≥512² (507 vs 544 µs native large —
the 64-B-aligned rstsr buffer is at least as good as a plain `Vec`), is
within noise at odd, and carries ~0.2 µs of dispatch machinery at 64²
(0.33 vs 0.12 µs — sub-µs absolute). **No ≥10 % kernel-only headroom exists
in the contiguous fill path.** faer16 fill never beats serial fill
(write bandwidth saturates per-core; at medium it is 18 % slower —
`PARALLEL_SWITCH=16384` lets marginal sizes into the parallel path).

## Device-level strided fill (the kernel's `else` branch — layout iterator)

| bench (µs) | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|
| diag 2048×2048 (shape [2048], stride [2049]) — the `eye` fill | 0.72 | 0.70 | 0.70 | 0.73 |
| stride-2 both axes (shape [1024]², stride [4096, 2]; 1.05 M elems) | 399.0 (20.5 GB/s w) | 409.2 (20.5) | 180.9 (44.8) | 187.2 |

The per-element layout-iterator cost (~0.39 µs/elem) matches the T1/T4'
strided-branch anatomy; the faer twin is 2.2× faster here. Headroom exists
(~2× serial), but no tensor-API surface reaches this branch (owned tensors
are contiguous; `eye` fills 2048 elements in 0.7 µs), so a patch would have
no user-visible effect — EDIT-GUIDE note, not a patch.

## Kernel-vs-rider split for `rt::full` (native, `results/perf/`)

| measurement | value |
|---|---|
| `rt::full` 2048² criterion | 4654 µs |
| `rt::full` perf wall (200 it) | 4.59–4.83 ms/iter, 8193 faults/iter, sys 90 % |
| `rt::full` perf wall + MALLOC_MMAP_THRESHOLD_=67108864 + MALLOC_TRIM_THRESHOLD_=134217728 | **0.62 ms/iter, ~0 faults/iter steady** (8271 startup faults / 200 it) |
| reuse `c.fill` (kernel-only) | 507–530 µs (criterion native, serial/faer) |
| raw slice-fill bound | 544 µs |
| numpy `np.full` 2048² (T0 context, alloc incl.) | 1.19 ms |

Conclusion: **the rider is ~4.0 ms of the 4.65 ms (86–88 %); the remaining
0.5–0.6 ms broadcast kernel is at the write-bandwidth floor** (tunables
`rt::full` 0.62 ms ≈ reuse `c.fill` 0.51 ms ≈ bound 0.54 ms, and all three
beat numpy's allocating fill). T7's remedy levels fully cover creation ops:
env tunables recover the equivalent of ~7.5× for idiomatic `rt::full` users,
and `c.fill` into a reused buffer is the in-crate idiom.
