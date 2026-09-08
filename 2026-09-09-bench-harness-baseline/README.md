# T0 — bench-harness-baseline (no patch)

Campaign task T0: a **reusable benchmark harness pattern** plus an **honest
baseline** for the candidate ops of the rstsr CPU-efficiency campaign, framed
against measured bandwidth ceilings, ndarray/numpy anchors, and `perf` evidence.

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5` (path dep
  `../../rstsr/rstsr-core`, strictly read-only).
- **This is a NO-PATCH task.** There is no `proposed.patch` here by design:
  nothing in the rstsr tree was touched. Verified at the end:
  `git -C ../../rstsr status --short` is empty and HEAD is `386948b`.
- Campaign plan: [../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
  (this README follows its §3 measurement contract and §8 deliverables).

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores / 32 threads, AVX-512; caches: 48 kB L1d, 1 MB L2 per core, 128 MiB total L3 (X3D) |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)` (nightly channel, matches rstsr's pinned nightly) |
| cargo | 1.97.1 |
| criterion | 0.5.1 |
| ndarray | 0.16.1 (+ ndarray-stats 0.6 for the argmax anchor) |
| numpy | 2.5.1 (conda env `torch`) |
| perf | 7.0.14 (`/usr/bin/perf`) |
| Parallelism | `RAYON_NUM_THREADS=16` (physical cores); harness asserts `DeviceFaer::get_num_threads() == 16` at every bench start |
| Target configs (D7) | `portable` = no RUSTFLAGS (x86-64 SSE2 baseline); `native` = `-C target-cpu=native` |
| Cargo profile | stock release defaults, stated explicitly in Cargo.toml (opt-level 3, no LTO, codegen-units 16) |

## Directory layout

```
Cargo.toml            standalone crate (no workspace); rstsr-core path dep with
                      default features mirrored explicitly
src/lib.rs            harness: devices, thread check, criterion config,
                      deterministic fixtures (macros), size classes
benches/triad.rs      pure-Rust triad + memcpy ceiling (no rstsr)
benches/transpose.rs  a.t().to_contig(RowMajor)   (forces the copy; .t() is a view)
benches/reduce.rs     sum_axes(0) / sum_axes(-1) / sum_all
benches/vecdot.rs     1-D dot 1e3/1e5/1e7, f32 spot, batched (4096,512)
benches/elementwise.rs  a+b contig / broadcast / strided(=+b.t view) / spots
benches/fill_argmax.rs  rt::zeros, rt::full, rt::argmax (1e6/1e7)
benches/anchors_ndarray.rs  ndarray anchors (same sizes)
examples/correctness.rs   correctness gate vs naive scalar loops (both devices)
examples/profile_ops.rs   fixed-iteration op runner for `perf stat`
numpy_ref/anchors.py     numpy anchor script (conda `torch` env)
numpy_ref/results.txt    saved numpy anchor output
results/                 raw criterion outputs (portable/, native/, native_d8/),
                         perf outputs (perf/ + derived_summary.txt),
                         tables.md (all tables below, regenerable)
results/make_tables.py   regenerates tables.md from the raw outputs
reproduce.sh             self-contained: correctness -> portable -> native -> d8 -> perf -> numpy
```

## Measurement contract (what every number here means)

1. **Release mode** (`cargo bench`, bench profile = release defaults), both
   target configs run from separate `cargo` builds via explicit RUSTFLAGS.
2. **Correctness gate before perf**: `examples/correctness.rs` compares every
   op against naive scalar loops on deterministic fixtures — including the
   odd size 1000x777, the explicit zero-stride broadcast view
   (`brow.broadcast_to([m,n])`), the auto-broadcast `[1,n]` form, transpose
   views, f64 + f32 spots — for **both** `DeviceCpuSerial` and `DeviceFaer`
   (16 threads). It passed under **both** RUSTFLAGS configs
   (`results/correctness_portable.txt`, `results/correctness_native.txt`).
3. **Anti-cheat**: inputs AND outputs pass through `std::hint::black_box`;
   per-iteration outputs are freshly allocated by the op itself and moved into
   `black_box`. Fixture data comes from a deterministic integer hash (no
   const-foldable constants).
4. **Allocation policy: allocation included.** Every benched op allocates its
   output inside the op (`to_contig`, `sum_axes`, `+`, `zeros`, `full`,
   `vecdot`); this is part of the measured time and is identical across
   devices, configs, and anchors. The ndarray `add_zip_prealloc` anchor is
   the only deliberately different variant (preallocated output; labeled as
   such) and is compared against `add_alloc` (same policy as rstsr), not
   against rstsr's number directly.
5. **Both devices**: `DeviceCpuSerial` explicitly, and `DeviceFaer::new(0)`
   (exactly what `Device::default()` builds under `faer_as_default`); the
   rayon pool size comes from `RAYON_NUM_THREADS=16` and is asserted, not
   assumed. Note `DeviceCpuRayon::generate_pool(0)` sizes the pool from
   `rayon::current_num_threads()` (rstsr-core/src/feature_rayon/device.rs),
   so the env var must be set before process start — the harness enforces it.
6. **Streaming honesty and the L3 caveat**: the large class (2048x2048 f64 =
   32 MiB; vecdot 1e7 two-buffer set = 160 MB) is comparable to this chip's
   128 MiB X3D L3, so repeated-iteration benches are partially L3-fed and
   their derived GB/s can exceed the DRAM ceiling. **triad/memcpy large
   (240/160 MB working sets) are the honest DRAM numbers.** The L3 effect
   applies equally to rstsr, ndarray, and numpy anchors, so cross-op ratios
   stay fair; single-op "GB/s" above the triad number should be read as L3
   assistance, not DRAM bandwidth. This is a consequence of the plan's D9
   size classes on this particular machine; candidate-op experiments that
   need pure-DRAM streaming should rotate buffers or use sizes > L3.
7. **Statistics**: criterion defaults (100 samples), 2 s measurement +
   0.7 s warm-up per bench (uniform, minutes-scale total per config).

## Reusable harness snippet (what later experiment dirs copy)

`Cargo.toml` fragment (path + features visible in the manifest; feature
pass-through for the D8 column):

```toml
[dependencies]
rstsr-core = { path = "../../rstsr/rstsr-core", default-features = false, features = [
    "row_major", "aligned_alloc", "faer", "faer_as_default",
] }
criterion = { version = "0.5", features = ["cargo_bench_support"] }

[features]
dispatch_dim_layout_iter = ["rstsr-core/dispatch_dim_layout_iter"]

[[bench]]
name = "candidate_op"
harness = false
```

Bench-file skeleton (the full pattern lives in `benches/*.rs`):

```rust
use std::hint::black_box;
use bench_harness_baseline::{assert_faer_threads, configure_group, faer_device, serial_device};
use bench_harness_baseline::gen_mat_f64;                 // fixture macro
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;

fn bench(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);                  // RAYON_NUM_THREADS=16 guard
    let mut group = c.benchmark_group("candidate_op");
    configure_group(&mut group);                          // 2 s + 0.7 s uniform
    for &(label, m, n) in bench_harness_baseline::MAT_SIZES {
        let a = gen_mat_f64!(m, n, 1, &dev_serial);       // deterministic fixture
        group.bench_function(BenchmarkId::new(format!("{label}_{m}x{n}-serial_f64"), "op"), |b| {
            b.iter(|| black_box(/* the op, on black_box(&a) */))
        });
        // ... same for &dev_faer with id `...-faer16_f64`
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
```

Conventions baked into the harness: `black_box` on inputs and outputs;
deterministic fixtures via `gen_value`/`gen_vec_*`/`gen_mat_*!` macros
(macros, not generic fns — they inherit rstsr's trait bounds at the call
site); name benches `<sizeclass>_<shape>-<device>_<dtype>/<variant>` so the
D8 filter `"large|odd"` works; run order and env in `reproduce.sh`.

## Bandwidth ceiling (pure Rust, serial core)

| op | portable | GB/s | native | GB/s |
|---|---|---|---|---|
| triad large (c = a + 3b, 3x1e7 f64 = 240 MB traffic) | 6.05 ms | 39.6 | 6.08 ms | 39.5 |
| memcpy large (copy 1e7 f64, 160 MB traffic) | 3.73 ms | 42.9 | 3.91 ms | 40.9 |
| triad medium (1e6, 24 MB — L2/L3-scale reference) | 190 µs | 126 | 182 µs | 132 |

**The single-core DRAM streaming ceiling on this machine is ~40 GB/s.** All
streaming numbers below are framed against it. (numpy's single-thread
memcpy anchor agrees: 44.9 GB/s; numpy's alloc-temp-bound triad: 20.6 GB/s
nominal, ~34 GB/s real after counting its `3.0*b` temporary.)

## Baseline tables

Full tables (all classes, incl. f32 spots) are in
[`results/tables.md`](results/tables.md), regenerated by
`results/make_tables.py` from the raw criterion outputs. The decision-relevant
extracts:

### Large class 2048x2048 f64 (32 MiB) — time and derived GB/s (allocation included)

| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| transpose copy | 21.73 ms | 3.1 | 22.13 ms | 3.0 | 3.51 ms | 19.1 | 3.54 ms | 19.0 |
| add contiguous | 6.33 ms | 15.9 | 6.30 ms | 16.0 | 2.87 ms | 35.1 | 2.72 ms | 37.0 |
| add broadcast row | 5.91 ms | 17.0 | 5.67 ms | 17.7 | 2.55 ms | 39.5 | 2.47 ms | 40.7 |
| add strided (b.t) | 28.07 ms | 3.6 | 28.25 ms | 3.6 | 4.29 ms | 23.5 | 4.39 ms | 22.9 |
| sum axis0 | 1.38 ms | 24.3 | 1.90 ms | 17.7 | 444.66 µs | 75.5 | 341.06 µs | 98.4 |
| sum axislast | 468.03 µs | 71.7 | 483.55 µs | 69.4 | 98.23 µs | 341.6 | 94.81 µs | 353.9 |
| sum all | 461.56 µs | 72.7 | 433.46 µs | 77.4 | 56.66 µs | 592.2 | 55.65 µs | 603.0 |
| fill: full | 4.49 ms | 7.5 | 4.63 ms | 7.2 | 4.67 ms | 7.2 | 4.60 ms | 7.3 |
| fill: zeros (calloc-lazy, no touch) | 2.27 µs | - | 2.33 µs | - | 2.31 µs | - | 2.34 µs | - |
| argmax 1e7 | 93.53 ms | 0.9 | 93.95 ms | 0.9 | 10.76 ms | 7.4 | 11.69 ms | 6.8 |
| vecdot 1e7 | 2.81 ms | 56.9 | 2.82 ms | 56.7 | 2.91 ms | 55.0 | 2.79 ms | 57.4 |
| vecdot batched 4096x512 | 465.78 µs | 72.0 | 461.78 µs | 72.7 | 147.85 µs | 226.9 | 144.41 µs | 232.4 |
| transpose copy f32 | 15.43 ms | 2.2 | 15.55 ms | 2.2 | 1.34 ms | 25.0 | 1.37 ms | 24.5 |
| add contiguous f32 | 789.44 µs | 63.8 | 772.47 µs | 65.2 | 238.28 µs | 211.2 | 255.77 µs | 196.8 |
| sum axis0 f32 | 426.49 µs | 39.3 | 454.12 µs | 36.9 | 177.59 µs | 94.5 | 95.19 µs | 176.3 |
| vecdot 1e7 f32 | 1.20 ms | 66.9 | 1.22 ms | 65.8 | 1.20 ms | 66.8 | 1.25 ms | 64.1 |

GB/s above the 40 GB/s triad ceiling (sum_*, batched vecdot, f32 spots) are
L3-assisted per the caveat above — read them as "upper context", and compare
ops against their anchors, which sit under the same effect.

Key context numbers (same table generation):

- **native vs portable is ~1.00x for nearly every rstsr kernel** — the
  per-element closure/MaybeUninit/index-loop code does not auto-vectorize
  even with AVX-512 enabled. The exceptions are instructive: ndarray's
  sum_axis0 speeds up 1.7x (large) to 1.96x (odd) under native, and
  medium-class add 1.26x — well-written reduction/elementwise loops do pick
  up AVX-512. The ceiling is code structure, not ISA.
- **faer16 (rayon) hurts at small sizes**: sum_all 64x64 is 7.2 µs faer vs
  0.41 µs serial (~7 µs dispatch overhead); medium 512x512 add is 63 µs faer
  vs 49 µs serial. The crossover to "parallel wins" sits between medium and
  large for elementwise/reduce ops, as expected from rstsr's
  PARALLEL_SWITCH thresholds.
- **vecdot 1e7 vs analytic peak**: 7.1 GFLOP/s serial = **3.9% of the
  32-DP-FLOP/cycle/core peak** (182 GFLOP/s at ~5.7 GHz); `perf` puts it at
  1.06 FLOP/cycle. It reads 160 MB in 2.81 ms (56.9 GB/s) which is above the
  DRAM ceiling — i.e. partly L3-fed; a truly DRAM-bound 1-D dot at 40 GB/s
  would take ~4.0 ms. ndarray `dot` ties (2.75 ms) and numpy `einsum` ties
  (2.69 ms); numpy `np.dot` (multithreaded BLAS ddot) wins at 1.19 ms
  (134.8 GB/s, 16 threads).
- **batched vecdot (4096,512) is the fixable one**: serial 465.78 µs vs
  faer16 147.85 µs (3.1x), and per-element work ~14 cycles — this is the
  code map's contiguous-remaining branch (per-element MaybeUninit
  read-modify-write inside the contraction loop), clearly visible.

### Medium 512x512 f64 (2 MiB, L2-class) — serial (portable)

| op | serial | faer16 | ndarray |
|---|---|---|---|
| transpose copy | 811.8 µs (5.2 GB/s) | 205.1 µs | 28.0 µs (73 GB/s) |
| add contiguous | 49.4 µs (127 GB/s) | 62.6 µs | (not benched at medium) |
| sum axis0 | 51.6 µs | 71.0 µs | 32.1 µs |
| sum axislast | 21.6 µs | 45.5 µs | 15.5 µs |
| sum all | 15.9 µs | 16.3 µs | 15.4 µs |

At L2-resident sizes the serial kernel is within ~1.5-3x of ndarray except
transpose (29x) — the transpose gap is worst exactly where blocking matters.
Note faer16 loses to serial at this size for everything except transpose.

### Odd 1000x777 f64

transpose copy serial 2.43 ms (5.1 GB/s), faer16 362.5 µs; sum axis0 serial
163 µs / faer 172 µs portable (faer does NOT win strided reductions at this
size portable — PARALLEL_SPLIT overhead) but 84.8 µs under native.
sum_axislast/all match the large-class pattern.

## numpy / ndarray anchors (headline gaps)

numpy numbers are single-threaded except `np.dot` (BLAS); from
[numpy_ref/results.txt](numpy_ref/results.txt). All rstsr and ndarray anchor
numbers in this table are the **portable config** (ndarray native differs only
marginally; where noted explicitly it is called out).

| op (large class) | rstsr serial | rstsr faer16 | ndarray | numpy | verdict |
|---|---|---|---|---|---|
| transpose copy | 21.73 ms | 3.51 ms | 5.33 ms | **99.8 ms** (see caveat below) | rstsr loses to ndarray 4x; numpy's `.T.copy()` is anomalously slow here — context only |
| sum axis0 | 1.38 ms | 444.7 µs | 629 µs (581 µs native) | **460 µs** | 1.5-3x gap to anchors; faer16 already beats both |
| sum axislast | 468 µs | 98.2 µs | 455 µs | 655 µs | serial already at parity/better; faer 4.6x |
| sum all | 462 µs | 56.7 µs | 454 µs | 791 µs | parity serial; faer 8x |
| add contiguous | 6.33 ms | 2.87 ms | 6.56 ms (alloc) / **2.17 ms** (zip prealloc) | **2.77 ms** | serial ties ndarray-with-alloc; numpy and zip-prealloc show what allocation-free would give |
| add strided (b.t) | 28.07 ms | 4.29 ms | - | 87.4 ms | rstsr already 3x faster than numpy; still 5x below its own contig path |
| add broadcast row | 5.91 ms | 2.55 ms | - | **2.49 ms** | numpy matches faer16; serial 2.4x behind |
| fill (full) | 4.49 ms | 4.67 ms | 4.58 ms | **1.19 ms** | 3.8x gap to numpy's memset-class fill |
| zeros | 2.3 µs | 2.3 µs | 2.4 µs | 0.003 ms | all lazy (calloc); NOT a fill benchmark |
| argmax 1e7 | **93.53 ms** | 10.76 ms | **7.18 ms** | **1.32 ms** | 13x behind ndarray, 71x behind numpy — the outlier of the whole survey |
| vecdot 1e7 | 2.81 ms | 2.91 ms | 2.80 ms (dot) | 2.69 ms (einsum) / 1.19 ms (BLAS dot) | at parity for serial-class engines |

Caveat on the numpy `.T.copy()` number: the ~99.8 ms anomaly reproduced across
repeats within this run, but across repeat runs the rstsr-serial advantage over
numpy on this op ranged ~2.6x-4.6x, so treat it as **context only** — the
transpose verdict (4x behind ndarray) rests on the ndarray anchor, not numpy.

## perf evidence (native, DeviceCpuSerial; `results/perf/`)

Derived metrics in `results/perf/derived_summary.txt` (bytes from the fixed
iteration counts in `examples/profile_ops.rs`; GB/s = bytes / task-clock):

| op | t_ms | GB/s | cyc/elem | ins/elem | IPC | L1d-miss% | LLC-miss | FLOP/cyc | %pk32 |
|---|---|---|---|---|---|---|---|---|---|
| triad | 1857.9 | 38.8 | 3.53 | 0.81 | 0.23 | 43.3 | 55.8M | 0.567 | 1.8% |
| transpose | 919.2 | 2.9 | 30.36 | 122.46 | 4.03 | 2.8 | 42.3M | - | - |
| sum_axis0 | 154.9 | 17.3 | 2.61 | 2.90 | 1.11 | 12.9 | 19.1M | 0.383 | 1.2% |
| sum_axis1 | 143.9 | 69.9 | 0.64 | 0.84 | 1.31 | 20.2 | 3.3M | 1.559 | 4.9% |
| vecdot | 331.4 | 48.3 | 1.88 | 3.13 | 1.66 | 22.5 | 3.8M | 1.063 | 3.3% |
| add_contig | 674.1 | 14.9 | 9.15 | 12.87 | 1.41 | 8.0 | 91.8M | - | - |
| argmax | 14584.8 | 0.8 | 53.60 | 283.35 | 5.29 | 0.1 | 2.5M | - | - |
| zeros | 1.3 | (calloc-lazy) | 0.00 | 0.01 | 1.77 | 2.5 | - | - | - |

What this says about where the time goes:

- **transpose copy is instruction-bound, not bandwidth-bound**: 122
  instructions and 30 cycles per element at IPC 4.0 with only 2.8% L1d
  misses. The generic strided assign burns its time in per-element index
  arithmetic + clone, not in memory. Any blocked kernel removes ~10x of
  work, so T1's upside is real and its mechanism is confirmed (and the D8
  column shows part of it is available from the already-shipped
  monomorphized iterator, see below).
- **argmax is an instruction pathology**: 283 ins/element, 53.6 cycles/element,
  IPC 5.29 (the machine is executing an instruction mountain, never waiting
  on memory — 0.1% L1d miss). A contiguous fast path with simple index
  tracking should reclaim ~10-50x.
- **add_contig is ~2.6x below triad with a kernel-side and an
  allocation-side component**: 12.9 ins/element (should be ~1-2 when
  vectorized) and 844k page faults in one profile run (~8.2k/iter = one fresh
  32 MiB output per iteration), with **sys time (0.43 s) exceeding user time
  (0.24 s)** — the zero-page mapping of the freshly-allocated output is a
  first-class cost in the allocation-included policy. rstsr's
  `aligned_alloc` path (`std::alloc::alloc`, 64 B) and ndarray's Vec-based
  `add_alloc` show the same cost (6.30 vs 6.56 ms) while numpy and
  ndarray-zip-prealloc demonstrate the allocation-free ceiling (2.77 /
  2.17 ms). Candidate experiments should (a) fix the kernel branch
  (chunks_exact) and (b) treat output-buffer reuse / allocation policy as a
  separate, possibly bigger lever.
- **sum_axis0 (strided reduction) is latency-limited** (IPC 1.11, 12.9%
  L1d miss, 17-24 GB/s): it improves 1.3-2x under native only after
  `dispatch_dim_layout_iter` (see D8) or in ndarray's hand-unrolled loop —
  rstsr's clone-through-closure accumulate blocks vectorization.
- **vecdot 1-D** runs at 1.06 FLOP/cycle = 3.3% of the 32 FLOP/cycle peak
  with IPC 1.66 — for a 2-buffer read stream the DRAM ceiling alone would
  cap a perfect kernel at ~2.5 FLOP/cycle here (160 MB / 40 GB/s), so the
  1-D dot is within ~2.3x of its memory-bound limit; the remaining gap is
  scalar (non-vectorized) accumulation. The batched case's serial path is
  the clearly-inefficient one (3.1x slower than faer16 on the same data).
- **LLC-miss counts on this X3D part are not a usable DRAM signal** (they
  stay suspiciously low even for triad; likely L3 partitioning/sectored
  counting). L1d-miss rate, IPC, ins/elem and task-clock-derived GB/s carry
  the analysis; treat the LLC column as decoration.

## D8 secondary column (native + `dispatch_dim_layout_iter`, large|odd subset)

Full table in `results/tables.md`; highlights vs plain native:

| case | native | native+D8 | speedup |
|---|---|---|---|
| transpose copy odd 1000x777 serial | 2.44 ms | 1.23 ms | **1.98x** |
| transpose copy odd 1000x777 faer16 | 366.6 µs | 197.3 µs | **1.86x** |
| add strided large serial | 28.25 ms | 20.79 ms | **1.36x** |
| sum axis0 large faer16 | 341.1 µs | 249.1 µs | **1.37x** |
| transpose copy large faer16 | 3.54 ms | 3.14 ms | 1.13x |
| transpose copy large serial f32 | 15.55 ms | 10.66 ms | 1.46x |
| vecdot 1e7 f32 serial | 1.22 ms | 2.15 ms | **0.57x (regression)** |
| vecdot 1e7 f32 faer16 | 1.25 ms | 2.12 ms | **0.59x (regression)** |
| everything else | ~1.0x | ~1.0x | neutral |

Reading: the runtime-dimension-dispatch helps exactly the irregular-strided
cases (odd transpose, strided add, strided faer reductions, up to ~2x) and is
neutral on contiguous streaming, but it *hurts* f32 vecdot noticeably in this
build. It stays a per-caller decision, not a default-on feature (consistent
with rstsr's own doc comment on the feature: overhead for small tensors,
large compile-time cost).

Carry-forward for the vecdot task (T3): keep vecdot benches and candidate
kernels clear of `dispatch_dim_layout_iter` (or re-validate f32 1-D dot under
it first) — in this build the feature costs up to 0.57x on f32 vecdot.

## Conclusions — confirmed / re-ranked code-map targets

Measured re-ranking for the campaign (code-map rank in parentheses):

1. **argmin/argmax (was #6) -> promote to top tier.** 13x behind ndarray
   (portable anchors), 71x behind numpy at 1e7; 283 ins/element scalar
   pathology with zero vectorization. Small, self-contained kernel
   (`reduce_all_unraveled_arg_cpu_serial`), largest measured headroom per
   line of code. Note ndarray's argmax itself is ~5x behind numpy — numpy is
   the honest target here, not ndarray.
2. **elementwise contiguous branch (was #3) — confirmed, with a structural
   rider.** 12.9 ins/element, 2.6x below triad; native AVX-512 buys nothing
   (1.00x). The rider: with allocation included, >50% of add_contig wall
   time is kernel page-fault handling for the fresh output — an
   allocation-reuse / buffer-pool experiment may pay off across ALL
   allocating ops at once (add, to_contig, reductions, fill), independent of
   kernel work. Recommend a T4a (kernel) + T4b (allocation policy) split.
3. **transpose copy (was #5) — confirmed, mechanism refined.** 3.1 GB/s vs
   40 GB/s ceiling and vs ndarray's 12 GB/s; instruction-bound (122 ins/el).
   Plan T1's blocked kernel is justified; also cheap interim wins: route
   through `dispatch_dim_layout_iter`-style monomorphized iterators (+13%
   large, +98% odd, already in-tree behind the feature flag) and use the
   dead `orderchange_out_r2c_ix2_cpu_serial` 64x64 blocked kernel.
4. **reduce_axes sum_axis0 (was #4) — confirmed.** 2-3x behind numpy/ndarray
   on the strided axis; gets *worse* under native (1.38 -> 1.90 ms) with the
   default iterator — fragile codegen that a local-accumulator rewrite fixes;
   faer16 already recovers 2-4x, so the serial path is the target.
5. **vecdot contiguous-remaining branch (was #1) — confirmed but scoped
   down.** The 1-D path is at memory/L3 limits and within ~2.3x of a perfect
   DRAM-bound kernel (3.9% of FMA peak is misleading framing for a
   bandwidth-bound op). The *batched* case (4096x512: 465.8 µs serial vs
   147.9 µs faer16, ~14 cyc/output) is where the MaybeUninit RMW
   restructure pays; expected win ~2-3x serial.
6. **inner_dot_naive_cpu_rayon (was #2) — untested here.** T0's bench matrix
   covers `vecdot` but not matmul `%` (which routes to inner_dot). The
   code-map hypothesis stands untested; add a `%`-on-1-D bench to T3.
7. **fill_promote (was #8) — confirmed, with a caveat.** `full` 2048^2 is
   3.8x slower than numpy's fill (7.5 vs 28 GB/s write); but `zeros` is
   calloc-lazy everywhere (2.3 µs) and must NOT be "optimized" into a
   touching fill. Fix `full`/`ones`/fill, leave `zeros` alone.
8. **vecdot general/strided branch (was #7) — stays low priority** (1-D dot
   already ties ndarray/numpy-einsum; strided contraction unmeasured here).

New items surfaced by T0 worth queueing:

- **Allocation/page-fault policy study** (see #2 rider): glibc maps/unmaps
  the 64-B-aligned 32 MiB blocks (8.2k soft faults per 32 MiB output, sys >
  user in add_contig). Buffer pools, malloc trim/threshold tuning, or
  output-reuse APIs could win 1.5-2.5x on every allocating op at streaming
  sizes, orthogonal to kernel quality.
- **faer16 small-size overhead**: ~7 µs per op dispatch at 64x64 (sum_all
  7.2 µs vs 0.41 µs serial) — worth a look only if small-tensor users matter;
  otherwise the PARALLEL_SWITCH thresholds are doing their job.

## Deviations from the brief

- `cargo bench` rejects `--release` (bench profile already inherits release);
  reproduce.sh relies on the profile being release by default — same effect.
- ndarray has no `argmax` in 0.16 — the anchor uses `ndarray-stats 0.6`
  (`QuantileExt::argmax`), added as a dev-dep.
- The strided-add bench is `a + b.t()` where b is an independent tensor
  (rather than literally `a + a.t()`) so the odd-size correctness case is
  expressible (`a + a.t()` does not broadcast for non-square shapes); for
  square fixtures the two are identical in layout and measured cost.
- argmax "medium" is labeled medium (1e6) per the brief's ARGMAX sizes even
  though its 8 MB working set is large-class; it filters into the D8 column
  accordingly.
- The D8 stage originally passed the filter through cargo incorrectly
  (`cargo bench --features X -- "large|odd"` split across the `--`); fixed in
  reproduce.sh (`cargo bench --bench B --features X -- "large|odd"
  --save-baseline native_d8`). First D8 attempt produced empty logs; the
  committed `results/native_d8/` is from the fixed run.
- Extra perf targets beyond the required six: argmax and zeros (both proved
  diagnostic). LLC-miss counters proved unreliable on this X3D part (kept in
  raw outputs, not used for conclusions).
- `results/make_tables.py` is added tooling beyond the brief (table
  regeneration for the README); its output snapshot is committed as
  `results/tables.md`.

## No patch

This experiment makes **no code changes to rstsr** and ships **no
proposed.patch** — `git -C ../../rstsr status --short` was verified empty at
the start and at the end, HEAD `386948b`.

## Reproduce

```bash
cd 2026-09-09-bench-harness-baseline
./reproduce.sh              # everything: ~35-45 min
./reproduce.sh correctness  # gate only (both configs, ~2 min)
./reproduce.sh portable     # criterion suite, portable (~8 min)
./reproduce.sh native       # criterion suite, native (~8 min)
./reproduce.sh d8           # native + dispatch_dim_layout_iter, large|odd (~8 min)
./reproduce.sh perf         # perf stat pass, serial (~1 min)
./reproduce.sh numpy        # numpy anchors via conda torch (~1 min)
```

Requires: rustc 1.97.1 nightly active (rstsr's rust-toolchain.toml handles it
via the path dep builds), `RAYON_NUM_THREADS=16` (set by the script), conda
`torch` env for the numpy stage, `perf` for the perf stage. `target/` is
gitignored and left in place for reviewer re-runs; `Cargo.lock` is committed.
