# Streaming H2D / compute / D2H overlap — CuPy prototype (2026-11-23)

Question: is the "separate the transfers from the compute and pay for both"
cost real, and can a chunked multi-stream pipeline hide the transfer time behind
the device computation? This directory answers it in Python (CuPy) first, so the
target numbers for a future `rstsr-cuda` test are known before any Rust work.

**No rstsr code is involved.** For reference, the `rstsr` checkout used for the
porting notes is branch `261021/cuda` at `6d17d04`.

## Verdict

Yes, this is achievable in CuPy, and the effect is large:

| flow | wall time | vs separated |
|---|---|---|
| separated phases (load all, compute all, store all) | 83.9 ms | 1.00× |
| per-chunk multi-stream pipeline (pinned host) | 47.9 ms | **1.75×** |

Across six runs of the same configuration the pipeline landed at **1.75–1.85×**,
i.e. **95–97 % of the `max(compute, h2d+d2h)` bound** and **94–97 % of the
transfer time hidden**. The residual is the pipeline ramp (one chunk of H2D
before the first compute can start, one chunk of D2H at the end).

Three ingredients are each *necessary*; dropping any one returns the
system to the separated time (measured, see [negative controls](#negative-controls)):

1. **pinned (page-locked) host buffers** — pageable memory makes every copy
   synchronous;
2. **non-blocking D2H** (`ndarray.get(..., blocking=False)`) — a blocking `get`
   syncs the stream per chunk;
3. **more than one stream** — one stream serializes H2D → compute → D2H.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, sm_120, 36 SMs, 8 GiB |
| host link | PCIe 4.0 ×8 (~16 GB/s theoretical; the link idles at gen1/gen2 and trains up under load) |
| driver / CUDA | 595.99.02 / 13.2 |
| CuPy / NumPy / Python | 14.1.1 / 2.4.6 / 3.11.15 |
| CPU | AMD Ryzen 9 9955HX (16 cores) |

Measured with this GPU at its current state: H2D **14.0–14.2 GB/s**, D2H
**13.9–14.4 GB/s**, f64 FMA chain **320–335 GFLOP/s** (clock-up state; see
the hazard note).

> Benchmark hazard: this laptop GPU idles at 630 MHz and the f64 rate moves
> **~2×** within seconds (166 ↔ 335 GFLOP/s observed). Every number here is
> therefore taken from **interleaved (round-robin) rounds** — see
> [Method](#method) — and speedups are only compared within one process.

## Method

`bench_streaming.py`, default configuration: **32 chunks × 8 MiB = 256 MiB per
direction**, f64, 4 streams, 7 timed rounds, total runtime ~5.5 s (budget: 20 s).

- **Payload**: one dependent FMA chain per element
  (`y = fma(y, c, 0.5)`, `C` iterations) — a pure device-compute kernel whose
  length is a knob, not a memory-bandwidth test.
- **Balancing**: `C` is autotuned at start-up so the compute phase costs
  ≈ 1.15 × (`h2d` + `d2h`). Without that the phases are not comparable across
  machines/clocks, and the pipeline either wastes the copy engine or stalls on
  it.
- **Warm-up**: a sustained f64 loop first, so the clocks are out of idle before
  anything is timed.
- **Timing**: every probe (3 phases + 6 modes) runs once per round, and the
  per-probe median over rounds is reported. Timing each mode in its own block
  mixes clock state with mode.
- **Correctness**: the pipeline's host output is compared bitwise with the
  `serial_all` output; the run fails if they differ.

Modes:

| mode | what it does |
|---|---|
| `h2d` / `compute` / `d2h` | each phase alone over all chunks (the model's terms) |
| `serial_all` | all H2D, then all compute, then all D2H (the naive flow) |
| `serial_chunk` | per chunk: H2D, compute, D2H, everything blocking, one stream |
| `pipeline` | per chunk: async H2D, compute, async D2H on a rotating stream, pinned host |
| `pipeline_pageable` | `pipeline`, but ordinary pageable NumPy host buffers |
| `pipeline_blocking` | `pipeline`, but the D2H call blocks |
| `pipeline_1stream` | `pipeline`, but all chunks share one stream |

## Results (canonical run, `results/bench-streaming.txt`)

```
phase totals (isolated, interleaved median)
  h2d         19.18 ms   14.00 GB/s
  compute     45.61 ms    322.2 GFLOP/s (f64 FMA chain, 219 FMA/elem)
  d2h         19.33 ms   13.89 GB/s

predicted serial      84.12 ms  (h2d + compute + d2h)
predicted pipeline    45.61 ms  (max of compute and h2d+d2h)

mode                       ms  vs serial_all  vs serial_chunk  pipeline eff
---------------------------------------------------------------------------
h2d                     19.18          4.37x            4.32x             -
compute                 45.61          1.84x            1.82x             -
d2h                     19.33          4.34x            4.29x             -
serial_all              83.90          1.00x            0.99x             -
serial_chunk            82.87          1.01x            1.00x             -
pipeline                47.92          1.75x            1.73x         95.2%
pipeline_pageable       86.08          0.97x            0.96x         53.0%
pipeline_blocking       83.20          1.01x            1.00x         54.8%
pipeline_1stream        82.95          1.01x            1.00x         55.0%

transfer time hidden by the pipeline:  94.0% of h2d+d2h
speedup, naive separation -> pipeline: 1.75x
```

### The model

The separated flow is exactly additive — `83.90 ≈ 19.18 + 45.61 + 19.33`
(measured; the `serial_chunk` variant that interleaves the phases on one stream
is the same, as expected). The pipeline collapses to the slowest single
resource, `max(compute, h2d + d2h)`, plus a one-chunk ramp.

Why `h2d + d2h` and not `max(h2d, d2h)`: the two copy directions only *partly*
overlap on this GPU (`results/transfer-concurrency.txt`, 8 × 64 MiB each way):

```
h2d only      38.20 ms   14.05 GB/s
d2h only      37.37 ms   14.37 GB/s
concurrent    49.65 ms   21.63 GB/s aggregate
sum = 75.57 ms,  max = 38.20 ms
```

Aggregate 21.6 GB/s against 14.2 GB/s one-way: the copy engine is one shared
resource with partial duplex, not two independent engines. So the honest
pipeline bound here is `max(compute, ~0.66 × (h2d + d2h))`; with the autotuned
compute at ~1.8× that duplex transfer time the run is compute-limited, which is
why it reaches 95 % of the simpler `max(compute, h2d+d2h)` bound. If the compute
were the smaller term, the pipeline would flatten at the copy-engine time and
the speedup would cap at `(h2d + compute + d2h) / (0.66 × (h2d + d2h))` ≈ 1.5×.

A second consequence, visible in the timeline: copies from *different* streams
do **not** run in parallel on this part. Four concurrent H2D copies of 16 MiB
finish 1.2 ms apart in issue order — the copy engine serializes them, and a
later chunk's H2D window includes the queueing wait.

### Timeline

`bench_streaming.py` replays the pipeline once with CUDA events, one lane per
stream; `h` = H2D, `c` = compute, `d` = D2H (canonical run):

```
  0 ms                                                                                      47.6 ms
  s0 |hcccdhcccccccccccdhcccccccccdhcccccccccdhcccccccccccddhccccccccddhccccccccddhcccccccccdd        |
  s1 |hhcccccdhhccccccccccdhcccccccccdhcccccccccdhcccccccccccdhhccccccccdhhccccccccdhccccccccccdd     |
  s2 |hhhcccccccdhccccccccccddhccccccccddhcccccccccdhhccccccccccdhcccccccccdhcccccccccdhccccccccccdd  |
  s3 |hhhhhccccccccdhccccccccccdhhccccccccdhccccccccccccdhcccccccccdhcccccccccdhcccccccccddhccccccccdd|
      h = H2D   c = compute   d = D2H   (1 char ~ 0.50 ms)
```

Every lane is always in some phase, and at a given column different lanes are in
different phases: that is the overlap. In the separated modes the same three
phases occupy three disjoint time slabs.

## Negative controls

| removed ingredient | mode | result |
|---|---|---|
| pinned host memory | `pipeline_pageable` | 0.97× (86.1 ms) — pageable copies block the host |
| non-blocking D2H | `pipeline_blocking` | 1.01× (83.2 ms) — one sync per chunk |
| multiple streams | `pipeline_1stream` | 1.01× (82.9 ms) — one stream serializes the phases |

That is the useful negative result: **CuPy's defaults do not stream.**
`cp.ndarray.set(host_array)` from a plain NumPy array and
`cp.ndarray.get(out=...)` with `blocking=True` (the default) are both
synchronous, and `cp.asarray`/`.get()` on pageable memory stage through an
internal pinned buffer *on the host thread* — the pipeline then runs at
exactly the separated time.

## Porting notes for `rstsr-cuda`

What a Rust test of the same experiment needs, against `261021/cuda`:

1. **Pinned host memory — missing today.** Every transfer path takes a plain
   `Vec<T>` (`crates-device/rstsr-cuda/src/transfer.rs`: `d2h_full`, `h2d_full`,
   `h2d_scalar`, `d2h_scalar`) and ends with `stream.synchronize()`, i.e. it is
   the `pipeline_pageable` + `pipeline_blocking` row of the table above.
   `cudarc` 0.19.10 already ships the primitive:
   `CudaContext::alloc_pinned::<T>(len)` → `PinnedHostSlice<T>`
   (`driver/safe/core.rs`), plus `alloc_pinned_with_flags`.
   Note the default there is `CU_MEMHOSTALLOC_WRITECOMBINED`: excellent for D2H,
   potentially slow for CPU *reads* (write-combined memory is uncached on the
   host side). Measure `PORTABLE` vs `WRITECOMBINED` before fixing a default.
2. **Non-blocking transfer entry points — missing.** A streaming test needs
   `h2d_full_async(stream, &pinned_src)` / `d2h_full_async(stream, &mut
   pinned_dst)` that only enqueue (`cuMemcpyHtoDAsync` / `DtoHAsync`) and leave
   the sync to the caller's stream `synchronize()` after the loop. The
   per-element `d2h_scalar` (one sync per element) is the pathological case.
3. **More than one stream — a design blocker.** `crate::registry` caches one
   `CudaStreamObj` per ordinal, and `DeviceCuda::stream()` returns it, so every
   `DeviceCuda::new(0)` in a process shares one stream. Streams are also part
   of `same_device` (strict-streams decision), which means the pipeline has to
   be arranged as *all ops of one chunk on one stream* — exactly the Python
   pattern — rather than mixing operands across streams. Either expose a
   user-visible stream handle on `DeviceCuda`, or let a tensor carry its
   allocation stream (it already records it in `CudaBuffer::alloc_stream()`).
4. **Allocation** is stream-ordered `cuMemAllocAsync` (cudarc `CudaStream::alloc`),
   so interleaving allocations with copies does not sync the host. Still
   preallocate the chunk buffers outside the timed region: a pool miss during
   the loop inserts a real `cuMemAlloc` and pollutes the measurement, and the
   pool is not per-stream aware in the same way CuPy's arenas are.
5. **The test itself** should assert on the *ratio* (`pipeline < 0.7 ×
   serial_all`, say) rather than on absolute milliseconds, and take medians of
   interleaved rounds — the clock drift above is larger than the effect on a
   machine without a fixed clock. Keep the negative controls: they are what
   makes a "streaming works" test meaningful, since a mis-wired pipeline
   silently degrades to the serial time.

## How to run

```bash
python bench_streaming.py                  # ~5.5 s, 32 x 8 MiB, 4 streams
python bench_streaming.py --chunk-mib 4 --chunks 64
python bench_streaming.py --no-timeline --reps 5
python probe_transfer_concurrency.py       # H2D vs D2H duplex on this GPU
```

`--fma 0` (default) autotunes the compute length; pass e.g. `--fma 128` to pin
it. `--budget-guard` aborts before running if the estimated wall time exceeds
the 20 s budget.

## Artifacts

| file | content |
|---|---|
| `bench_streaming.py` | the benchmark (all modes, timeline, verification) |
| `probe_transfer_concurrency.py` | H2D/D2H duplex probe |
| `results/bench-streaming.txt` | output of the canonical run quoted above |
| `results/transfer-concurrency.txt` | output of the duplex probe |
