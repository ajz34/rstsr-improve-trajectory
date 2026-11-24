# Streaming H2D / compute / D2H overlap with pinned host tensors — Rust bench (2026-11-24)

Question (B6 of `../2026-11-23-pinned-host-buffer/PLAN.md`): does the pinned
host tensor type (`DeviceCudaPinned`, committed `71d3d89` on `261021/cuda`)
deliver the overlap the CuPy prototype measured, and do the negative controls
confirm that each ingredient is necessary?

## Verdict

Yes. On the same bytes as the CuPy prototype (32 chunks × 8 MiB f64), with
pinned host buffers, non-blocking transfers and three streams:

| flow | wall time | vs naive |
|---|---|---|
| `serial_all` — pageable host, blocking transfers (today's `to_device`) | 105.6 ms | 1.00× |
| `pipeline_1stream` — pinned, async edges, one stream | 80.0 ms | 1.32× |
| `pipeline_blocking` — pinned, 3 streams, `synchronize()` per stage | 84.9 ms | 1.24× |
| **`pipeline`** — pinned, async edges, 3 streams, issued per chunk | **46.1 ms** | **2.29×** |
| `pipeline_pageable` — 3 streams, but host side is a plain `Vec` | 43.7 ms | 2.42× |
| `d2h_writecombined` — WC pages as D2H destinations (isolated phase) | 18.6 ms | = default |

The pipeline lands at **92 % of the `max(compute, h2d + d2h)` bound** (42.2 ms)
and hides **90 % of the transfer time**. The two negative controls sit exactly
at the sum of the isolated phases (~80 ms): pinned memory alone, or three
streams alone, buy nothing. Canonical run: `results/bench-streaming.txt`; a
repeat (`results/bench-streaming-repeat.txt`, earlier revision without the
pageable probe) agrees within 0.4 %.

### The `pipeline_pageable` surprise — an honest divergence from CuPy

CuPy's `pipeline_pageable` collapses to the serial time (its pageable staging
blocks the host per chunk). rstsr's pageable `to_device` does **not**: it
synchronizes the *device* stream at the end of each transfer, but the host
only ever waits for its own chunk's copy, so the pageable flow also overlaps —
in these runs it even landed slightly *below* the pinned pipeline (43.7 vs
46.1 ms, within clock noise). Consequences:

- The pinned ingredient cannot be validated by a pageable negative control in
  rstsr, unlike CuPy; the controls that do isolate it are
  `pipeline_1stream` (no streams) and `pipeline_blocking` (no async edges).
- What pinned *does* buy here is host observability with correctness: the
  bytes are stable the moment an event sync returns, without the
  staging-copy dance CuPy needs, and without ever reading bytes a DMA is
  still writing.
- The numbers are within noise of each other; no performance claim is made
  for pinned vs pageable *in this artificial benchmark*. Under host contention
  (another thread doing real work between enqueues) the pageable path's
  host-synchronous staging would show up; that is the workload the pinned
  type exists for.

Write-combined pages as D2H destinations transfer at exactly the default rate
(18.6 ms both): the link is the limit on this machine, so decision D8 (default
`cudaHostAllocDefault`) costs nothing.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, sm_120, 8 GiB (driver 595.99.02, CUDA 13.2) |
| host link | PCIe 4.0 ×8; measured H2D 14.1 GB/s, D2H 14.4 GB/s |
| CPU | AMD Ryzen 9 9955HX |
| toolchain | nightly (workspace pin), release profile (opt-level 3, no LTO) |
| rstsr | branch `261021/cuda` @ `71d3d89` **plus one uncommitted fix** in `crates-device/rstsr-cuda/src/pinned/transfer.rs` (see below) |

## Method

`src/main.rs` (prototype crate, path-dependency on the local rstsr checkout;
written to be dropped into `rstsr-cuda/benches/bench_streaming.rs`,
`harness = false`, unchanged).

- **Payload**: `compute` is a chain of `rt::sin` kernels (one launch per
  link). The link count is autotuned at start-up so the compute phase costs
  ≈ 1.15 × (`h2d` + `d2h`) — here 11 links, ~43 ms for 256 MiB. (This
  differs from the CuPy prototype's dependent-FMA chain: `sin` saturates
  throughput rather than latency, but the balance target is the same.)
- **Streams**: `dev_h` (H2D), `dev_c` (compute), `dev_d` (D2H), built from one
  base value with `new_stream` / `with_stream`; no `set_stream`.
- **Pipeline**: per chunk, H2D on H → compute on C (re-tagged input, the
  kernel's `order_input` makes C wait for H2D(c) only) → D2H on D into a
  preallocated pinned output chunk. Every device buffer is kept alive until
  the final sync.
- **Timing**: 7 round-robin rounds, every probe once per round, a 150 ms
  clock-guard between rounds (the laptop GPU's clock moves ~2× when idle),
  per-probe medians; ratios only.
- **Correctness**: pipeline output compared bitwise with `serial_all`.
- **Assertions**: bitwise equal; `pipeline < 0.7 × serial_all`; `pipeline <
  0.75 × pipeline_1stream`; both controls ≥ 0.95 × (sum of phases);
  `pipeline_pageable` ≥ 0.95 × pipeline (informational, see the surprise
  section).

Run: `cargo run --release` (~15 s). Knobs: `RSTSR_BENCH_CHUNKS`,
`RSTSR_BENCH_ROUNDS`, `RSTSR_BENCH_ITERS` (0 = autotune), `RSTSR_BENCH_TRACE=1`
(print enqueue vs final-sync time per pipeline probe).

## Two findings that decide whether the pipeline overlaps at all

1. **Issue per chunk, not per phase.** The first version issued all 32 H2D
   copies, then all compute, then all D2H, relying on the kernels' cross-stream
   events. Result: exactly the serial time (80.9 ms). An event recorded when
   compute(c) is enqueued covers *everything* enqueued on H so far, so
   compute(0) waited for H2D(31). Per-chunk issue (h(c), c(c), d(c)) restored
   the dependency structure.
2. **Do not make the source's allocation stream wait for the D2H.** The
   committed `d2h_pinned_async` ordered the source both ways
   (`order_input_raw` + `order_output_raw`), i.e. it made the compute stream C
   `cuStreamWaitEvent` on the D2H copy of every chunk. Result: 62.7 ms =
   compute + d2h — H2D overlapped, D2H did not, because every later kernel on
   C queued behind the previous chunk's copy. The fix (uncommitted in rstsr):
   take the source pointer through cudarc's `DevicePtr::device_ptr(slice,
   exec_stream)`, which records a **read event** on the copy stream, dropped
   after the enqueue; `CudaSlice::drop` then makes the allocation stream wait
   for it *at drop time*, before the stream-ordered free. Same lifetime
   guarantee (a dropped temporary is not reused mid-DMA), zero blocking of C
   while the owner keeps the buffer alive. Pipeline: 62.7 → 47.1 ms.

A third, expected one: a device buffer dropped while a foreign-stream kernel
still reads it corrupts data (hit by the B5 test first); the bench keeps
every cross-stream buffer alive until the final sync.

## Porting notes

- Move `src/main.rs` to `rstsr-cuda/benches/bench_streaming.rs` with
  `[[bench]] name = "bench_streaming", harness = false` and `use rstsr::prelude::*`
  (the crate's dev-dependency); `rstsr_cuda::` paths stay as they are.
- Commit the `pinned/transfer.rs` change (D2H source ordering through the
  cudarc read event) — it is the difference between 1.33× and 2.27×.
- The same "wait at drop, not at use" idea applies to the kernel-level
  foreign-stream read-lifetime hole (`order_output` on read-only inputs); not
  done here.
