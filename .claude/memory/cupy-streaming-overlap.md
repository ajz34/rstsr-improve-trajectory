---
name: cupy-streaming-overlap
description: CuPy H2D/compute/D2H streaming prototype (2026-11-23) - 1.75-1.85x, all three ingredients required, copy engine partly duplex
metadata:
  type: project
---

Experiment `2026-11-23-streaming-overlap/`: CuPy prototype of chunked
multi-stream H2D -> compute -> D2H overlap, before any `rstsr-cuda` work.

Numbers (RTX 5070 Laptop, PCIe 4.0 x8, CuPy 14.1.1):

- separated (h2d + compute + d2h) is exactly additive; pipeline reaches
  `max(compute, h2d+d2h)` + one-chunk ramp -> **1.75-1.85x**, 95-97 % of bound.
- All three of pinned host memory, `get(..., blocking=False)`, and >1 stream are
  required; removing any one lands at 1.00x. CuPy's defaults do NOT stream:
  `set(np_array)` / `get()` on pageable memory are synchronous host-side.
- The copy engine is one shared resource with **partial duplex**: concurrent
  H2D+D2H runs at 21.6 GB/s aggregate vs 14.2 GB/s one-way (0.66 x the sum).
  Copies from *different* streams serialize (four 16 MiB H2D finish 1.2 ms
  apart in issue order), so a per-chunk stream lane is what gives the overlap.
- Benchmark hazards: this laptop GPU idles at 630 MHz and the f64 rate moves
  ~2x within seconds -> time all modes in interleaved round-robin rounds and
  report per-probe medians; assert on ratios, never absolute ms.

`rstsr-cuda` streaming status (at `261021/cuda`, `6d17d04`) — CORRECTED after
reading the code; the streaming/async half already exists:
- async by design (`src/lib.rs:12-13`); 9 `synchronize()` in the crate, 6 of them
  in `transfer.rs` (h2d/d2h/dtod/fill), **zero in any kernel or operator**;
- `DeviceCuda` carries a per-value stream + `set_stream`; `same_device` does NOT
  compare the stream, and `order_input`/`order_output` (`src/kernel.rs`) insert
  cross-stream events in every kernel — so produce-on-A / consume-on-B is legal
  today. `CudaStreamObj::new` is `pub(crate)`; only `wrap(ordinal, Arc<CudaStream>)`
  is public, so an outside crate cannot make a stream without cudarc.
- allocation is stream-ordered `cuMemAllocAsync` (no host stall).
Real gaps = host boundary only: (1) pinned host owner type (host tensors are
`DataOwned<Vec<T>>`; a `Vec` cannot own `cuMemHostAlloc` memory) — cudarc 0.19.10
has `CudaContext::alloc_pinned::<T>` -> `PinnedHostSlice<T>`, default flag
`CU_MEMHOSTALLOC_WRITECOMBINED` (good D2H, bad CPU reads); (2) non-syncing
`h2d`/`d2h` entry points (note `dtod_full` also syncs, so device `clone()` would
serialize a loop). Recommended shape: 3 streams H/C/D, not per-chunk rotation.
Details + hazards in `2026-11-23-streaming-overlap/RSTSR-FEASIBILITY.md`.
