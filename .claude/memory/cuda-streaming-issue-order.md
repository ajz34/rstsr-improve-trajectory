---
name: cuda-streaming-issue-order
description: Two laws for rstsr-cuda multi-stream pipelines - issue work per chunk (events cover everything enqueued so far) and never make a source's allocation stream wait for its consumer (stalls later kernels); use cudarc read-event + drop-time wait
metadata:
  type: project
---

Measured 2026-11-24 in `2026-11-23-pinned-streaming-bench/` (RTX 5070 Laptop).

1. **Issue per chunk, not per phase.** `cuEventRecord` captures everything
   enqueued on the stream so far; `order_input` records at enqueue time. If all
   H2D copies are enqueued before any compute, compute(0) waits for H2D(N-1):
   the "pipeline" measures as exactly serial (80.9 ms). Interleave
   h(c), c(c), d(c) per chunk.
2. **Never block a producer stream on its consumer.** `order_output_raw(src,
   exec)` = `cuStreamWaitEvent(alloc_stream, exec)`; for a D2H whose source
   lives on the compute stream this stalls every later kernel on that stream
   (62.7 ms = compute + d2h, H2D overlapped but D2H not). For *read* uses, order
   the free instead of the stream: `DevicePtr::device_ptr(slice, exec)` returns
   a `SyncOnDrop` that records a read event on `exec`; `CudaSlice::drop`
   (cudarc, multi-stream mode + event tracking, both default-on) makes the
   alloc stream wait for it right before `free_async`. Same lifetime guarantee,
   zero stall while the owner keeps the buffer alive. The same idea applies to
   the kernel-level foreign-stream read-lifetime hole (`cuda-write-target-
   ordering` item 2) — `order_output` is only needed for *writes* into an
   existing buffer.
3. Keep every cross-stream device buffer alive until the final sync
   (user-side guard until the kernel-level fix).
4. The copy-engine FIFO theory (a waiting D2H blocking later H2Ds) was WRONG:
   D2H lag 0/1/2/4 made no difference once (2) was fixed.

Numbers: pipeline ~46 ms vs ~106 naive (2.3x), 91% of
`max(compute, h2d+d2h)`, ~90% of transfer hidden; controls flat. Write-combined
pinned pages = default for D2H (link-bound).

**Pageable does NOT collapse in rstsr (unlike CuPy).** rstsr's pageable
`to_device` syncs the device stream per transfer but the host only waits for
its own chunk -> the pageable 3-stream flow also overlaps (43.7 ms, within
noise of pinned). So a pageable negative control cannot validate the pinned
ingredient in rstsr; use `pipeline_1stream` / `pipeline_blocking`. What pinned
buys is host observability (stable bytes after one event sync, no staging
copy), not raw overlap in this synthetic bench.
