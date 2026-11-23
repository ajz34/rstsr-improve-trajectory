---
name: cuda-clone-async
description: rstsr-cuda device Clone is already async (cudarc try_clone -> memcpy_dtod_async on the alloc stream); dtod_full's sync is the cross-ordinal path only
metadata:
  type: project
---

`CudaBuffer::clone_f` (`crates-device/rstsr-cuda/src/buffer.rs`) calls
`self.slice.try_clone()` -> cudarc `CudaStream::clone_dtod`
(`cudarc-0.19.10 driver/safe/core.rs:1795`) -> `alloc` + `memcpy_dtod_async`
**on the buffer's own allocation stream** (`:1750-1766`, same-stream branch).
No host barrier: the deep D2D clone returns after enqueue, matching decision D1
("deep D2D copy on the owning stream") of `2026-10-21-cuda-decisions/01-device-basic`.

Do not add a `synchronize()` to clone:
- value semantics only need an independent buffer, and device contents are only
  observed by device work, which is stream-ordered;
- the ordering is already covered by the allocation stream (copy before free),
  `order_input` (consumer waits), `order_output` (free waits for the consumer);
- a sync drains the whole stream and would make clone the one blocking op in an
  otherwise fully async op set (kills any pipeline);
- CuPy and PyTorch both agree: `ndarray.copy()` -> `astype` ->
  `elementwise_copy` kernel on the current stream (`cupy/_core/core.pyx:863,782`);
  `torch.clone` = `empty_like` + `copy_` (TensorFactories.cpp:2252-2276) ->
  `copy_device_to_device` -> `CUDACachingAllocator::memcpyAsync` on the current
  stream (Copy.cu:598,644), event barriers only for `src_device != dst_device`.

The sync in `transfer::dtod_full` is a different code path: cross-ordinal
(multi-GPU) moves only (`src/conversion.rs:228,245`), never `clone`.

Related: [[cuda-write-target-ordering]] (the foreign-stream read-lifetime hole
that a sync would hide, not fix), [[cuda-device-basic-grill-r1]] (F1/Q1),
[[cupy-streaming-overlap]].
