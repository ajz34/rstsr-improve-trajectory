---
name: cuda-write-target-ordering
description: rstsr-cuda host reads sync only the buffer's allocation stream, so every op writing an existing buffer needs two-way event ordering; byte-erased storage also forces a T: Copy POD gate
metadata:
  type: project
---

Two coupled invariants in `rstsr-cuda`:

1. **Streams.** Ops run on the device value's execution stream; host
   reads (`to_cpu_vec`, `get_index`, `Display`) synchronize only the
   buffer's *allocation* stream (the stream inside `CudaSlice`).
   Therefore a kernel that writes an existing buffer must (a)
   `order_input(target, exec)` before the launch — which also covers a
   still-pending `cudaMallocAsync` allocation — and (b) record an event on
   `exec` after the launch and make the allocation stream wait
   (`order_output`). Out-of-place ops are fine because their output is
   allocated on `exec`.
2. **POD.** Device storage is byte-erased; any path that crosses the
   host boundary (`d2h_full`, `get_index`, CPU/Faer conversions) must be
   `T: Copy`. The CPU-device precedent does not transfer: a CPU device
   stores a real `Vec<T>`, so non-POD values are representable there.

**Why:** a re-tagged buffer (`set_stream` + `to_device`) showed stale
reads in a loop (4089/4096 wrong) without two-way ordering; and
`Tensor<String, DeviceCuda>` was constructible from safe code and
reconstructed dangling host pointers (double free).

**How to apply:** new device ops that write into an existing buffer add
both ordering calls; new transfer/conversion impls stay `T: Copy`-bounded
(`CudaDType: Copy` already covers kernel dtypes). Related:
[[cuda-assign-copy-semantics]].
