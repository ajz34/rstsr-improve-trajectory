---
name: cuda-write-target-ordering
description: rstsr-cuda host reads sync only the buffer's allocation stream, so every op writing an existing buffer needs two-way event ordering; byte-erased storage also forces a T: Copy POD gate
metadata:
  type: project
---

Three coupled invariants in `rstsr-cuda`:

1. **Write ordering.** Ops run on the device value's execution stream;
   host reads (`to_cpu_vec`, `get_index`, `Display`) synchronize only the
   buffer's *allocation* stream (the stream inside `CudaSlice`).
   Therefore a kernel that writes an existing buffer must (a)
   `order_input(target, exec)` before the launch — which also covers a
   still-pending `cudaMallocAsync` allocation — and (b) record an event on
   `exec` after the launch and make the allocation stream wait
   (`order_output`). Out-of-place ops are fine because their output is
   allocated on `exec`.
2. **Read lifetime (FIXED 2026-11-24).** For a foreign-stream *input* there
   is only `alloc -> exec`; without a read event the allocation stream can
   free (`free_async`) and reuse the block while the consuming kernel still
   reads it. Reproduced 2026-10-25: `rt::add(&a_on_S0, &rt::arange(...,
   &dev_on_S2))` corrupted 4,194,303/4,194,304 elements once the temporary
   was dropped, and the next S2 allocation returned the same pointer. Fix:
   `CudaBuffer::read_ptr_on` returns the cudarc `SyncOnDrop` guard to the
   caller; `FlatLaunch::arg_buf`/`arg_ptr` hold every argument's guard and
   drop them only after `launch()` enqueues the kernel, so the read event
   covers the launch and `CudaSlice::drop` orders the free behind it.
   Critical cudarc detail: the event is recorded at GUARD DROP, not at
   `device_ptr` call time — dropping the guard before the enqueue records
   the wrong window (this was the original bug shape). Regression test:
   `tests/test_kernels.rs::foreign_stream_temporary_kernel_input_ordered`
   (fails on the pre-fix code).
3. **POD.** Device storage is byte-erased; any path that crosses the
   host boundary (`d2h_full`, `get_index`, CPU/Faer conversions) must be
   `T: Copy`. The CPU-device precedent does not transfer: a CPU device
   stores a real `Vec<T>`, so non-POD values are representable there.

**Why:** a re-tagged buffer (`set_stream` + `to_device`) showed stale
reads in a loop (4089/4096 wrong) without two-way ordering; and
`Tensor<String, DeviceCuda>` was constructible from safe code and
reconstructed dangling host pointers (double free).

**How to apply:** new device ops that write into an existing buffer add
both ordering calls; new transfer/conversion impls stay `T: Copy`-bounded
(`CudaDType: Copy` already covers kernel dtypes). Do not treat
`order_input` as a read-lifetime barrier: either keep input temporaries
alive past the launch or add the missing `exec -> alloc` event for
read-only inputs. Related: [[cuda-assign-copy-semantics]].
