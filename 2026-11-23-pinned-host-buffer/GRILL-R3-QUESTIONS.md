# GRILL R3 — creation surface, lifetime details, stream scope

## Agent-authored background: answers received on screen (transcribed)

- **Q2'**: agreed — new type wrapping `DeviceCpuSerial`.
- **Q3**: rename to **`DeviceCudaPinned`** (pinned first, "CUDA's handling"
  second; leaves room for other GPU backends). Check whether
  `PinnedBuffer` has an origin in cudarc.
- **Q4–Q6**: okay.
- **Q7**: `device: &B` (RSTSR passes devices by reference); confirm that the
  view wraps a `ManuallyDrop` fabricated `Vec`, not a plain `DataRef`.
- **Q8–Q10**: okay.
- **Q11**: leaning accept; wants a recap of pool vs allocator.
- **Q12**: okay.
- **Q13**: yes; asks what a non-minimal version would add.

### Agent notes on the answers

- **`PinnedBuffer` name**: cudarc's type is `PinnedHostSlice<T>`
  (`driver/safe/core.rs:1459`); the *concept* (page-locked memory from
  `cuMemHostAlloc`, one `CudaEvent` per allocation, `Drop` = sync +
  `cuMemFreeHost`) is cudarc's, the *name* is ours, matching `CudaBuffer`
  (`CudaSlice` in cudarc). Keeping `PinnedBuffer<T>` in `rstsr-cuda`
  follows the same naming shift; no objection unless you prefer
  `CudaPinnedBuffer`.
- **Q7 confirmed**: every tensor-level device transfer takes `&BOut`
  (`tensor/device_conversion.rs:22-99`), so `host_view(&self, device: &B)`
  is the house form. And yes — the view's storage is
  `DataRef::from_manually_drop(ManuallyDrop::new(Vec::from_raw_parts(ptr, len, len)))`
  (or the `DataMut` twin), exactly the `asarray(&[T])` / `asarray(&mut [T])`
  construction (`tensor/asarray.rs:525-541, :700-716`); a plain
  `DataRef::TrueRef` cannot exist because there is no real `Vec` to borrow.
  With Q2' embedding a `DeviceCpuSerial`, the argument-free form
  `host_view(&self) -> TensorView<T, DeviceCpuSerial, D>` is the primary one;
  `host_view_on(&self, device: &B)` the generic form.

### Recap for Q11 — allocator vs pool (agent-authored)

- **Allocator** = *how memory is obtained from the driver*. For pinned
  memory that is one `cuMemHostAlloc` (CUDA runtime `cudaHostAlloc`) per
  buffer. It is expensive: the driver page-locks physical RAM (walks the
  pages, faults them in, pins them), typically 0.1–1 ms per call plus
  per-page cost, and the pages become unswappable. `cuMemFreeHost` is the
  matching slow unpin. Our `PinnedBuffer` *is* this layer: new buffer →
  `alloc_pinned_with_flags`; drop → free.
- **Pool / caching allocator** = *a reuse layer in front of the allocator*.
  Instead of freeing a block back to the driver, keep it in a free list keyed
  by (rounded) size; the next allocation of that size reuses it without a
  driver call. CuPy: `PinnedMemoryPool` (power-of-two buckets, 512-B unit,
  no splitting; `cupy.get_default_pinned_memory_pool()`). PyTorch:
  `CachingHostAllocatorImpl` (power-of-two rounding up to a threshold, no
  splitting, optional reserved segment, `torch.cuda.host_memory_stats()`).
  Both are *always on* for their internal staging.
- **The pool's one hard problem**: a block may be "freed" by the Rust/Python
  side while a DMA is still writing into it. Handing it out again then
  corrupts data. Both frameworks therefore tie reuse to *events*: a freed
  block carries the events of the streams that used it and is reusable only
  once they are done (PyTorch: `record_event` + `process_events`; CuPy:
  `_EventWatcher` holds a reference until the event is done). Our Q6
  one-event-per-buffer is exactly the information a pool needs at free time
  (`event.query()`), so adding a pool later is a change to
  `PinnedBuffer`'s constructor/drop only — hence "pool-friendly internals":
  keep the `PinnedHostSlice<u8>` + event private, add a `HostAlloc` trait
  (like the planned-but-unbuilt `CudaAlloc`) when a pool lands.
- **Why it matters for the bench**: the pipeline allocates 32–64 pinned
  chunks. At ~0.5 ms per `cudaHostAlloc` that is ~16–32 ms outside the timed
  region if preallocated, or a confound inside it. CuPy/PyTorch would hide
  this in their pool; we preallocate explicitly. For *users*, many short-lived
  pinned tensors would be slow without a pool — the reason to revisit.

---

❓ **Q14** - **Creation and fill surface on `DeviceCudaPinned`**: implement
`DeviceCreationAnyAPI` (`empty` = uninit pinned, `zeros` = host memset,
`full` = host fill, `outof_cpu_vec` = copy `Vec` → pinned) and
`asarray((Vec<T>, shape), &DeviceCudaPinned)` → pinned copy; plus direct
wrapping `Tensor::new_f(Storage::new(DataOwned::from(buf), dev), layout)`
with `PinnedBuffer::from_vec(&DeviceCudaPinned, Vec<T>)` /
`PinnedBuffer::into_vec(self)`. `arange`/`linspace` via host fill. No
kernels — everything is host-side code over the synced slice.

➡️ Yes.

---

❓ **Q15** - **Two lifetime details on the async edges (F6, Q8)**:
(a) D2H `cuda.to_device(&pinned)`: the copy runs on the `DeviceCuda` value's
*execution* stream, but the source `CudaBuffer` is freed on its *allocation*
stream; if they differ (a `set_stream` device value), a temporary source can
be reused while the DMA still reads it (the known foreign-stream read-lifetime
hole). The pinned edge must therefore record an event on the exec stream and
make the source's allocation stream wait before returning — a transfer-local
fix, not the general kernel fix.
(b) `into_device` from pinned consumes the source; dropping it synchronizes
its event, so `into_device` is effectively blocking. Implement it as
`to_device` + drop (correct, simple) and document that pipelines use the
borrowing `to_device`.

➡️ Yes to both.

---

❓ **Q16** - **Event flag**: `PinnedBuffer`'s event is created with
`CU_EVENT_BLOCKING_SYNC` (cudarc's choice: the waiting thread sleeps) rather
than the default spin-wait. Spin-wait burns a core for the duration of each
host access; blocking-sync wake-up latency (~10–20 µs) is negligible against
a chunk-sized copy (>100 µs).

➡️ Yes: blocking-sync event.

---

❓ **Q17** - **Stream scope, non-minimal answer to Q13**: the minimal edit is
`DeviceCuda::new_stream(ordinal)`. The non-minimal set would be:
(a) `CudaStreamObj::new_f(&DeviceCuda)` public + a per-ordinal stream registry
(reuse, `Display`, `CudaStreamObj` identity by `Arc`), (b) a `Tensor::
synchronize()` for CUDA tensors (wait the exec stream, then the buffer's
events) as the "data is ready" primitive, (c) `CudaEvent` wrapper made public
so users can order their own cross-stream work, (d) `with_stream(self, stream)`
builder alongside the mutating `set_stream`. Only (a) is needed by the bench.
Take `new_stream` + (b) now (the pipeline's final wait is exactly (b));
(c)/(d) later.

➡️ Yes: minimal + `Tensor::synchronize()`; (c)/(d) deferred.

---

❓ **Q18** - **Blocking control in the bench**: `pipeline_blocking` is
produced by the bench calling `DeviceCuda::synchronize()` after each transfer
(existing public method), not by adding blocking transfer entry points.

➡️ Yes.

---

❓ **Q19** - **Pool decision (Q11)**: no pool this round; `PinnedBuffer`
internals private and event-carrying so a caching host allocator (with
event-gated reuse) can be added later without API change; revisit when a
user workload shows many short-lived pinned tensors.

➡️ Yes.
