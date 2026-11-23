# GRILL R1 — pinned host memory buffer tensor type

Round 1. Facts below are agent-sourced (2026-11-23): two read-only reports
(rstsr device/storage architecture at `261021/cuda` @ `6d17d04`; CuPy @
`8f7a35d` and PyTorch @ `ff4f371` pinned-memory model, local `~/Git-Others`
checkouts) plus direct reads of `rstsr-cuda/src/{transfer,conversion,stream}.rs`
and cudarc 0.19.10 `driver/safe/core.rs`. Answers to `GRILL-R1-ANSWERS.md`
(user-owned).

## Fact register (what the questions lean on)

**rstsr type system**

- F1. `DeviceRawAPI<T>` has one associated type `Raw`; `DeviceAPI<T> =
  DeviceBaseAPI + DeviceRawAPI<T> + DeviceStorageAPI<T> + Clone + Default`
  (`rstsr-core/src/storage/device.rs:9-11, :190`). **`Default` is mandatory.**
- F2. `TensorView<'a,T,B,D>` holds `Storage<DataRef<'a, B::Raw>, T, B>` — a
  reference to the *whole buffer type* (`&'a Vec<T>`), not a slice; offset
  lives in the layout (`tensorbase.rs:53`, `storage/data.rs:16-19`).
- F3. Zero-copy slice → view exists only through a **fabricated `Vec`**:
  `Vec::from_raw_parts(ptr, len, len)` inside `ManuallyDrop`, wrapped as
  `DataRef::from_manually_drop` / `DataMut::from_manually_drop`
  (`tensor/asarray.rs:519-544, :693-720`; contract `storage/data.rs:87-100`).
  The Vec must never be dropped or resized.
- F4. Op-level device traits are `Raw`-generic, but every CPU impl and 142
  tensor-level conveniences (iteration, indexing, `as_ptr`, slice `asarray`,
  blanket `DeviceFormatAPI`, BLAS driver tier, tblis) are hard-bound to
  `Raw = Vec<T>`. A non-Vec device gets **none** of them (CUDA implements each
  op trait itself). So "pinned participates in CPU compute only through a
  Vec-typed view" is not a choice — it is the only route.
- F5. Only non-Vec `Raw` today: `DeviceCuda::Raw = CudaBuffer<T>`
  (byte-erased `CudaSlice<u8>` + `len`, `T: Copy` at the host boundary).
  `rstsr-cuda` exists only on branch `261021/cuda`, not on `main`.
- F6. `rstsr-cuda/src/transfer.rs`: every path (`h2d_full`, `d2h_full`,
  `h2d_scalar`, `d2h_scalar`, `dtod_full`, `fill_naive`) takes/returns a plain
  `Vec<T>` and ends with `stream.synchronize()`. The module doc says "the async
  API surface supersedes that" — **no such surface exists**. CPU↔CUDA edges are
  `DeviceChangeAPI` impls in `conversion.rs:22-80` (`T: Copy + Send + Sync +
  'static`, `R: DataCloneAPI<Data = Vec<T>>`).
- F7. `CudaStreamObj::new` is `pub(crate)`; outside code can only `wrap(ordinal,
  Arc<CudaStream>)` with a cudarc stream (`stream.rs:77-87`). `DeviceCuda::
  default()` = `new(0)` (panics without CUDA, `device.rs:131-137`).
- F8. No pinned/host-register code and no transfer bench anywhere in the pack.
  The one bench is `rstsr-cuda/benches/bench_elementwise.rs` (plain `main`,
  `Instant`, CSV output, `RSTSR_BENCH_ITER/REPEAT`). Earlier decision record:
  "True async H2D/D2H on side streams: separate explicit API with pinned
  staging + events — designed in the transfer round; deferred with intent"
  (`01-device-basic/DECISIONS.md:147-151`). B6 (allocator) was never built.

**cudarc 0.19.10**

- F9. `CudaContext::alloc_pinned::<T: DeviceRepr>(len)` is `unsafe` (uninit),
  default flag `CU_MEMHOSTALLOC_WRITECOMBINED`; `alloc_pinned_with_flags`
  accepts `PORTABLE | DEVICEMAP | WRITECOMBINED` (any u32).
- F10. `PinnedHostSlice<T> { ptr, len, event: CudaEvent }`: before a copy the
  stream does `wait(event)`, after it `SyncOnDrop::Record(event)`; `as_slice()`
  synchronizes the event first; `Drop` = event sync + `cuMemFreeHost`. One
  event per buffer → sequential multi-stream use is safe, concurrent use of
  one pinned buffer by two streams is not.

**CuPy / PyTorch**

- F11. **CuPy: pinned memory is a raw untyped buffer only.**
  `alloc_pinned_memory(size) -> PinnedMemoryPointer` (`cupy/cuda/
  pinned_memory.pyx`, not `memory.pyx`); it has buffer protocol + `ptr`, and
  **no copy methods** — transfers are issued from the device-side
  `MemoryPointer.copy_from_host_async(mem, size, stream=None)` /
  `copy_to_host_async`. A `cupy.ndarray` can never live in pinned memory;
  the host side is `numpy.frombuffer(mem)`. Lifetime: `_EventWatcher.add(
  stream.record(), obj)` keeps the pointer alive until the event is done.
  Default allocator is a `PinnedMemoryPool` (power-of-two, 512-B unit, no
  event tracking inside the pool); `asarray` stages pageable inputs through a
  pooled pinned buffer; `get()` allocates a **pageable** `numpy.empty` unless
  `out=` is given, and `blocking=True` by default. Flags: `cudaHostAllocPortable`.
- F12. **PyTorch: pinned is a normal CPU tensor; pinned-ness is invisible.**
  `empty(pin_memory=True)` → `empty_cpu` picks `getPinnedMemoryAllocator()`
  but dispatch key set is plain `CPU`; no flag in `TensorImpl`/`StorageImpl`;
  `is_pinned()` queries `cudaPointerGetAttributes` on the pointer. It
  participates in CPU ops unchanged (docs neither recommend nor forbid).
  `copy_(non_blocking=True)` = `cudaMemcpyAsync` on the current stream +
  `record_event(host_ptr, ctx, stream)`; on free, an event is recorded per used
  stream and the block is reused only after `cudaEventQuery` succeeds for all
  (caching host allocator, `cudaHostAllocDefault`, or `cudaHostRegister`).
  `non_blocking=False` = memcpyAsync + `cudaStreamSynchronize`. No warning for
  pageable + non_blocking in current main ("the argument has no effect").
  Requires an accelerator; otherwise `RuntimeError`.

## Agent assessment of the user's proposal (fact check)

- `DeviceAPI<Raw = PinnedBuffer<T>>` as a **distinct device** is neither CuPy's
  (raw buffer) nor PyTorch's (invisible allocator swap) model, but it is the
  only *sound* Rust shape: `Vec<T>` frees through the global allocator, so a
  pinned tensor cannot be a `DeviceCpuSerial` tensor with a different
  allocator (memory `rstsr-external-storage-repr`). Confirmed viable.
- `host_view<B>(&device: B) -> TensorView<T, B, D>` with `B: DeviceAPI<T, Raw =
  Vec<T>>` **is achievable zero-cost, but only via F3's fabricated-`Vec`
  pattern** (the view must hold `&Vec<T>`). It also must synchronize the
  buffer's event before handing out host access (F10), so it is not free when
  a DMA is in flight.
- "No op-trait impls on pinned data" is forced by F4, not merely a preference.

---

## Questions

❓ **Q1** - **Crate placement**: Where is the pinned buffer + device defined?
(A) `rstsr-cuda` only — pinned memory needs a CUDA context (cudarc
`alloc_pinned`), the crate exists only on `261021/cuda`. (B) a `PinnedBuffer`
abstraction in `rstsr-core` with the CUDA implementation in `rstsr-cuda`.
(B) buys nothing until a second backend (ROCm/oneAPI) exists, and core has no
CUDA-free way to allocate pinned pages.

➡️ (A): `PinnedBuffer<T>` + `DeviceCpuPinned` live in `rstsr-cuda`; revisit a
core abstraction only when a second backend arrives.

---

❓ **Q2** - **Type identity**: Confirm the model: a *new device type*
`DeviceCpuPinned` (name in Q3) with `Raw = PinnedBuffer<T>`, whose tensors are
transfer endpoints + host-compute-via-view only, versus the two reference
models (CuPy raw buffer, PyTorch invisible flag). PyTorch's model is
unavailable (see assessment); CuPy's model would not be a tensor at all and
would force the pipeline to be hand-written Rust.

➡️ Yes: distinct device type, typed tensors, no op traits on it.

---

❓ **Q3** - **Naming**: `PinnedBuffer<T>` (user's), device `DeviceCpuPinned`,
methods `host_view` / `host_view_mut`. Alternatives: `DevicePinnedHost`,
`DeviceCudaHost` (DLPack's `kDLCUDAHost` name; makes the backend dependency
visible). "Cpu" in the name signals the "CPU tensor counterpart" relationship
and matches `DeviceCpuSerial`.

➡️ `PinnedBuffer<T>`, `DeviceCpuPinned`, `host_view`, `host_view_mut`.

---

❓ **Q4** - **`Default` (F1) and device object contents**: `DeviceCpuPinned`
must be `Default`. Proposed: `{ ordinal: Arc<CudaOrdinalObj>, default_order }`
— **no stream, no flags**; `default()` = ordinal 0, panicking on a CUDA-less
machine exactly like `DeviceCuda::default()`. The stream for a copy comes from
the *device-side operand* (`DeviceCuda` value); pinned allocation is
per-context, and PyTorch/`#21081` note pinned memory from one device is usable
by any other under UVA, so `same_device` compares ordinal only (like
`DeviceCuda`).

➡️ Yes.

---

❓ **Q5** - **Buffer representation and dtype gate**: mirror `CudaBuffer`
(F5): `PinnedBuffer<T> { inner: PinnedHostSlice<u8> (or raw ptr + event), len,
PhantomData<T> }`, byte-erased, `T: Copy` required only at the host boundary
(alloc, `host_view*`, transfers). This dodges cudarc's `T: DeviceRepr` bound
(F9) and keeps the POD gate consistent with
`cuda-write-target-ordering`.

➡️ Yes, byte-erased like `CudaBuffer`.

---

❓ **Q6** - **Lifetime / ordering model**: adopt cudarc's one-event-per-buffer
model (F10): every async copy into/out of the buffer does `stream.wait(event)`
before and records `event` on that stream after; `Drop` synchronizes the event
then frees; `host_view*`, `get_index`, `to_cpu_vec` synchronize the event
before touching bytes. Consequence: sequential use across streams is safe;
two streams *simultaneously* using one pinned buffer is not (it is aliasing
anyway). PyTorch's per-stream event *set* is more general but needs our own
host allocator; defer.

➡️ Yes: one event per buffer now; document the no-concurrent-use rule.

---

❓ **Q7** - **`host_view` construction (F2/F3)**: `host_view` must fabricate a
`Vec<T>` (`Vec::from_raw_parts(ptr, len, len)` in `ManuallyDrop`) to produce a
`TensorView<T, B, D>`; `host_view_mut` the `DataMut` twin. The house precedent
is `asarray(&[T])` (F3). Proposed signature on `TensorAny<R, T, DeviceCpuPinned, D>`
where `R: DataAPI<Data = PinnedBuffer<T>>`:

```rust
fn host_view<'a, B>(&'a self, device: B) -> TensorView<'a, T, B, D>
    where B: DeviceAPI<T, Raw = Vec<T>>;
fn host_view_mut<'a, B>(&'a mut self, device: B) -> TensorViewMut<'a, T, B, D>
    where R: DataMutAPI<Data = PinnedBuffer<T>>, B: DeviceAPI<T, Raw = Vec<T>>;
```

both synchronize the event first (F10) and copy the layout. The fabricated
`Vec` must never be dropped/resized — same contract as `asarray`.

➡️ Yes.

---

❓ **Q8** - **Transfer edges and their default synchronicity**: implement
`DeviceChangeAPI` for `DeviceCpuPinned ↔ DeviceCuda` as **non-blocking by
default** (enqueue `memcpy_*_async` on the `DeviceCuda` value's stream,
wait/record the buffer event, return immediately; PyTorch `non_blocking=True`
semantics) — correctness comes from Q6, and the 1.75× result requires it. The
data is observable on the host only after an event sync (automatic on any host
access). `to_device(&DeviceCuda)` from a *pageable* `DeviceCpuSerial` tensor
stays blocking (today's path, unchanged).

➡️ Yes: pinned↔CUDA edges async by default; pageable↔CUDA unchanged.

---

❓ **Q9** - **Host-side edges**: `DeviceCpuPinned ↔ DeviceCpuSerial/DeviceFaer`
via `DeviceChangeAPI` are **copies** (host memcpy, like PyTorch `pin_memory()`
which is a blocking copy); zero-copy is only through `host_view*`. Also
implement `DeviceToCpuAPI` for `DeviceCpuPinned` (copy to `Vec`) and
`DeviceStorageAPI::{get_index, set_index, get_index_ptr}` as plain host
access after event sync.

➡️ Yes.

---

❓ **Q10** - **Allocation flags (F9/F11/F12)**: default `0` (`cudaHostAllocDefault`,
PyTorch's choice) rather than cudarc's `WRITECOMBINED`, because CPU compute on
the buffer through `host_view` is a stated goal and WC memory is uncached on
the host. Expose `PinnedBuffer::with_flags(len, PinnedFlags)` for the bench to
measure WC vs default for pure D2H sinks.

➡️ Yes: default flags `0`; `with_flags` for experiments.

---

❓ **Q11** - **Pooling**: `cudaHostAlloc` is slow (ms-scale), and CuPy/PyTorch
both pool pinned memory by default. rstsr-cuda has no host allocator at all
(F8). Proposed: **no pool in this round**; `PinnedBuffer` stays opaque so a
caching host allocator can be added non-breakingly; the bench preallocates all
pinned buffers outside the timed region.

➡️ Yes: no pool now, pool-friendly internals.

---

❓ **Q12** - **Bench placement and form (user: "bench test, not usual test")**:
develop and record the numbers in a new task dir here
(`2026-11-23-pinned-streaming-bench/`, prototype crate depending on
`rstsr-cuda` by path), then port into `rstsr-cuda/benches/bench_streaming.rs`
in the style of `bench_elementwise.rs` (plain `main`, medians of interleaved
rounds). Modes mirror `bench_streaming.py`: `h2d`/`compute`/`d2h`,
`serial_all` (pageable + blocking = today's `to_device`), `pipeline`
(pinned + async + 3 streams), and negative controls `pipeline_blocking`,
`pipeline_1stream`. Assert `pipeline < 0.7 × serial_all` and that controls
do not improve; report ratios only.

➡️ Yes. A *correctness-only* usual test (bitwise pipeline vs serial) still
goes in `rstsr-cuda/tests/`.

---

❓ **Q13** - **Public stream creation**: the bench needs >1 stream from outside
the crate's internals (F7). Bring a minimal `DeviceCuda::new_stream(ordinal)`
(or `DeviceCuda::with_new_stream`) into scope with this work, since without
it the pipeline is not expressible from an ordinary crate.

➡️ Yes, in scope.
