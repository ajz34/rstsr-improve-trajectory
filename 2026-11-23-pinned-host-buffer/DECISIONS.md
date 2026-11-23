# DECISIONS — pinned host memory tensor type (grill closed 2026-11-23)

Five rounds (`GRILL-R1..R5-QUESTIONS.md`; answers received on screen,
transcribed by the agent). Baseline: rstsr `261021/cuda` @ `6d17d04`, cudarc
0.19.10 @ `ec1cac7`; CuPy @ `8f7a35d`, PyTorch @ `ff4f371` as references.
Fact register F1–F18 lives in the round files.

Reference models (F11/F12): CuPy exposes pinned memory as a raw untyped
buffer only; PyTorch as a normal CPU tensor with an invisible allocator swap.
rstsr can do neither (`Vec<T>` frees through the global allocator; one `Raw`
per device type), so the design is a **third model: a distinct device type
whose tensors are typed transfer endpoints plus zero-copy host views**.

## D1 — Placement and names

- Everything lives in `rstsr-cuda` (branch-only crate; pinned allocation
  needs a CUDA context). No `rstsr-core` change. Revisit a core abstraction
  when a second GPU backend exists.
- Buffer: `PinnedBuffer<T>` (concept from cudarc `PinnedHostSlice<T>`; name
  ours, as `CudaBuffer` vs `CudaSlice`).
- Device: **`DeviceCudaPinned`** — "pinned first, CUDA's handling second",
  leaves room for other GPU backends.
- Methods: `host_view`, `host_view_mut`, `host_view_on`, `synchronize`.

## D2 — Device object

```rust
pub struct DeviceCudaPinned { ordinal: Arc<CudaOrdinalObj>, base: DeviceCpuSerial }
impl<T> DeviceRawAPI<T> for DeviceCudaPinned { type Raw = PinnedBuffer<T>; }
```
- Wraps `DeviceCpuSerial` (the `DeviceFaer { base }` pattern); `default_order`
  / `same_device` delegate to `base` plus ordinal. No stream, no flags.
- `Default` = `new(0)`, panicking on a CUDA-less machine like
  `DeviceCuda::default()`; `new_f(ordinal)` fallible.
- Neither `DeviceCpuSerial` nor `DeviceCuda` can host a second `Raw`
  (coherence, F13); a flag on `DeviceCpuSerial` would put a CUDA handle in
  core. Both rejected.

## D3 — Buffer representation and dtype gate

`PinnedBuffer<T> { inner: PinnedHostSlice<u8>, len, PhantomData<T> }`,
byte-erased like `CudaBuffer`; `T: Copy` only at the host boundary (alloc,
views, transfers). Sidesteps cudarc's `T: DeviceRepr` bound; same POD gate as
the rest of `rstsr-cuda`.

## D4 — Ordering and lifetime: one event per buffer (cudarc model)

- Each async copy: `stream.wait(event)` before, record `event` on that stream
  after. `Drop` = event sync + `cuMemFreeHost`.
- Every host access (`host_view*`, `get_index*`, `set_index`, `into_vec`)
  synchronizes the event first.
- Event created with `CU_EVENT_BLOCKING_SYNC` (sleeping wait; wake-up latency
  negligible against a chunk-sized copy).
- Sequential cross-stream use is safe; simultaneous use of one buffer by two
  streams is not — documented, not enforced. PyTorch's per-stream event set
  deferred until we own a host allocator.

## D5 — Host views (zero-cost, fabricated `Vec`)

`TensorView` holds `&Vec<T>`, so the view's storage is
`DataRef::from_manually_drop(ManuallyDrop::new(Vec::from_raw_parts(ptr, len, len)))`
(or the `DataMut` twin) — exactly the `asarray(&[T])` construction; the Vec
is never dropped or resized. No op traits are implemented for
`DeviceCudaPinned` (forced: 142 core conveniences are `Raw = Vec<T>`-bound);
CPU compute happens only through these views.

```rust
pub trait TensorPinnedHostViewAPI<'a, T, D: DimAPI> {
    fn host_view(&'a self) -> TensorView<'a, T, DeviceCpuSerial, D>;
    fn host_view_mut(&'a mut self) -> TensorViewMut<'a, T, DeviceCpuSerial, D>;
    fn host_view_on<B>(&'a self, device: &B) -> TensorView<'a, T, B, D> where B: DeviceAPI<T, Raw = Vec<T>>;
}
impl<'a, R, T: Copy, D> TensorPinnedHostViewAPI<'a, T, D> for TensorAny<R, T, DeviceCudaPinned, D>
    where R: DataAPI<Data = PinnedBuffer<T>>;
```
Must be a local extension trait: `rstsr-cuda` cannot write inherent impls on
`TensorAny` (E0116, F17). Exported from the crate prelude.

## D6 — Transfer edges

- `DeviceChangeAPI` for `DeviceCudaPinned ↔ DeviceCuda`, **non-blocking by
  default**: enqueue on the `DeviceCuda` value's execution stream, wait/record
  the buffer event, return (PyTorch `non_blocking=True` semantics). Data is
  observable on the host only after an event sync, taken automatically by any
  host access.
- D2H source safety (the known foreign-stream read-lifetime hole): the
  transfer records an event on the exec stream and makes the source
  `CudaBuffer`'s allocation stream wait before returning — transfer-local fix.
- `into_device` from pinned = `to_device` + drop of the source; the drop
  syncs, so it is effectively blocking — documented; pipelines use the
  borrowing `to_device`.
- `DeviceCpuPinned ↔ DeviceCpuSerial/DeviceFaer`: copies (like
  `pin_memory()`); zero-copy only via `host_view*`.
- `DeviceToCpuAPI` for `DeviceCudaPinned` (→ `Vec`); `DeviceStorageAPI`
  `get_index`/`set_index`/`get_index_ptr` = host access after sync.
- Pageable `DeviceCpuSerial ↔ DeviceCuda` stays blocking (unchanged).

## D7 — Construction surface: inherent only, no creation traits

```rust
impl DeviceCudaPinned {
    new(ordinal) / new_f(ordinal);
    alloc_f<T: Copy>(&self, len) -> Result<PinnedBuffer<T>>;            // uninit
    zeros_f<T: Copy>(&self, len) -> Result<PinnedBuffer<T>>;            // host memset
    from_vec_f<T: Copy>(&self, vec: Vec<T>) -> Result<PinnedBuffer<T>>; // host memcpy
    tensor_f<T: Copy, D: DimAPI>(&self, buf, shape: D) -> Result<Tensor<T, Self, D>>;
}
impl<T> PinnedBuffer<T> { len, is_empty, into_vec_f(self) -> Result<Vec<T>>, sync(&mut self) -> Result<()> }
```
No `DeviceCreationAnyAPI`, no `asarray`, no `arange`/`linspace`. Fill,
compute, reshape happen through `host_view*`. (No edge references a creation
trait, F16.)

## D8 — Allocation flags and pooling

- Default flags `0` (`cudaHostAllocDefault`, PyTorch's choice) — not cudarc's
  `WRITECOMBINED`, because CPU compute through `host_view` is a goal.
  `PinnedBuffer::with_flags(len, flags)` for WC-vs-default benching of D2H sinks.
- No pool this round. `cudaHostAlloc` is ms-scale; CuPy/PyTorch pool by
  default with event-gated reuse. Internals stay private and event-carrying so
  a caching host allocator can be added later without API change; the bench
  preallocates. Revisit on a real many-short-lived-pinned-tensors workload.

## D9 — Stream and synchronize API (all four, each with a test)

- (a) `DeviceCuda::new_stream_with_ordinal(ordinal) -> Self`,
  `DeviceCuda::new_stream(&self) -> Self`; public
  `CudaStreamObj::new_f(&DeviceCuda) -> Result<Arc<CudaStreamObj>>`.
  Tests: two-stream producer/consumer without `set_stream`; the bench.
- (b) `synchronize` as local extension trait
  `CudaSyncAPI { synchronize_f, synchronize }` for `DeviceCuda` (exec stream),
  `DeviceCudaPinned` (no-op), `TensorAny<.., DeviceCuda, ..>` (exec stream,
  then buffer alloc stream), `TensorAny<.., DeviceCudaPinned, ..>` (buffer
  event). Tests: pipeline final wait; `to_device` + `synchronize` +
  `host_view` compare. Promote to `rstsr-core` when a second async backend
  lands.
- (c) Public `CudaEvent` wrapper (`record`, `wait`, `synchronize`, `query`).
  Tests: D6 source-ordering case; one stream's copy ordered behind another's
  kernel without tensors.
- (d) `DeviceCuda::with_stream(self, &Arc<CudaStreamObj>) -> Self` builder
  (immutable twin of `set_stream`; keeps flags/order). Test: the bench builds
  `dev_h/dev_c/dev_d` from one base value.

## D10 — Tests and bench

- Usual correctness tests, `rstsr-cuda/tests/test_pinned.rs` (Q23 list):
  round-trip; `host_view*` read/write reach the bytes and a CPU op lands in
  the buffer; pinned→CUDA→pinned on default and side stream; async `to_device`
  + immediate `host_view`; foreign-stream temporary source ×100 with pool
  reuse; 3-stream pipeline bitwise equals serial; non-contiguous layout;
  empty buffer; drop with in-flight D2H.
- Bench (prototype + numbers in `rstsr-improve-trajectory/2026-11-23-
  pinned-streaming-bench/`, then ported to `rstsr-cuda/benches/
  bench_streaming.rs` in the `bench_elementwise.rs` style): modes `h2d`,
  `compute`, `d2h`, `serial_all` (today's pageable blocking path),
  `pipeline` (pinned + async + 3 streams), controls `pipeline_blocking`
  (explicit `DeviceCuda::synchronize()` per transfer — no blocking entry
  points added), `pipeline_1stream`. Interleaved rounds, medians, assert
  `pipeline < 0.7 × serial_all`, controls do not improve, pipeline equals
  `serial_all` bitwise; report ratios only.

## Deferred with intent

Core `DeviceSyncAPI`; per-stream event sets; caching host allocator; WC
default; `host_view` on rayon devices beyond the generic `host_view_on`;
general kernel-level foreign-stream read-lifetime fix; DLPack `kDLCUDAHost`.
