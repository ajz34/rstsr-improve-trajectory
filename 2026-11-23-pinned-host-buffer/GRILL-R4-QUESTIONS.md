# GRILL R4 — construction surface, stream/synchronize API, test list

## Agent-authored background: answers received on screen (transcribed)

- **Q14**: no creation traits needed (especially no `arange`/`linspace`);
  `DeviceCudaPinned` is "storage of pinned memory, additionally transform it
  to tensor view/mut; other usage is not required, unless necessary".
- **Q15**: okay, try that.
- **Q16, Q18, Q19**: okay.
- **Q17**: "if you can find useful test cases, may try allow all to be
  implemented. Also `DeviceCuda::new_stream_with_ordinal(ordinal)` and
  `DeviceCuda::new_stream(&self)` probably better to my feel."

### Fact (agent-sourced)

- F16. `DeviceChangeAPI<'l, BOut, R, T, D>` and `TensorDeviceChangeAPI` are
  bounded by `DeviceRawAPI` only (`storage/conversion.rs:3-21`,
  `tensor/device_conversion.rs:14-99`); `TensorAny::new_f` needs
  `B: DeviceAPI<T>`, whose only extra is `DeviceStorageAPI` (`len`,
  `get_index*`, `set_index`) — already agreed in Q9. No creation trait is
  referenced by any edge we plan, so dropping them costs nothing.
  Existing `DeviceCuda` stream API: `stream()`, `set_stream(Arc<CudaStreamObj>)`
  / `set_stream_f` (panics on foreign ordinal), `default_stream()`,
  `synchronize()` (`rstsr-cuda/src/device.rs:65-118`).

---

❓ **Q20** - **Minimal construction surface (no creation traits)**: only
inherent constructors, no `DeviceCreationAnyAPI`, no `asarray`:

```rust
impl DeviceCudaPinned {
    pub fn new(ordinal: usize) -> Self;                 // panicking, like DeviceCuda::new
    pub fn new_f(ordinal: usize) -> Result<Self>;
    pub fn alloc_f<T: Copy>(&self, len: usize) -> Result<PinnedBuffer<T>>;   // uninit
    pub fn zeros_f<T: Copy>(&self, len: usize) -> Result<PinnedBuffer<T>>;   // memset 0
    pub fn from_vec_f<T: Copy>(&self, vec: Vec<T>) -> Result<PinnedBuffer<T>>; // copy
    pub fn tensor_f<T: Copy, D: DimAPI>(&self, buf: PinnedBuffer<T>, shape: D) -> Result<Tensor<T, Self, D>>;
}
impl<T> PinnedBuffer<T> { pub fn len; pub fn is_empty; pub fn into_vec_f(self) -> Result<Vec<T>>; pub fn sync(&mut self) -> Result<()>; }
```

`zeros` is a plain `cuMemsetD8`-free host memset after alloc; `from_vec` a
host memcpy. Everything else (fill, compute, reshape) is done through
`host_view*` on a `DeviceCpuSerial` view.

➡️ Yes.

---

❓ **Q21** - **Stream API (Q17), with your names**: implement all four with
the test cases that justify them:
(a) `DeviceCuda::new_stream_with_ordinal(ordinal) -> Self` and
`DeviceCuda::new_stream(&self) -> Self` (clone on a fresh stream, same
ordinal/flags/order); `CudaStreamObj::new_f(&DeviceCuda) -> Result<Arc<..>>`
public. Test: `set_stream`-free two-stream producer/consumer; the bench.
(b) `Tensor::synchronize()` — see Q22. Test: pipeline final wait; a
`to_device(&pinned)` followed by `synchronize()` then `host_view` compare.
(c) `CudaEvent` wrapper (`CudaEventObj::record(stream)`, `wait(stream)`,
`synchronize`, `query`) public. Test: the Q15(a) source-ordering case is the
first consumer; a unit test orders one stream's copy behind another's kernel
without tensors.
(d) `DeviceCuda::with_stream(self, &Arc<CudaStreamObj>) -> Self` builder
(immutable twin of `set_stream`, keeps flags/order). Test: build `dev_h`,
`dev_c`, `dev_d` from one base value in the bench without `&mut`.

➡️ Yes: all four, each with the test named.

---

❓ **Q22** - **Shape of `Tensor::synchronize()`**: a small public trait in
`rstsr-cuda`, `DeviceSyncAPI { fn sync_storage<R: DataAPI>(&self, raw: &R::Data) -> Result<()>; }`
implemented for `DeviceCuda` (exec stream, then the buffer's allocation
stream) and `DeviceCudaPinned` (buffer event), with an inherent
`TensorAny<R, T, B, D>::synchronize(&self) -> Result<()>` for
`B: DeviceSyncAPI`. Not a core trait yet (core has no notion of async
devices); promote to `rstsr-core` when a second async backend appears.

➡️ Yes.

---

❓ **Q23** - **Correctness test list (usual tests, `rstsr-cuda/tests/
test_pinned.rs`)**: (1) `from_vec` round-trip `into_vec`; (2) `host_view` /
`host_view_mut` read and write propagate to the pinned bytes, and a CPU op on
the view (`rt::add` into it) lands in the buffer; (3) pinned → CUDA → pinned
round-trip equals the source, on both default and a side stream;
(4) async `to_device` immediately followed by `host_view` (implicit sync)
gives correct data; (5) the Q15(a) ordering case (source temporary on a
foreign stream, repeated 100× with pool reuse); (6) pipeline (3 streams)
bitwise equals serial; (7) non-contiguous pinned layout transfers byte-equal;
(8) empty buffer; (9) drop of a buffer with an in-flight D2H does not
corrupt (event sync). Bench correctness: pipeline vs `serial_all` bitwise
inside the bench too.

➡️ Yes.
