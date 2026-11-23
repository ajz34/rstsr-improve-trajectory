# PLAN — pinned host memory tensor type (`DeviceCudaPinned` + `PinnedBuffer<T>`)

Implements `DECISIONS.md` (D1–D10). Work happens in two places:

- **rstsr** `crates-device/rstsr-cuda`, on the existing branch `261021/cuda`
  (base `6d17d04`; new commits on top, no auto-commit — owner reviews).
- **rstsr-improve-trajectory** `2026-11-23-pinned-streaming-bench/` — the
  prototype crate + numbers, before the bench is ported.

Each batch ends with a gate; do not start the next batch before the previous
gate is green. Build/test invocation for the crate:
`cargo test -p rstsr-cuda --features faer -- --test-threads=1`
(symlinks/LD path per `rules/testing.md`; `TMPDIR=$HOME/.cache/tmp-cargo`).

## B1 — `PinnedBuffer<T>` and `DeviceCudaPinned` (D1–D3, D7, D8)

Files: `src/pinned.rs` (new), `src/device_pinned.rs` (new), `src/lib.rs`
(module list, `prelude_dev`, crate doc).

- `PinnedBuffer<T> { inner: PinnedHostSlice<u8>, len, _m: PhantomData<*const T> }`
  - `new_f(ctx: &Arc<CudaContext>, len) -> Result<Self>` via
    `alloc_pinned_with_flags::<u8>(bytes, 0)`; `with_flags_f(ctx, len, flags)`.
    Re-create the event with `CU_EVENT_BLOCKING_SYNC` — cudarc's
    `alloc_pinned_with_flags` already does this (F10), so wrap as-is.
  - Event protocol (D4): `wait(&stream)` / `record(&stream)` helpers; `sync()`
    = event synchronize; `Drop` is `PinnedHostSlice`'s (sync + `cuMemFreeHost`).
  - `len`, `is_empty`, `into_vec_f`, `as_slice_synced`/`as_mut_slice_synced`
    (`&[T]` after sync; `T: Copy`). Manual `Send`/`Sync` with safety comments
    (bytes touched only after the event sync; aliasing impossible from the
    owning repr). Deep `Clone` = alloc + host memcpy (needed by `Raw: Clone`
    on materializing paths, as `CudaBuffer`).
  - `uninit` twin: the tensor layer transmutes `Raw<T> ↔ Raw<MaybeUninit<T>>`;
    layout is T-independent so this needs no code beyond the `PhantomData`
    trick (same as D1 of device-basic).
- `DeviceCudaPinned { ordinal: Arc<CudaOrdinalObj>, base: DeviceCpuSerial }`
  - `new(ordinal)` / `new_f(ordinal)` / `Default` = `new(0)`;
    `ordinal()`, `context()`, `base()`.
  - `impl<T> DeviceRawAPI<T> { type Raw = PinnedBuffer<T> }`;
    `DeviceBaseAPI` delegates `default_order`/`set_default_order` to `base`,
    `same_device` = `Arc::ptr_eq(ordinal) && base.same_device`.
  - `DeviceStorageAPI<T>` for `T: Copy`: `len`, `get_index`, `get_index_ptr`,
    `get_index_mut_ptr`, `set_index` — each syncs the buffer event first;
    `DeviceToCpuAPI<T>` = sync + `to_vec`.
  - `DeviceAPI<T>` for `T: Copy` (matches `DeviceCuda`'s `T: Copy` impl).
- Inherent constructors (D7): `alloc_f`, `zeros_f` (host memset), `from_vec_f`
  (host memcpy), `tensor_f(buf, shape) -> Tensor<T, Self, D>` through
  `TensorAny::new_f(Storage::new(DataOwned::from(buf), dev), layout)`.

**Gate B1**: unit tests in `src/pinned.rs` — alloc/zeros/from_vec round-trip,
`into_vec_f`, `get_index/set_index` after sync, empty buffer, Drop with no
pending work. `cargo clippy -p rstsr-cuda` clean.

## B2 — Host views via extension trait (D5)

Files: `src/pinned_view.rs` (new), prelude export.

- `TensorPinnedHostViewAPI<'a, T, D>` with `host_view`, `host_view_mut`,
  `host_view_on<B>(&self, device: &B)` (and `_mut_on`), implemented for
  `TensorAny<R, T, DeviceCudaPinned, D>`, `R: DataAPI<Data = PinnedBuffer<T>>`
  (`DataMutAPI` for the mut forms). Body: sync event →
  `Vec::from_raw_parts(ptr, len, len)` in `ManuallyDrop` →
  `DataRef::from_manually_drop` / `DataMut::from_manually_drop` →
  `TensorAny::new_f(storage, layout.clone())`. Use `as_mut_ptr`-style
  provenance for the mut path (mirror `asarray(&mut [T])`, memory
  `rstsr-owned-iter-review`).
- Safety comment states the never-drop/never-resize contract and why the
  pinned owner outlives `'a`.

**Gate B2**: tests — view reads equal `into_vec`; writes through
`host_view_mut` land in the bytes; `rt::add(&a_view, &b_view)` result assigned
into a view mutates the buffer; a rayon (`DeviceFaer`, `--features faer`)
`host_view_on` works; miri is not applicable (CUDA), so a provenance review
by eye against `asarray` is the check.

## B3 — Async transfer edges + source-ordering fix (D6)

Files: `src/transfer.rs` (new async fns), `src/conversion.rs` (new impls),
`src/pinned.rs` (copy helpers).

- `transfer.rs`: `h2d_pinned_async(dev: &DeviceCuda, src: &PinnedBuffer<T>)
  -> Result<CudaBuffer<T>>` and `d2h_pinned_async(dev, src: &CudaBuffer<T>,
  dst: &mut PinnedBuffer<T>) -> Result<()>`: on `dev.stream()`; buffer event
  `wait` before, `record` after (via cudarc `HostSlice::stream_synced_slice`
  which already does exactly this for `PinnedHostSlice`).
- D2H source safety: `order_input_raw(src, exec)` before the copy (alloc →
  exec) **and** `order_output_raw(src, exec)` after (exec → alloc) so a
  dropped temporary is not reused mid-DMA; same for the H2D destination
  buffer (its alloc stream may differ from `dev.stream()` — the alloc is on
  `dev.stream()` so this is a no-op unless we allocate elsewhere).
- `DeviceChangeAPI` impls:
  - `DeviceChangeAPI<'l, DeviceCuda, R, T, D> for DeviceCudaPinned`:
    `change_device`/`into_device` consume the `PinnedBuffer`, copy, drop the
    source (documented blocking); `to_device` borrows, async.
  - `DeviceChangeAPI<'l, DeviceCudaPinned, R, T, D> for DeviceCuda`:
    allocates the destination on `self` (pinned), `d2h_pinned_async`, returns
    the pinned tensor (not host-ready until an event sync).
  - `DeviceChangeAPI` for `DeviceCpuPinned ↔ DeviceCpuSerial` (host memcpy)
    and `↔ DeviceFaer` under `feature = "faer"`.
  - `Repr`/`ReprTo` follow the existing CPU↔CUDA impls (`DataOwned` /
    `DataCow`).
- Existing pageable paths untouched.

**Gate B3**: tests — pinned→CUDA→pinned equals source (default stream);
side-stream device value (`new_stream`) same result; async `to_device` then
immediate `host_view` (implicit sync) correct; foreign-stream temporary
source ×100 with a tight device pool (the Q15(a) regression); non-contiguous
pinned layout transfers byte-equal; `into_device` blocking note in a doc
test. All existing `rstsr-cuda` tests still green.

## B4 — Stream, event, synchronize API (D9)

Files: `src/stream.rs`, `src/device.rs`, `src/event.rs` (new), `src/sync.rs`
(new).

- `CudaStreamObj::new_f(dev: &DeviceCuda) -> Result<Arc<Self>>` public
  (wraps the existing `pub(crate) new(ctx)`).
- `DeviceCuda::new_stream_with_ordinal(ordinal) -> Self`,
  `DeviceCuda::new_stream(&self) -> Self` (clone, fresh stream),
  `DeviceCuda::with_stream(self, stream: &Arc<CudaStreamObj>) -> Self`
  (immutable `set_stream` twin; panics on foreign ordinal like `set_stream`).
- `CudaEvent` thin wrapper over the cudarc `Event` already used in
  `kernel.rs`: `new_f(dev)`, `record(&stream)`, `wait(&stream)`,
  `synchronize`, `query`.
- `CudaSyncAPI { synchronize_f, synchronize }` for `DeviceCuda` (exec
  stream), `DeviceCudaPinned` (no-op), `TensorAny<.., DeviceCuda, ..>` (exec
  stream, then the buffer's allocation stream), `TensorAny<..,
  DeviceCudaPinned, ..>` (buffer event). `funcs::synchronize_tensor`
  free-function alias for discoverability.

**Gate B4**: tests — two-stream producer/consumer built only from
`new_stream`/`with_stream` (no `set_stream`); `CudaEvent` orders one
stream's copy behind another stream's kernel without tensors; `synchronize`
on a pinned tensor after async D2H makes `host_view` correct without an
implicit sync; `synchronize` on a CUDA tensor then `to_cpu_vec` equals a
fully synced run.

## B5 — Correctness test file (D10)

`tests/test_pinned.rs`: the nine Q23 cases, one test each, named
`pinned_*`. Also add the two B4 tests and the B3 regression there (unit tests
stay in-module; integration file is the owner-facing record).

**Gate B5**: `cargo test -p rstsr-cuda --features faer -- --test-threads=1`
green; `cargo test -p rstsr-cuda --no-default-features` (no faer) green;
`cargo doc -p rstsr-cuda` warning-free for the new items (docstrings per
`rules/api-doc-conventions.md`).

## B6 — Bench (D10, in this repo first)

Dir `2026-11-23-pinned-streaming-bench/` with a prototype crate depending on
`rstsr-cuda` by path (branch `261021/cuda` with B1–B5 applied):

- `benches/bench_streaming.rs` (plain `main`, `Instant`, CSV `mode,median_ms`,
  `RSTSR_BENCH_ITER/REPEAT`, same style as `bench_elementwise.rs`).
  Default 32 chunks × 8 MiB f64, 3 streams (`dev_h`, `dev_c`, `dev_d` from
  one base via `with_stream`), FMA-chain compute kernel with autotuned
  iteration count so compute ≈ 1.15 × (h2d + d2h), interleaved round-robin
  probes, per-probe medians, clock warm-up loop (laptop GPU clock hazard).
- Modes: `h2d`, `compute`, `d2h`, `serial_all` (pageable `to_device`/
  `to_cpu_vec`), `pipeline`, `pipeline_blocking` (explicit
  `synchronize()` per transfer), `pipeline_1stream`. Optional
  `pipeline_writecombined` via `with_flags` (D8 experiment).
- Assertions: pipeline bitwise equal to `serial_all`; `pipeline < 0.7 ×
  serial_all`; controls within 1.15× of `serial_all`. Output ratios only.
- README: rstsr commit, environment, numbers table (before/after = serial vs
  pipeline), negative controls, conclusion; `results/` gitignored if large.
- Port to `rstsr-cuda/benches/bench_streaming.rs` + `[[bench]]` entry after
  the prototype numbers are accepted.

**Gate B6**: assertions hold on the local RTX 5070; numbers recorded; README
written.

## Handoff

After B6: patch series on `261021/cuda` left **uncommitted or committed per
owner instruction**, PR only with explicit permission (`pr-writeup` skill);
bench report stays in this repo. Memory updates to
`.claude/memory/cuda-pinned-host-buffer-grill.md` after each batch with
measured facts and gotchas.

## Risks / watch-list

- Fabricated-`Vec` provenance on the mut path: mirror `asarray(&mut [T])`
  exactly (`as_mut_ptr`), never let the Vec reach `into_raw`/drop.
- cudarc `PinnedHostSlice::as_slice` requires `ValidAsZeroBits`; we only use
  its `HostSlice` impl and our own raw-pointer access under `T: Copy`.
- Stream-ordered `cuMemAllocAsync` pool exhaustion inside the loop: the bench
  preallocates device chunk buffers.
- `--all-features` workspace build: no new feature gates beyond `faer`.
