# STATUS — implementation B1–B6 (2026-11-24)

Executed against `PLAN.md` on rstsr branch `261021/cuda` (base `6d17d04`).
Committed on `261021/cuda` as `71d3d89` (2026-11-24, on the owner's instruction),
13 files under `crates-device/rstsr-cuda/`, +1413/-1.

## What landed (layout after the `pinned/` restructure of 2026-11-24)

```
crates-device/rstsr-cuda/src/
  pinned/mod.rs         module docs + re-exports (PinnedBuffer, DeviceCudaPinned, view traits)
  pinned/buffer.rs      PinnedBuffer<T>, pinned_flags, fill_host, copy_vec_into        (B1)
  pinned/device.rs      DeviceCudaPinned, storage traits, inherent constructors         (B1)
  pinned/view.rs        TensorPinnedHostViewAPI / ...MutAPI                              (B2)
  pinned/transfer.rs    h2d_pinned_async, d2h_pinned_async (non-blocking, two-way source ordering) (B3)
  pinned/conversion.rs  DeviceChangeAPI: pinned<->CUDA (async), pinned<->CPU/faer (copy) (B3)
  event.rs              CudaEvent                                                            (B4)
  sync.rs               CudaSyncAPI                                                           (B4)
  stream.rs / device.rs CudaStreamObj::new_f; DeviceCuda::{new_stream_with_ordinal[_f], new_stream[_f], with_stream} (B4)
  transfer.rs           pageable paths unchanged; only the two async fns moved out
  lib.rs                module list, prelude_dev, funcs::synchronize_tensor, crate docs
tests/test_pinned.rs    12 tests                                                              (B5)
```

| Batch | Gate tests in `tests/test_pinned.rs` |
|---|---|
| B1 | `pinned_buffer_roundtrip`, `empty_pinned_buffer`, `drop_with_in_flight_d2h_is_safe` |
| B2 | `pinned_host_views_read_write_and_cpu_ops`, `pinned_faer_host_view_on` |
| B3 | `pinned_cuda_roundtrip_default_and_side_stream`, `foreign_stream_temporary_source_ordered`, `noncontiguous_pinned_layout_transfers_byte_equal` |
| B4 | `stream_api_without_set_stream`, `event_orders_copy_behind_foreign_kernel`, `synchronize_readiness` |
| B5 | `pipeline_three_streams_bitwise_equals_serial` + the whole file |

Total: ~1 300 new lines. Exports: `prelude_dev::{DeviceCudaPinned, PinnedBuffer,
CudaEvent, TensorPinnedHostViewAPI, TensorPinnedHostViewMutAPI, CudaSyncAPI}`.

## Gates (all green on RTX 5070 Laptop, CUDA 13.2)

```
cargo test -p rstsr-cuda --features faer -- --test-threads=1
  lib 72, test_device_basic 13, test_kernels 54, test_matmul_tiled 5,
  test_ops_wave 26, test_pinned 12   -> 0 failed
cargo test -p rstsr-cuda --test test_pinned            (no faer) 11 passed
cargo test -p rstsr-cuda --doc                        ok
cargo clippy -p rstsr-cuda --all-targets [--features faer] -- -D warnings
  clean for rstsr-cuda (pre-existing rstsr-core faer lints are untouched)
RUSTDOCFLAGS="-D warnings" cargo doc -p rstsr-cuda --no-deps   clean
rustfmt: new files clean; modified files differ only in pre-existing hunks
```

## Deviations from PLAN.md / DECISIONS.md

1. **`PinnedBuffer` owns its allocation** (`result::malloc_host` / `free_host`
   + a cudarc `CudaEvent`) instead of wrapping `PinnedHostSlice<u8>` — see
   "Why not cudarc `PinnedHostSlice`" below. Same protocol as decided (wait →
   copy → record, sync on host access, sync on drop), `CU_EVENT_BLOCKING_SYNC`.
   Byte access for copies is through `&[MaybeUninit<u8>]` / `&mut [MaybeUninit<u8>]`
   (`uninit_bytes[_mut]`), so no reference to initialized data is ever formed
   over a fresh or DMA-target buffer; fills/copies use raw-pointer writes.
2. `zeros_f` is bounded by cudarc's `T: ValidAsZeroBits` (all-zero bytes must
   be a valid `T`), not by `Copy` alone.
3. `full_f(len, value)` added (one-line host fill, used by a test) — not in
   D7; drop it if unwanted.
4. `DeviceChangeAPI<DeviceCudaPinned> for DeviceCuda`: the trait methods have
   no `self`, so the D2H stream is the *source tensor's* device value
   (`tensor.storage().device()`), i.e. the stream the data was produced on.
5. `DeviceCuda::synchronize()` resolves to the pre-existing inherent method
   (same behaviour as the trait); `CudaSyncAPI` adds the tensor forms.
6. `into_vec_f(self)` does not consume (it cannot move out of `&mut self` of
   the drop path); it is a plain copy like `to_vec_f`, then the value drops.

## Why not cudarc `PinnedHostSlice` (owner question, 2026-11-24)

Checked against cudarc 0.19.10 `driver/safe/core.rs:1452-1590`:

- **Everything is private.** `ptr`, `len`, `event` are `pub(crate)`; the
  event is not exposed, so `CudaSyncAPI`/`PinnedBuffer::event()` and a
  later event-gated pool (D8) cannot be built on it. `Drop` is fixed
  (sync + `cuMemFreeHost`) — fine today, but not replaceable.
- **Every accessor synchronizes the host.** `as_ptr`, `as_mut_ptr`,
  `as_slice`, `as_mut_slice` all call `event.synchronize()` first. A pinned
  H2D source would therefore block the host on the previous use of the
  buffer before the copy is even enqueued; our `h2d_pinned_async` only
  makes the *stream* wait (`stream.wait(event)`), which is what keeps the
  pipeline non-blocking.
- **The one non-syncing path is `HostSlice::stream_synced_{,mut}_slice`**,
  an `unsafe` fn that does wait + enqueue-side record via `SyncOnDrop`. It
  covers the two copy directions, but returns `&[u8]`/`&mut [u8]`, i.e. a
  reference to initialized data — invalid for a fresh allocation (H2D
  source after `alloc`) or a D2H destination that is still uninitialized.
  cudarc gates the `as_*` family on `T: ValidAsZeroBits` for the same
  reason (only "all-zero bytes are valid" types), which still does not make
  a reference to *uninitialized* bytes defined behaviour.
- **No `Clone`, no fill, no `into_vec`** except through `as_slice()` (host
  sync + `ValidAsZeroBits`), and `alloc_pinned` is `unsafe` with `T:
  DeviceRepr` — fine for `u8`, but the wrapper gives us nothing on top of
  the two driver calls it makes (`malloc_host` + `new_event`).
- Net: ~40 lines of our own (`with_flags_f`, `Drop`, the two `wait`/`record`
  helpers) buy private-but-ours internals, non-blocking raw access, and the
  pool hook; `PinnedHostSlice` would have forced a host sync into every
  accessor and a reference-into-uninit into the async paths.

## B6 — bench (2026-11-24)

Prototype + numbers: `../2026-11-23-pinned-streaming-bench/` (README has the
table and the two issue-order findings). Result: pipeline 47.1 ms vs 106.9 ms
for the naive flow (**2.27×**), 91 % of the `max(compute, h2d+d2h)` bound,
89 % of the transfer time hidden; controls at the phase sum; bitwise equal.

It exposed one defect in the committed D2H edge: `order_output_raw(src)`
made the *source's allocation stream* wait for each copy, serializing every
later kernel on the compute stream (62.7 ms instead of 47.1). **Uncommitted
fix** in `crates-device/rstsr-cuda/src/pinned/transfer.rs`: the source
pointer is taken through cudarc `DevicePtr::device_ptr(slice, exec_stream)`
so a read event is recorded on the copy stream and honored at `CudaSlice::drop`
(free waits; the stream does not). `test_pinned` (12) green with the fix.
Port of the bench into `rstsr-cuda/benches/` awaits acceptance. Cross-check
against the CuPy prototype (`../2026-11-23-streaming-overlap/`): all CuPy
probes are covered except `serial_chunk` (== `serial_all` there, omitted) and
the event-timeline Gantt (not ported); one divergence — rstsr's pageable
pipeline still overlaps, so CuPy's `pipeline_pageable` control does not
transfer; see the bench README.

## Lessons for B6 (bench)

- **Keep every cross-stream device buffer alive until the final sync.** The
  first pipeline test dropped the H2D buffer at the end of its loop iteration;
  the compute stream's kernel then read a block the allocation stream had
  already returned to the pool, and chunk 3 received chunk 1's data
  (intermittently — one of two runs). This is the documented foreign-stream
  read-lifetime hole (`cuda-write-target-ordering`), not a pinned-edge bug;
  the pinned D2H edge orders its source both ways (`d2h_pinned_async`). The
  bench must hold H2D inputs and compute results in Vecs across the loop.
- `to_device` is the panicking form (`to_device_f` fallible) — no `.unwrap()`.
- `TensorAny::to_vec` needs creation traits the pinned device does not have;
  read pinned data through `raw().to_vec_f()` or a `host_view()`.
- Empty pinned buffers use a `T`-aligned dangling pointer (debug builds assert
  a non-null slice base); `Drop` skips `cuMemFreeHost` for them.

## Closing (2026-11-24)

Both pending rstsr changes committed on `261021/cuda`, splitting fix from
bench as `07e3d2e` (D2H drop-time read event) and `f11eeda`
(`benches/bench_streaming.rs` + `[[bench]]` entry, port of this task's
`src/main.rs` with a `bench_elementwise`-style header; `pipeline_pageable`
documented as informational). Verified in-crate: all assertions green,
pipeline 46.08 ms vs serial 105.38 ms (2.29x), 91.4% efficiency, 89.5% of
transfer hidden, controls at the 79.7 ms phase sum, WC == default, bitwise
equal. Note the bench's overlap assertions depend on the fix commit; the two
must land together on any clean checkout.
