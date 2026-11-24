# STATUS — implementation B1–B5 (2026-11-24)

Executed against `PLAN.md` on rstsr branch `261021/cuda` (base `6d17d04`).
**Nothing is committed in rstsr** (owner instruction); `git status` there
shows 4 modified + 7 new files, all under `crates-device/rstsr-cuda/`.

## What landed

| Batch | Files | Gate |
|---|---|---|
| B1 | `src/pinned.rs` (`PinnedBuffer<T>`, `pinned_flags`), `src/device_pinned.rs` (`DeviceCudaPinned`, storage traits, `alloc_f`/`zeros_f`/`full_f`/`from_vec_f`/`tensor_f`) | unit behaviour covered by `tests/test_pinned.rs` |
| B2 | `src/pinned_view.rs` (`TensorPinnedHostViewAPI`, `TensorPinnedHostViewMutAPI`; fabricated `ManuallyDrop<Vec<T>>`) | `pinned_host_views_read_write_and_cpu_ops`, `pinned_faer_host_view_on` |
| B3 | `transfer.rs` (`h2d_pinned_async`, `d2h_pinned_async` with two-way source ordering), `src/conversion_pinned.rs` (pinned↔CUDA async; pinned↔CPU copies; faer variants) | round-trips, foreign-stream temporary ×100, non-contiguous, empty, in-flight drop |
| B4 | `src/event.rs` (`CudaEvent`), `src/sync.rs` (`CudaSyncAPI`), `stream.rs` (`CudaStreamObj::new_f`), `device.rs` (`new_stream_with_ordinal[_f]`, `new_stream[_f]`, `with_stream`), `funcs::synchronize_tensor` | `stream_api_without_set_stream`, `event_orders_copy_behind_foreign_kernel`, `synchronize_readiness` |
| B5 | `tests/test_pinned.rs` (12 tests) + crate doc (`lib.rs`) | see below |

Total: ~1 270 new lines. Exports: `prelude_dev::{DeviceCudaPinned, PinnedBuffer,
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
   + a cudarc `CudaEvent`) instead of wrapping `PinnedHostSlice<u8>`:
   cudarc's `as_slice` requires `ValidAsZeroBits` and forms a reference to
   possibly-uninitialized memory; its pointer is private; its event flag is
   not adjustable. Same protocol (wait → copy → record, sync on host access,
   sync on drop), `CU_EVENT_BLOCKING_SYNC` as decided.
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
