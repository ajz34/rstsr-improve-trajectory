# Review round 2 + fixes — rstsr branch `261021/cuda` @ `ee70fe4`

Second review of the CUDA kernel layer, run 2026-10-23 with
`/code-review high` against the branch through **`ee70fe4`** ("rstsr-cuda:
address kernel-layer review findings"), followed by an independent
re-evaluation and an empirical re-check on the GPU before any fix was
accepted.

- Review target: commits `cdadd12..ee70fe4` on `261021/cuda`.
- Re-evaluation: every finding was checked against the code; the
  consequential ones were reproduced on the RTX 5070 Laptop (driver
  595.91.07) with a throwaway harness outside the repo.
- Fixes: **`b44cbef`** on `261023/cuda-review-fixes` (branched from
  `ee70fe4`), applied only after the re-evaluation below.
- No rstsr commit was made before the owner approved the fixes.

Severity tags as in `REVIEW-6b56c18.md`: **[BUG]**, **[RACE]**,
**[PERF]**, **[API]**, **[NIT]**.

---

## Re-evaluation of the 8 findings

| # | Finding | Verdict | Outcome |
|---|---|---|---|
| 1 | Copy/cast kernels index the source by the *output* rank/shape | **Confirmed** | Reproduced exactly; fixed |
| 2 | Non-POD dtypes storable and reconstructed from device bytes | **Confirmed** | Fixed by POD gating |
| 3 | Write targets not ordered against their allocation stream | **Confirmed** | Reproduced; fixed, wider than reported |
| 4 | `memcpy_fast` accepts an f-contig pair on a row-major device | **Confirmed** | Reproduced exactly; fixed |
| 5 | CUDA→CUDA `to_device` deep-clones then copies | **Confirmed** | Fixed |
| 6 | Format walker is mirrored by hand | **Partially valid** | Mirror exact today; left open |
| 7 | `make_key` re-assembles/hashes source every launch | **Partially valid** | Real but small; left open |
| 8 | `get_index_ptr` returns a device address under a host-pointer API | **Valid (latent)** | Contract documented |

The review missed one bug, found during re-evaluation and more common
than several of the eight (see "Missed bug" below).

## Confirmed findings and fixes

### 1. [BUG] Rank/shape-changing assigns read out of bounds

`copy_same_dtype`/`copy_cast` decomposed each thread's linear id by the
output shape and applied the source strides at that same rank. Valid only
when both layouts have identical shapes; `OpAssignArbitaryAPI` explicitly
allows different shapes/ranks.

Reproduced (`arange(12).into_shape([3,4]).swapaxes(0,1).reshape([2,6])`):

```
CUDA: [0,4,8,0,0,0,1,5,9,0,0,0]
CPU : [0,4,8,1,5,9,2,6,10,3,7,11]
```

Source offsets reached 21 for a 12-element buffer (zeros = OOB reads).

Fix: both copy kernels take **both** shapes plus an order flag; each side
decomposes by its own shape in the device's default order
(`rstsr_iter_offset`). This reproduces the CPU devices' positional-zip
semantics for rank/shape-changing assigns.

### 4. [BUG] `memcpy_fast` ignores the device's default order

The gate accepted `(c_contig && c_contig) || (f_contig && f_contig)`
regardless of `default_order`, and regardless of shape equality.
Reproduced (`swapaxes(0,1).reshape([12])`): CUDA `[0..11]`, CPU
`[0,4,8,1,...]`.

Fix: the fast path now requires same dtype (see missed bug), same shape,
contiguity in the **device's** order, full buffers, zero offsets.
Note the two findings interlock: an order-only gate fix would have sent
this case into the broken copy kernel, so both were fixed together.

### Missed bug — equal-size cross-dtype assigns byte-copied

`assign_impl` took `memcpy_fast` whenever `size_of::<TC>() ==
size_of::<TA>()`, *before* checking dtype identity, so `i64→f64`,
`i32↔f32`, `bool↔i8` etc. were raw byte copies:

```
i64 arange(8) -> f64 zeros: CUDA [0, 5e-324, 1e-323, ...], CPU [0,1,2,...]
```

The existing `assign_cast_pairs` test only used different-size pairs, so
it never reached the path. Fixed by adding `TypeId` equality to the gate;
cross-dtype assigns always go through the cast kernel.

### 3. [RACE] Write targets not ordered against host reads

Every host read synchronizes only the buffer's **allocation** stream,
while kernels run on the device value's **execution** stream. Inputs were
ordered with events, but buffers an op writes into were not.

Reproduced with the documented `set_stream` + `to_device` re-tag flow
(40-iteration loop): 4089/4096 elements stale at iteration 22.

The review scoped this to in-place unary and re-tagged assign/fill; it is
in fact every write into a pre-existing buffer (copy, cast, fill,
in-place unary, and the `dtod_full` source). Fix:

- `order_input(target, exec)` before the launch (covers a still-pending
  allocation and prior writes),
- new `order_output(target, exec)` after the launch: record on the
  execution stream, make the allocation stream wait, so host reads on
  the allocation stream observe the write.

Out-of-place ops were already correct (their output is allocated on the
execution stream), which is why the existing stream tests passed.

### 2. [BUG] Non-POD storage is unsound

Byte-erased storage plus unbounded `DeviceAPI<T>`/`DeviceChangeAPI` meant
`Tensor<String, DeviceCuda>` was constructible from safe code and read
back with `ptr::read` from freed host pointers (double free).

Fix: `CudaDType: Copy`; `DeviceStorageAPI`, `DeviceAPI`,
`DeviceCreationAnyAPI`, `DeviceToCpuAPI` and all `DeviceChangeAPI` impls
are `T: Copy`-bounded; transfer helpers too. The old "CPU-device
precedent" comment was wrong: the CPU device stores a real `Vec<T>`, CUDA
stores bytes. Verified the String conversion no longer compiles.

### 5. [PERF] `to_device` doubled the copy

`to_device` deep-cloned the buffer and then `dtod_full`ed the clone.
Fix: copy directly from the source buffer (with event ordering). Also:
same-ordinal conversions now relabel for order/flags/stream-only
differences, as `change_device` already did; only cross-ordinal transfers
allocate and copy.

### 8. [API] `get_index_ptr` contract undocumented

CPU impls return host pointers, the CUDA impl a device address; the
shared trait said nothing. Documented the backend-dependent contract on
`DeviceStorageAPI` rather than changing the signature.

## Left open

- **#6 (walker mirror)**: `prefix_axis_visited`/`last_axis_visited` still
  mirror `print_vec_with_layout_dfs_inner`; the mirror is exact today,
  but the `expect` in `format_with` would panic if it ever drifts. A fix
  needs the native-impl walker to report visited indices — its own
  reviewed change.
- **#7 (launch-path host cost)**: per-launch source assembly + SipHash of
  the full source is real (≈0.5–1 µs vs a ~2–5 µs enqueue), but the
  review's "comparable to kernel time" claim was overstated, and a
  previous `make_key` memo was deliberately deleted in `ee70fe4`. Needs a
  proper interning design (cheap discriminator key), not a re-add.

## Verification of `b44cbef`

New regressions in `test_kernels.rs`:
`assign_reshape_of_strided_view`, `assign_reshape_flatten_of_strided_view`,
`assign_shape_change_on_col_major_device`, `assign_same_size_cast_pairs`,
`inplace_after_stream_retag` (40 cross-stream iterations).

- `test_kernels` 18 passed; `test_device_basic` 13 passed; lib 3 passed.
- `cargo clippy -p rstsr-cuda --all-targets` clean; `--features faer`
  check clean.
- The four previously failing reproductions now pass; sin comparisons use
  `1e-12` (CUDA vs Rust libm differ by 1 ULP, not a staleness signal).

## Still to do

- Re-run `bench_elementwise` vs cupy on the fixed tree — the copy kernel
  now carries a second shape array and an order flag, and the assign fast
  path is narrower; `to_contig(T)` was the only cell at the 2× gate
  (2.0×) before this round.
- Decide whether `261023/cuda-review-fixes` is folded into `261021/cuda`
  (fast-forward) or kept as a review branch.
