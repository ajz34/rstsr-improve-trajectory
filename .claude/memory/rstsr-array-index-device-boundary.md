---
name: rstsr-array-index-device-boundary
description: "rstsr array_index device boundary — resolved index entries live in device storage, the op is layout-generic and takes the device default order"
metadata:
  type: project
---

`rt::array_index` (2026-10-08-array-indexing, commit `1a23cc0`, manual review in
that task dir's `REVIEW-R1-RESPONSE.md`):

- `ArrayAuxIndexer<'a, B>` carries the **resolved** entries in device storage
  (`&<B as DeviceRawAPI<usize>>::Raw`) plus their `Layout<IxD>`; host-level
  `&[usize]` was rejected by the maintainer. The op is layout-generic: a device
  reads entries through the layout (any strides/offset), never assuming a
  row-major buffer — the tensor tier hands over contiguous buffers in the
  **device default order**, but that is not part of the contract.
- `DeviceArrayIndexAPI::array_index` takes `order: FlagOrder`; the kernel visits
  the broadcast dimensions in that order. **Placement rule** (2026-10-09,
  `5a42771`): NumPy's rule measured in the device's access order — a run of
  advanced indexers that other indexers *displace* goes to the **back** under
  `ColMajor` instead of the front (the broadcast block keeps its contiguity
  role: the most-strided axis), while a run that stays *together* keeps its
  subscript position. So the **shape** is order-dependent exactly when the
  indexers are apart; the arrangement (C- vs F-contiguous) always follows the
  device for rank ≥ 2; 1-D results are identical. Pinned by the `entry_row_cpu`
  integration module `core_func::indexing::test_array_index::device_order` —
  `test_array_index_order_arrangement` (placement + arrangement, against NumPy's
  `ravel()` / `ravel(order='F')`), `test_array_index_order_invariance` (together
  runs and 1-D values) and `test_array_index_order_displaced_deep_base`. (Moved
  there 2026-10-09 from the in-src unit module, which keeps only quick smoke
  tests; the other order-behavior tests of the suite live in `device_order`
  modules too.)
- **Boolean masks in a key** (2026-10-09, rstsr `60d3e18`): lowered at the tensor
  tier to their `nonzero` coordinates (one 1-D index array per mask axis, kept in
  device storage), so a mask consumes `ndim(mask)` axes and contributes one
  `(count,)` block — the placement rule transfers verbatim and the driver is
  untouched (masks arrive as ordinary index arrays; the new bound is
  `OpNonzeroAPI<bool, IxD>`). A zero-dimensional boolean stays declined (it would
  add a `{0, 1}`-sized block without consuming an axis). Under `ColMajor` a mask
  of rank ≥ 2 permutes its count axis (the device `nonzero` sequence) — values,
  not only arrangement; 1-D masks are order-independent.
- **`mask_select` needs no placement change**: its count axis replaces the
  leading axes (the leading-together configuration), so `(count, *trailing)` is
  kept. Its selection *sequence* still follows the mask visit order (device
  order) — a separate, registered sequence divergence, not a placement question.
- Do **not** generalize that order-independence to `mask_select`: its selection
  sequence *is* the mask visit order, so it follows the device (visible even in
  1-D results). Its Row/Column Major Notice says so; `order_semantics.md` and
  the `col-major-transfer` tracking entry once claimed otherwise and were fixed.
  Project policy: only the **row-major default order** is held to NumPy / the
  array API; column-major divergence is the registered `col-major-transfer`
  convention.
- Index tensors are resolved element by element through `Storage::get_index` (no
  `raw()[..]` reads, no `Raw = Vec<isize>` pin) and the resolved `usize` buffer
  goes back into device storage with `outof_cpu_vec`.
- When handling a column-major device, do **not** "revert" index layouts with
  `Layout::reverse_axes`: mirroring flips the trailing alignment of the index
  arrays against the broadcast dimensions whenever their rank differs from the
  broadcast rank. Canonicalizing (copying through the layout in row-major order)
  or addressing the entries through their layouts both work and agree.
- **Rayon advanced indexing is now parallel** (implemented 2026-10-09, rstsr
  working tree; design + numbers in
  `2026-10-08-array-indexing/FOLLOWUP-rayon-array-index.md`). `array_index`,
  `mask_select`, `mask_fill`, `take_along_axis` got rayon kernels in
  `rstsr-native-impl/src/cpu_rayon/` (`index_select` was already parallel;
  `bool_select` rides it). `DeviceRayonAutoImpl` (reached by `DeviceFaer` and
  the five BLAS crates) wires them via `get_current_pool()`.
  - Kernel speedups ~5-6x (1-D and 2-indexer gather); `mask_select`/`mask_fill`
    ~3x end-to-end. But **`array_index` is only ~1.5x end-to-end** because the
    *tensor tier* (resolve every index into device storage + allocate output)
    is serial and comparable in cost — the kernel is not the bottleneck for the
    public call. Parallelizing `tensor/array_indexing.rs`'s resolution is the
    next lever.
  - Two design gotchas that cost most of the speedup if missed: build the four
    offset tables **in parallel** (`for_each_init` scratch), and **never** flatten
    with a per-element `k / n_bulk` integer division (nest the smaller dim
    inside instead).
  - Correctness pinned by in-src `*_parallel_matches_serial` tests (rayon vs
    serial, sizes above the switch) and a scratch NumPy cross-check; no effect
    on the serialize/`device_order` semantics or the array-API numbers
    (1216/84/82).
