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
