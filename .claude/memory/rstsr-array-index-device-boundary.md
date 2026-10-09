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
  the broadcast dimensions in that order. Values/shape are order-independent;
  the result arrangement follows the device, and so does the flattened visit
  sequence (`to_vec()`); in-src `test_array_index_colmajor_iteration` pins this
  against NumPy's `ravel()` / `ravel(order='F')`.
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
