# Response to the manual review (`REVIEW-R1-PROMPT.md`)

Agent-authored response to the maintainer's manual review of the array-indexing
device boundary and the row/column-major handling. All items are addressed in
the rstsr branch `261009/array-indexing` (see the commits after `853279d`).

## 1. `ArrayAuxIndexer` carries device-level entries

The aux struct is now device-generic and holds the resolved entries in device
storage, in the sketched shape:

```rust
pub struct ArrayAuxIndexer<'a, B>
where
    B: DeviceRawAPI<usize>,
{
    pub src_axis: usize,
    /// Resolved index entries, in device storage.
    pub indices: &'a <B as DeviceRawAPI<usize>>::Raw,
    /// Layout of the index array's own shape.
    pub layout: Layout<IxD>,
}
```

`DeviceArrayIndexAPI<T>` gained the matching bound
(`Self: DeviceAPI<T> + DeviceRawAPI<usize> + DeviceRawAPI<MaybeUninit<T>>`) and
is documented as layout-generic: a device reads the entries **through**
`layout` (arbitrary strides and offset) and never assumes a host slice or a
C-contiguous buffer; any layout that represents the same index array gives the
same result.

The old host-level `indices: &'a [usize]` (and the `Raw = Vec<..>` pinning in
the tensor-tier bounds) is gone. For the CPU devices the adapter is the only
place that turns the raw into a slice (`ix.indices.as_slice()`), which is
exactly the statement "this device's `usize` raw is host memory".

## 2. Tensor tier: no extraction to host slices, data moved into device storage

`array_indexing.rs` no longer reads index tensors through the raw buffer.
Per array indexer:

- the raw (unresolved) carrier is kept in a private `RawIndex<B>` enum
  (`Host(Vec<isize>)` for `OneDimIndex`, `Device(Tensor<isize, B, IxD>)` for
  index tensors);
- entries are resolved **element by element through the device storage**
  (`Storage::get_index`), visited in the device default order — no `raw()[..]`
  indexing, no host slice, no `Raw = Vec<isize>` pin;
- the resolved `usize` entries are moved into device storage with
  `DeviceCreationAnyAPI::outof_cpu_vec` (a move for CPU-like devices, an upload
  for devices with their own memory) and the layout handed to the device op is
  the canonical contiguous layout **in the device default order**
  (`shape.new_contig(None, order)`), so fill order and layout always agree;
- a zero-dimensional integer tensor is lowered to a scalar `Select` using
  `get_index(layout.offset())`.

Laziness is preserved and re-checked against NumPy (2.5.1): an empty broadcast
never touches the index arrays (no bounds error), while an empty *subspace* is
still validated (`np.zeros((0, 3))[:, np.array([7, 8, 9])]` does raise).

## 3. `order: FlagOrder` in the device op, and the two column-major strategies

`DeviceArrayIndexAPI::array_index` now takes `order` (the device default order)
as its last parameter, and the tensor tier passes `device.default_order()`. The
kernel visits the broadcast dimensions in that order; the values it produces do
not depend on it (the arrangement is carried by `lc`), so this is a
traversal/locality choice.

Both strategies offered in the review were implemented and compared:

- **(1) "revert the index layout axes order to row-major, apply row-major
  indexing"** — implemented as a device-side canonicalization: copy the entries
  through the layout in row-major coordinate order into a C-contiguous buffer
  and run the kernel with a row-major enumeration;
- **(2) "use a column-major indexing iterator"** — the shipped path: address the
  entries through their own layouts (no copy) and visit the broadcast dimensions
  in the device order, enumerating each index array's entries in that same
  order.

**Result: they agree, and (2) is kept.** (2) does no copy and keeps the
layout-generality contract literal, so the kernel cannot silently depend on a
canonical buffer.

A trap worth recording: the *literal* axis-reversal reading of (1) — mirroring
`ArrayAuxIndexer.layout` with `Layout::reverse_axes` — is **not** sound in
general. Mirroring flips the trailing alignment of the index arrays against the
broadcast dimensions, which only coincides with the correct pairing when every
index array has the broadcast rank. The discriminating case (an index array of
rank 2 and one of rank 1 broadcasting to a rank-2 bulk) is now a regression
test.

## 4. Column-major behaviour today

- values and shape: order-independent (NumPy-parity; hand-computed,
  NumPy-cross-checked expectations in `test_array_index_order_equivalence`);
- arrangement: follows the device order, unchanged from before (C-contiguous on
  a row-major device, F-contiguous on a column-major one);
- new: the index arrays are read, resolved and traversed in the device order,
  and the op receives `order`;
- row-major behaviour is byte-for-byte what it was (fresh 4000-case NumPy
  differential, full test batteries below).

## Verification after the rework

- 547 row-major entry-binary tests, 253 doctests;
- lib tests: 145 (default features) / 144 (col-major features, the CI job);
  44 `rstsr-common` tests;
- `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D
  warnings` clean (this also re-checks the five BLAS device crates that symlink
  the rayon auto-impl module, and `rstsr-faer-py`);
- fresh 4000-case NumPy differential (new seed) passing;
- array-api conformance: `test_array_object.py` 27/27 (the four
  `test_getitem_arrays_and_ints_*` nodes stay green) and a full-suite re-run;
- the two-strategy comparison above (strategy 1 run once, then reverted).
