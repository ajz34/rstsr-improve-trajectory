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
  NumPy-cross-checked expectations in `test_array_index_order_invariance`);
- arrangement: follows the device order, unchanged from before (C-contiguous on
  a row-major device, F-contiguous on a column-major one);
- new: the index arrays are read, resolved and traversed in the device order,
  and the op receives `order`;
- row-major behaviour is byte-for-byte what it was (fresh 4000-case NumPy
  differential, full test batteries below).

## 5. Follow-up audit (the col-major framing)

Reviewing my own framing after the reviewer's reminder that **only the
row-major default order is held to NumPy/the standard** — a column-major device
is allowed (and registered) to diverge — two tracking statements were wrong and
are fixed:

- `order_semantics.md` lumped `mask_select` in with the "order-independent"
  gathers. It is not: its selection *sequence* is the mask visit order, i.e.
  device-dependent, so on a column-major device even a 1-D result carries the
  column-major sequence (its own Row/Column Major Notice always said so). The
  page now separates the coordinate-described gathers (`index_select`,
  `take_along_axis`, `array_index` — values/shape order-independent) from the
  visit-sequence ones (`mask_select`).
- the `col-major-transfer` entry in `numpy_differences.md` claimed "shape and
  element values never differ; only the memory arrangement does", which is true
  for `array_index` but false for `mask_select`; both are now stated separately,
  with the policy spelled out: NumPy/array-API parity holds under the row-major
  default order, the column-major device-order convention is the registered
  transfer.

No *behaviour* was found to be wrongly deferred under this label: the
order-independence notices on `array_index` and the sole-index boolean path hold
in the row-major case (differential + conformance + the order-equivalence test).

## 6. Column-major behaviour is now tested (not just documented)

Follow-up ask: make tests that validate the column-major claim at the fancy
indexing axis, and restructure them around a sharper question — when do the two
device orders actually differ?

**The bounded claim** (what the restructured tests encode): for the same logical
inputs the two orders always gather the same *logical* result (shape + values at
every position); what differs is the *arrangement*, and that differs exactly
when the result has two or more dimensions. A 1-D result is identical on either
device. An n-dimensional (n > 1) index array always broadcasts to an
n-dimensional bulk and so always forces a multi-dimensional result — hence
always lands in the difference — but **the index arrays' rank is not the
criterion**: a 1-D index array only has to meet a slice (or another 1-D array
across a slice) to produce a multi-dimensional result that also differs.

Two in-src tests (both run in the row-major and the col-major CI unit-test jobs):

- `test_array_index_order_invariance` — the "same logical result" half.
  Arrangement-blind comparisons (element reads via `.i(...)`, plain flattenings
  for 1-D results) across the two devices: a 1-D gather consuming every axis
  (`a[1, [0,1,2], [2,0,1]]`), a slice keeping a base dimension
  (`a[[1,0], .., 1]`), a mixed-rank trailing-aligned broadcast
  (`a[1, [[2],[0]], [3,0,1,2]]`), and the *same logical* multi-dimensional index
  arrays on both devices (a host listing is read in the device order, so the
  column-major device is handed the F-order listing). It also documents the
  construction trap: the *same* flat listing on the other order is the
  transposed pair of index arrays, and the gather faithfully returns other
  elements — construction, not gather.
- `test_array_index_order_arrangement` — the "what differs" half. Each device is
  compared against NumPy's flattening in the corresponding order:
  1-D result ⇒ identical strides and sequence (`a[1, [0,1,2], [2,0,1]]`);
  3-D result from a *1-D* index array plus slices ⇒ C-contiguous vs
  F-contiguous strides and `ravel()` vs `ravel(order='F')` (`a[:, :, [0,1]]`);
  two 1-D index arrays separated by a slice (`a[[0,1], :, [2,0]]` — the sharpest
  case, since no multi-dimensional index array is involved: row-major strides
  `[3, 1]` with buffer `[2, 6, 10, 12, 16, 20]`, column-major strides `[1, 2]`
  with buffer `[2, 12, 6, 16, 10, 20]` = NumPy's `ravel()` / `ravel(order='F')`
  of the same `[[2, 6, 10], [12, 16, 20]]`);
  multi-dimensional index arrays with the broadcast dimension in the middle
  (`consec = 1`, `a[0:2, [0,1], [1,0], :]` — the fancy dimension carries
  stride 2 under column-major) and leading (`consec = 0`,
  `a[[0,1], :, [1,0], :]` — the fancy axis is the fastest varying one).

Both test bodies carry the NumPy expressions their constants come from, and both
record that the kernel's *internal* visit order is a locality choice that cannot
show up in the result (each output position is written from exactly one source
position): the arrangement is what carries the device order.

Also: the `col-major-transfer` entry pointed at the `doc_draft` twin for the
array-indexing order caveat, which only pins RowMajor — the pointer now names
`test_array_index_order_arrangement`.

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
