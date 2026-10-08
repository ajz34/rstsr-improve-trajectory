# Grill Round 3 — array indexing (fancy indexing) (2026-10-08)

Agent-authored. Answers go in `GRILL-R3-ANSWERS.md` (user-owned) or on screen.

## Round-2 answers (background, agent-transcribed from screen — not authoritative)

- **Q1:** agreed — no transpose copy **and** no transpose view; compute the
  output shape, construct the output layout + storage directly. **New
  constraint: the op must be *layout-generic*** — do not assume row/col
  contiguity for source, output, or index arrays. If a transpose is ever
  needed, `into_transpose` (layout-only) on the owned result suffices
  (`TensorCow::Owned`).
- **Q2:** agreement (follow NumPy) **plus**: do **not** give a 0-d integer array
  a `From` path into `ArrayIndexer::ArrayIndex` — 0-d integer arrays take the
  scalar/`Select` route.
- **Q3/4/5:** agreed — input `impl TryInto<AxesIndex<ArrayIndexer<B>>>`; trait
  `DeviceArrayIndexAPI`; error taxonomy (`IndexError` for OOB /
  broadcast-mismatch / too-many-indexers, `AxisError` for axis errors).
- **Q6:** agreed — exact NumPy-parity test mechanics from the skills/memory
  (below).
- **Q7:** delivery = branch **`261009/array-indexing`** (already checked out in
  `/home/a/rstsr_pack/rstsr` at `89556bf`, based on `#126` merged); **auto-commit
  allowed; no push, no PR**. getitem only this pass.

## Facts now pinned (agent-gathered; no decision needed)

- Test home: **`rstsr-core/tests/core_func/indexing/test_array_index.rs`** (that
  dir already exists: `test_indexing.rs`, `test_take_along_axis.rs`); register in
  `core_func/indexing/mod.rs`. Doc twin:
  `tests/doc_draft/indexing/test_array_index_doc.rs`. Parity ref **NumPy v2.5.2**;
  provenance header `// numpy: v2.5.2 | <path>::<Class>::<method> (L<line>)`
  must byte-match a `tests/tracking/numpy_coverage.csv` row; add the row.
- Parity bodies pin `device.set_default_order(RowMajor)` per test fn.
- `array_index` has **no NumPy public counterpart** — the test module must be
  `numpy_array_index` only if it cites real `test_indexing.py` cases, else
  `custom_array_index`.
- Run: `cargo test -p rstsr-core --test entry_row_cpu --features "backtrace row_major" --no-default-features -- numpy_array_index -- --nocapture`.

## Round-3 frontier

❓ **Q1 — confirm the placement algorithm and its edge cases.**

Proposed (NumPy `mapiter_fill_info` / `_get_transpose` translated): parse the
tuple to a per-source-axis list; then

- `fancy_ndim` = max ndim over *array* indexers (0-d int arrays contribute 0);
- walk the tuple tracking `curr_dim` (source axes consumed) and `result_dim`
  (base output dims emitted — one per `Slice`, one per `Insert`, 0 per
  `Select`); an entry is **"advanced"** if it is an array indexer **or a
  `Select`/0-d-int-array** (NumPy counts scalars as advanced for grouping);
- `consec` = `result_dim` at the first advanced entry; if a *basic*
  slice/ellipsis/newaxis intervenes and a later advanced entry appears →
  `consec = 0`;
- `base_shape` = subspace dims in tuple order; `broadcast_shape` = mutually
  broadcast index shapes (trailing-aligned); output shape =
  `base_shape[..consec] ++ broadcast_shape ++ base_shape[consec..]`;
- edge cases: `Ellipsis` expands to remaining full slices (repeated → error);
  empty tuple → identity (view); all-basic tuple → view; empty index arrays →
  valid empty output.

➡️ Recommend: **as written** (this is the NumPy rule). Flag any case you want
handled differently (esp. `Insert`/`NewAxis` interaction with `consec`, and
whether `Insert` between two array indexers should break contiguity).

❓ **Q2 — the device kernel: layout-generic gather, and reuse vs new kernel.**

The op must be stride-generic (Round-2 Q1). Decomposition of the gather: the
output's axes are `[bulk (broadcast) dims at consec .. consec+fancy_ndim]` then
the subspace axes; each output multi-index splits into a bulk sub-index and a
subspace sub-index. The kernel reads, for each advanced source axis, its index
array element at the bulk sub-index (via that array's own strides), and offsets
the source by `src_stride * resolved_index`; non-advanced source axes come from
the subspace sub-index. `take_along_axis` is the `fancy_ndim`=1, same-rank,
single-axis special case.

Options: **(a)** a fresh `array_index_cpu_serial` stride-generic kernel (serial)
+ `array_index_cpu_rayon`; **(b)** decompose at the tensor tier into the
existing `index_select`/`take_along_axis` (multiple device calls — rejected,
loses placement + is O(axes) calls).

➡️ Recommend **(a)** a fresh layout-generic kernel, with the device trait
`DeviceArrayIndexAPI` taking `(&mut out_raw, &Layout<IxD>, &in_raw, &Layout<DA>,
indexers: &[(usize src_axis, &[usize], &Layout<IxD>)])` (indices resolved to
`usize` at the tensor tier). Rayon: serial first, parallel kernel as a
follow-up if the gather is embarrassingly parallel over the output space.

❓ **Q3 — module/type layout: `array_indexer.rs`, `array_indexing.rs`, device aux
type, and 0-d/single-array ergonomics.**

Proposed split (your Q7 naming): `rstsr-core/src/tensor/array_indexer.rs` — the
public `ArrayIndexer<B>` enum + all `From`/`TryFrom`; `rstsr-core/src/tensor/
array_indexing.rs` — `array_index`/`array_index_f` + methods + the lowering
(uint tensor/form → `(src_axis, Vec<usize>, Layout<IxD>)`). Device trait extends
`rstsr-core/src/operators/adv_indexing.rs`. Open: where the device-side aux type
(`ArrayAuxIndexer<V>` from INIT, or just the tuple) lives, and the ergonomics —
does `array_index(x, arr)` (single array, no tuple) resolve, can a 0-d `Tensor`
be passed as a scalar index, and does a `Tensor<isize>` index require
`Dim = IxD` exactly or any rank?

➡️ Recommend: `ArrayIndexer<B>` in `tensor/array_indexer.rs`; device trait +
aux type in `operators/adv_indexing.rs`; lower everything at the tensor tier to
resolved `usize` + contiguous index layouts (so the device never sees a tensor).
Single-array form resolves via tuple-of-one. A 0-d integer `Tensor` lowers to
`Basic(Select)` (no `From`→`ArrayIndex`); any-rank integer tensors accepted and
lowered to `ArrayIndex`. Confirm the file split and the aux-type home.

❓ **Q4 — col-major divergence tests vs the deferred col-major test infra.**

You asked for tests showing the col-major divergence is "reasonable, not
buggy". But the current tree has **no `entry_col_cpu.rs`, no `tests/col_major/`
and no `col_major` `[[test]]` gate**, and skill `core-col-major-transfer` is an
explicit STUB ("Design deferred — do not create `tests/col_major/` contents
yet"). So there is nowhere to run col-major tests today. Options: **(a)** defer
the col-major divergence tests with the col-major infra (record intended
semantics in `numpy_differences.md` + the order-semantics doc for now); **(b)**
wire a minimal `entry_col_cpu` + `tests/col_major/` in *this* PR (a separate
infra task, contradicts the stub); **(c)** author them order-neutral in
`core_func/` (they would run under row-major and assert the row behavior — does
not actually exercise col-major).

➡️ Recommend **(a) defer**: land row-major `array_index` now, and record the
expected col-major divergence as documentation + a `col-major-transfer`-tagged
`numpy_differences.md` entry, to be turned into real tests when the col-major
entry binary lands. Flag if you want (b) as part of this PR.

❓ **Q5 — scope of the NumPy parity subset.**

Candidate sources in `test_indexing.py` (v2.5.2): `TestIndexing`
(`test_single_int_index`, `test_empty_fancy_index`, `test_broaderrors_indexing`,
`test_nontuple_ndindex`), `TestBroadcastedAssignments` (broadcast rules; the
assignment side is deferred, the get side applies), `TestMultiIndexingAutomated`
(`_get_multi_index` semantics — the placement rule executable spec), the
`basics.indexing.rst` placement examples (`x[..., ind, :]`, `x[:, ind1, ind2]`
vs `x[:, ind1, :, ind2]`), duplicate indices, negative indices, dtype
preservation.

➡️ Recommend: transfer the **get-side subset**: `test_single_int_index`,
`test_empty_fancy_index`, `test_broaderrors_indexing`, representative
`TestMultiIndexingAutomated` cases, and the doc placement examples as
`numpy_array_index` cases; add hand-authored `custom_array_index` cases for
mixed slice+array placement, duplicates, negatives, 0-d handling, and stride
(non-contiguous views). Setitem/broadcasted-assignment cases → registered
`todo`/`partial` in `numpy_differences.md`.

❓ **Q6 — trait-name micro-confirm + public surface registration.**

Your Round-2 Q1 message said "`OpArrayIndexAPI`", but Round-2 Q4 agreed on
`DeviceArrayIndexAPI`. Confirm which. And confirm the public registration: free
fns `array_index`/`array_index_f` → `prelude::rstsr_funcs` → `rt::`; the
`ArrayIndexer<B>` type → `prelude::rstsr_structs`/traits; docstrings per
`api-doc-conventions` (Overload table incl. the 0-d/single-array forms).

➡️ Recommend: **`DeviceArrayIndexAPI`** (matches the index-op family), public
surface as `rt::array_index` + `TensorAny::{array_index, array_index_f}` with
`ArrayIndexer<B>` re-exported; the INIT's `OpArrayIndexAPI` name is superseded.
