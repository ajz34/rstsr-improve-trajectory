# Array indexing (fancy indexing) — decisions (2026-10-08)

Consolidated, closed design after grilling rounds 1-3. Supersedes the open
questions in `GRILL-INIT-PROMPT.md`. Gaps: rstsr gap **G-039** / checklist
**C6b**. Reference: NumPy **v2.5.2**; array-api spec 2025.12 (`ff497ed8`);
array-api-tests `6c0b59f9`.

## Naming & scope

- **"array indexing"** = fancy indexing (integer-array indices). **"advanced
  indexing"** = the broader mask/bool/integer family.
- Semantics target: **NumPy vectorized indexing** incl. the **placement rule**,
  row-major parity. The array-api reduced subset (all-int tuples, mutual
  broadcast, zipped gather) is the special case.
- **This pass: getitem only.** Advanced-key `setitem` (scatter) deferred.
- **Boolean indexers:** a lone bool array routes to existing `mask_select`;
  bool *mixed into a tuple* deferred (and may be skipped if too difficult).
- **0-d integer arrays:** advanced for the placement computation, but lowered as
  `Basic(Select)`; **no `From` path into `ArrayIndexer::ArrayIndex`**.

## Public API (tensor tier)

- Free fns `array_index` / `array_index_f` + `TensorAny::{array_index,
  array_index_f}`. `i`/`slice`/`[]` **untouched** (`i` stays view-only).
- Input type: `impl TryInto<AxesIndex<ArrayIndexer<B>>, Error: Into<Error>>`.
  Single-array form resolves (`array_index(x, arr)` == `array_index(x, (arr,))`).
  No separate `*Args` struct — the tuple is the group.
- Return: **`TensorCow<'_, T, B, IxD>`**. Borrowed view iff, after lowering, no
  non-trivial array indexer remains (pure basic tuple → delegate to `i`);
  owned otherwise.
- `From`/`TryFrom` on `ArrayIndexer<B>` for `Vec`/`&Vec`/`&[T]`/`[T; N]` with
  `T ∈ {isize, usize, i64, u64, i32}` and `Tensor`/`TensorView`/
  `&TensorAny<isize, B, IxD>`. A 0-d integer `Tensor` lowers to `Select`
  (no `From`→`ArrayIndex`).
- File split: `rstsr-core/src/tensor/array_indexer.rs` (public enum +
  conversions); `rstsr-core/src/tensor/array_indexing.rs` (`array_index*` +
  methods + lowering). Device trait + device aux type in
  `rstsr-core/src/operators/adv_indexing.rs`.
- Registration: fns → `prelude::rstsr_funcs` → `rt::`; `ArrayIndexer<B>` →
  prelude structs/traits; docstrings + Overload table per `api-doc-conventions`.

## Device tier (3-tier house pattern)

- Trait **`DeviceArrayIndexAPI`** (matches `Device{IndexSelect,TakeAlongAxis,
  MaskIndex}API`; the INIT's `OpArrayIndexAPI` name is superseded). The
  codebase's `Op*API`/`Device*API` inconsistency is noted to fix later.
- **Layout-generic**: no row/col-contiguity assumption for source, output, or
  index arrays (arbitrary strides/offset).
- Boundary (final, after the manual review — see `REVIEW-R1-RESPONSE.md`):
  `(&mut out_raw, &Layout<IxD>, &in_raw, &Layout<DA>, &Layout<IxD> base,
  indexers: &[ArrayAuxIndexer<{src_axis, indices: &DeviceRawAPI<usize>::Raw,
  layout: Layout<IxD>}>], consec, order: FlagOrder)`. All indices are resolved to
  `usize` and validated at the tensor tier; the entries live in **device
  storage** and are read through their layout (the review's INIT-prompt tuple
  `(axis, V data, Layout<IxD>)`, with `V` the device raw). The tensor tier keeps
  host carriers as `Vec<isize>` and index tensors as tensors until the broadcast
  shape is known, resolves through `Storage::get_index` in the device order, and
  moves the resolved entries into device storage with `outof_cpu_vec`.
  (Superseded: the first implementation passed `&[usize]` host slices and pinned
  `DeviceAPI<isize, Raw = Vec<isize>>`.)
- **Default-order aware**: the op takes the device default order; the index
  arrays are resolved and the broadcast dimensions visited in that order, and
  the result arrangement follows it. Values and shape stay order-independent.
- Kernel: **fresh `array_index_cpu_serial`** (serial); rayon kernel as a
  follow-up. `take_along_axis` is the `fancy_ndim = 1`, same-rank special case.
- **No transpose copy and no transpose view**: compute the output shape and
  construct the output layout + storage directly (bulk dims already at
  `consec`). If a transpose is ever needed, `into_transpose` (layout-only) on
  the owned result — still `TensorCow::Owned`.

## Placement algorithm (NumPy `mapiter_fill_info` / `_get_transpose`)

- `fancy_ndim` = max ndim over *array* indexers (0-d int arrays contribute 0).
- Walk the tuple tracking `curr_dim` (source axes consumed) and `result_dim`
  (base output dims emitted: 1 per `Slice`, 1 per `Insert`, 0 per `Select`). An
  entry is **advanced** iff it is an array indexer **or** a `Select`/0-d-int
  array (NumPy counts scalars as advanced for grouping).
- `consec` = `result_dim` at the first advanced entry; reset to `0` if a basic
  slice/ellipsis/newaxis intervenes and a later advanced entry appears.
- `base_shape` = subspace dims in tuple order; `broadcast_shape` = mutually
  broadcast index shapes (trailing-aligned). Output shape =
  `base_shape[..consec] ++ broadcast_shape ++ base_shape[consec..]`.
- Edges: repeated `Ellipsis` → error; empty tuple → identity (view); all-basic
  tuple → view; empty index arrays → valid empty output.

- **Column-major placement** (decided 2026-10-09, `5a42771`; see
  `PLAN-col-major-placement.md`): the placement rule is measured in the device's
  access order — a run of advanced indexers that other indexers displace goes to
  the **back** under `ColMajor` instead of the front (the broadcast block keeps
  its contiguity role: the most-strided axis), while a run that stays together
  keeps its subscript position in both orders. Row-major is unchanged; 1-D
  results are identical; the shape differs between the orders exactly when the
  advanced indexers are apart. `mask_select` needs no change (its count axis
  replaces the leading axes, i.e. the leading-together configuration).

## Errors

- `IndexError`: out-of-bounds integer index, non-broadcastable index arrays,
  too many indexers (NumPy-parity messages where cheap, e.g. "shape mismatch:
  indexing arrays could not be broadcast together with shapes …").
- `AxisError` (structured): axis-related errors.
- `DeviceMismatch`: index tensors not on the input's device.
- No artificial advanced-indexer cap unless a real device limit exists.

## Tests (agent-decided per round-3 Q5: comprehensive, honest)

- Parity: `rstsr-core/tests/core_func/indexing/test_array_index.rs` (register in
  `core_func/indexing/mod.rs`); doc twin
  `tests/doc_draft/indexing/test_array_index_doc.rs`. Provenance header
  `// numpy: v2.5.2 | <path>::<Class>::<method> (L<line>)`, byte-matching a new
  `tests/tracking/numpy_coverage.csv` row; module `numpy_array_index` (cited
  cases) + `custom_array_index` (hand-authored edges); parity bodies pin
  `device.set_default_order(RowMajor)`.
- Coverage: NumPy get-side subset — `test_single_int_index`,
  `test_empty_fancy_index`, `test_broaderrors_indexing`, `test_nontuple_ndindex`,
  representative `TestMultiIndexingAutomated`, the `basics.indexing.rst`
  placement examples — plus `custom_*` for mixed slice+array placement,
  duplicates, negatives, 0-d handling, non-contiguous strides, empty index,
  dtype preservation. Be comprehensive; register any genuine shortfall
  (framework/construction limits) in `tracking/numpy_differences.md`, honestly.
- **Col-major divergence tests deferred** with the col-major test infra
  (`entry_col_cpu` / `tests/col_major/` do not exist; skill
  `core-col-major-transfer` is a STUB). Record the intended col-major
  divergence now as a `col-major-transfer`-tagged `numpy_differences.md` entry
  + order-semantics doc note; promote to tests when the col entry binary lands.
- rstsr-faer-py: extend `parse_key`/`api.py::__getitem__` to accept `Array` keys
  → `rt::array_index`; `__setitem__` with advanced keys deferred. `capabilities()`
  unchanged (no test reads a relevant flag).

## Delivery

- Branch **`261009/array-indexing`**, checked out in `/home/a/rstsr_pack/rstsr`
  at `89556bf` (based on `#126` merged). **Auto-commit allowed; no push, no PR.**
- Work in stages; commit per stage with the `git-commit-coauthor` trailers.
