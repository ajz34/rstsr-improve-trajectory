# Array indexing (fancy indexing) — outcome (2026-10-08)

Closes the grilling of `GRILL-INIT-PROMPT.md` (rounds 1-3, `DECISIONS.md`),
tracked as gap **G-039** / checklist **C6b**. Implementation lives in the rstsr
checkout, branch **`261009/array-indexing`** (based on `89556bf`, the `#126`
boolean-mask merge); not pushed, no PR.

## Commits (rstsr)

- `d53128f` — core: `ArrayIndexer` / `ArrayIndexArgs`, `rt::array_index` /
  `array_index_f` (+ `TensorAny` methods), `DeviceArrayIndexAPI` and the serial
  kernel; the `Layout::dim_narrow` negative-step fix; core tests.
- `0e789b1` — rstsr-faer-py: integer-array keys route through `rt::array_index`.
- `97456a1` — docs (full-tier `array_index` docstring + doc_draft twin), tracking
  rows (numpy/doc coverage, sync surface, numpy-differences registry).

## What was built

- NumPy **vectorized indexing** for ungrouped keys: basic indexers mixed with
  integer index arrays, index arrays mutually broadcast (trailing-aligned), and
  the **placement rule** for the broadcast dimensions (in place for a contiguous
  advanced run, front otherwise; integers count as advanced for the grouping).
- `array_index` returns `TensorCow`: a view when no index array is present,
  an owned gather otherwise. The device op is layout-generic (arbitrary strides)
  and the output layout is constructed directly — no transpose copy or view.
- Out of scope (registered): grouped ("parenthesized") index tuples, boolean
  index arrays mixed into a tuple (a lone bool array stays `mask_select`), and
  advanced-key assignment.

## Deviations from the grilled plan (flagged)

1. **Input type**: `impl TryInto<AxesIndex<ArrayIndexer<B>>>` is impossible —
   `From`/`TryFrom` impls targeting `AxesIndex<..>` cannot be written outside
   `rstsr-common` (orphan rule E0117). Replaced by the local argument type
   `ArrayIndexArgs<B>` (the house `*Args` idiom, `RepeatArgs`-shaped); the
   `AxesIndex` form is still accepted through `TryFrom`.
2. **Trait name**: `DeviceArrayIndexAPI` (the index-op family convention), not
   the INIT prompt's `OpArrayIndexAPI` — confirmed in round 3, Q6.
3. **Basic-slicing fix**: `Layout::dim_narrow`'s `step < 0` branch mishandled
   explicit negative bounds (`a[4:-1:-1]` reversed instead of emptying,
   `a[-4::-1]` returned one element). Fixed to Python's slicing rules; recorded
   in `numpy_differences_resolved.md`.
4. **Col-major tests**: deferred with the col-major entry binary (still a stub);
   the intended divergence is recorded in `numpy_differences.md`.

## Verification

- rstsr-core row-major suite: **543 passed / 0 failed** (entry_row_cpu).
- Doctests: **253 passed / 0 failed** (`cargo test -p rstsr-core --doc`).
- `cargo clippy` clean (default and `faer` feature, all targets).
- NumPy differential: 1000 NumPy-generated cases (`gen_array_index_cases.py`)
  agree on shape and values; 4000-case runs were used during development.
- array-api conformance (rstsr_faer.api @ suite `6c0b59f`): **1216 passed /
  84 failed / 82 skipped** — from 1212/88/82, i.e. exactly the four
  `test_getitem_arrays_and_ints_{1,2}[{1,None}]` nodes, no regressions
  (`test_array_object.py`: 27 passed).

## Code review (max effort) and its outcome

A max-effort review of `89556bf..80231bc` fanned out over ten angles and returned
15 findings; all are addressed in `80231bc` (or were already correct and are now
pinned by a test):

**Correctness / build**
1. The five BLAS device crates (`rstsr-{openblas,mkl,blis,aocl,kml}`) symlink
   `rayon_auto_impl/mod.rs` and refused to compile (E0583) — the module symlink
   was missing in each; added.
2. *Real semantic bug*: a **zero-width ellipsis** (every axis consumed
   explicitly) was dropped from the expanded index, so it no longer separated
   the advanced indexers around it: `x[None, [0,1,2], ..., 2]` on a (4,5) tensor
   returned `(1,3)` where NumPy gives `(3,1)`. Fixed (the ellipsis is kept as a
   separator) and pinned by `custom_array_index::test_zero_width_ellipsis`; the
   generator now emits zero-width ellipses (it structurally could not before).
3. "Too many indices" is now an `IndexError` on both the array path and the
   basic-slicing delegation (the Python exception kind at the faer-py boundary).
4. The in-src unit tests failed under the CI **col-major** job; they now pin
   `RowMajor`.
5. `to_indexers` in the shim had a reachable `unreachable!()` for array keys
   (via the setitem entry points); it now returns a `TypeError`.
6. `array_index` now rejects index arrays on another device (`DeviceMismatch`).

**Perf** — the kernel precomputes per-bulk (source delta, output offset) and
per-base tables; the inner loop is two adds per element instead of recomputing
the base unravel, the output multi-index and every indexer offset.

**Tests / docs (policy MUSTs)** — NumPy assertion comments and `IndexError`-kind
pins in the transferred tests, value assertions in the doc_draft twin,
`TestMultiIndexingAutomated::test_1d` tracked (`partial`, covered by the
generated differential), the resolved-difference note corrected to the real
reproducers (`a[4:-1:-1]`, `a[-6::-1]` — `a[-4::-1]` never diverged), the
`# See also` "Similar function" subsection, verbatim variant titles, and an
`asarray`-style Overloads table.

Re-verified after the fixes: 544 row-major suite tests, 253 doctests, col-major
lib tests 143, rustfmt/clippy/rustdoc clean, 4000-case NumPy differential,
array-api conformance still **1216 / 84 / 82**.

## Second code review (max effort, 2026-10-09) and its outcome

A fresh max-effort pass over the branch (finders + a ~100k-case randomized NumPy
differential probe) found **no reachable functional bug beyond those fixed in
`80231bc`** and returned three amendments, all addressed in `853279d`:

1. *Residual in the in-flight slicing fix*: `dim_narrow`'s `step > 0` branch
   evaluated emptiness before the stop was clamped, so `start == len_prev < stop`
   (`rev[2:5]` on a negative-stride axis) still produced a wrapped offset; the
   length is now computed after clamping (mirroring the `step < 0` branch) and
   pinned by `custom_indexing::test_negative_step_and_empty_slices` (including the
   offset itself).
2. *Lazy index validation*: an empty broadcast never touches the index arrays, so
   NumPy returns an empty result instead of raising for values that are never
   gathered (`a[np.array([], int), np.array([99])]`); resolution and bounds
   checking now happen only when the broadcast is non-empty.
3. Test/doc/contract gaps: the NumPy fixture + Python generator moved out of
   rstsr-core (its `CONTEXT.md` forbids external fixtures and Python regen steps in
   core) and the cases are inlined as `NUMPY_CASES`; new strided-source,
   `()`/`None` and laziness tests; `rstsr-faer-py`'s getitem rides one path so the
   error kind is `IndexError` for both spellings, and `__setitem__` marshals a 0-d
   integer-array key; `DeviceArrayIndexAPI` documents the
   write-every-element-exactly-once obligation; the repeated-ellipsis and
   `AxesIndex::None` panics are documented and raise `IndexError`; the stale module
   docs and the `# Panics` list are complete; the col-major divergence is
   registered (`col-major-transfer`) and noted on `order_semantics.md`.

Known, deliberately unfixed (perf-only, recorded as follow-ups): the
`From<&TensorAny>` index conversion copies the index tensor (the public
`ArrayIndexer<B>` carries an owned tensor by design), and the tensor tier still
walks the indexers itself rather than reusing `Layout::dim_slice` (it must keep a
zero-width ellipsis, which `dim_slice` elides).

Re-verified after `853279d`: 547 row-major suite tests, 253 doctests, col-major lib
143, rstsr-common tests, fmt/clippy/rustdoc clean, a 4000-case NumPy differential,
and array-api conformance still **1216 / 84 / 82**.

## Follow-ups (not done)

- boolean index arrays mixed into a tuple; advanced-key `setitem` (scatter);
- a rayon kernel for the gather (the rayon device currently delegates to serial);
- col-major divergence tests, once `entry_col_cpu` exists;
- perf pass on the kernel (per-element multi-index recomputation).

## Manual review round (`REVIEW-R1-PROMPT.md`, 2026-10-09)

The maintainer's manual review asked for the device boundary to carry the
resolved index entries in **device storage** (`ArrayAuxIndexer.indices:
&DeviceRawAPI<usize>::Raw`, any layout allowed) instead of host slices, for the
device op to take the **device default order**, and for the tensor tier to move
data into device storage rather than extracting index tensors to host slices.
It also asked to try both column-major implementations (revert the index layouts
to row-major vs. a column-major iterator) and keep one.

Implemented and verified — the response is `REVIEW-R1-RESPONSE.md`; the
superseded boundary line in `DECISIONS.md` was updated. Headlines:

- `ArrayAuxIndexer<'a, B>` + `order: FlagOrder`; entries resolved element by
  element through `Storage::get_index` (no raw-slice reads, no `Raw = Vec<..>`
  pin) and moved into device storage with `outof_cpu_vec`.
- Both strategies agree; the layout-generic one (no copy, entries addressed
  through their layouts, broadcast dims visited in the device order) is kept. A
  naive axis reversal of the index layouts is *not* equivalent — it flips the
  trailing alignment for index arrays whose rank is below the broadcast rank;
  that case is now a regression test.
- New in-src order tests (they run in both CI order jobs), split by the question
  they answer: `test_array_index_order_invariance` (the two orders gather the
  same logical result — arrangement-blind comparisons, mixed-rank broadcast, the
  same logical multi-dimensional index arrays on both devices, and the
  flat-listing construction trap) and `test_array_index_order_arrangement` (the
  arrangement differs exactly for results of rank ≥ 2; a 1-D result is
  identical; both devices checked against NumPy's `ravel()` /
  `ravel(order='F')`).

Re-verified: 547 row-major suite tests, 253 doctests, lib 145 (default) / 144
(col-major), 44 common tests, fmt/clippy/rustdoc clean, fresh 4000-case NumPy
differential, array-api **1216 / 84 / 82** (unchanged; the four
`test_getitem_arrays_and_ints_*` nodes still pass).
