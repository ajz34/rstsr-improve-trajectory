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

## Follow-ups (not done)

- boolean index arrays mixed into a tuple; advanced-key `setitem` (scatter);
- a rayon kernel for the gather (the rayon device currently delegates to serial);
- col-major divergence tests, once `entry_col_cpu` exists;
- perf pass on the kernel (per-element multi-index recomputation).
