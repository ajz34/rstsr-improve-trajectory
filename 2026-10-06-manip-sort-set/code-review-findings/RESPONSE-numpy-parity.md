# Response — NumPy test-parity review (2026-10-07)

Second review round on branch `261006/manip-sort-set`, scope **NumPy test parity
only** (rstsr-core tests + tracking vs `main`). The review found 16 issues (all
independently confirmed); fixes landed in the main repo `/home/a/rstsr_pack/rstsr`
(working tree, **uncommitted** — no auto-commit in rstsr). Each NumPy behavior
claim was re-probed first-hand on numpy 2.5.1 + `~/Git-Others/numpy` v2.5.2
before acting.

## Gates after the fix

- `entry_row_cpu`: **499 passed** (was 462 → +37), 0 failed.
- lib / allocatable_dtype / tensor_sum: 114 / 10 / 2, all pass.
- doctests: 242 pass (default features). The two known failures are
  pre-existing and unrelated: `op_binary_arithmetic.rs` (linker SIGBUS on this
  machine) and `creation.rs` `DeviceFaer` (only under `--no-default-features`).
- clippy `--all-targets`: 0 warnings; `cargo fmt --check`: clean.
- `sync_numpy.py ~/Git-Others/numpy`: **0 MISSING** (oracle resolves every
  tracked surface entry).

## Dispositions

| # | finding | action |
|---|---|---|
| 1 | CSV/SURFACE gap for the whole wave (sort/argsort/searchsorted/nonzero/unique/isin/take_along_axis; also TestRepeat + 3 TestTile methods) | Added **46 CSV rows** (`numpy_coverage.csv` 204→250) and extended `sync_numpy.py` SURFACE with the sort/searching/set/indexing/diff families + TestRepeat + all 4 TestTile methods. Oracle now MISSING-free. |
| 2 | `diff` had no parity/custom/doc_draft test and no CSV row; scalar-side divergence unregistered | New `core_func/manipulation/test_diff.rs` (TestDiff: basic/axis/nd/n/prepend/append + 2 custom edges) and `doc_draft/manipulation/test_diff.rs`; wired both `mod.rs`; 6 CSV rows; new `numpy_differences.md` entry for the no-scalar-expansion / no-`bool` divergences. |
| 3 | `unique`/`isin`/`nonzero`/`take_along_axis` written as `custom_*` though NumPy classes exist | Split into NumPy-cited `numpy_unique` / `numpy_isin` / `numpy_nonzero` / `numpy_take_along_axis` modules (provenance headers + CSV rows), keeping the `custom_*` edge cases separate. |
| 4 | `test_nonzero_0d_errors` claimed a nonexistent divergence | Replaced by `numpy_nonzero::test_nonzero_zerodim`: NumPy **raises** for 0-d too (`TestNonzero::test_nonzero_zerodim`, L1651) — it is parity, not a deviation. |
| 5 | `unique_*` differences entry + docstring misstated NumPy | Rewrote the entry and the `unique_values` docstring: `np.unique` sorts/collapses NaNs, the array-api aliases pass `equal_nan=False` and are unordered since 2.3; rstsr targets the aliases. |
| 6 | `take_along_axis` docstring claimed a phantom NaN/broadcast deviation | Docstring corrected — rstsr broadcasts outside `axis` (NumPy + array-api parity). |
| 7 | `tile` docstring claimed NumPy skips copying all-ones | Removed the false "deviation"; NumPy copies too (gh4679 fix). |
| 8 | `numpy_differences.md` cited a nonexistent `test_sort.py::TestSortComplex` | Re-pointed to `_core/tests/test_multiarray.py::TestMethods::test_sort_complex`. |
| 9 | `searchsorted` header cited nonexistent class `TestNumeric` | Corrected to `TestNonarrayArgs::test_searchsorted`. |
| 10 | sort/argsort descending + stable NumPy tests untranslated | Added `numpy_sort_descending` / `numpy_argsort_descending` (signed/unsigned/floats/stable-duplicates, size-0). |
| 11 | `searchsorted` n-elements/1-element/all-equal, resetting, sorter case 1 missing | Added `test_searchsorted_n_elements`, `test_searchsorted_resetting`, and sorter case 1 (b.searchsorted(k) == a.searchsorted(k, sorter)). |
| 12 | `take_along_axis` tensor-side size-1 broadcast + empty-middle-axis missing | Added `TestTakeAlongAxis::{test_broadcast,test_empty}` (a=(3,4,1)×idx=(1,2,5)→(3,2,5)). |
| 13 | `unique` signed-zero/complex assertions vacuous | Signed-zero now asserts `is_sign_negative()`; added `test_unique_1d`, `test_unique_complex_signed_zeros`, and a strengthened complex first-occurrence value check. |
| 14 | `argsort_custom` comparator made every comparison Equal | Scrambled input + `x % 4` key so the comparator determines the order; added an explicit within-row check. |
| 15 | `test_rank_promotion_up` missing `specify_test!` and misfiled in `numpy_tile` | Added the macro and moved it to `custom_tile` (no NumPy test for rank promotion). |

Also normalized every wave provenance header to the `// numpy: v2.5.2 | <path>::<Class>::<method> (L<n>)`
form (test_repeat/roll/tile/sort/argsort/searchsorted).

## Notes

- The reviewer's unrelated-doctest observation (`op_binary_arithmetic.rs` link
  bus error) was reproduced here; it touches no file in this branch.
- The `test_unique_signed_zero_merge` sign assertion now pins the first-seen
  encoding (`-0.0` first stays negative), matching `np.unique([-0., 1., 0.])`.
- No rstsr commit made (standing no-auto-commit rule); the change set is the
  working tree of `/home/a/rstsr_pack/rstsr`.
