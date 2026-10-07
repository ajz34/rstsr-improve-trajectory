# SUMMARY — manip/sort/set wave (2026-10-06 → 2026-10-07)

Implementation of 17 array-API functions across manipulation, sorting, and
searching/set families, per DECISIONS.md (this directory; grill closed R4).
Work landed in the main repo `/home/a/rstsr_pack/rstsr`, branch
`261006/manip-sort-set` (base `e7cdc6a`), 12 commits, all gates green.
No push, no PR (wave grants).

## Functions delivered

| family | functions | architecture |
|---|---|---|
| manipulation | `repeat`, `roll`, `tile` | device-independent composition (uninit + assign_arbitary_uninit + assume_init) |
| sorting | `sort`, `argsort` (SortArgs), `sort_custom`, `argsort_custom` | 3-tier: OpSortAPI → DeviceCpuSerial/DeviceRayonAutoImpl → rstsr-native-impl kernels |
| searching | `searchsorted` (SearchArgs, side+sorter), `nonzero` | searchsorted 3-tier; nonzero two-pass count+fill, host-side coordinate split |
| set | `unique_values/counts/inverse/all`, `isin` | dual algorithm: naive (PartialEq) + TypeId-dispatched sorted fast path (ExtSortCmp) |
| indexing | `take_along_axis` | gather kernel; isize indices, negatives resolved tensor-side |
| misc | `diff` | device-independent composition; n passes of sliced subtraction via op_mutc_refa_refb |

Supporting traits: `ExtSortCmp` (rstsr-dtype-traits: NaN-last ordering,
Equal-on-== for ±0, complex part-wise lexicographic per NumPy numpy_tag.h),
`ExtZero` (nonzero membership for all dtypes), `AxisIndex<T>` appended to
rstsr-common.

## Commits (rstsr, branch 261006/manip-sort-set)

| commit | content |
|---|---|
| `9eed544` | stage 1: repeat/roll/tile (composition ops) |
| `a34dd1d` | stage 2a: ExtSortCmp; AxisIndex |
| `a4fa20c` | stage 2 review fix: complex ExtSortCmp NaN ordering |
| `05a5da4` | stage 3: sort/argsort (+custom), 3-tier |
| `0458dab` | stage 3 review fix: empty-rest guard, complex gate, docs |
| `359a1da` | stage 4: searchsorted 3-tier |
| `d6edec8` | stage 4 review fix: sorter validation, complex-NaN caveat, methods |
| `caf794a` | stage 5: unique_*, isin — dual naive + TypeId-fast |
| `b7ed266` | stage 5 review fix: counts bug, col-major outputs, exact-init |
| `f37dc74` | stage 6: nonzero + take_along_axis; ExtZero |
| `e6ed99a` | stage 6 review fix + stage 7: diff |
| `e0bcb9f` | stage 8: faer-py bindings (all 17) + review fixes (G-072 rust fix, roll broadcast parity, diff docstring, sorter min-bound) |

## Verification

- **rstsr-core entry tests**: 460 passed (row-major entry binary; col-major
  entry binary not yet wired — ADR-0002), 243 doctests, clippy 0 warnings,
  cargo fmt clean at every stage close.
- **faer-py conformance suite** (official array-api-tests @ pin, census
  baseline 1014/286/82 from 2026-10-06): **1059 / 241 / 82 of 1382** after
  stage 8 — **+45 flips, 0 regressions** (test-for-test on nodeids, two
  independent runs). Flips: 14 test_has_names + 14 test_signatures + 17
  runtime (repeat, roll, tile, searchsorted+scalars, nonzero, isin+scalars,
  unique_*×4, sort, argsort, take_along_axis, diff, diff_append_prepend).
- **Capabilities**: `data-dependent shapes` flipped to True (accurate since
  nonzero/unique now exist).
- **Docs/tracking**: array_api_standard.md Y-rows, api_specification.md
  families, numpy_coverage.csv hashes, 3 new numpy_differences.md entries
  (complex sort decline, kind= absence, unique first-occurrence order for
  non-orderable dtypes).

## Review findings ledger (deepseek/deepseek-flash, haiku slot, effort high)

One review round per stage (single pass, no loops; reviewer never edits):

| stage | must-fix | should-fix | notes | disposition |
|---|---|---|---|---|
| 1 | — | — | — | (review bundled into stage-3 round) |
| 2 (ExtSortCmp) | complex NaN-vs-NaN ordering | — | — | applied `a4fa20c` |
| 3 (sort) | empty-rest-dim panic | complex silently sortable; docs | — | applied `0458dab` |
| 4 (searchsorted) | sorter length/bounds unchecked | complex-NaN binary-search caveat; missing methods | — | applied `d6edec8` |
| 5 (unique/isin) | naive counts +1; ColMajor output order; uninit tail UB | — | — | applied `b7ed266` |
| 6 (nonzero/take_along) | usize indices → isize + tensor-layer negative resolution; nonzero flat index = walk position not offset | — | — | applied `f37dc74`+`e6ed99a` |
| 7 (diff) | — | Panics docstring over-claims n>axis error; bounds non-minimal claim | bounds claim rebutted by compile error (into_owned needs OpAssignAPI<T,D>, concat_f needs Default + OpAssignAPI<T,Vec<usize>>) — bounds restored; docstring fixed in `e0bcb9f` |
| 8 (bindings) | — | roll shift/axis mismatch "sums instead of raising"; uint64×signed IndexError in _common_int_dtype | finding 1 **rejected with evidence**: NumPy probe shows np.roll broadcasts tuple-shift on single axis (sums) and len-1 shift across axes — rstsr behavior already parity; parity pinned with 2 new tests, which exposed a real rust-side gap (len-1 Vec-shift rejected → fixed, broadcast now). Finding 2 applied (clean TypeError for non-promotable pairs). G-072 (found by binding subagent) fixed in `e0bcb9f` | applied `e0bcb9f` |

Rejected/applied split is recorded per the standing rule: every finding
noticed; unapplied ones justified (stage-8 finding 1 — NumPy probe evidence;
stage-7 bounds nit — compile-proven required).

## Bugs found during binding & review (GAP-REGISTER of 2026-10-04-rstsr-faer-py)

- **G-071** (fixed, shim): `default_dtypes()` lacked the 2025.12
  `"indexing"` key.
- **G-072** (fixed, rust-side, this wave): assignment contig fast path
  panicked on zero-size assigns — offsets applied against empty storages.
  Early `size == 0` return in cpu_serial + cpu_rayon; regression test in
  test_tile.rs; `test_tile` flipped green.
- **G-073** (fixed, shim): 0-d integral arrays marshal through `__index__`
  (scalar-index semantics).
- **G-074** (fixed, shim, pre-existing): all/any truthiness zero is 0-d now
  (0-d input with keepdims=True returned `(1,)`).

## Notes for follow-up waves

- Complex `searchsorted` remains a registered follow-up (NaN-key binary
  search deviates from part-wise lexicographic order; decline? align? decide
  with owner).
- `nonzero` 0-d input raises (NumPy returns empty coordinate arrays);
  deviation registered in DECISIONS.md.
- `unique` non-orderable dtypes return first-occurrence order (documented
  dual-order contract).
- Col-major entry test binary still absent (ADR-0002); row-major only this
  wave, with ColMajor default-order runtime tests inside the row binary.
- Suite remainder in wave scope: `test_concat`/`test_stack` red via
  pre-existing G-009 (cross-dtype) — out of this wave's scope.

## Trajectory-repo commits

- `59de84b` grill artifacts (R1–R4 + DECISIONS) 
- `91b619f` stage-6 amendment (negative take_along indices, nonzero scratch)
- `8ae2810` GAP-REGISTER v10 (stage-8 entries + G-072 fix)
- this SUMMARY + final STATUS
