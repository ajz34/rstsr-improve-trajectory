# Code review — rstsr branch `261006/manip-sort-set` (vs `main`)

- **Target**: `/home/a/rstsr_pack/rstsr`, branch `261006/manip-sort-set` (HEAD `e0bcb9f`,
  merge-base `e7cdc6a`), `git diff main...HEAD` = 104 files, +8284/−33, 12 commits.
- **Scope**: the 17 array-API functions (repeat, roll, tile, sort/argsort(+custom),
  searchsorted, nonzero, unique_values/counts/inverse/all, isin, take_along_axis, diff),
  the 3-tier device plumbing, new traits `ExtSortCmp`/`ExtZero`/`AxisIndex`, the
  rstsr-faer-py bindings (+ `python/rstsr_faer/api.py`), tests and tracking files.
- **Method**: 10 independent finder angles (line-by-line ×2, removed-behavior, cross-file,
  language pitfalls, wrapper/proxy, reuse, simplification, efficiency, altitude, conventions)
  → 1 sweep pass for gaps. Every finding below was verified with direct evidence
  (scratch crate compiled against the branch, `cargo check` on the device crates, probes of
  the installed `rstsr_faer.api` wheel and NumPy 2.5.1, NumPy/array-API pinned sources).
  NumPy probes used the local v2.5.2 checkout at `~/Git-Others/numpy` and numpy 2.5.1 in
  the environment; spec text from the pinned `~/Git-Others/array-api` (`spec/2025.12`).
- **Note on the wave's own gates**: the branch gates were `cargo test -p rstsr-core`
  (460 entry tests, 243 doctests, green — re-run here and confirmed) plus the array-api
  conformance suite. Those gates do **not** compile `crates-device/*`, and the suite's
  isin/take_along_axis value checks are `TODO`s, which is why the findings below survived.

## Findings (15, most severe first)

### 1. `crates-device/*` no longer compile — `half::` in a symlinked module
`rstsr-core/src/feature_rayon/auto_impl/set.rs:33-34` (symlinked into
`crates-device/rstsr-{openblas,mkl,blis,aocl,kml}/src/rayon_auto_impl/set.rs`) names
`half::f16`/`half::bf16` in `use_fast_path`, but the device crates have no `half`
dependency (only `rstsr-core` does).
**Evidence**: `cargo check -p rstsr-aocl|rstsr-mkl|rstsr-blis|rstsr-kml|rstsr-openblas`
all fail with `error[E0433]: cannot find module or crate 'half'` (exit 101). Adding
`half` to one device Cargo.toml makes that crate compile, i.e. this is the sole break.
Fix: delete the two `half::` arms here (the `half` feature of rstsr-dtype-traits already
gates them) or add the dep to the five crates. The module's own comment ("kept local so
the device crates' symlinked module stays self-contained") is the opposite of what happened.

### 2. `take_along_axis` reads the index tensor in storage order, not logical order
`rstsr-core/src/tensor/adv_indexing.rs:466` walks `indices.raw()` (the whole underlying
buffer, in memory order) and then hands that vector to the kernel with a freshly built
C-contiguous layout (`:485`), so only a C-contiguous, zero-offset index tensor works.
- transposed/sliced index view → silently wrong gather
  (`idx_full.t()`: got `[0,1,2,3...]` identity instead of `[0,3,2,1,...]`);
- broadcast (stride-0) index view → **panic** `index out of bounds: the len is 3 but the
  index is 3` at `rstsr-native-impl/src/cpu_serial/adv_indexing_take_along.rs:67`;
- offset/sliced view → spurious `IndexError` from out-of-view raw elements;
- on a `col_major`-feature device (or after `set_default_order(ColMajor)`) an owned
  F-contiguous index tensor (e.g. built by `rt::asarray` inside the faer-py shim) is
  reinterpreted as C-order → wrong results, contradicting the docstring's "behaves
  identically under RowMajor and ColMajor".
Fix: build `resolved` through the index layout (`IndexedIterLayout`/`indices.iter()`),
or `to_contig()` the index tensor at the boundary.

### 3. `unique_all` naive path undercounts multiplicities (complex)
`rstsr-native-impl/src/cpu_serial/set.rs:99` (`if slot == count - 1 { continue; }`) treats
"repeat of the newest unique entry" as "fresh entry" and skips the increment. Introduced by
the stage-5 fix `b7ed266` that was meant to fix the double-increment.
**Evidence** (branch build): `xp.unique_counts(asarray([1+0j,1+0j,2+0j])).counts` →
`[1, 1]` (NumPy `[2, 1]`); `unique_all` on `[1,1,2,2,3]` → counts `[1,1,1]` vs NumPy `[2,2,1]`.
Reachable from `rstsr_faer.api` for complex64/128 (the naive path is all complex dtypes; the
sorted fast path is ints/reals only). The wave's only complex test checks
`unique_values` length, so nothing caught it. Fix: increment unless the entry is genuinely
fresh (use a flag from the match, not `slot == count - 1`).

### 4. `unique_all`/`unique_counts`/`unique_inverse` fail on broadcast (stride-0) views
`rstsr-core/src/device_cpu_serial/set.rs:78` and the rayon twin
`feature_rayon/auto_impl/set.rs:78` do `inverse.truncate(a.len())` — `a.len()` is the raw
*storage* length, not the element count, so a view whose storage is shorter than its shape
leaves `inverse` too short.
**Evidence**: `rt::to_broadcast(&[1,2,3], [2,3])` → `unique_inverse_f` returns
`Err(ValueOutOfRange "idx_max = 6 not match to pattern ..=len_data = ..=3")`;
`unique_values` on the same view works, so the family is internally inconsistent.
Fix: truncate to `la.size()` (what the `OpUniqueAPI` contract intends).

### 5. `isin` treats NaN as a member of any set containing NaN
`rstsr-native-impl/src/cpu_serial/set.rs:125-129`: `isin([nan], [nan])` → `[true]`
(verified through `rstsr_faer.api` and the rust API). NumPy 2.5.1 gives `[false]`, and the
2025.12 spec requires membership "based on value equality (see `array_api.equal`)", under
which NaN is never equal. The branch's own test `test_isin_nan_membership` pins the
non-conforming behavior and the deviation is **not** registered in `numpy_differences.md`
(only complex-sort / `kind=` / unique-order are). The docstring sentence "NaNs are distinct
members" also reads as the opposite of what the code does.

### 6. `isin`'s NaN fast path is wrong for complex
`rstsr-native-impl/src/cpu_serial/set.rs:128` decides a NaN key by testing only
`x2_sorted[m-1]`, which holds only when NaN-bearing values sort last. Complex values with a
NaN component sort by their finite parts (`ExtSortCmp`), so they can sit mid-array.
**Evidence**: `x1=[Complex(5,NaN)]`, `x2=[Complex(1,NaN), Complex(2,0)]` → `[false]`,
although x2 contains a NaN and the in-scope docstring (`tensor/set.rs:625`) promises "a NaN
element of x1 matches iff x2 contains any NaN" (the f64 analogue returns `[true]`).

### 7. `searchsorted` silently truncates a float/NaN scalar `x2`
`crates-interop/rstsr-faer-py/python/rstsr_faer/api.py:1515` casts a scalar `x2` to `x1`'s
dtype (`asarray(x2, dtype=h1.dtype())`), so `xp.searchsorted(asarray([1,2,3]), 2.5)` → `1`
(NumPy `2`; the spec's own condition `x1[i-1] < v <= x1[i]` fails for 2.5 at i=1) and
`xp.searchsorted(asarray([1,2,3]), nan)` → `0` (NumPy `3`). The 2025.12 mixing rule only
sanctions cast-to-array-dtype for *compatible* scalars; mixed int-array/float-scalar is
explicitly promote-or-raise, never truncate. The suite only draws int scalars
(`hh.array_and_py_scalar(dh.all_int_dtypes)`), so it is not caught.

### 8. `take_along_axis` rejects broadcast-compatible indices (spec requires broadcasting)
`rstsr-core/src/tensor/adv_indexing.rs:452-461` enforces exact shape equality outside the
axis. The 2025.12 stub says `indices` "**must** be compatible with `x`, except for the axis
specified by `axis` (see broadcasting)" and the output shape "**must** be determined
according to broadcasting". NumPy broadcasts too.
**Evidence**: `xp.take_along_axis(x(3,4), idx(1,2), axis=1)` → `ValueError InvalidLayout`,
NumPy returns `[[3,0],[7,4],[11,8]]`. The rustdoc asserts the opposite of the standard
("the array-api standard requires matching shapes outside the axis"), so the deviation is
both non-conforming and mis-documented; the suite carries a TODO for this case.

### 9. Complex `ExtSortCmp` deviates from NumPy, contradicting its own documentation
`rstsr-dtype-traits/src/ext_sort_cmp.rs:61-79` compares complex part-wise, so
`(0, NaN) < (2, 0)`. NumPy's `numpy_tag.h` `complex_type::less` puts **any** NaN-bearing
value after every finite value (`np.sort([1+0j, nan+0j, 0+0j, 0+nanj])` →
`[0+0j, 1+0j, 0+nanj, nan+0j]`, verified; the source is quoted in the finding log). The
trait doc bullet, the inline comment (":64") and the unit-test comment (":145", "any
NaN-bearing value sorts after every finite one") all claim the NumPy behavior; the test only
compares NaN values against `inf`, so it passes. Affects `sort_custom` on complex (the
documented use case), `isin` complex (finding 6) and `searchsorted` complex.

### 10. `roll` rejects a shift tuple with a one-element axis list
`rstsr-core/src/tensor/manipulation/roll.rs:117`: `rt::roll_f(&v, vec![1,2], vec![0])`
→ `Err(InvalidValue "shift and axis must have the same length")`, whereas NumPy broadcasts
and sums: `np.roll(arange(5), (1,2), axis=(0,))` → `[2,3,4,0,1]` (verified). The docstring
claims NumPy parity "including the tuple-shift / tuple-axis combinations". The python shim
normalizes a length-1 axis list to a scalar, so only the Rust API is affected.

### 11. `diff` rejects `n == 0` in the Python shim
`python/rstsr_faer/api.py:1490` raises `ValueError("n must be a positive integer")` for
`n=0`, while `np.diff(x, n=0)` returns a copy (verified) and the spec defines the output
size as `M+N1+N2-n` without a positivity clause; the rust `diff_f` handles `n=0` correctly.
The restriction exists only in the binding, i.e. the shim narrows the API below its own
implementation. (The suite draws `n >= 1`, so it is not caught.)

### 12. Negative Python ints leak `OverflowError` from the pyo3 `usize` parameters
`crates-interop/rstsr-faer-py/src/manipulation.rs:310` (`tile`), the `repeat` pyfunction's
`all_count: Option<usize>`, and `searchsorted`'s `sorter: Option<Vec<usize>>` receive raw
pyo3 `usize` conversions. **Evidence**: `xp.tile(a, -1)`, `xp.tile(a, (-1,1))`,
`xp.repeat(a, -1)` → `OverflowError: can't convert negative int to unsigned` (a
meaningless ArithmeticError), NumPy raises `ValueError`. No shim-side sign check exists.

### 13. `take_along_axis` docstring contradicts its own signature and tests
`rstsr-core/src/tensor/adv_indexing.rs:506-508` documents `indices` as
`&TensorAny<RI, usize, B, DI>` with "entries within range (no negative values — use `take`
…)", while the function takes `isize` (`:419`) and resolves negatives from the back
(`:467`), and the shipped `test_negative_indices` asserts NumPy-equivalent negative
semantics. A caller following the doc gets a compile error for `usize` indices and wrongly
believes negatives are rejected.

### 14. `unique_values` docstring states the wrong signed-zero encoding
`rstsr-core/src/tensor/set.rs:175`: "the `+0.0` encoding for the sorted path". The sorted
path is a *stable* sort plus `==` dedupe, so it keeps the first-seen encoding.
**Evidence**: `unique_values([-0.0, 1.0, 0.0])` → `[-0.0, 1.0]` (sign bit set), which is
also what NumPy returns and what the 2025.12 spec allows. The docstring (and the tracking
note) should say first-seen.

### 15. `repeat`/`roll` with `axis=None` copy one element at a time
`rstsr-core/src/tensor/manipulation/roll.rs:44-68` (the identical block is copied in
`repeat.rs:145-175`) flattens by iterating the layout and, per element, rebuilding a
`dim_narrow` + a `dim_select` chain + `broadcast_layout_to_first` and issuing a 1-element
`assign_arbitary_uninit` — on the order of ten heap allocations and one device call per
input element. `repeat(x, 2, axis=None)` on a 10^6-element tensor ≈ 10^7 allocations for
work a single `to_contig(RowMajor)` (or the slab/broadcast trick the `Some(axis)` branch at
`repeat.rs:208` already uses) performs in one call. `roll(x, 0, axis=None)` (a no-op shift)
pays the same cost. The registered "fused kernels" follow-up does not cover the per-element
layout reconstruction, which is avoidable today.

## Below the cut (noted, not in the top 15)

- `OpUniqueAPI` (`rstsr-core/src/operators/set.rs:10`) documents "the tensor level truncates
  to `u` elements", but `tensor/set.rs` does **not** truncate — it relies on the impl
  truncating. A device implementing the trait as documented yields `assume_init` over
  uninitialized memory (latent UB; no current impl affected).
- `tensor/sorting.rs:102` (argsort doc): "Complex dtypes sort lexicographically …, whereas
  NumPy raises for complex sort" is self-contradictory and false (NumPy 2.5 sorts complex
  lexicographically) — it contradicts the branch's own `numpy_differences.md` entry.
- `tensor/exports` (`tensor/mod.rs:49`) re-exports the wave's diff/searching/set/sorting but
  omits `nonzero::*`, so `rstsr_core::tensor::exports::nonzero` does not resolve.
- `numpy_coverage.csv` gained duplicate rows for `TestRoll::{test_roll1d,test_roll2d,
  test_roll_empty,test_roll_big_int}` and `TestTile::test_basic`, whose stale copies still
  say "rstsr has no roll/tile" (contradictory statuses for the same NumPy test).
- `numpy_differences.md` (new unique entry) closes with "NaNs are distinct entries … and
  signed zeros merge in both paths, matching NumPy": NumPy *collapses* NaNs
  (`np.unique([nan,nan])` → `[nan]`), so the asserted parity is wrong and the rstsr
  NaN-distinct behavior (spec-conformant) is not registered as a deviation.
- `test_unique_signed_zero_merge` (`tests/core_func/set/test_set.rs:95`) asserts
  `values[0] == 0.0`, which holds for both zero encodings — the test cannot fail on the
  property its comment claims.
- New rustdoc warnings (3) from the diff: ambiguous `[concat]` link
  (`manipulation/repeat.rs:327`) and two redundant explicit link targets
  (`operators/adv_indexing.rs:27`, `operators/searching.rs:7`).
- `feature_rayon/auto_impl/set.rs:1-7` claims "the isin binary search parallelizes per x1
  value"; the code calls the serial kernel (no parallel switch), duplicating
  `device_cpu_serial/set.rs` verbatim (including a second 17-dtype `use_fast_path` ladder
  that must be kept in sync across 7 crates).
- `diff` has no `core_func` tests (only two inline doctests); `n` > axis length,
  prepend/append validation and the empty-axis early stop are untested; no
  `numpy_coverage.csv` rows.
- Efficiency/cleanup family: rayon sort allocates a `(T, usize)` pairs buffer per line
  (`cpu_rayon/sorting.rs:84`; the serial twin hoists it); `sort_line_cpu_serial` uses stable
  `sort_by` although the `(value, position)` order is total (`sort_unstable_by` is
  byte-identical and allocation-free); `nonzero_f` clones its whole flat-index buffer
  (`tensor/nonzero.rs:38`) and hand-rolls `stride_c_contig`/`unravel_index_c`
  (`rstsr-common/src/layout/shape.rs`); `take_along_axis` materializes an extra O(n)
  resolved-index Vec; `SortArgs.stable` is threaded into the device traits but ignored by
  every impl; `AxisIndex` duplicates the `AxesIndex` conversion matrix; faer-py
  `dispatch_unique_values!`/`pair!`/`quad!` re-hand-roll the existing `dispatch_t!` macro;
  the wave's 17 new anchors ship no `doc_draft` twins and several lack the mandatory
  row/col-major notice and the `## Similar function from other crates/libraries`
  See-also subsection; `test_tile.rs::test_rank_promotion_up` omits `specify_test!`;
  `test_tile.rs:54-58` is a 5-line non-doc comment (CLAUDE.md caps at 4).

## Reproduction index

| # | one-line repro |
|---|---|
| 1 | `cargo check -p rstsr-aocl` → E0433 `half` |
| 2 | `cargo run` on a scratch crate: `take_along_axis(&a2x3, &idx.t(), -1)`; `to_broadcast(idx,…)` → panic |
| 3 | `xp.unique_counts(asarray([1+0j,1+0j,2+0j])).counts` → `[1,1]` |
| 4 | `rt::unique_inverse_f(&rt::to_broadcast(&[1,2,3],[2,3]))` → `Err(ValueOutOfRange)` |
| 5 | `xp.isin(asarray([nan]), asarray([nan]))` → `[true]` (NumPy `[false]`) |
| 6 | `rt::isin((&[Complex(5,NaN)], &[Complex(1,NaN), Complex(2,0)]))` → `[false]` |
| 7 | `xp.searchsorted(asarray([1,2,3]), 2.5)` → `1` (NumPy `2`); `nan` → `0` |
| 8 | `xp.take_along_axis(asarray([[1,2],[3,4]]), asarray([[0]]), axis=-1)` → ValueError |
| 9 | `np.sort([1+0j, nan+0j, 0+0j, 0+nanj])` vs `ext_total_cmp` on the same values |
| 10 | `rt::roll_f(&arange(5), vec![1,2], vec![0])` → Err |
| 11 | `xp.diff(asarray([1,4,9]), n=0)` → ValueError |
| 12 | `xp.tile(asarray([0,1]), -1)` → OverflowError |
| 13 | read `tensor/adv_indexing.rs:506-508` vs the `isize` signature at `:419` |
| 14 | `rt::unique_values(&[-0.0, 1.0, 0.0])` → first element has the sign bit set |
| 15 | read `roll.rs:44-68` / `repeat.rs:145-175` (one assign call per element) |
