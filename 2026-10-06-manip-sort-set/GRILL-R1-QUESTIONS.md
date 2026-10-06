# Grill Round 1 — manip/sort/set wave (2026-10-06)

Agent-authored. Answers go in `GRILL-R1-ANSWERS.md` (user-owned) or on screen.

Evidence base: 4 comprehension reports (rstsr-core 0.9.0 manipulation layout;
searching/3-tier device pattern; rstsr-faer-py binding patterns; array-api
2025.12 stub signatures + test enforcement), census stamp `20261006-183350`
(41 in-scope nodeids), GAP-REGISTER v9.

## Settled by evidence (not asked, flag if wrong)

- All 16 names are absent rust-side; `src/docs/array_api_standard.md` already
  reserves empty-status rows for each.
- 3-tier house pattern for device-dependent ops: `Op*API` trait in
  `src/operators/` → impls in `device_cpu_serial/` + `feature_rayon/auto_impl/`
  (+ symlink into `device_faer/rayon_auto_impl/`) + kernels in
  rstsr-native-impl `cpu_serial`/`cpu_rayon` → tensor-level free fns +
  `TensorAny` methods → `prelude::rstsr_funcs` → `rt::`.
- Device-independent house pattern: single tensor-level impl generic over
  `B: DeviceAPI<T> (+ DeviceCreationAnyAPI + OpAssignAPI)`, composing existing
  device ops (`concat`, `bool_select` precedent).
- Index dtype: `usize` rust-side (argmax `TOut` precedent), i64 py-side via
  `idx_lift`; searchsorted output must equal `default_dtypes()["indexing"]`.
- All new ops take views (`&TensorAny`/`TensorView`) and return freshly
  allocated tensors (user pre-granted; no into_/COW variants needed).
- Spec exacts: `stable=True` default ⇒ default sort is stable; `roll.shift` is
  a *named pos-or-kw* python param; `unique_all` namedtuple fields
  `values/indices/inverse_indices/counts` with `inverse_indices` shaped like
  `x`; NaNs are distinct in `unique_*`; signed zeros collapse; `isin` has
  `invert` (2025.12); `take_along_axis(axis=-1)` with negative indices;
  `searchsorted` x2 widened to scalars in 2025.12; `sorter=` exercised.
- rstsr-faer-py is interface-only: 3-point registration (pyfunction →
  `lib.rs` → `api.py` shim + `__all__`), no algorithms; `Vec<NativeArray>` +
  py-side `tuple(...)` is the multi-output precedent (no namedtuple yet).
- No auto-commit in rstsr; branches `YYMMDD/<slug>` off main; push fork
  `ajz34`; PRs to `RESTGroup/rstsr`. Work in `~/rstsr_pack/rstsr-local-workspace`.

## Questions

❓ **Q1 — scope boundary: is `diff` in?**
The stated scope is the 16 names (41 nodeids). `diff` (4 nodeids, utility,
2024.12-gated) is adjacent: composable from concat/slicing/subtraction but the
n-fold loop is algorithmic, so it would need a small rust-side `diff` too.
Everything else red in the census (clip, log1p, dtype fns, linalg/fft
namespaces) is out — registered separately.

➡️ Recommend: **exclude `diff`** this wave (keep the wave one coherent
theme — "manip + sort + set"; diff becomes a trivial follow-up PR). Include it
only if you want the utility file green in the same pass.

❓ **Q2 — delivery slicing: one PR or three?**
Options: (a) single PR: 3 manip + 3 sort/search + 7 set/index + bindings +
tests; (b) three staged PRs, each = branch off main, core + faer-py binding +
tests + suite measurement, in dependency order:
  PR-A manipulation: `repeat`, `roll`, `tile` (no rust deps on the others);
  PR-B sorting: `sort`, `argsort`, `searchsorted` (new Op traits + kernels);
  PR-C set/indexing: `unique_*`×4, `isin`, `nonzero`, `take_along_axis`
  (composes argsort from PR-B, searchsorted from PR-B, index_select).
Each PR is independently mergeable and reviewable; W4/W5/`where` waves were
each ≈ this size.

➡️ Recommend: **(b) three staged PRs** — 16 names + 2 device kernel sets +
~10 new rstsr-core test files in one PR is over review size, and PR-C wants
PR-B's argsort landed anyway. Order A → B → C (A has no rust-side trait work,
so it can go first and fastest).

❓ **Q3 — device-dependence classification (the core architecture decision).**
Proposed split:

| op | class | mechanism |
|---|---|---|
| `repeat`, `roll`, `tile` | device-independent | concat-style: `uninit_impl` + `assign_uninit` composition (OpAssignAPI already has serial+rayon impls ⇒ parallel for free); no new operator traits, no new kernels |
| `sort`, `argsort` | 3-tier device-dependent | new `OpSortAPI`/`OpArgSortAPI` in `operators/sorting.rs`; kernels in rstsr-native-impl (serial `sort_by`/`sort_unstable_by` + NaN-aware comparator; rayon `par_sort_by`/`par_sort_unstable_by` over lines) |
| `searchsorted` | 3-tier device-dependent | `OpSearchSortedAPI`; per-x2-line binary search, rayon over x2 lines |
| `take_along_axis` | 3-tier device-dependent | extend `operators/adv_indexing.rs`: gather kernel with an index *tensor* (natural generalization of `index_select`) |
| `unique_*` ×4 | device-independent composition | argsort (reuses PR-B trait) → permute → host sweep over sorted values (O(n) serial host read) → `outof_cpu_vec`; data-dependent length decided on host (`bool_select` precedent) |
| `nonzero` | device-independent | elementwise `!= 0` → host scan in C-order → per-dim index tensors |
| `isin` | device-independent | sort+unique x2 + searchsorted over x1 (reuses PR-B), reshape, invert |

Caveat to acknowledge: the device-independent bucket (unique/nonzero/isin via
host reads) ties those ops to CPU-family devices (`Raw = Vec<T>`); a future
GPU device would need real kernels. That is acceptable today and the 3-tier
ops (sort/search/take_along) are the future-proof surface.

Sub-decisions to confirm: (i) repeat/roll/tile as assign-compositions rather
than fused 3-tier copy kernels (more device calls per op, but zero new device
surface — "make things easy" per your framing); (ii) `take_along_axis` as a
real kernel (vs a tensor-level per-line `index_select` loop, which would be
O(lines) device calls); (iii) unique's host sweep (vs a two-pass device trait).

➡️ Recommend: **the table as written** — 3-tier for sort/argsort/searchsorted/
take_along_axis, device-independent composition for the rest. It matches the
existing codebase's own split (argmax/where/index_select are device-dependent;
concat/bool_select are compositions) and keeps the new-kernel count minimal
while still demonstrating the strict interface/implementation separation.

❓ **Q4 — rust API surface: signatures, args, placement, multi-output types.**
Proposal (all with `_f` fallible + panicking twins + `TensorAny` methods +
prelude/`rt::` wiring + api-doc-conventions docstrings):

- `tensor/manipulation/repeat.rs` — `repeat_f(x, repeats: impl Into<Repeats>, axis: impl TryInto<AxesIndex<isize>>) -> Tensor<T,B,IxD>`;
  `Repeats = All(usize) | Elems(Vec<usize>)` (From<usize>/From<Vec<usize>>);
  `AxesIndex::None` ⇒ flattened semantics.
- `tensor/manipulation/roll.rs` — `roll_f(x, shift: AxesIndex<isize>, axis: Option<AxesIndex<isize>>)` (int or tuple shift; axis None ⇒ flatten).
- `tensor/manipulation/tile.rs` — `tile_f(x, repetitions: Vec<usize>)`.
- `tensor/sorting.rs` — `sort_f(x, args: impl Into<SortArgs>) -> Tensor<T,B,IxD>`;
  `argsort_f(x, args) -> Tensor<usize,B,IxD>`; `SortArgs { axis: isize (-1 default), descending: bool, stable: bool }` with `ReduceArgs`-style `From` impls.
- `tensor/searching.rs` — `searchsorted_f(x1, x2, side: SearchSide, sorter: Option<&[usize]>) -> Tensor<usize>` (numpy-parity `sorter` param — see Q6); `nonzero_f(x) -> Vec<Tensor<usize,B,IxD>>` (one per dim, row-major).
- `tensor/set.rs` — `unique_values_f(x) -> Tensor`; `unique_counts_f(x) -> UniqueCounts{values,counts}`; `unique_inverse_f(x) -> UniqueInverse{values,inverse_indices}`; `unique_all_f(x) -> UniqueAll{values,indices,inverse_indices,counts}`; `isin_f(x1, x2, invert: bool) -> Tensor<bool>`.
- `take_along_axis_f` appended in `tensor/adv_indexing.rs`.
- Operator traits: `operators/sorting.rs` (`OpSortAPI`, `OpArgSortAPI`, `OpSearchSortedAPI`); `operators/adv_indexing.rs` extended.
- New structs (`Repeats`, `SortArgs`, `SearchSide`, `Unique*`) → `prelude::rstsr_structs`.

The genuinely new shapes here (no precedent yet): `Repeats` int-or-array enum,
`SearchSide` string-backed enum, multi-output structs `UniqueCounts`/
`UniqueInverse`/`UniqueAll` (nothing in core returns named multi-tensor
results today), and `nonzero`'s `Vec<Tensor>` per-dim return. Python side
builds the namedtuple in `api.py` from the struct fields / Vec.

➡️ Recommend: **as proposed** — args structs follow the `ReduceArgs` `From`
idiom; multi-output via named structs (clearer than positional `Vec`, and
`unique_all` needs 4 distinguishable fields anyway); module placement mirrors
the array-api sections (manipulation/sorting/searching/set) rather than
shoehorning into reduction/indexing.

❓ **Q5 — semantics policy: dtype coverage, NaN, stability details.**
- sort/argsort dtypes: `T: PartialOrd` ⇒ bool + ints + floats (exactly what
  the suite tests). Complex **declined** this wave (error at tensor level;
  numpy does lexicographic complex sort — record as numpy-difference).
  Alternative: implement lexicographic now via a `ComplexOrd` wrapper.
- NaN in sort (untested by suite, numpy-parity for core tests): NaN sorts
  last ascending; descending = full comparator reversal ⇒ NaN first when
  descending. Alternative: pin NaN last in both directions.
- Stability: default `stable=true` ⇒ stable sort in BOTH directions (ties
  keep original order; descending is reversed-comparator, not reversed-array).
  `stable=false` may use unstable sort.
- ±0.0: `-0.0 == 0.0` in sorting (PartialOrd gives this); suite assumes ±0
  away anyway. unique: signed zeros collapse to one entry (either sign kept);
  each NaN is a distinct entry (sweep dedupes with `==`, which never merges
  NaNs); `values` sorted ascending (numpy parity; suite leaves order free but
  requires alignment with counts/inverse/indices).
- `indices` in unique_all = first occurrence in flattened C-order (suite
  enforces); `inverse_indices` keeps x's shape.
- searchsorted/isin dtypes: real numeric only (PartialOrd); complex declined,
  registered. nonzero: all dtypes incl. complex/bool (PartialEq + zero-test).
  repeat/roll/tile: all dtypes (pure copy).

➡️ Recommend: **as listed** — decline complex for sort/searchsorted/isin
(register as intentional gaps), numpy-parity NaN-last-ascending with plain
comparator reversal for descending, stable-in-both-directions.

❓ **Q6 — `searchsorted` `sorter=`: numpy-parity rust param or shim composition?**
The suite passes `sorter=xp.argsort(x1)` half the time. Options:
(a) rust pyfunction composes: apply sorter via `take(argsort)` then plain
searchsorted (shim-side composition of two rt calls — `sum_bool` precedent);
(b) `rt::searchsorted_f` takes `sorter: Option<&[usize]>` natively
(np.searchsorted has this param, so it is numpy-parity) and applies it in the
kernel wrapper before searching.

➡️ Recommend: **(b)** — one rust-side place applies the permutation (reusing
`index_select`), the py wrapper just forwards `sorter`; keeps the shim
interface-only in the strict sense and gives numpy users the same knob.

❓ **Q7 — flip capabilities `data-dependent shapes` to True with this wave?**
Currently `False` in `__array_namespace_info__.capabilities()`. After
nonzero/unique the namespace genuinely produces data-dependent shapes, so
`False` becomes a lie. **Fact verified this round**: array-api-tests reads
that capability value NOWHERE — the only `capabilities()` call site
(test_inspection_functions.py:15) asserts the dict is well-formed
(keys present, bool-typed) and never branches on the values; the
`data_dependent_shapes` pytest mark skips only via the CLI flag
`--disable-data-dependent-shapes` (conftest.py:238), not via the capability;
no hypothesis strategy consults it. So flipping True/False cannot change any
test outcome.

➡️ Recommend: **flip to True in PR-C** (with nonzero landing) — it is free,
measured-risk-zero, and honest. (The same fact implies the current `False`
costs nothing either, so if you prefer minimal-diff PRs, keeping False +
registering is also safe.)

❓ **Q8 — performance deliverable for this wave?**
Correctness-first with free parallelism where it falls out (rayon `par_sort_by`
for sort, existing rayon assign for repeat/roll/tile composition, rayon search
over x2)? Or does this wave also need criterion benchmarks / kernel tuning
(multithreaded-first is the standing priority, but prior waves shipped without
bench deliverables and tuned later via trajectory experiments)?

➡️ Recommend: **correctness-first, no bench deliverable**; note obvious perf
follow-ups (fused repeat kernel vs assign-composition, two-pass device-side
unique) in the gap register for later trajectory experiments.

❓ **Q9 — rstsr-core test placement and depth.**
Placement proposal: new category dirs `tests/core_func/sorting/`
(test_sort, test_argsort, test_searchsorted) and `tests/core_func/set/`
(test_unique, test_isin); `nonzero` → `tests/core_func/searching/` (new, or
into indexing/?); `take_along_axis` appended to `tests/core_func/indexing/`;
repeat/roll/tile → `tests/core_func/manipulation/`. Depth: curated
NumPy-cited parity subset per core-test skill (NumPy sources:
test_sort.py is large; test_shape_base.py for repeat/tile; roll in
test_numeric.py) + `custom_*` modules for suite-specific edges (stable ties,
descending, NaN, signed zero, negative axes, 0-d errors, empty inputs) +
doc_draft doctests per api-doc-conventions.

➡️ Recommend: **as proposed**, with `searching/` as a new dir (mirrors the
array-api section; keeps `indexing/` about indexing-with-indices). Curated
subset, not full NumPy translation — full translation of test_sort.py alone
is a wave by itself.
