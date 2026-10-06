# DECISIONS — manip/sort/set wave grill (closed 2026-10-06)

Four rounds (R1–R4, files `GRILL-R*.md` in this directory). Evidence base:
4 codebase-comprehension reports + 3 fact-finders; census stamp
`20261006-183350` (1014/286/82); spec/tests pinned array-api-tests `6c0b59f`
/ spec `5f847a3` (2025.12). Implementation awaits user go.

## Scope

In: `repeat`, `roll`, `tile`, `sort`, `argsort`, `searchsorted`, `nonzero`,
`isin`, `unique_values/counts/inverse/all`, `take_along_axis`, `diff`
(17 names, 45 nodeids: 41 census + diff's 4).
Out: clip, log1p, dtype functions, linalg/fft namespaces (registered
elsewhere; fft out for this project at this time).

## Architecture (interface ⇄ implementation strictly separated)

- Tiering per op:
  - 3-tier device-dependent (new `Op*API` traits + device impls +
    rstsr-native-impl kernels cpu_serial + cpu_rayon + device_faer symlink):
    `sort`/`argsort` (+`*_custom`), `searchsorted`, `unique_*`, `isin`,
    `nonzero`, `take_along_axis` (extend `operators/adv_indexing.rs`).
  - Device-independent tensor-level (no new device traits): `repeat`,
    `roll`, `tile` (concat-style `uninit_impl` + `assign_uninit`
    composition), `diff` (concat + slice-pair subtraction, `tensor/diff.rs`,
    plain non-overloaded signature).
- Kernels: `&inputs`/`&mut output`, O(1) auxiliary memory beyond
  inputs/outputs (matmul-style exceptions must be efficiency-motivated and
  registered). Registered exceptions: `isin`'s O(m) sorted-x2 temp;
  `unique_all`'s O(u)-bit `seen` vector (≤ the `indices` output itself);
  sort/argsort's O(axis_size) per-line `(value, position)` pairs scratch
  (index-sort cannot write through in place; the buffer is reused per line,
  not per output element — stage-3 review 2026-10-06).
- sort/argsort on 0-d input: errors (AxisError — there is no axis to sort
  along; NumPy's np.sort(0-d) succeeds returning a copy, deviation
  registered, stage-3 review 2026-10-06).
- Layout: kernels layout-generic; iteration with explicit order —
  C-order-flat contracts (repeat/roll axis=None, nonzero scan, unique
  flatten, first-occurrence indices) use `IndexedIterLayout::new(la,
  RowMajor)` (argmax idiom), never `reshape(-1)` (not C-order on col-major
  devices), no tier-2 axes-reversal this wave (documented fallback for
  future row-major-hardwired kernels). Output layout is the high tier's
  call: allocate `new_contig(None, device.default_order())` — C-contig
  preferred on row-major devices, F-contig on col-major; sort output also
  default-order.
- Inputs: every new op takes `impl TensorViewAPI` (established idiom for
  data-consuming compute free fns — reduce/matmul family), returns owned
  tensors.
- `unique_*` algorithm (R4): **naive + fast dual path**. Naive = general
  bound (`Clone + PartialEq`), first-occurrence scan, O(n·u), covers all
  dtypes incl. complex. Fast = sorted algorithm under `ExtSortCmp`-typed
  kernels, substituted per-dtype via **TypeId dispatch + `from_raw_parts`
  reinterpretation** (exactly the DeviceFaer matmul pattern,
  `device_faer/matmul.rs:15,53-79` — `same_type` gate, SAFETY comments
  citing TypeId proof) for the dtypes where it is a win (ints, bool,
  floats; complex optional at implementation's discretion). Tensor API is
  single-path; only the device impl dispatches internally.

## rstsr-dtype-traits

- New trait **`ExtSortCmp`** (suffix-less, `Ext*` family style), method
  `fn ext_total_cmp(&self, other: &Self) -> std::cmp::Ordering`.
  Impls: signed/unsigned ints (`Ord::cmp`), `bool`, f32/f64 (+half):
  NaN-last, `Equal` when `a == b` (covers ±0.0 — NOT raw `total_cmp`,
  which orders −0 < +0 and payloads); `Complex<f32/f64>` lexicographic
  (re, then im), NaN components last. Standalone sibling of `ExtReal`
  (whose `ext_min/ext_max` commit to IEEE semantics — do not mix).
- rstsr-common: new **`AxisIndex<isize>`** (exactly one axis, negative
  normalized via `rstsr_check_axis!`, own reduced From/TryFrom surface)
  beside `AxesIndex`; used by `SortArgs.axis`, `take_along_axis`, `diff`;
  no retrofit of existing single-axis fns.

## Tensor API (rstsr-core)

- `repeat_f(x, repeats: impl Into<RepeatArg>, axis: impl
  TryInto<AxesIndex<isize>>)`; `RepeatArg { All(usize), Elems(Vec<usize>)
  }` (no tensor variant — py marshals via tolist; `Elems` len ∈ {1, axis
  size | total size}); axis=None flattens C-order.
- `roll_f(x, shift: AxesIndex<isize>, axis: Option<AxesIndex<isize>>)`.
- `tile_f(x, repetitions: Vec<usize>)` (spec rank-promotion rule).
- `sort_f(x, args: impl Into<SortArgs>)`, `argsort_f(...) ->
  Tensor<usize>`; `SortArgs { axis: -1, descending: false, stable: true }`,
  ReduceArgs-style `From` impls; sort/argsort take `impl TensorViewAPI`,
  real dtypes only (bool+ints+floats); complex declined at tensor layer
  (spec "real-valued" — spec-aligned, registered).
- `sort_custom_f`/`argsort_custom_f(x, cmp: Fn(&T,&T)->Ordering + Send +
  Sync, args)` mirroring `OpReduceCustomAPI` closure style; built-in sort
  = same machinery with default comparator; complex sort = documented
  docstring example + doctest, not shipped as a named fn; NumPy `kind=`
  recorded as intentional deviation; not exposed in the python wrapper.
- `searchsorted_f(x1, x2, args: impl TryInto<SearchSortedArgs>)` (deviation
  from `Into`: side strings are fallible to parse; stage-4 review 2026-10-06);
  `SearchSortedArgs { side: SearchSide = Left, sorter: Option<Vec<usize>>
  }` (numpy-parity rust-native `sorter`, length/bounds validated at the
  tensor layer); output dtype must equal `default_dtypes()["indexing"]`
  (= int64 py-side, via idx_lift). Complex-NaN keys: binary search hoists
  NaN-bearing keys after finite entries (differs from ExtSortCmp's
  part-wise complex order); complex searchsorted remains a registered
  follow-up.
- `nonzero_f(x) -> Vec<Tensor<usize, B, IxD>>` (one per dim, strict C
  order, 0-d raises); two-pass count+fill kernels; all dtypes.
- `unique_values_f(x) -> Tensor`; `unique_counts_f -> UniqueCounts`;
  `unique_inverse_f -> UniqueInverse`; `unique_all_f -> UniqueAll{values,
  indices, inverse_indices, counts}`; structs get `Into<(Tensor,Tensor)>`
  (2-field ones); `inverse_indices` keeps x's shape, `indices` index
  flattened C-order first-occurrence; NaNs distinct; signed zeros merge.
- `isin_f(x1, x2, invert: bool) -> Tensor<bool>` (sorted-x2 + binary
  search; real dtypes).
- `take_along_axis_f(x, indices, axis: AxisIndex)` — gather kernel with
  index tensor (generalizes `index_select`); strict
  same-shape-except-axis, no broadcast; negative indices resolved.
- `diff(x, axis: impl TryInto<AxisIndex<isize>>, n: usize, prepend:
  Option<&TensorAny>, append: Option<&TensorAny>) -> Tensor` (+`_f` twin).
- All with `_f` + panicking twins + `TensorAny` methods + prelude/`rt::`
  wiring + api-doc-conventions docstrings; placement mirrors array-api
  sections (`manipulation/{repeat,roll,tile}.rs`, `sorting.rs`,
  `searching.rs`, `set.rs`, `diff.rs`, adv_indexing extended).

## Semantics

- Stability: `stable=true` default and honored in both directions
  (descending = reversed comparator, not reversed array).
- NaN in sort: **last in BOTH directions** (NumPy 2.5 `numpy_tag.h`: "NaN
  sorts to the end in reverse too"; `np.sort([3,nan,1,2], descending=True)
  -> [3,2,1,nan]`). Amended by stage-2 review 2026-10-06 — plain comparator
  reversal would put NaN first descending and fail translated parity tests;
  the descending comparator must keep NaN pinned last, not reverse the
  NaN relation.
- Complex sort via `ExtSortCmp`: NaN-bearing complexes order by finite
  parts (NumPy `numpy_tag.h` complex comparator), NaN part last within
  each part; implemented in ext_sort_cmp.rs per stage-2 review.
- Indices: `usize` rust-side everywhere; i64 py-side.

## rstsr-faer-py (interface-only)

- 3-point registration per name (pyfunction → lib.rs → api.py shim +
  `__all__`); signatures exactly per 2025.12 stubs (`roll.shift` named
  pos-or-kw; kw-only `axis/descending/stable/side/sorter/invert/n/...`);
  enums validated in api.py, `&str`-free rust signatures where practical;
  index outputs via idx_lift→int64; multi-output via `Vec<NativeArray>` +
  py-side namedtuple construction (first namedtuple precedent:
  `unique_all`).
- Capabilities `"data-dependent shapes"` flipped True with the nonzero
  stage (verified: suite never reads the value).
- Update `src/docs/array_api_standard.md` + `api_specification.md`
  tracking rows; `numpy_differences.md` entries (complex sort decline,
  unique order under naive path, kind= absence).

## Tests

- rstsr-core: new category dirs `tests/core_func/{sorting,set,searching}/`;
  take_along_axis → indexing/; repeat/roll/tile → manipulation/; NumPy
  parity per skills `core-test`/`test-conventions` (NumPy v2.5.2 sources
  at `~/Git-Others/numpy`: test_sort.py, test_shape_base.py,
  test_numeric.py, test_function_base.py, test_setops.py, test_indexing
  ...), curated subset + `custom_*` edges (ties/descending/NaN/±0/
  negative axes/0-d/empty) + doc_draft doctests.
- Conformance: suite re-run per stage group (skill rstsr-faer-py-tests;
  wheel in rstsr-local-workspace); expectation ≈ +45 flips (41 + diff 4),
  0 regressions.

## Delivery & workflow (user R1/R2/R3 grants)

- Single branch **`261006/manip-sort-set`** off main in
  `~/rstsr_pack/rstsr-local-workspace`. Stage-sliced commits (9 stages,
  each green before next): (1) manip repeat/roll/tile; (2) `ExtSortCmp` +
  `AxisIndex`; (3) sort/argsort/sort_custom + kernels; (4) searchsorted;
  (5) unique/isin (naive + TypeId fast dispatch); (6) nonzero +
  take_along_axis; (7) diff; (8) faer-py bindings + capabilities flip +
  tracking tables; (9) trajectory reports.
- Auto-commit allowed this session in the rstsr repos (explicit user
  instruction overriding the standing no-auto-commit rule); **no push, no
  `gh pr`**. Commit trailers per `git-commit-coauthor` skill.
- Models (claude-code-router): main = glm/glm-5.3 high; implementation
  subagents = glm/glm-5.3-flash (sonnet slot) effort max; per-stage code
  review = DeepSeek/deepseek-flash (haiku slot) effort high, **once, no
  implement-review loops**; final review after everything = DeepSeek/
  deepseek-flash effort max. Not all review comments must be applied;
  unapplied ones must be noticed and justified in the stage report.
- Main session orchestrates only; subagents implement; sequential stages.

## Registered follow-ups (not this wave)

- Fused 3-tier copy kernels for repeat/roll/tile (vs assign composition);
  two-pass device-side unique fast path for non-CPU devices; complex
  sort/argsort/searchsorted/isin if ever spec-required; tier-2
  axes-reversal adapter if a future fused kernel is row-major-hardwired;
  `AxisIndex` retrofit of existing single-axis fns; NumPy `kind=` knob;
  u64 scalar-carrier fix (G-013) and the 16 shim-side failures remain
  separate tracks.
