---
name: rstsr-reduction-apis-pr2
description: PR #111 open (branch 261005/custom-reduce: 24a09ef + 113dc5e + col-major fix 472eddd), all 13 CI checks green; NOT merging without owner go.
metadata:
  type: project
---

PR2 of the 2026-10-05 reduction-apis task: branch `261005/custom-reduce`
(based on main f736987 = merged PR1 #110). COMMITTED part: `24a09ef` custom
user reduction (`reduce_all`/`reduce_axes`/`reduce_with_args` +
`OpReduceCustomAPI` + device impls + reference-table docstrings). ON TOP,
UNCOMMITTED (owner: "you are not going to auto git commit" - everything sits
in the working tree awaiting review): the cumulative wave, folded into the
SAME PR (owner: "Not separating two PRs").

Cumulative surface (D8, array-api 2025.12 semantics): `rt::cumulative_sum` /
`rt::cumulative_prod` (+`_f`) taking `impl Into<CumulativeArgs>`
(`{axis: Option<isize>, include_initial: bool}`; Default derived; overloads
`()`, isize/i32/i64/usize, `Option<isize>`, bool = include_initial,
`(axis, bool)`); `*_with_dtype` variants (method spelling
`b.cumulative_sum_with_dtype::<f64>(...)` - the FREE fn turbofish fills `T`
first, `<T, TOut, B, D>`, so free-form TOut turbofish is impossible; same
gotcha as sum_with_dtype). Full docstrings on the two non-fallible anchors
(§ per api-doc-conventions, `# Panics` ends with fallible pointer); device
traits `OpCumSumAPI`/`OpCumProdAPI` (TOut=T) + `OpCumSumDtypeAPI`/
`OpCumProdDtypeAPI`; kernels `cumulative_cpu_serial`/`cumulative_cpu_rayon`
in rstsr-native-impl (scan lines parallel over remaining axes via
AtomicPtr+par_iter; PARALLEL_SWITCH fallback), impls in device_cpu_serial +
feature_rayon/auto_impl (symlinked to device_faer).

Kernel gotchas (2026-10-05): `Layout<IxD>` accessors are `shape()`/`stride()`
(NOT strides); `IxD = Vec<usize>` so `.as_ref()` on them is AMBIGUOUS
(E0282) - use `.clone()`; iterator offsets are usize, strides isize (cast
arith); K-order output via `layout_for_array_copy` on a PROBE layout that
reuses input strides with the GROWN scan-axis shape - the probe fails
`Layout::new` stride validation (overlap check), use
`Layout::new_unchecked` (never iterated, only feeds greedy permutation);
`dim_chop` (not dim_eliminate) removes the axis without size-1 check.

Semantics decided: axis=None valid for 1-D only, n-D raises InvalidValue
(np.cumsum would flatten - intentional divergence, numpy_differences.md);
0-D rejected; dtype=None keeps T (NO platform-int widening - intentional
divergence, use with_dtype); include_initial grows axis to M+1 with 0/1
prepended; output layout = input K arrangement, fresh storage.

Tests: core_func/reduction/test_cumulative.rs (numpy_cumulative: TestCumsum/
TestCumprod test_basic + test_cumulative_include_initial, hashes afcd46564c4e
/ 0c50f1f6f8b1 / bed67d65850e; custom_cumulative: overloads, axis=None
contract, transposed view, broadcast scan axis, with_dtype anti-overflow u8
->u64 (overflow of plain u8 PANICS in debug - never test raw), zero-size,
negative axis/methods) + doc_draft twin (Display format is single-space
`[ 1 3 6 10]`, floats print without `.0`) + DeviceFaer runtime test in
tensor/reduction.rs (incl. 2048-element parallel-branch case).

Gates all green: entry_row_cpu 341, faer lib 135, doc 194, clippy 0
(default/rayon/faer), nightly fmt, rustdoc renders (no broken links).

PR #111 opened 2026-10-05 (fork ajz34:261005/custom-reduce -> RESTGroup/rstsr
main); first CI run failed ONE check, unittests-col-major: DeviceFaer::default()
follows the GLOBAL default order, which is ColMajor under the col_major feature
- the unpinned arange/into_shape fixture filled col-major (4093 = 4j+1 vs
expected 3070). Fix 472eddd pins device.set_default_order(RowMajor) (house
convention for order-dependent faer tests). LESSON: local gate for faer tests
needs `cargo test -p rstsr-core --lib --no-default-features --features
"faer col_major"` too (col_major is mutually exclusive with row_major, so
drop default features). All 13 checks green on 472eddd; NOT merging (owner
checks status, no auto-merge).

OPEN question flagged to owner: `array_api_standard.md` lines ~261-262 still
list cumulative_sum/cumulative_prod as unimplemented - D15 said the file
"stays untouched by this task", so left alone; flipping those cells is a
one-line follow-up if the owner wants it.

Next: owner reviews PR #111 and merges (or asks for changes); no auto-merge.
PR3 `261005/cumulative` as a separate branch is OBSOLETE (folded in). Fresh subagent drafts
`rstsr-code-style` skill at task wrap-up (Q9).

Related: [[rstsr-reduction-apis-pr1]], [[rstsr-reduction-apis-grill]].
