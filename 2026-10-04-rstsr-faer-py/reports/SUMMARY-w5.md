# W5 — searching & indexing wave + branch review (2026-10-06)

Task (owner): bind the searching/indexing entries rstsr already provides
(`argmax`, `argmin`, `count_nonzero`, `take`); then review the whole
branch against the session's initial prompt (`/code-review max`), fix the
rust-side root causes with owner authorization, and carry the branch to a
PR.

## Result

| run | passed | failed | skipped |
|---|---|---|---|
| W4 (start of wave) | 984 | 316 | 82 |
| **W5 (this wave)** | **996** | **304** | **82** |

Same branch/wheel as W4; full map stamp `20261006-131221`, re-verified
`20261006-132433` (identical totals), per-test table
`harness/reports/COMPLIANCE-FULL-20261006-131221.csv`. **12 flips / 0
regressions**: `test_argmax`, `test_argmin`, `test_count_nonzero` and
`test_take` plus their `has_names` and signature entries — all passing on
the first run.

## Bindings (wrapper discipline kept)

- `argmax`/`argmin`: `axis=None|int` + keepdims over
  `arg*_with_args_f`; ordered dtypes only (complex has no ordering —
  undefined in the standard too); ties resolve to the first occurrence
  (rstsr's own contract).
- `count_nonzero`: `axis=None|int|tuple` + keepdims over
  `count_nonzero_with_args_f`; complex is served. Bool routes to the
  bool-specialized sum (`TensorSumBoolAPI::sum_with_args_f`, `TOut =
  usize`) — a rust kernel, not a Python fallback (the one documented
  exception, on the owner's request; the generic count kernel is
  `Zero`-bound, and counting `True`s is the 0/1 sum by definition).
- `take`: `rt::take_f` over a one-dimensional integer index array (the
  Python layer flattens via `tolist`); negative-index resolution and
  bounds checks are rstsr's; `axis=None` is accepted only for 1-D input
  (spec); the output keeps the input dtype.
- rstsr's index reductions return `usize` tensors while the standard
  requires the default index dtype, so `ops::idx_lift` re-materializes
  the values as int64 (a lossless element cast of the astype class, not
  an algorithm). Two dispatch macros split by kernel bound:
  `dispatch_t_index_ord!` (ordered family) and `dispatch_t_index_zero!`
  (count_nonzero).

Still missing after this wave: `where` (G-037), `nonzero`,
`searchsorted`, `take_along_axis`, `isin`, `unique_*`, `sort`/`argsort`,
`repeat`/`roll`/`tile`.

## Review round (commit `4bce438`)

`/code-review max` over the W4+W5 branch: no scope deviation from the
initial prompt; five real edge-case bugs surfaced (in the branch's own
edits and in the code the new bindings exposed), each fixed rust-side
with a regression test and a `tracking/numpy_differences_resolved.md`
entry, plus one shim-side input fix. Committed `4bce438` (17 files,
+244/−26; branch total 23 files, +1173/−31).

1. `Layout::diagonal` gated the super-diagonal range on rows instead of
   cols — `eye(2, 4, k=2)` produced an all-zero matrix and
   `eye(3, 1, k=2)` a bogus layout-overflow error; every `rt::diagonal`
   consumer (eye, diag, indexing) shares the fix (G-064).
2. `squeeze` validated only the head of the descending-sorted axis list:
   a mixed list like `(-4, 0)` kept the invalid `-1` and succeeded; every
   mapped axis is validated now (G-065).
3. `take`/`index_select` bounds-checked `max().unwrap_or(&0)`, testing a
   sentinel against an empty axis; empty indices on a zero-length axis
   now return the empty selection (G-066).
4. `argmax`/`argmin` rejected an empty *output* (`zeros((2, 0))` with
   `axis=0`); the guard now applies to the split *reduced* axes, so an
   empty output is legal while an empty reduced axis still raises
   (G-067).
5. `tril`/`triu` surfaced a bare `AxisError` for rank-1 input
   (IndexError through the array-API wrapper); the kernels assert
   `ndim >= 2` up front, giving a clean dimension error (G-068).

Also in the review commit: `meshgrid()` with zero vectors now returns
`()` at the binding (legal input per 2025.12 — "the `indexing` keyword
has no effect"); the Python-side validations the review found were moved
into the kernels/binding so `api.py` is back to the committed
wrapper-only state; the linspace "NumPy parity" claims were scoped to
float64.

Conformance after the fixes: **996/304/82** with a test-level 0-flip
diff versus the pre-review run (`20261006-132433` → `20261006-144402`,
`harness/reports/COMPLIANCE-FULL-20261006-144402.csv`) — the five fixed
cases sit outside the suite's generators (the eye/tril/triu value checks
wait on `where`, G-037). Core gates re-run locally: `entry_row_cpu`
350/350, rstsr-common 40/40, doctests 194, col-major 134 (release),
fmt + clippy `-D warnings` clean (3 configs).

## PR #113 (merged)

Branch pushed to the fork; PR #113 `rstsr-faer-py (feat):
creation/manipulation + searching surface, with core fixes` against
`main` — all 13 checks green on the first run (rustfmt, clippy, 4×
unittests, unittests-col-major, unittests-faer-linalg, unittests-pthread,
doctests, integration-tests, 2× no-std). Merged by the owner 2026-10-06
as squash `c0ea36b`; no fix-up commits were needed.

## Reproduce

```bash
cd crates-interop/rstsr-faer-py
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
cd <task>/harness && NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh
```
