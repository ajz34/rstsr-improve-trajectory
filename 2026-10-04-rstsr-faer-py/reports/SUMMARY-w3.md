# W3 — statistical functions wave (2026-10-06)

Task (owner): "implement rstsr-faer-py to get closer to array API test
coverage; main task: statistical functions; shim stays a wrapper; fix bugs
and missing algorithms on the rust side; be honest about failures."

## Result

| run | passed | failed | skipped |
|---|---|---|---|
| W2 (start of wave) | 902 | 398 | 82 |
| **W3 (this wheel)** | **934** | **366** | **82** |

Wheel: rstsr branch `261005/faer-py-stats` (off main `cc65a48` = merged PR
#111), opt-level 0 for iteration; pins unchanged (suite `6c0b59f`, spec
`5f847a3`, standard 2025.12, 100 examples, derandomized).

**test_statistical_functions.py: 0/9 -> 9/9.** No failing test anywhere in
the suite is attributed to a statistical function anymore.

## Flips (32 failed->passed, 1 surfaced bug)

- `test_statistical_functions` 9/9: cumulative_sum, cumulative_prod, max,
  mean, min, prod, std, sum, var.
- `test_has_names` +8 (statistical names now exist).
- `test_signatures` +9 (the statistical signatures match the 2025.12 stubs:
  arg names, kinds, and order).
- `test_utility_functions` +2 (all/any with axis/keepdims — G-041).
- `test_special_cases` +4 net (test_empty_arrays for mean/prod/std/var,
  minus the round case below).
- 1 test flipped the OTHER way and exposed a real rust-side bug (G-059):
  `round` was ties-away-from-zero (`f64::round`), the spec requires
  ties-to-even. The derandomized example set drew an exact halfway value
  this run for the first time. **Fixed rust-side** (`round_ties_even_f` in
  both device kernel tables + regression test
  `test_round_ties_to_even` + `numpy_differences_resolved.md` entry);
  re-run green.

## How the bindings work (wrapper discipline kept)

- Rust (`ops.rs`): thin generic wrappers over the rstsr `*_with_args_f`
  families (`ReduceArgs`: axes + keepdims), `var/std_with_args_f`
  (`VarArgs`: + correction), `cumulative_{sum,prod}_f`
  (`CumulativeArgs`: axis + include_initial). `all`/`any` keep the
  registered truthiness cast (`x != 0`) now feeding
  `all/any_with_args_f`. Dispatch macros unchanged in kind:
  `dispatch_t_numeric_same` (sum/prod/cumulative), new
  `dispatch_t_real_numeric_same` (max/min — complex ordering unimplemented,
  conformant decline), float/complex arms by hand for var/std (complex
  input -> real dtype output, `TOut = T::Real`, value path is
  `|x-mean|^2`-based, complex-correct).
- Python (`api.py`): signatures copied from the 2025.12 stubs (kwarg order
  included); `_norm_axes` (None | int | tuple); accumulation dtype rule
  (`_accumulation_dtype`: signed->int64, unsigned->uint64, floats/complex
  keep; `dtype=` passthrough) applied by casting via the existing astype
  path BEFORE the same-dtype reduction — the exact order the standard
  recommends ("the input array **should** be cast ... before computing the
  sum"). No promotion table, no numeric algorithm in either layer.
- Not needed after all: `median` is not in the 2025.12 statistical surface
  (checked the spec stubs), so the sort gap (G-001) does not block this
  category.

## What is achieved vs lacking (statistical category)

| spec function (2025.12) | state | suite evidence |
|---|---|---|
| sum | **achieved** — full signature (axis/dtype/keepdims), accumulation rule | test_sum + test_empty_arrays pass |
| prod | **achieved** — same | test_prod passes |
| max / min | **achieved** — axis/keepdims, dtype preserved | both pass |
| mean | **achieved** — axis/keepdims; ints promote to float64 | passes |
| std / var | **achieved** — axis/correction/keepdims; correction int-or-float; complex -> real dtype | both pass (+ empty-arrays cases) |
| cumulative_sum | **achieved** — axis (None only for 1-D)/dtype/include_initial; no-overflow via cast-first | passes |
| cumulative_prod | **achieved** — same | passes |
| all / any (utility) | **achieved** — axis/keepdims on every dtype (truthiness cast) | test_all/test_any pass |
| round (elementwise, adjacent fix) | **fixed rust-side** — ties-to-even per spec | special-case round family passes |

Nothing statistical remains on the failure list. The remaining 366
failures belong to other categories (missing surface 217: manipulation/
creation/linalg/sorting/searching/set/fft names; wrong-value 77: the G-055
complex-transcendental cluster and friends; unexpected-exception 60:
declined mixed-dtype arithmetic G-009; plus data_type_functions G-027/
G-032, astype device= G-046, reshape pos-or-kw G-047, info keys G-048/
G-050, boolean indexing G-038/G-051, diff).

## Rust-side state (branch `261005/faer-py-stats`, NOT committed — awaiting owner)

Changes, all uncommitted in the working tree:

1. `crates-interop/rstsr-faer-py/src/ops.rs` — statistical section
   (wrappers + pyfunctions; `sum`/`all`/`any` signatures extended).
2. `crates-interop/rstsr-faer-py/src/any_tensor.rs` — new
   `dispatch_t_real_numeric_same` macro (ExtReal reductions).
3. `crates-interop/rstsr-faer-py/src/lib.rs` — register the 9 new
   pyfunctions.
4. `crates-interop/rstsr-faer-py/python/rstsr_faer/api.py` — statistical
   namespace + marshalling helpers + `__all__`.
5. `rstsr-core/src/device_cpu_serial/operators/op_binary_common.rs` +
   `rstsr-core/src/feature_rayon/auto_impl/op_binary_common.rs` —
   `round_ties_even_f` (G-059 fix).
6. `rstsr-core/tests/core_func/math/test_unary_math.rs` — regression test.
7. `rstsr-core/tests/tracking/numpy_differences_resolved.md` — archived
   divergence entry.

Gates on the rstsr checkout: entry_row_cpu 342 passed, faer lib 135,
doctests 194, clippy clean, nightly fmt clean.

## Reproduce

```bash
cd crates-interop/rstsr-faer-py
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
cd <task>/harness && NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh
```
