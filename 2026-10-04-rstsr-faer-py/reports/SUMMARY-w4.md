# W4 — creation & manipulation complement (2026-10-06)

Task (owner): bind the 2025.12 creation category (completing it) and the
manipulation complement over rstsr's existing entries; shim stays a
wrapper; rust-side bugs found on the way are fixed with review.

## Result

| run | passed | failed | skipped |
|---|---|---|---|
| W3 (start of wave) | 934 | 366 | 82 |
| **W4 (this wave)** | **984** | **316** | **82** |

Wheel: rstsr branch `261006/faer-py-creation-manip` (worktree
`tmp/faer-py-w4`, base main `2ce2afd` = merged PR #112), release
opt-level 0 for iteration; pins unchanged (suite `6c0b59f` + spec
`5f847a3`, standard 2025.12, 100 examples, derandomized). Full map stamp
`20261006-121144`; per-test table
`harness/reports/COMPLIANCE-FULL-20261006-121144.csv`; flips computed as
a per-test outcome diff against the W3 stamp `20261006-002157`. The
branch later grew W5 + the review round and was merged as PR #113
(squash `c0ea36b`).

**50 flips / 0 regressions:**

- functional: creation `empty_like`, `full_like`, `ones_like`,
  `zeros_like`, `meshgrid`; `broadcast_shapes` (incl. its empty/error
  cases), `broadcast_arrays`, `expand_dims` (+ tuples), `flip`,
  `moveaxis`, `squeeze` — 11 of the 18 new functions pass their
  functional tests.
- `test_has_names`: all 9 creation names + all 9 manipulation names.
- `test_signatures`: the same 18.

Remaining blocked value checks: `eye`/`linspace`/`tril`/`triu`/`unstack`
wait on `where` (G-037 — the suite's own float value-checker calls it);
`concat`/`stack` on mixed-dtype draws (G-009). `repeat`/`roll`/`tile`
have no rstsr primitive (G-002/G-003).

## Surface (wrapper discipline kept)

- creation, Rust: `eye`, `linspace`, `meshgrid`, `tril`, `triu` — thin
  dtype-dispatch wrappers; linspace rides the `ComplexFloat` kernel
  (f32/f64/c64/c128), eye/tril/triu the `Num`-bound creation kernels
  (bool declined, G-062; linspace non-float declined, G-063).
- creation, Python: `empty_like` / `zeros_like` / `ones_like` /
  `full_like` reuse the existing entries with `x.shape` / `x.dtype` — no
  new numeric path.
- manipulation, Rust: `broadcast_shapes`, `concat`, `stack`, `unstack`,
  `expand_dims`, `squeeze`, `flip`, `moveaxis`; the view-returning rstsr
  ops are materialized via `into_owned` (the handle model has no shared
  storage, G-036).
- manipulation, Python: `broadcast_arrays` composes rstsr's
  `broadcast_shapes` + `broadcast_to` so every input keeps its dtype
  (rstsr's own helper is same-dtype-only); `concat(axis=None)` flattens
  through reshape first (the spec's own definition).
- joins stay same-dtype: mismatched parts decline with the G-009
  reference (no shim promotion table, per policy).

## Rust-side fixes (found by this wave; committed on the branch)

1. `triu_ix2_cpu_serial` computed `j_end = max(i + k, 0)` without
   clamping it to `ncol`, so any diagonal leaving the matrix
   (`k >= ncol`, or `M > N` with a negative `k`) indexed past the buffer
   and panicked (`rt::triu(ones((3, 3)), 2)`). The zero range is now
   clamped to the row and `i + k` is saturating; `tril` got the same
   treatment (a huge `k` used to wrap and zero the wrong range). Tests
   `custom_tril_triu::{test_k_outside_row_bounds, test_extreme_k}`
   (`4bd2ab5`; register G-060).
2. `linspace` left the last value at `start + (n - 1) * step`, which
   rounds to a neighbor of `stop` (`linspace(0, 6.4913965932284536e16,
   25)` was one ulp short), and the serial kernel accumulated
   `v += step`, drifting a few ulp (`linspace(2, 10, 100)[-1]` was
   `9.999999999999996`). Both kernels now compute `y[i] = start + i *
   step` and assign the endpoint directly (`6fcabe6`; test
   `custom_linspace::test_endpoint_exact`; register G-061). NumPy parity
   holds bit-for-bit for float64 only — float32/complex compute in the
   output dtype and may differ by a few ulp.

## Reproduce

```bash
cd crates-interop/rstsr-faer-py
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
cd <task>/harness && NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh
```
