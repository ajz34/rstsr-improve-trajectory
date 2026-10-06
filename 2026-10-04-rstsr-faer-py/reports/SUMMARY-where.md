# `where` — the searching-surface slice (2026-10-06)

Task (owner): implement `where` in rstsr-faer-py — rstsr-core's `rt::where`
landed as PR #114 (merged `076270b`; scalar-bound follow-up #115 merged as
`1b09497` during this session) and the binding was the remaining G-037 item of
the searching wave. Wrapper-only rule in force: the shim marshals and calls
rstsr; any missing capability is a rust-side gap. Rust-side *bug* fixes are
authorized for this task series (W3–W5 directive).

Wheel = rstsr branch **`261006/faer-py-where`** (worktree `tmp/faer-py-where`,
later relocated to `~/rstsr_pack/rstsr-local-workspace` on 2026-10-06; base
`1b09497` = merged #115). Two commits (`155e48f` trait fix, `8d4cb28`
binding), pushed to the fork; **PR #116, all 13 checks green on the first
run** (rustfmt, clippy, 4× unittests, col-major, faer-linalg, pthread,
doctests, integration, 2× no-std), **merged by the agent on the owner's go
as squash `05cf6ce`** — no fix-up commits needed.

## Result

| run | passed | failed | skipped |
|---|---|---|---|
| W5 (start of session) | 996 | 304 | 82 |
| **`where` (this wave)** | **1014** | **286** | **82** |

Full maps, 19/19 chunks, `NO_EXPLAIN=1`:

- warm-DB stamp `20261006-180325` — 18 flips, 0 regressions
- **canonical FRESH stamp `20261006-180651`** (hypothesis DB deleted first) —
  totals identical and a test-level 0-flip diff versus the warm run
- final-source wheel (a comment-only edit to the section header landed after
  the recorded build) rebuilt and re-run chunked: same 1014/286/82, 19/19
  chunks; smoke probe and both `where` tests re-verified

Table `harness/reports/COMPLIANCE-FULL-20261006-180651.csv`; per-test diff
against W5 (`…-144402.csv`) printed 18 `failed → passed`, nothing else.

Flips (18): the 12 tests blocked by `"where is not defined"` —
`test_where`, `test_where_with_scalars`, `has_names[searching-where]`,
`func_signature[where]`, and the verification paths of `test_getitem`,
`test_asarray_arrays`, `test_eye`, `test_tril`, `test_triu`, `test_linspace`,
`test_unstack`, `test_positive[positive/__pos__]` — plus
`test_nan_propagation[{mean,prod,std,sum,var}]`, whose generators build the
NaN-positions input with `xp.where`. (`test_nan_propagation[{max,min}]` stay
red on the independent reduction-NaN gap, below.)

## Binding (wrapper-only)

- Rust (`src/ops.rs`): `op_where` is a thin `rt::where_f` call whose output
  type is rstsr's own promoted type (`TX: DTypePromoteAPI<TY>`); the pyrfunction
  is `where` (Rust keyword forces the `where_` item name; the attribute
  restores it). The condition must be a bool tensor — other dtypes raise.
- Dispatch (`src/any_tensor.rs`): new `dispatch_where!`, a 169-arm ternary
  table over the 13 canonical dtypes (every pair in rstsr's promotion matrix),
  each arm calling the same generic fn item; the result variant is derived from
  the promoted type, so **no promotion table is duplicated in the shim** (same
  construction rule as the W2 `dispatch_bin_promote*` macros).
- Python (`api.py`): `where(condition, x1, x2, /)` validates a bool Array
  condition, then marshals `x1`/`x2` through the existing `_operands`
  weak-scalar path (scalar → 0-d array of the array operand's dtype, spec
  "Mixing arrays with Python scalars"); both-scalar calls raise (spec: at
  least one of `x1`/`x2` must be an array). No value logic in Python.

Known declined cases (registered, not worked around):

- non-bool condition → TypeError (standard: condition "should have a boolean
  data type"; the suite draws bool only; rstsr's kernel is bool-typed).
- complex Python scalar against a real array (`where(cond, f32_arr, 1j)`) →
  TypeError — the pre-existing G-009 weak-scalar family limit shared by every
  binary op, not where-specific.

## Rust-side fix: the missing `i8 x i16` promotion (G-069)

The 169-arm table surfaced exactly one missing pair: `DTypePromoteAPI<i16> for
i8` (and its `DTypeCastAPI<i16> for i8` twin) — `rstsr-dtype-traits/src/
promotion.rs` had `impl_promotion_asable!(i8, i16, …)` swallowed by a section
comment (`// internal typeimpl_promotion_asable!(…)`) since the 2025-09-29
promotion commit `ce977a1`; every `DTypePromoteAPI`-bound op (maximum/minimum/
floor_divide/atan2/copysign/hypot/nextafter/logaddexp, comparisons, and now
where) declined the pair while the mirror row `i16 x i8` existed. Fixed by
restoring the line (flags `false, true, i16` mirror `i16, i8, true, false`,
`Res = i16`, NumPy's `result_type` and the suite's own table agree) and adding
an inline `#[cfg(test)]` module to `promotion.rs`: a compile-time completeness
check over the full 13 × 13 matrix, `Res` checks for representative pairs, and
value checks for the restored row.

The suite draws both `(int8, int16)` and `(int16, int8)` orderings, so
`test_where` fails on ~half of its int8×int16 draws without this fix.

## Verification

- Smoke probe (dev-time, numpy-referenced): same-dtype basics, 3-way
  broadcasting, the mixed pairs `int8×int16`, `int16×int8`, `int8×uint8`,
  `float32×float64`, `complex64×float64`, `uint64×int64`, `complex64×complex128`
  (value + dtype), weak scalars both sides, bool dtype, 0-d, zero-size, and the
  two error paths — all pass.
- Core gates (worktree): `rstsr-dtype-traits` 8 passed (3 new); `rstsr-core
  --test entry_row_cpu` 364 passed incl. all `core_func::operators::test_where`;
  `cargo fmt --check` and `clippy --all-targets -D warnings` clean on both
  changed crates.

Left red around the new surface (unchanged, registered): reduction NaN
propagation for `max`/`min` (G-056 family, verified independent of `where`:
`xp.max([1.0, nan]) → 2.0`), `nonzero`/`searchsorted`/`sort` still absent
(G-029 remainder).

## Reproduce

```bash
cd <workspace `/home/a/rstsr_pack/rstsr-local-workspace`>/crates-interop/rstsr-faer-py
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
cd <task>/harness && NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh   # FRESH=1 for canonical
```
