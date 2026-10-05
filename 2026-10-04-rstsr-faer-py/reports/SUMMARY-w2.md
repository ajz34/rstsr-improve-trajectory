# SUMMARY — W2 (elementwise + operator dunders), 2026-10-05

Wave W2 of `FAILURE-REDUCTION-PLAN.md`, executed shim-side under the
wrapper-only rule, with one owner-authorized rust-side addition (`positive`).

## Result

| stamp | passed | failed | skipped | note |
|---|---|---|---|---|
| `20261004-233924` (W0/W1 baseline) | 320 | 980 | 82 | before this pass |
| `20261005-020258` (chunked, 19/19 chunks; final wheel) | **902** | **398** | 82 | after W2 |

Net: **+582 passed / −582 failed**, no new skips (a ±1–2 wobble between
stamps comes from hypothesis DB replay, per the harness notes).

**Post-review follow-up (2026-10-05).** The owner asked for `positive` to be
implemented like `OpNegAPI` (the same trait machinery, but no operator trait
bound on the element type); the implementation was reworked (details below)
and the wheel rebuilt. Full chunked re-run on the rebuilt wheel (stamp
`20261005-094602`, 19/19 chunks): **902 / 398 / 82 — identical to the table
above**. The same pass fixed all remaining compile warnings (shim: unused
`py` + three dead wrapper fns; clippy: needless `Ok(?)`, empty line after doc
comment; workspace: two unused-dependency warnings). Per file:

| suite file | before (p/f) | after (p/f) |
|---|---|---|
| test_special_cases.py | 135 / 482 | 507 / 110 |
| test_has_names.py | 60 / 154 | 122 / 92 |
| test_operators_and_elementwise_functions.py | 15 / 140 | 100 / 55 |
| test_signatures.py | 51 / 124 | 113 / 62 |
| all others | unchanged | unchanged (W3–W7 territory) |

## What landed (shim only, no rstsr change unless noted)

- **Elementwise surface, 56 names** bound over `rt::`: the transcendental
  family (bool rejected; integers promote to float64 as rstsr's
  `DTypeIntoFloatAPI` prescribes), the dtype-preserving family, predicates,
  and the binary family (`maximum minimum floor_divide atan2 copysign hypot
  nextafter logaddexp remainder pow bitwise_* logical_*` + `square`).
- **Operator dunders** on `Array`: `__pow__ __floordiv__ __mod__ __and__
  __or__ __xor__ __lshift__ __rshift__ __invert__ __neg__ __pos__ __abs__`
  (+ reflected/in-place forms). In-place dunders are spec-legal
  `self = op(self, other)` forms.
- **Mixed-dtype pair dispatch** for the `DTypePromoteAPI`-bound ops
  (generated from rstsr's own promotion impls, not a shim-side table).
- **Scalar operands** for every binary function (namespace + dunders),
  same-kind weak-scalar policy as before; cross-kind scalars stay declined
  (G-009).
- **`positive` added rust-side** (owner-authorized; reworked after owner
  review to follow the `neg`/`not` machinery): `TensorPositiveAPI` +
  device kernel `OpPositiveAPI` in `rstsr-core` — no operator trait bound on
  the element type (the kernel only clones). Borrowed/view inputs produce a
  fresh tensor through the device kernel; an owned input is returned as-is
  (the in-place identity is a no-op). Fulfillment table updated; bound in
  the shim via `rt::positive_f`. Committed 2026-10-05 on branch
  `261004/rstsr-faer-py`: `1546e5e` (rstsr-core) + `799c4b3` (shim).

## Deliberate declines (registered, not worked around)

- Integer inputs to `ceil/floor/trunc/round/conj` (owner ruling: decline now,
  rust-side fix later) — G-052.
- `pow` for integer/complex dtypes — G-053.
- `signbit`: rstsr's kernel computes `is_positive()` (inverted); the shim
  raises instead of returning wrong values — G-054.
- `expm1(complex)` and the complex special-value/accuracy gaps — G-055.
- Mixed-dtype arithmetic (`add/sub/mul/div/rem`), bitwise/shifts, and
  mixed-kind scalars — G-009 (rust-side: those device kernels are bounded on
  same-type Rust ops).

## Remaining 398 failures (census)

| class | n | dominant owners |
|---|---|---|
| missing surface (W3+ names) | 246 | linalg 23, log1p 20, fft 14, where 9, reductions (mean/prod/std/var/max/min/cumulative_*) ~40, manipulation (concat/stack/expand_dims/…), creation (`*_like`, eye, tril…), sort/argsort, searching/set, dtype queries (result_type/can_cast/isdtype) |
| wrong value | 80 | complex special cases (G-055), maximum/minimum NaN (G-056), remainder semantics (G-057), `where`-based verification paths |
| unexpected exception | 60 | complex-kernel declines 12, signbit 9, mixed-dtype arithmetic/bitwise ~24, pow 5, finfo complex 4 |
| indexing gap | 6 | mask/fancy (G-038/G-039) |
| dtype-fn bug | 4 | finfo complex (G-032) |
| other | 3 | 2 unclassified (tan/tanh complex NaN path), 1 axes binding |

Nothing in the remaining set is a W2 name: every elementwise/operator name
is either bound or carries a register entry.

## Observations for the next pass

- Compile time (measured 2026-10-05 with `CARGO_PROFILE_RELEASE_OPT_LEVEL`,
  `maturin build --release`, same tree):

  | opt-level | full build (deps + shim) | shim-only rebuild | full suite time |
  |---|---|---|---|
  | 0 | 1m26s | 1m14s | 60 s |
  | 2 | 6m08s | 5m47s | 59 s |
  | 3 | ≥6m19s (deps cached) | 6m05s | 59 s |

  The pair-dispatch expansion (270 promote arms + 170 equality arms × op) is
  the whole difference vs the pre-W2 ~30 s rebuild. The suite is
  hypothesis-dominated (arrays ≤ 1024 elements, per-array cost far below
  dispatch overhead), so the opt-level is free on the test side — **use
  opt-level 0 for dev iterations** (~5× faster builds, same suite verdict),
  opt-level 3 only for the record wheel. Runtime cost of opt 0 on real
  compute (1e6-element f64, best-of-blocks): add 0.79 ms / exp 3.2 ms at
  opt 0 vs add 0.14 ms / exp 0.55 ms at opt 2/3 (numpy reference: 0.16 /
  0.38 ms) — so opt 0 is unsuitable for perf measurements, but fine for
  correctness runs. opt 2 and opt 3 are indistinguishable in both compile
  and run time. Revisit the arm count if this ever runs per-commit in CI.
- Two shim-side bugs found earlier (G-040 isnan/isfinite/isinf 0-d shape,
  G-041 `all`/`any` keepdims) were left untouched per the "W2 strictly"
  scope; they are 1-line fixes in `ops.rs` and remain registered.
- The `where` dependency (G-037) still blocks `test_positive`'s float
  verification and the test_array_object verification paths; it is the
  highest-value rust-side item for the *next* wave.

## Reproduce

```bash
cd 2026-10-04-rstsr-faer-py/harness
NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh
python3 compliance_table.py reports/rstsr_faer_api-MERGED-20261005-020258.json \
    -o reports/COMPLIANCE-FULL-20261005-020258.csv --census
```
