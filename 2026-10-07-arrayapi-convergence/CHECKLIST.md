# Convergence checklist — rstsr_faer.api × array-api-tests (2026-10-07)

Ordered by ROI and by blocking. Counts are "suite nodes moved" (fails flipped,
plus currently-skipped tests that get *activated* where noted). Register ids
(G-nnn) refer to `../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

Working baseline to beat: **1178 / 122 / 82 of 1382** (stamp `20261008-061227`,
`FRESH=1`, `CHUNKED=1`, `NO_EXPLAIN=1`).

**Progress (2026-10-08).** C4 (`remainder`) merged — rstsr PR #123, squash
`8a09076` — moved 1144 / 156 / 82 → 1156 / 144 / 82 (+12). C5 (`signbit`) and
C7 (`pow`) are done on rstsr branch `261008/signbit-pow` (`bacc07b`, PR
pending) and move 1156 / 144 / 82 → 1170 / 130 / 82 (+14). C9 (NaN
propagation) is done on the stacked branch `261008/maxmin-nan` and moved
1170 / 130 / 82 → 1174 / 126 / 82 (+4). C8 (integer `ceil`/`floor`/`trunc`/
`round` dtype preservation) is done on the further-stacked branch
`261008/int-round-dtypes` and moves 1174 / 126 / 82 → **1178 / 122 / 82** (+4,
0 regressions, node set identical). See C5, C7, C8, C9. Group A (shim quick
wins) is still the cheapest remaining block.

## A. Shim-side quick wins — do first (~18 fails, no rust changes)

Wrapper-only edits in `crates-interop/rstsr-faer-py`; each is a register entry,
no algorithms. Lowest risk, immediate payoff.

- [ ] **A1. `finfo` accepts complex dtypes** (G-032) — 4
- [ ] **A2. namespace-info** — `devices()` → tuple, add `default_device()`,
      `dtypes(kind=)`, fix `default_dtypes()` keys (G-048/49/50) — 4
- [ ] **A3. 0-d result shape** in `isnan/isfinite/isinf` int/bool fallback (G-040) — 3
- [ ] **A4. creation `empty/full/ones/zeros` `shape` pos-or-kw** (G-047) — 4
- [ ] **A5. `astype(..., device=)` acceptance** (G-046) — 1
- [ ] **A6. u64 PyScalar carrier** for `bitwise_invert` (G-013) — 2

Exit check: full chunked run, expect **1178 + ~18 = ~1196 passed / ~104 failed
/ 82 skipped**, 0 regressions (test-for-test diff).

## B. Decisions needed before more binding work (each is a fork, not code)

- [ ] **B1. `linalg` namespace** — implement `xp.linalg` vs. scope-out via
      task-side `SKIPS_FILE`. Implement: **flips 37 fails + activates 49 skips
      = 86 tests** — the single biggest lever. Most kernels exist rust-side;
      QR/slogdet/solve_symmetric do not (G-004/G-005). (G-023)
- [ ] **B2. `fft`** — build an FFT subsystem vs. declare out of scope (W7 open).
      **42 tests** hinge (14 fails + 28 skips).
- [ ] **B3. dtype functions `can_cast/isdtype/result_type`** (G-027/G-008) —
      rstsr promotion exists only as associated types, no token-level query;
      needs a rust-side design decision. 12 fails.
- [ ] **B4. `clip`** (no rstsr primitive) — rust-side; 3 fails. (`log1p` done —
      see C3.)

## C. Rust-side queue — ordered by failures per work item

Register + request; never fix agent-side.

- [x] **C1. G-055 complex transcendentals — DONE** (rstsr PR #122, squash
      `b2e22ab`): new `rstsr-dtype-traits::c99_complex` carries the C99 Annex G
      routines for `sqrt/cosh/sinh/tanh/tan/acos/asin/acosh/asinh/atanh` — the
      inverse trig via the Hull–Fairgrieve–Tang crossover for large `|z|`;
      `ExtComplexFloat` gained one method per function (real `f32`/`f64`
      delegate to libm, so only complex takes the new path) and both device
      tables were re-pointed. All 53 fixed (48 special values + 5 elementwise).
      Also fixed the *draw-dependent* `expm1(±0±0i)` and `tanh(±inf+iy)`
      zero-sign cases (G-075/G-076). Note: the oracle is the spec stub
      docstrings, **not** numpy — see the register's v11 method note.
- [ ] **C2. G-009 mixed-dtype arithmetic & joins — 29**
      (`add/sub/mul/div`, `bitwise_*`/shifts, `concat`/`stack`: Rust bounds are
      same-type only)
- [x] **C3. G-058 `log1p` kernel — DONE** (rstsr PR #121, branch
      `261007/log1p-kernel`): new `rstsr-dtype-traits` trait `ExtComplexFloat`
      (`ext_log_1p` / `ext_exp_m1`) with one device-table row per backend
      serving real + complex; real via `libm::log1p`/`expm1`, complex via the
      compensated `ln(u) − rho/u` and `2 exp(z/2) sinh(z/2)`. Fixed `log1p` (20
      nodes) **and** complex `expm1` (12). Also required exporting
      `log1p`/`log1p_f`/`TensorLog1pAPI` from the rstsr-core prelude.
- [x] **C4. G-057 `remainder` signed-zero / infinite-divisor — DONE** (rstsr
      PR #123, squash `8a09076`): `ExtNum::ext_rem` carries the array-API
      floored remainder (sign of the divisor; unsigned unchanged, signed the
      floored lift of `%`, floats fully special-cased, complex → num-complex's
      Gaussian `%`). `OpRemAPI` keeps the `Rem` bound but dispatches the float
      dtypes by `TypeId` to `ext_rem` (f32/f64; f16/bf16 on the serial device
      only — the shared rayon module cannot name `half`); both device tables.
      Floats now match numpy incl. the 4 special cases; integers keep Rust's
      `%` for now (noted in the operator module docs). All 12 fixed
      (`remainder` / `__mod__` / `__imod__` × 4).
- [x] **C5. G-054 `signbit` inverted semantics — DONE** (rstsr branch
      `261008/signbit-pow`, `bacc07b`, PR pending): the kernel wrote
      `is_positive()` — the inverse of the sign-bit test — under a `Signed`
      bound. New `ExtReal::ext_signbit` (unsigned `false`, signed `< 0`,
      float/half `is_sign_negative`, so `-0.0`/`-NaN` are correct);
      `OpSignBitAPI` split out of the boolean-output table, bound re-pointed
      `Signed` → `ExtReal`, so unsigned/half are covered. All 9 fixed
      (`test_signbit` + 8 `test_special_cases::test_unary[signbit(±0/±inf/±NaN)]`).
- [ ] **C6. G-038/G-039 mask & fancy indexing — 6**
- [x] **C7. G-053 `pow` int/bool/complex bases — DONE** (rstsr branch
      `261008/signbit-pow`, `bacc07b`, PR pending): `OpPowAPI` moved off the
      `num::Pow` special case to a promoted binary op (`TOut = TA::Res`, the
      `atan2`/`maximum` shape) with the kernel in `ExtNum::ext_pow` — integer
      bases with integer exponents and complex bases now work, and a negative
      integer exponent is rejected with an `InvalidValue` error (NumPy raises;
      the array API leaves it unspecified) instead of an unrepresentable value.
      All 5 fixed (every `test_pow` param).
- [x] **C8. G-052 int `ceil/floor/trunc/round` dtype preservation — DONE**
      (rstsr branch `261008/int-round-dtypes`, stacked on `261008/maxmin-nan`):
      the four were in the into-float `duplicate_item` table
      (`T: DTypeIntoFloatAPI`, `TOut = T::FloatType`), so integer inputs widened
      to f64. Moved them out: `ExtReal::ext_ceil`/`ext_floor`/`ext_trunc`
      (int/uint identity, `Float::…` for float/half) and `ExtNum::ext_round`
      (int/uint identity, `libm::roundeven` ties-to-even for float/half,
      componentwise for complex — spec 2024.12) with `TOut = T`; the four
      rounding impls now sit in the same `/* #region same type */` right after
      the into-float table, on both devices. The hand-rolled
      `rstsr_native_impl::scalar_math::round_ties_even_f` was relocated into
      `ExtNum::ext_round` and deleted (its only callers were the op tables). All
      4 fixed (`test_ceil`/`test_floor`/`test_trunc`/`test_round`) → 1178 / 122 /
      82 (stamp `20261008-061227`, 0 regressions, node set identical). Note
      `conj` (the other half of G-052) is untouched — `test_conj` draws complex
      dtypes only, so its integer declination is ungraded.
- [x] **C9. G-056 NaN propagation (`max/min`, `maximum/minimum`) — DONE**
      (rstsr branch `261008/maxmin-nan`, stacked on `261008/signbit-pow`): the
      float/half `ExtReal::ext_max`/`ext_min` delegated to the IEEE 754-2008
      `minNum`/`maxNum` (Rust `f32::min`/`max`), which *drop* NaN; they now
      propagate NaN (IEEE 754-2019 `maximum`/`minimum`), the array-API
      contract. Elementwise `maximum`/`minimum` and the `max`/`min` reductions
      both route through these two methods on both devices, so the kernel-body
      change alone fixed all 4 nodes (`test_binary[maximum/minimum(NaN)->NaN]`
      + `test_nan_propagation[max/min]`) → 1174 / 126 / 82 (stamp
      `20261008-053535`, 0 regressions, node set identical).
- [ ] **C10. G-044 `negative` on unsigned — 2; G-045 empty `setitem` — 1**

## D. Process / measurement hygiene

- [ ] Build with `--release` at any opt-level, **never** the dev profile
      (debug-assertions flip exactly 10 integer-overflow nodes).
- [ ] `CARGO_PROFILE_RELEASE_OPT_LEVEL=0` for dev iterations; opt-3 only for a
      wheel whose numbers are recorded.
- [ ] Red runs always `CHUNKED=1`; canonical counts with `FRESH=1`;
      `NO_EXPLAIN=1` for speed.
- [ ] After every chunked run: confirm chunks == 19 `test_*.py` files.
- [ ] Before recording a flip: confirm the run's node set is test-for-test
      stable vs. the previous stamp.
- [ ] Fix the wheel-provenance gap (README caveat): point the local workspace at
      the state that builds the recorded wheel.
- [ ] Every divergence gets a `GAP-REGISTER` entry (rust-side / shim-side /
      suite); no rstsr-core edits without owner permission.
- [ ] Heavy Rust builds — notably `cargo test --doc` — need
      `TMPDIR=$HOME/.cache/tmp-cargo`: `/tmp` is a tmpfs with a per-user quota,
      so rustdoc's parallel temp writes hit `EDQUOT` and surface as phantom
      "Couldn't compile the test" failures (env, not code; CI is green).
      
      Human user note: only when error happens, then use this way as last resort.
      In common case, simply `cargo test --doc` should enough.

## Suggested next session

Take **group A** (≈18 fails, low risk, all shim-side) and, in parallel, put the
**B1 (linalg)** decision to the owner — it moves the most tests of any single
item. On the rust queue the order is now **C2** (mixed-dtype arithmetic/joins,
29 — note `pow` already gained the promotion path, so the arithmetic ops can
follow the same shape), then **C6** (mask/fancy indexing 6), then **C10**
(`negative` on unsigned 2, empty `setitem` 1 — another small `ExtReal`
element-trait fix alongside C5/C8/C9).
