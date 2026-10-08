---
name: rstsr-pow-signbit-decisions
description: "C5/C7 (2026-10-08): pow is now a promoted binary op with the kernel in ExtNum::ext_pow, and a negative integer exponent is rejected (InvalidValue), deliberately NOT a numpy divergence; signbit lives on ExtReal."
metadata:
  type: project
---

Decisions settled with the owner while fixing `signbit` (C5, G-054) and `pow`
(C7, G-053) — rstsr branch `261008/signbit-pow`, commit `bacc07b`:

- **`pow` output dtype is the operand promotion** — `OpPowAPI` follows the
  `atan2`/`maximum` shape (`TOut = TA::Res`, kernel `ExtNum::ext_pow`) instead of
  the old `num::Pow` special case. Consequence: `rt::pow(f32, i32)` now returns
  `f64` (consistent with `rt::atan2`); the owner accepted this.
- **a negative integer exponent is rejected**, not given a value. The array API
  leaves `int ** int` with a negative exponent unspecified and numpy raises
  `ValueError`; rstsr returns `None` from `ext_pow` -> an `InvalidValue` error
  (the panic form `rt::pow` unwraps it; the Python binding surfaces `ValueError`).
  The first draft returned `0` and was rejected as a silently-wrong answer.
  Because it matches numpy's intent it is **not** a `numpy_differences.md` entry
  (that file records open divergences only).
- **`ext_signbit` belongs on `ExtReal`**, not `ExtNum` (complex has no sign bit);
  `ext_rem` and `ext_pow` are on `ExtNum` (complex has both). The integer-pow
  reject path carries an `AtomicBool` per device method because the rayon closure
  bound is `Fn + Send + Sync` — a plain flag will not compile.

Enabling a real `signbit` also flips the suite's `_has_functional_signbit()`
probe to true, which switches on the ±0 sign checks inside every strict float
comparison — a latent-regression surface that this wave showed clean.

Related: [[arrayapi-convergence-harness]], [[arrayapi-special-value-oracle]]
