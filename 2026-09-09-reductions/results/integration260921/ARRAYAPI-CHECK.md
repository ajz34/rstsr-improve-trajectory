# Array-API 2024.12 compliance check — reduction NaN semantics (2026-09-21)

Follow-up to the variant-V decision (strip A2): check rstsr against the
Python array API standard for the min/max reduction family. Spec text taken
from the normative sources (`src/array_api_stubs/_2024_12/{statistical,
searching}_functions.py` at data-apis/array-api, the docstrings rendered into
the 2024.12 pages rstsr's docstrings link to). Empirical behavior measured by
`examples/arrayapi_check.rs` in the T2' dir (rstsr @ variant V, but the
min/max NaN behavior is identical on master — A2 never changed values).

## Verdict table

| case | rstsr actual | array-api 2024.12 | verdict |
|---|---|---|---|
| `min/max_all` mid-NaN `[1,NaN,3]` | `1` / `3` (skip) | **NaN (propagation REQUIRED)** | **DIVERGENT** |
| `min/max_all` NaN-first `[NaN,5,3]` | `3` / `5` (skip) | NaN | **DIVERGENT** |
| all-NaN | `f64::MAX` / `f64::MIN` (seed) | NaN | **DIVERGENT** |
| `min/max_axes` column with NaN | skips (per-column value) | NaN for that column | **DIVERGENT** |
| `sum/mean` with NaN | NaN | NaN | ✓ |
| zero-size `min_all` | Err/panic | implementation-defined (error permitted) | ✓ |
| zero-size `sum_all` | `0` | implementation-defined (NumPy: 0) | ✓ |
| ±0.0 tie order | returns the LATER operand's zero | implementation-defined | ✓ |
| `argmax` NaN `[1,NaN,3]` | `2` (skip mid-stream) | **unspecified** (no special case) | ✓ permissible (NumPy says 1; divergence documented in test_argmax.rs) |
| `argmax` ties | first occurrence | first occurrence | ✓ |

## Key asymmetry in the standard

- **`min`/`max` (statistical): NaN propagation is REQUIRED** — "If `x_i` is
  `NaN`, the minimum/maximum value is `NaN`". rstsr's `ExtReal::ext_min/
  ext_max` (floats → `Float::min`/`max`, minnum NaN-skip; ints → `Ord::min/
  max`) diverges. This is pre-existing (386948be → present, unchanged by any
  campaign patch).
- **`argmax`/`argmin` (searching): the spec is silent on NaN** — rstsr's
  documented skip-divergence is standard-compliant there; only NumPy parity
  is affected.

So "array-api comply first" bites exactly one place in this family: the
value min/max NaN policy, which lives in `rstsr-dtype-traits/src/ext_real.rs`
(float impls) plus the all-NaN seed choice in the reduction wiring
(`f_init = T::ext_max_value` / `ext_min_value`).

## Notes for a future compliance fix (owner decision, not in this patch)

1. Semantics: propagate NaN — accumulate with NaN-poisoning (any-NaN OR-mask
   + final select, or total-order accumulate) and seed choice becomes moot;
   `nanmin`/`nanmax` would then be the skip variant (rstsr currently has
   nanargmin/nanargmax only; array-api has no nan* functions at all).
2. Consistency win: fixes the current internal inconsistency where
   `argmax` reports "no NaN position" while NumPy's argmax points at the
   NaN, and where `max_all` and `sum_all` disagree on whether NaN matters.
3. Cost risk: the arg campaign measured first-NaN-wins as de-vectorizing
   (+55..+100% kernel-level). For VALUE min/max a branchless
   `(x > acc) ? x : acc` + OR-reduce of `is_nan` should be much cheaper
   (one extra cmpeq/or lane + final blend), but unmeasured — needs its own
   mini-bench before committing.
4. Blast radius: `ext_real.rs` semantics change touches min/max wiring in
   rstsr-core serial + rayon-auto impls (and `min_axes`/`max_axes`
   fixups); integers unaffected. The T2' experiment gates lock
   CURRENT-BEHAVIOR (skip) — they'd need the same semantic flip.
5. Main-suite coverage: `tests/core_func/reduction/test_min.rs` etc. have NO
   NaN cases today (NumPy-parity harness only ports non-NaN NumPy cases),
   so no test currently enshrines the skip behavior upstream — clean to
   change.
