---
name: num-complex-naive-division
description: "num::Complex division is the naive |b|^2 form and returns 0/NaN/inf when |b| is outside about 1e±154 (f64) or 1e±19 (f32); NumPy/SciPy divide overflow-safe (scale + Smith); needed for LAPACK GGEV alpha/beta (beta scales with norm(B))."
metadata:
  type: reference
---

`num::Complex<T>`'s `Div` (num-complex 0.4.6) computes
`(a.re*b.re + a.im*b.im) / |b|^2` — no scaling, no zero handling.

- `|b|^2` overflows to inf for `|b| > ~1.3e154` (f64) / `~1.8e19` (f32):
  the quotient becomes `0` (finite numerator) or NaN.
- `|b|^2` underflows to 0 for `|b| < ~1e-154` (f64) / `~1e-19` (f32):
  the quotient becomes inf/NaN although it is representable.
- NumPy (`npy_cdiv`: scale by `1/(|re|+|im|)`, then Smith's algorithm) and
  therefore SciPy return the true value in all these cases.

Where it matters: `w = alpha / beta` of LAPACK GGEV. LAPACK's `BETA` is
"less than and usually comparable with norm(B)", so `B ~ 1e20 I` in f32 or
`B ~ 1e160 I` in f64 silently gave `[0, 0]` / NaN eigenvalues in rstsr
(found by review 2026-11-25, fixed with a local `cdiv`: real `b` divides
component-wise, complex `b` by Smith's algorithm). `Complex / real` (by a
real scalar) is component-wise in `num` and is fine.

Do not build a regression test with `B` tiny next to a normal `A`: LAPACK's
QZ deflation tolerances are relative, so `B = 1e-200 I`, `A = O(1)` is a
*singular* pencil to LAPACK (beta = 0, inf eigenvalues) — scale the pencil
as a whole (`(s A, s I)`) to keep the eigenvalues O(1).
