---
name: nvrtc-int-math-overloads
description: "NVRTC has no integer overloads for rint/signbit/isnan-class functions: `rint(long long)` and `signbit(long long)` are ambiguous compile errors, and `isnanf`/`isinff`/`isfinitef`/`signbitf` do not exist."
metadata:
  type: reference
---

Verified 2026-10-25 with NVRTC 13.2 (and end-to-end through rstsr-cuda on
sm_120):

- `rint(<int>)` → "more than one instance of overloaded function rint
  matches the argument list: rint(float), rint(double) … argument types
  are: (const long long)". `ceil`/`floor`/`trunc` are the exceptions that
  do accept integers (and are identity).
- `signbit(<int>)` → ambiguous among `signbit(float)`, `signbit(double)`,
  `signbit(long double)`.
- `isnan`, `isinf`, `isfinite`, `signbit` have no `f`-suffixed spellings in
  NVRTC (`isnanf`/`isinff`/`isfinitef`/`signbitf` are undefined); use the
  unsuffixed overloads, which accept float and double (and promote half via
  `__half2float`).
- `signbit(int)`/`rint(int)` are exactly what an integer-dtype device op
  produces if it reuses a float expression builder, so a dtype-bound
  (`CudaReal` including ints) alone is not enough to make the op compile.

**How to apply:** emit identity for integer `round`, `((x) < 0)` for
integer `signbit`, and keep float unaries on the unsuffixed overloads; add
a compile-level test per integer width when adding a unary family.

Related: [[ptx-maxnan-f32-sm80]], [[cuda-kernel-nvrtc-grill]]
