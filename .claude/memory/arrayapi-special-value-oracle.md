---
name: arrayapi-special-value-oracle
description: "The array API special_cases table grades against the spec stub docstrings, not numpy (numpy uses glibc complex math on Linux); the table is stricter than numpy in two zero-sign cases."
metadata:
  type: reference
---

`test_special_cases.py` builds its table by parsing the **spec stub docstrings**
(`array-api/src/array_api_stubs/_2025_12/elementwise_functions.py`), so that file
— not numpy — is the oracle for special values. Read it before implementing any
special-value rule.

Two traps found during C1 (2026-10-07, rstsr PR #122):

- **numpy on this Linux box calls the system complex math (glibc)**, not the
  vendored `numpy/_core/src/npymath/npy_math_complex.c.src` fallbacks it ships —
  the two disagree in the NaN corners (`cacosh(0+NaN i)`, `ccosh(0+i∞)`,
  `ctanh(0+i∞)`). Probing glibc from C must use **runtime** values (`volatile`
  inputs, `-fno-builtin`): constant-folding `INFINITY`/`NAN` literals gives a
  third, wrong answer and cost a debugging round.
- the stub table is **stricter than numpy** in the zero-sign cases
  `expm1(±0 ± 0i) -> +0 + 0j` and `tanh(±inf + iy) -> 1 + 0j` — those rows carry
  no "sign of the imaginary component is unspecified" note, unlike their
  neighbouring `y = ±inf` / `NaN` rows. numpy passes both only by hypothesis-draw
  luck; rstsr passes them deterministically. Expect such nodes to look green on a
  warm hypothesis DB and flip on `FRESH=1`.

Related: [[rstsr-arrayapi-compliance-notes]]
