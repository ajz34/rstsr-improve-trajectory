# Grill init prompt — test infrastructure redesign

Recorded (lightly edited for structure) from the session prompt of 2026-10-20.

Propose a test infrastructure change for the rstsr ecosystem (rstsr main repo +
rstsr-agents skills/rules/ADRs). Scope:

- Possibly equivalent code refactor in rstsr itself (better-organized tests).
- New `rstsr-agents/rules/testing.md` as the general testing guideline (policy
  level, like code style; exact commands stay in skills).
- Domain-specific test guidelines / skills where missing.

Requirements possibly in play (not an exhaustive list):

- Core tests: row-major numpy-style correctness, device correctness, doc tests
  (doc_draft twins), custom tests.
- Core col-major tests (currently probably stub).
- Regression tests for reported issues.
- Device-specific / BLAS-LAPACK-specific tests (OpenBLAS/MKL/BLIS/AOCL/KML).
- Efficiency benchmarks (future).

Some parts are not implemented yet but need a prepared guideline. Existing test
skills are preserved but should be referenced by the new testing rule. CUDA
device is coming; the guideline should be ready for it. Soft-linking (symlinks)
is common in this project; some tests should not be rewritten.
