---
name: rstsr-arrayapi-compliance-notes
description: Array API conformance workstream seeded 2026-10-04 — study doc + NumPy grading harness transferred to 2026-10-04-arrayapi-compliance-notes; rstsr-side testing not started.
metadata:
  type: project
---

2026-10-04: transferred the external study notes + NumPy grading harness into
`2026-10-04-arrayapi-compliance-notes/` (from `~/Documents/notes-repo/array-api`;
the pdf/typst build stayed behind, `numpy-compliance/reports/` is gitignored).

Contents: `TESTING-ARRAY-API.md` (what and how to test the 2025.12 standard:
10 layers + the unspecified list + §6 Phase 1–3 plan) and `numpy-compliance/`
(harness grading a Python library against `data-apis/array-api-tests` @
`6c0b59f`; recorded NumPy 2.5.1 baseline ~1335 passed / ~42 failed / 5 skipped
= 6 stable rule-violation clusters).

Broader task (stated by the owner): test rstsr's compliance to the Array API
standard. Key doc conclusions: numerical *semantics* (Layer 1) is
language-neutral and the prize; API *surface* (Layer 2) names port; the Python
protocol (Layer 3) is only needed for the official suite (Option A = binding).
Passing the official suite is necessary but not sufficient — no accuracy bound,
no values for fft/linalg/mean/std/var/remainder etc. Phase 1 = spec-derived
tests (surface/signatures from the stubs, promotion tables, port the
`**Special cases**` docstring parser into our own edge-case tables).

Follows [[rstsr-arrayapi-nan-compliance]] (min/max NaN family already landed
under the compliance-beats-efficiency directive). Spec source rule:
`~/Git-Others/array-api` @ `ff497ed8` (2025.12).
