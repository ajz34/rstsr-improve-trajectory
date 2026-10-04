---
name: rstsr-faer-py-grill
description: 2026-10-04 grill closed - validation-only pyo3 shim (DeviceFaer) graded by array-api-tests; zero rust-side edits; plan S0-S4 with review gates; implementation awaiting user go.
metadata:
  type: project
---

Grill for **rstsr-faer-py** closed 2026-10-04 (3 rounds;
`2026-10-04-rstsr-faer-py/DECISIONS.md` is the full record). Implementation
ON HOLD for the user's explicit go.

- **Purpose**: validation-only instrument — thin pyo3 + Python shim
  exposing rstsr (DeviceFaer, 13 canonical dtypes, defaults int64/f64/c128)
  as an array-api module graded by the official suite. A-first (Phase 3 of
  [[rstsr-arrayapi-compliance-notes]] carved out first). Not a product.
- **Hard rules**: ZERO rust-side edits (problems → register `rust-side`
  entries, fixes only on permission); no Python algorithms/promotion
  tables/fallbacks (missing ⇒ skip); rstsr repo carries package code only
  (harness/register here; repos → `~/Git-Others`); no publishing, no CI.
- **Pins**: standard 2025.12, suite `6c0b59f` + submodule `5f847a3`
  (identical to NumPy baseline; suite needs `git submodule update --init`;
  `ARRAY_API_TESTS_MODULE` env, no numpy dependency; pytest-json-report
  mandatory; `__array_namespace_info__()` + per-array `__dlpack_device__()`
  are import-time hard requirements; kwargs must match spec names).
- **Plan**: S0 harness+NumPy-baseline gate → S1 skeleton + first red map
  (review 1) → S2 DLPack via pyo3 PyCapsule over `rt::dlpack` (copy-only
  import) → S3 surface by category (per-category reviews, skippable) →
  S4 close (report + register + report-only fulfillment-table diff; issues
  batched only on user go). Known expected gaps: sort/argsort, roll,
  repeat/tile, QR, slogdet/solve_symmetric (faer), `__pos__`/`__ifloordiv__`/
  `__ipow__`; unconfirmed: where/nonzero/unique/searchsorted/result_type/astype.
- **S0 DONE 2026-10-04** (gate passed): harness in
  `2026-10-04-rstsr-faer-py/harness/`; suite @ `~/Git-Others/array-api-tests`
  `6c0b59f`, submodule offline-init'd from `~/Git-Others/array-api`
  (submodule NAME is `array_api_tests/array-api`; needs
  `-c protocol.file.allow=always`); deps (pytest/hypothesis/ndindex/
  pytest-json-report) installed directly into conda env `torch` (no venv,
  user instruction). Baseline: 1335/42/5 of 1382 in 39 s, 6 known clusters
  (`reports/SUMMARY-s0-baseline.md`). Next: S1 skeleton (branch
  `261004/rstsr-faer-py`, crates-interop/rstsr-faer-py, rstsr_faer.api),
  review point 1 after first red map. Machine rule: github via SSH only
  (CLAUDE.md).
- **Repo conventions learned**: `GRILL-*-ANSWERS.md` are user-owned —
  agent never creates/edits/deletes them (verbatim commits OK); agent
  transcriptions go into the next round's questions file as background.
  Fulfillment table = `rstsr-core/src/docs/array_api_standard.md`
  (Y/C/P/D/T/blank; README "full support" is a future-list claim, already
  stale re moveaxis). See [[rstsr-cpu-dlpack-implementation]],
  [[rstsr-arrayapi-nan-compliance]].
