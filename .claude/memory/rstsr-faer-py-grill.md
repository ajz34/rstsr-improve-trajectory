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
  (`reports/SUMMARY-s0-baseline.md`). Machine rule: github via SSH only
  (CLAUDE.md).
- **S1 DONE 2026-10-04, review point 1 pending**: crate
  `rstsr/crates-interop/rstsr-faer-py` on branch `261004/rstsr-faer-py`
  (UNCOMMITTED in rstsr — repo rule; user review first). pyo3 0.29.3
  (features abi3-py310 + py-clone; Py<T>: Clone is feature-gated), maturin
  1.15, wheel installed into torch env; python-side edits = cp to
  site-packages, rust edits = wheel rebuild (~30 s incremental). First red
  map: **253 passed / 1026 failed / 87 skipped of 1366** — suite runs
  END-TO-END. **OOM corrected post-S1 (user question)**: NOT json-report
  (whole-suite failure longreprs = 3.1 MB) — it was an **rstsr-native-impl
  arange infinite loop** (G-030, rust-side, candidate issue #1):
  `arange(0, 4.15e9, step=-1.3e8)` → f64 fast path bails (`ceil(neg).to_usize()`
  → None) → generic `while current < end` fallback has no direction guard →
  marches to −∞, ~48 GB in <1 min. Int path survives by accident
  (`(0..neg_isize)` empty). Two shim bugs fixed same session:
  step dropped when stop=None (arange(start, step=huge) → arange(huge)!),
  and sign-mismatch now returns empty (spec-exact guard in shim Rust).
  Corrected red map: **255/1040/87 of 1382 (19 chunks, stamp 20261004-194249)**;
  the earlier 253/1026/1366 silently lost the creation chunk (run.sh now
  hard-errors on missing chunk reports; collection = 1382 = NumPy baseline
  exactly, nothing "vanishes"). Probe tricks: pytest fd-capture swallows
  prints (use sys.__stderr__); Rust alloc failure aborts before hypothesis
  prints "Falsifying example" (wrap the shim with a call logger). Top reds:
  `__getitem__` (blocks hypothesis data generation — 0/155 operators),
  missing elementwise fns (241, all on rt::), xp.linalg, manipulation
  remainder. Rust-side facts in register G-016..G-020: zeros/full gate on
  `T: Num` (bool excluded), OpAllAPI/OpAnyAPI bool-only,
  TensorIsNanAPI float/complex-only, DTypeCastAPI subset-only, no tensor
  astype, empty_f unsafe. Next: user review → S2 DLPack or reordered S3
  (getitem first).
- **Repo conventions learned**: `GRILL-*-ANSWERS.md` are user-owned —
  agent never creates/edits/deletes them (verbatim commits OK); agent
  transcriptions go into the next round's questions file as background.
  Fulfillment table = `rstsr-core/src/docs/array_api_standard.md`
  (Y/C/P/D/T/blank; README "full support" is a future-list claim, already
  stale re moveaxis). See [[rstsr-cpu-dlpack-implementation]],
  [[rstsr-arrayapi-nan-compliance]].
