# DECISIONS — rstsr-faer-py grill (2026-10-04)

Grill CLOSED after 3 rounds (GRILL-R1/R2/R3 questions+answers in this
directory; user answers files are authoritative). Implementation is ON
HOLD awaiting the user's explicit go ("Wait on impl.").

## Objective

- rstsr-faer-py is a **validation-only instrument**: a thin pyo3+Python
  shim exposing rstsr (DeviceFaer) as a Python array module so the official
  array-api-tests suite can grade it. NOT a product, NOT a distributed
  multi-device wrapper.
- **A-first**: this is Option A / Phase 3 of the compliance-notes plan,
  carved out first. Phases 1–2 (spec-derived in-Rust tests, semantics
  beyond the suite) remain rust-side future work, prioritized by this
  suite's gap register.
- **M1 definition of done**: the suite runs END-TO-END against
  rstsr-faer-py producing an honestly-red map + complete gap register +
  zero rstsr-core edits.

## Hard rules

1. **Zero rust-side code edits** (R2-Q5 override; supersedes R1's "dumb
   little fixes" clause). Rust-side problems become register entries
   marked `rust-side`; fixes only by the user or on explicit permission.
2. **Python boundary**: allowed = signature shims, protocol objects
   (`__array_namespace_info__`/`capabilities()`, Device singleton, dtype
   equality objects, `__dlpack_device__` constant), marshalling (nested
   lists → Rust asarray, slice normalization, weak-scalar forwarding).
   Forbidden = promotion tables, numeric computation, fallbacks for
   missing ops (missing ⇒ recorded skip, never a Python workaround).
3. **rstsr repo stays clean**: package code only; harness, pins, reports,
   skips/xfails files, and the register live in this task dir; any needed
   repos clone to `~/Git-Others`.
4. **No publishing** (PyPI/conda) but packaging-ready (wheel builds
   locally). **No CI** in this task — revisited after rust-side work.

## Scope and pins

- Standard 2025.12; suite pinned `6c0b59f` + submodule `5f847a3` —
  identical to the NumPy 2.5.1 baseline run (~1382 items, 6 known stable
  failure clusters), so results are comparable.
- Expose xp core + `xp.linalg`; do NOT expose `xp.fft` (auto-skips).
  DLPack exchange IN SCOPE: `__dlpack__`/`from_dlpack` via pyo3 PyCapsule
  over merged `rt::dlpack`, import copy-only; both directions; early.
- DeviceFaer singleton (`device=` accepts only it); the 13 canonical
  dtypes (bool, i8–i64, u8–u64, f32/f64, c64/c128); declared defaults
  int64 / float64 / complex128. NOT exposed: i128/u128/f16/bf16.
- Pre-registered expected gaps: sort/argsort, roll, repeat/tile, QR,
  slogdet + solve_symmetric (faer impl), `__pos__`/`__ifloordiv__`/
  `__ipow__` (fulfillment-table D). Unconfirmed surface (where, nonzero,
  unique_*, searchsorted, result_type, astype) resolved by the first run.

## Architecture

- `crates-interop/rstsr-faer-py`: cargo workspace member, pyo3 + maturin +
  abi3-py310, Python module `rstsr_faer`, suite entry `rstsr_faer.api`
  via `ARRAY_API_TESTS_MODULE`.
- **Python-class-first**: Python `Array` wraps an opaque pyo3 tensor
  handle; ALL protocol (properties, `__getitem__` normalization, dunders,
  `__int__/__float__/__bool__/__index__` casts) Python-side; pyo3 layer =
  constructors + free op functions over handles.
- Bind through the facade crate `rstsr` (`rt::`, faer-as-default device
  feature); drop to `rstsr-core` calls only where facade overloads are
  awkward.
- Branch `261004/rstsr-faer-py` off rstsr main; push to fork `ajz34`; PRs
  against RESTGroup only with user permission, as usual.

## Plan (S0–S4) with review gates

- **S0 — Harness first**: adapt numpy-compliance harness into the task dir
  (overlay venv on conda `torch`; `git submodule update --init` in
  `~/Git-Others/array-api-tests`; pytest-json-report/hypothesis/ndindex).
  Gate: reproduce the NumPy baseline locally before any Rust exists.
- **S1 — Skeleton + first red map**: crate + namespace skeleton + Python
  `Array` + `asarray` + ~5-op vertical slice + `__dlpack_device__`; first
  suite run; register v0. → **Review point 1 (mandatory)**.
- **S2 — DLPack early**: PyCapsule wiring over `rt::dlpack`; light review.
- **S3 — Surface categories**: creation → elementwise → manipulation →
  statistical → searching/sorting → linalg; one suite run per category;
  register + skips/xfails grow. → **Per-category reviews (skippable)**.
- **S4 — M1 close**: final honestly-red report + complete register +
  report-only fulfillment-table diff proposal + issue batches; decision
  record + memory updates. → **Final review (mandatory)**.

## Gap mechanics

- Suite-native `--skips-file`/`--xfails-file` (entries reasoned
  `rstsr-gap: #<issue>`); missing ⇒ skip, implemented-but-divergent ⇒
  xfail.
- Register (task dir): cross-reference fulfillment-table row ↔ suite
  outcome ↔ issue; discrepancies in BOTH directions are findings (table-Y
  failing = overclaim; blank/D = expected gap; stale entries). The table's
  C-status column (`%` = matmul) is invisible to the Python suite — shim
  maps `__mod__` → `rt::rem`; register records Python-side conformance.
- Issues on RESTGroup/rstsr: **batched at review points, opened only on
  the user's explicit go** (or filed by the user).

## R3 answers (on screen, agent-transcribed)

- Q1(a) review granularity accepted; Q2(a) issue batching accepted.
- "You can proceed to finalize decisions. Wait on impl."

## Process record

- Grill files R1–R3 in this directory; answers files are user-owned
  (codified in repo CLAUDE.md: agent never creates/edits/deletes them;
  verbatim commits allowed).
- Inputs: three fact-finder reports (rstsr @ `69c97bf`; improve-trajectory
  history; array-api-tests @ `6c0b59f`).
