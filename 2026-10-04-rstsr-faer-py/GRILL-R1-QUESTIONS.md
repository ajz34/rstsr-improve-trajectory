# Grill round 1 — RE-PROMPTED after fact-finding (2026-10-04)

The original R1 (asked before this repo's conventions were loaded and before
fact-finding) is superseded by this version, which folds in the user's
mid-round partial answers and all three fact-finder reports. Answers go in
`GRILL-R1-ANSWERS.md` (or on screen; the agent transcribes them there).

## Partial answers already given by the user

- Validation-only. NOT a distributed multi-device Python wrapper
  (maintenance efficiency). `2026-10-04-rstsr-cpu-pyo3` abandoned.
- Package lives in the rstsr repo, "some proper subfolder" (which one: Q5).

## Fact base (condensed from the three fact-finder reports, 2026-10-04)

**rstsr repo** (main @ `69c97bf`, v0.9.0, nightly toolchain, MSRV 1.82):

- Devices: `DeviceCpuSerial` / `DeviceCpuRayon` (base) / `DeviceFaer`
  (own rayon pool + faer gemm). `DeviceFaer` elementwise/reduction/creation
  coverage is the generic bound set: i8..i64, u8..u64, i128/u128 (ExtNum),
  f32/f64, c32/c64, f16/bf16 (feature-gated), bool (storage, sum/all/any,
  comparison output, and/or/xor). faer gemm f32/f64/c32/c64 only, generic
  fallback otherwise. Bitwise ops exist for ints.
- Known op gaps vs array-api: no sort/argsort, no roll, no repeat/tile;
  linalg traits lack QR entirely; faer impl lacks slogdet + solve_symmetric
  (BLAS impl has both); no general eig/LU traits. Unconfirmed surface
  (first run will resolve): where, nonzero, unique_*, searchsorted,
  result_type, astype.
- `rstsr-core/src/docs/array_api_standard.md`(+`_extension.md`) documents
  the claimed surface; readme claims full array-api support natively —
  the suite may falsify a shipped claim.
- Placement precedent: `crates-interop/rstsr-cpu-dlpack`. Zero pyo3/python
  anywhere today; no python CI job.

**improve-trajectory history**:

- `2026-10-04-arrayapi-compliance-notes/TESTING-ARRAY-API.md` (1,143 lines)
  is the authoritative study: three layers (numerical semantics / surface /
  Python protocol); NumPy is NOT an oracle; the suite's own gaps are
  enumerated; NumPy 2.5.1 baseline = 1382 items, ~42 failed in 6 stable
  clusters; `numpy-compliance/` harness works (overlay venv on conda env
  `torch`, `--json-report`, summarize); Phase 1–3 plan with the Python
  binding as **Phase 3 / Option A — the gate, recommended last**; "official
  suite green is necessary but NOT sufficient".
- rstsr-cpu-dlpack MERGED to main (`7297bc5`, `rt::dlpack`): imports
  copy-only, import repr `!Send`/`!Sync`, view export shipped; Python
  boundary contract in its DESIGN.md §8.
- rstsr-cpu-pyo3 abandoned at design stage (nothing built); closing
  insight: a compiled binding pins the backend at compile time
  (`Device` type-generic, not object-safe). Carried design defaults:
  pyo3 + maturin + abi3-py310, no PyPI, thin standalone crate.
- Standing directives: compliance beats efficiency; do not re-litigate
  DLPack-over-rust-numpy, `rstsr-cpu-*` naming, no-pyo3-in-core, capsule
  discipline.

**array-api-tests checkout** (`~/Git-Others/array-api-tests` @ `6c0b59f`,
tag 2026.09.08+4):

- Cannot even import until `git submodule update --init` (array-api @
  `5f847a3`; stubs come from the suite's submodule, not the `ff497ed8`
  study clone).
- Module selection env-only: `ARRAY_API_TESTS_MODULE` (dotted path or
  `exec(...)` snippet). Version: `ARRAY_API_TESTS_VERSION` >
  `xp.__array_api_version__` > default "2025.12".
- NO numpy dependency. Requirements: pytest, pytest-json-report
  (mandatory), hypothesis>=6.151, ndindex>=1.8. Must run from repo root.
- 220 test functions. `test_has_names` parametrizes over every name in the
  standard. fft/linalg extensions auto-skip if `xp.fft`/`xp.linalg`
  attributes are absent.
- Hard import-time requirements: `xp.__array_namespace_info__()` with
  `devices()`, `dtypes()`, `default_dtypes()`, `capabilities()`; every
  array needs `__dlpack_device__()` (probed); hypothesis strategies
  namespace is built on xp.
- `asarray` must accept scalars + arbitrarily nested lists; dtype objects
  need only `__eq__`; 0-d arrays need `__bool__/__int__/__float__/__index__`
  returning Python scalars; `to_device` never called by name; `device=`
  kwargs probed with graceful fallback; `xp.bool` auto-patched from
  `xp.bool_`.
- Signatures: `inspect.signature` vs spec stubs; extra params allowed;
  kwarg NAMES must match the spec (tests call kwargs by name);
  uninspectable callables degrade gracefully.
- Gap machinery built into the suite: `--skips-file`/`--xfails-file`
  (substring nodeid match; defaults `skips.txt`/`xfails.txt`),
  `ARRAY_API_TESTS_SKIP_DTYPES`, `ARRAY_API_TESTS_XFAIL_MARK=skip|xfail`,
  `--json-report`.
- Dtype universe auto-built from names present on xp (absent ⇒ auto-skip).

## Questions

**Q1 — Task identity vs the Phase 1–3 plan.** The compliance-notes plan
recommends the official-suite gate (Option A) LAST, after in-Rust
spec-derived tests (Phase 1) and semantics coverage (Phase 2). This task
jumps straight to A. Challenge: why first? Defensible: the official suite
is the fastest comprehensive red map (it also covers surface + protocol
layers that in-Rust Phase 1 can't), and the stated objective is exactly
official-suite testimony. Proposal: rstsr-faer-py delivers Option A ONLY;
Phases 1–2 stay a separate future rstsr-core workstream, prioritized by
this suite's gap register; inside THIS task, zero rstsr-core feature work —
every missing op is an issue + skip, never a PR from this task.
➡️ Adopt A-first with that strict separation; gap-fixing happens as
rstsr-core PRs that re-run this harness (fix ⇒ re-run ⇒ register shrinks).

**Q2 — Python boundary rule, restated with concrete cases.** Allowed in
Python: shim functions with exact stub signatures; `__array_namespace_info__`
/`capabilities()`; `Device` singleton and dtype objects (equality-only);
`__dlpack_device__` returning the constant `(1, None)`; pure marshalling
(nested lists → Rust asarray, slice-tuple normalization, forwarding weak
Python scalars into Rust ops). Forbidden in Python: promotion tables
(`result_type` semantics must come from Rust or be a gap), any numeric
computation, any fallback for a missing op.
➡️ Adopt. Consequence accepted: if rstsr lacks result_type/astype/where/
nonzero/unique/searchsorted, those become recorded gaps, not Python helpers.

**Q3 — Scope and pins.** Expose xp core + `xp.linalg`; do NOT expose
`xp.fft` (extension auto-skips when absent). Pin suite @ `6c0b59f` +
submodule `5f847a3`, version 2025.12 — identical pins to the NumPy baseline
run, so results are comparable. DLPack: implement `__dlpack_device__` now
(import-time hard requirement); DEFER `__dlpack__`/`from_dlpack`
(test_dlpack's 3 tests recorded as gap) with `rt::dlpack` as the M2 path.
➡️ As stated. Alternative: wire DLPack exchange in M1 since the crate is on
main — rejected for M1 as capsule wiring risk for 3 tests' worth of signal.

**Q4 — Device and dtype exposure.** Bake a single `DeviceFaer` instance;
`device=` kwarg accepts only it; `__dlpack_device__` → `(1, None)`. Expose
exactly the 13 canonical dtypes: bool, i8/i16/i32/i64, u8/u16/u32/u64,
f32/f64, c64/c128. Do NOT expose i128/u128/f16/bf16 (outside the standard;
suite auto-skips absent names anyway).
➡️ As stated — the device genuinely supports all 13, so maximal honest
coverage; gaps will then reflect ops, not dtypes.

**Q5 — Placement and wiring.** `crates-interop/rstsr-faer-py` as a cargo
workspace member (shares nightly pin + dep versions), maturin project at
that directory, Python module name `rstsr_faer`. Carried from the abandoned
design: abi3-py310, no PyPI, thin standalone crate. Precedent:
`rstsr-cpu-dlpack-test.yml` workflow shape (an unmerged `d84f235` exists of
exactly that shape for dlpack).
➡️ Workspace member at `crates-interop/rstsr-faer-py`. Alternative
(standalone non-member) only if workspace builds break on the python env —
CI jobs build `-p` so this is unlikely.

**Q6 — Object model.** Python class `Array` wrapping an opaque pyo3 tensor
handle; ALL protocol on it lives Python-side (shape/ndim/size/dtype/device/
T/mT properties, `__getitem__` normalization, dunders, scalar casts
`__int__/__float__/__bool__/__index__`); the pyo3 layer exposes only
constructors + free op functions over handles. This deliberately puts more
in Python than a pyclass-first design — but every added piece is protocol
glue per Q2, and it keeps CPython semantics out of the Rust layer.
➡️ Python-class-first; pyo3 = ops library, not the object system.

**Q7 — Harness.** Clone the numpy-compliance harness pattern into the
package dir: setup.sh (overlay venv on conda `torch`, init suite submodule,
install pytest-json-report/hypothesis/ndindex), run.sh
(`ARRAY_API_TESTS_MODULE` exec-snippet importing `rstsr_faer`,
`--json-report`, summarize). Suite checkout stays in `~/Git-Others`, not
vendored into rstsr. Commit summarized reports, never raw.
➡️ As stated.

**Q8 — Gap mechanics.** Use the suite's native `--skips-file`/`--xfails-file`
(files live with the harness; every entry reason `rstsr-gap #<issue>`).
Missing feature ⇒ skip; implemented-but-divergent ⇒ xfail; every entry ⇒ a
real rstsr GitHub issue. Human-readable gap register (categorized counts +
issue links, doc_coverage.csv style) maintained in THIS task dir while the
workstream is active; promoted into the rstsr repo when stable.
➡️ As stated.

**Q9 — CI.** M2 fast-follow, not M1: workflow `rstsr-faer-py-test.yml` —
maturin build (nightly), pinned suite clone + submodule, run, upload json
report artifact. M1 is local-only.
➡️ As stated.

**Q10 — M1 definition of done (challenge).** "The suite runs END-TO-END
against rstsr-faer-py producing an honestly-red map + complete gap
register + zero rstsr-core changes." No gap-fixing inside this task, even
for cheap ones (sort/argsort will be tempting). Predictable first-red
list: sort/argsort, roll, repeat/tile, QR, slogdet, solve_symmetric, plus
unconfirmed surface (where/nonzero/unique/searchsorted/result_type/
astype) resolved by the first run.
➡️ Adopt. First fixes become follow-up rstsr-core PRs that prove the loop.

## Process notes (conventions, not questions)

- Branch `261004/rstsr-faer-py` off rstsr main; push to fork `ajz34`; PRs
  against RESTGroup; no auto-commit in the rstsr repo; this
  improve-trajectory dir commits autonomously with co-author trailers.
- Suite submodule init is part of harness setup, not a manual step.
- Session record + memory updates land in this directory per repo
  convention.
