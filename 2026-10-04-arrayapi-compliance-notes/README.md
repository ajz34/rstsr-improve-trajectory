# 2026-10-04 — Array API compliance: study notes + NumPy grading harness

- **Date**: 2026-10-04.
- **Task**: seed the "test rstsr's compliance to the Python Array API standard"
  workstream by transferring the external study notes and the NumPy grading
  harness from `~/Documents/notes-repo/array-api` (authored 2026-10-03/04).
- **Status**: transferred; rstsr source untouched. Follow-on rstsr-side
  conformance work not started.

## What this directory is

Two artifacts:

1. **`TESTING-ARRAY-API.md`** — the study: what the 2025.12 standard requires
   (10 test layers: surface, signatures, dtype promotion, broadcasting, the
   special-case value tables, indexing, errors, devices, Python protocol,
   extensions), what *not* to test (the unspecified list), what the official
   suite actually measures, and a strategy + concrete plan for a non-Python
   library (§5–6). Source-verified, with `path:line` citations.
2. **`numpy-compliance/`** — a working harness that grades a Python library
   against the official `data-apis/array-api-tests` suite, plus its recorded
   NumPy baseline. rstsr is not a Python module, so what transfers from this
   part is the *test content and suite architecture* (doc §5) — the harness is
   the reference for the eventual "official gate" (Option A) and documents the
   suite's four oracles and its verified gaps.

## Recorded baseline (NumPy 2.5.1 vs spec 2025.12, 2026-10-04)

Whole suite, suite-default examples, ~40 s, 1382 tests:
`~1335 passed / ~42 failed / 5 skipped` (±2 between runs; the violated *rule
set* is stable). Six clusters: floor_divide/`//` inf handling, `iinfo` array
input, `expm1` complex table, `finfo` Python-float attributes, `fftfreq`
`dtype=`, `clip` parameter names — details in `numpy-compliance/README.md`.
Calibration value: even NumPy deviates from the standard, so a mismatch in a
future differential check is a question, not an answer.

## Provenance

| | |
|---|---|
| source | `~/Documents/notes-repo/array-api` (2026-10-03/04; the pdf/typst build of the doc stays there) |
| spec checkout cited | `~/Git-Others/array-api` @ `ff497ed8` (2025.12 docs) |
| suite | `data-apis/array-api-tests` @ `6c0b59f` (2026-09-15), spec submodule `5f847a38` — pinned in `setup.sh` |
| numpy graded | 2.5.1, conda env `torch` (release, not dev); the `.venv` overlay is rebuilt locally |
| transferred | the doc + `numpy-compliance/{README.md,setup.sh,run.sh,probe_numpy.py,summarize_report.py}` |
| not transferred | `.pdf`/`.typ`/`md2typst.py` (kept in notes-repo), raw pytest JSON reports (`numpy-compliance/reports/` is gitignored here; regenerate with `./run.sh` in ~40 s), `.venv/`, the `array-api-tests/` checkout (cloned by `setup.sh`) |

## Files

| file | what |
|---|---|
| [TESTING-ARRAY-API.md](./TESTING-ARRAY-API.md) | main artifact: WHAT to test + HOW, 2025.12 sources cited |
| [numpy-compliance/README.md](./numpy-compliance/README.md) | harness usage + the recorded NumPy baseline (the six violated rule clusters) |
| [numpy-compliance/setup.sh](./numpy-compliance/setup.sh) | one-time: clone pinned suite + spec submodule, build `.venv` overlay on the torch env |
| [numpy-compliance/run.sh](./numpy-compliance/run.sh) | run the suite; writes a JSON report + summary into `reports/` (gitignored) |
| [numpy-compliance/probe_numpy.py](./numpy-compliance/probe_numpy.py) | what the library under test self-declares |
| [numpy-compliance/summarize_report.py](./numpy-compliance/summarize_report.py) | pytest JSON → failures grouped by spec function |

## Relation to the rstsr trajectory

- `../2026-10-02-arrayapi-nan-compliance/` fixed the min/max NaN family under
  the owner directive "array API compliance beats efficiency" — that is Layer 5
  of this study applied. That session explicitly postponed the systematic
  whole-standard audit and a test mechanism; this directory seeds exactly that
  workstream.
- The doc's headline: the standard has three layers; numerical *semantics*
  (Layer 1) is language-neutral and is the real prize; API *surface* names port
  (Layer 2); the Python protocol (Layer 3) is a shim needed only for the
  official suite. Priority for rstsr: own-language semantics tests (Phase 2)
  plus generated surface/signature tests (Phase 1); the official gate (Phase 3)
  waits on a Python binding.

## Next steps (doc §6, not started)

1. **Phase 1 — spec-derived, no Python**: surface/signature tests generated
   from the stubs; promotion tables as data; port the `**Special cases**`
   docstring parser into our own edge-case tables; the `must`-typed error table.
2. **Phase 2 — semantics in Rust**: property/identity tests; cover the suite's
   verified gaps (no accuracy bound; no values for mean/std/var/pow/astype/
   fft/…; `test_remainder` unconditionally skipped).
3. **Phase 3 — official gate**: Python binding (Option A) + `array-api-tests`.
