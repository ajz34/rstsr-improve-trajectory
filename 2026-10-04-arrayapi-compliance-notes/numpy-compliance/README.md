# Grading NumPy against the Python Array API standard

Hands-on companion to `../TESTING-ARRAY-API.md` §4 ("what the official suite actually
measures"). This directory runs the **official conformance suite**,
[`data-apis/array-api-tests`](https://github.com/data-apis/array-api-tests), against the
**NumPy that already lives in the `torch` conda env** — released NumPy **2.5.1** from
conda, not a dev/nightly build, not built from source.

```
/home/a/miniconda3/envs/torch/bin/python   ← the interpreter
numpy 2.5.1  (site-packages, release)      ← the library under test
```

Nothing is installed into the torch env. The test-only packages (pytest, hypothesis,
ndindex) go into an overlay venv, `.venv/`, which inherits the torch env's site-packages —
so the numpy that pytest imports is byte-for-byte the torch env's numpy (setup.sh asserts
this).

## TL;DR

```bash
cd /home/a/Documents/notes-repo/array-api/numpy-compliance

./setup.sh          # one-time: clone the pinned suite, check out spec submodule, build .venv
./run.sh            # grade numpy: whole suite, suite-default 100 hypothesis examples/test
./run.sh --max-examples 20   # same, faster, finds fewer edge cases
```

At the end, `run.sh` prints a summary (failed functions, representative messages, skips) and
writes the raw pytest JSON to `reports/`. Re-print a summary any time with:

```bash
./.venv/bin/python summarize_report.py            # newest report
./.venv/bin/python summarize_report.py reports/numpy-20261004-134806.json
```

## What exactly gets graded

| | |
|---|---|
| Library under test | `numpy` 2.5.1, imported by `ARRAY_API_TESTS_MODULE=numpy` |
| Spec version | **2025.12** — NumPy declares `numpy.__array_api_version__ == "2025.12"`, and the suite reads that attribute (`array_api_tests/__init__.py:88`). Override with `API_VERSION=2023.12 ./run.sh` |
| Suite revision | pinned commit `6c0b59f` (2026-09-15), spec submodule `5f847a38` — see "Why pinned" below |
| Oracle | **not NumPy.** Values are checked against Python-scalar reference implementations, spec-derived tables, and special-case tables regex-parsed out of the spec docstrings. See §4.2–4.3 of the doc for the (large) list of things it does *not* check |

Run `./.venv/bin/python probe_numpy.py` to see what NumPy self-declares before any test
runs — version, devices, default dtypes, `__array_namespace_info__()` capabilities.

Note the target is the **top-level `numpy` namespace**, not `numpy.array_api` (that module
was removed in NumPy 2.0; `numpy.__array_namespace__` is the public entry point now).

## Files

| file | purpose |
|---|---|
| `setup.sh` | clone the suite at the pinned commit, init the spec submodule, build `.venv` overlay on the torch env python. Idempotent |
| `run.sh` | run pytest with the right env/flags, write a JSON report + summary. `MODULE=…`, `MAX_EXAMPLES=…`, `API_VERSION=…`, `FRESH=1` override |
| `probe_numpy.py` | what numpy says about itself (`__array_api_version__`, capabilities, dtypes) |
| `summarize_report.py` | pytest JSON → pass/fail counts, failures grouped by spec function with a representative message, skip reasons |
| `reports/` | JSON reports (`numpy-<stamp>.json`) + sidecar `.meta.json` (library, version, spec version, suite commit, examples) |
| `array-api-tests/` | the suite checkout (git clone, pinned) |
| `.venv/` | overlay venv: suite deps only; numpy comes from the torch env |

## Reading the results

A useful first run is the API-surface check alone — it takes seconds and answers "does
NumPy have every name the 2025.12 spec requires?":

```bash
./run.sh array_api_tests/test_has_names.py
```

Then the full suite. Three outcomes matter:

- **failed** — a real disagreement with the spec's normative text (or a name the suite
  considers required and the library lacks).
- **skipped** — mostly the suite's own exclusions: `test_remainder` is skipped
  *unconditionally* as flaky (so the `%`-sign rule is never checked), plus version-gated
  tests when the spec version under test is older.
- **passed** — claims nothing about numerical accuracy: the default elementwise tolerance
  is `rel_tol=0.25, abs_tol=1`, and several functions (linalg, fft) are checked for
  dtype/shape only. See §4.3 before treating a green run as "correct".

## Baseline: NumPy 2.5.1 vs 2025.12 (2026-10-04)

Whole suite, suite-default examples (`./run.sh`), ~40 s, 1382 test items:

```
1334-1335 passed, 41-43 failed, 5 skipped
```

The exact count wobbles by ±2 between runs — each failing special-case test draws a value
for its case and Hypothesis explores different ones per run (and later runs replay the
persistent example database in `array-api-tests/.hypothesis`; `FRESH=1 ./run.sh` clears it).
**What is stable is the set of violated rules** — 6 of them, each verified by hand:

| # | spec function(s) | failing tests | the disagreement |
|---|---|---|---|
| 1 | `floor_divide`, `__floordiv__`, `__ifloordiv__` | 18 | `±inf // finite` must be `±inf`, `finite // ±inf` must be `∓0.0`; NumPy returns `nan` |
| 2 | `iinfo` | 8 | must accept `Union[dtype, array]`; `np.iinfo(np.asarray(1, dtype=np.int8))` raises `ValueError: Invalid integer data type 'O'` |
| 3 | `expm1` (complex) | 6-8 | the complex `inf`/`NaN` table: `expm1(+inf+0j)` → `inf+nanj` (spec: `+inf+0j`), `expm1(-inf+nanj)` → `nan+nanj` (spec: `-1+0j`), `expm1(-0+0j)` → `-0+0j` (spec: `0+0j`); plus one `tanh` case |
| 4 | `finfo` | 4 | `.eps`, `.max`, `.min`, `.smallest_normal` must be Python `float`; NumPy returns `np.float32`/`np.float64` scalars (`.bits` is a correct `int`) |
| 5 | `fftfreq`, `rfftfreq` | 4 | signature must accept `dtype=`; `np.fft.fftfreq` has no such parameter (2 signature tests + 2 value tests) |
| 6 | `clip` | 1 | parameters must be named `min`/`max`; NumPy uses `a_min`/`a_max` |

Everything else passes: the API-surface check (`test_has_names`: 214 required names, methods,
attributes, extension functions) is clean in 0.1 s, and all 5 skips are `test_remainder`
(the suite skips itself unconditionally as flaky).

Reproduce any cluster in three lines:

```python
import numpy as np
np.iinfo(np.asarray(1, dtype=np.int8))          # ValueError  (spec: Union[dtype, array])
type(np.finfo(np.float32).eps)                  # np.float32  (spec: Python float)
np.expm1(np.complex128(complex(np.inf, 0.0)))   # inf+nanj    (spec: +inf+0j)
np.float64(np.inf) // np.float64(1.0)           # nan         (spec: +infinity)
```

## Knobs

| what | how |
|---|---|
| fewer examples (faster, less thorough) | `--max-examples 20` passed to `run.sh` |
| from-scratch run (clear Hypothesis' counterexample DB) | `FRESH=1 ./run.sh` |
| pin an older spec | `API_VERSION=2023.12 ./run.sh` |
| run a single module | `./run.sh array_api_tests/test_creation_functions.py` |
| skip an optional extension | `./run.sh --disable-extension linalg fft` |
| grade a different library | `MODULE=array_api_strict ./run.sh` (needs `uv pip install --python .venv/bin/python array-api-strict`) |
| skip dtypes | `ARRAY_API_TESTS_SKIP_DTYPES=float32,complex64 ./run.sh` |

Full flag list: `.venv/bin/python -m pytest --help` inside `array-api-tests/`, or §4.4 of
the doc.

## Why pinned

The suite is calver-versioned and **new tests can break previously-passing libraries**
(it is written against the spec as it evolves). `setup.sh` therefore checks out a fixed
commit; bump `SUITE_COMMIT` deliberately, and re-read the diff.

## Notes

- **Dev-build trap.** If a future NumPy nightly is installed in the torch env, this setup
  would grade that instead. `probe_numpy.py` prints `dev build? yes/no`, and `setup.sh`
  prints the resolved path — check them if results look surprising.
- **The venv is an overlay.** `.venv` uses `--system-site-packages`, so `import numpy`
  resolves to the torch env. Only pytest/hypothesis/ndindex live in `.venv`; the torch env
  itself is untouched. To use the torch env directly instead:
  `uv pip install --python /home/a/miniconda3/envs/torch/bin/python -r array-api-tests/requirements.txt`.
- **Relevance to our project.** This is the gate a *Python* module under test has to pass.
  Our library isn't a Python module, so what transfers is the test *content* — the oracles
  and the spec-docstring-derived special-case tables — not this harness; see §5 of the doc.
