# Harness — array-api-tests gate for rstsr-faer-py

Adapted from `../../2026-10-04-arrayapi-compliance-notes/numpy-compliance/`
(read that README for the full guide: what the suite measures, its gaps,
reading the results, all knobs). Differences only:

- Suite checkout is shared at `~/Git-Others/array-api-tests` (user rule:
  repos live in `~/Git-Others`; task dirs and rstsr stay clean), pinned
  `6c0b59f` + spec submodule `5f847a3` — identical pins to the NumPy
  baseline, so numbers compare.
- Test deps (pytest, pytest-json-report, hypothesis, ndindex) are
  installed **directly into the conda env `torch`** — no overlay venv
  (user instruction); numpy 2.5.1 and, later, the `rstsr_faer` wheel all
  live in that one env.
- `reports/` is generated here and gitignored; commit only written
  summaries (e.g. `reports/SUMMARY-<subject>.md`).
- `run.sh` adds `SKIPS_FILE=` / `XFAILS_FILE=` knobs for the gap machinery
  (suite-native `--skips-file` / `--xfails-file`); the files themselves
  live in the task dir, not upstream.

## S0 gate

Reproduce the NumPy 2.5.1 baseline before any wrapper code exists:

```bash
./setup.sh
MODULE=numpy ./run.sh
```

Expected (recorded 2026-10-04, same pins): 1382 items, ~1335 passed /
41–43 failed / 5 skipped; 6 stable violated-rule clusters (floor_divide
±inf, iinfo array input, expm1 complex table, finfo Python-float attrs,
fftfreq dtype=, clip min/max names); all 5 skips = self-skipped
`test_remainder`. Counts wobble ±2 (hypothesis DB replay; `FRESH=1` clears).

## Later (S1+)

```bash
MODULE=rstsr_faer.api ./run.sh        # the subject under test
SKIPS_FILE=../skips.txt XFAILS_FILE=../xfails.txt MODULE=rstsr_faer.api ./run.sh
```
