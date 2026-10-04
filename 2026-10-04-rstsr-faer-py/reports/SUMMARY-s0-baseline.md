# S0 gate — NumPy 2.5.1 baseline reproduced (2026-10-04)

Harness: this task dir `harness/` (suite `~/Git-Others/array-api-tests`
@ `6c0b59f`, spec submodule @ `5f847a3`, test deps in conda env `torch`).
Run: `MODULE=numpy ./run.sh` (suite-default 100 examples, derandomized,
report `numpy-20261004-164346.json`).

## Result

```
1335 passed, 42 failed, 5 skipped   (1382 collected, 39 s)
```

Matches the recorded baseline in
`../../2026-10-04-arrayapi-compliance-notes/numpy-compliance/README.md`
(1334–1335 / 41–43 / 5) — **gate passed**. All six known violated-rule
clusters present:

| # | cluster | this run |
|---|---|---|
| 1 | `floor_divide` ±inf (spec: `±inf // finite → ±inf`, `finite // ±inf → ∓0.0`; NumPy: `nan`) | 25 across `test_special_cases` (binary/unary/iop) |
| 2 | `iinfo` must accept array input | 8/8 |
| 3 | `expm1` complex inf/NaN table (+1 tanh case) | 7 (`unary`) |
| 4 | `finfo` attrs must be Python floats | 4/4 |
| 5 | `fftfreq`/`rfftfreq` need `dtype=` | 4 across signatures+fft |
| 6 | `clip` param names `min`/`max` | 1 |

All 5 skips: `test_remainder` (suite skips itself as flaky — the `%` sign
rule is never graded).

## Meaning for the project

- The measurement instrument is calibrated: any rstsr-faer-py run with
  these pins is directly comparable to this number.
- Expected rstsr red map ≠ this list. rstsr will fail on *absent surface*
  (sort/argsort, roll, repeat/tile, …) — pytest errors/missing names —
  which NumPy never shows. The comparable quantity is the set of
  *semantic* disagreements among tests that actually ran.

## Environment note (recorded for setup.sh history)

GitHub HTTPS is blocked on this machine; the spec submodule was
initialized offline from the sibling clone `~/Git-Others/array-api`
(`-c protocol.file.allow=always`; submodule NAME is
`array_api_tests/array-api`, not its path). CLAUDE.md now codifies the
SSH-over-HTTPS rule.
