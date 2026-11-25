# lstsq for BLAS devices (2026-11-25)

Least-squares entry (`rt::linalg::lstsq`) for the BLAS/LAPACK device crates.

- rstsr: `ed1f62d` (branch `261125/blas-lstsq`; not committed by the agent)
- environment: 1-thread BLAS (`RSTSR_DEV=1`), OpenBLAS from the system,
  torch conda env for the Python validators (NumPy 2.5.4 / SciPy 1.18.1)

Files: `GRILL-INIT-PROMPT.md`, `GRILL-R1-QUESTIONS.md` (agent),
`GRILL-R1-ANSWERS.md` (user-owned), `DECISIONS.md` (D1–D16), `PLAN.md`
(B0–B6), `IMPLEMENTATION.md` (what landed, deviations, verification).

Driver choice is backed by `../.claude/memory/lstsq-driver-efficiency.md`
(GELSD is the only efficient four-output driver); reference semantics are in
`../.claude/memory/numpy-scipy-lstsq-semantics.md`.

## Results (OpenBLAS, 1 thread, `RSTSR_DEV=1`)

| suite | tests | wall |
|--|--|--|
| `tests_driver_impl` | 89 (6 lstsq) | 23.5 s |
| `tests_linalg_func` | 100 (6 lstsq) | 31.3 s |
| `tests_zero_size` | 30 (4 lstsq) | 0.01 s |
| Python validators | 3 files | < 5 s each |

`cargo clippy --all-targets -- -D warnings` clean for both trait crates, `rstsr`
(linalg), and all five device crates; `cargo test -p rstsr --features linalg`
13/13; `cargo doc -p rstsr --no-deps --features linalg` warning-free.

## Gotchas

- Row-major bridge for `M < N`: copy `max(M, N)` rows of `B` in and back
  (see `IMPLEMENTATION.md`); copying `M` rows silently truncates `x`.
- `NRHS = 0` must be padded to one zero column (`DLALSD` returns `info = -4`).
- The `info = -4` NaN check is unreliable on x86 (`DLANGE('Max')` swallows
  NaN); keep the mapping but do not assert the message in tests.
- Only OpenBLAS links on this machine; the other devices are compile-checked,
  and `--features lapacke` is compile-only (no `LAPACKE_*` in the system
  OpenBLAS, MKL not on the linker path).
