# lstsq for BLAS devices (2026-11-25)

Least-squares entry (`rt::linalg::lstsq`) for the BLAS/LAPACK device crates.

- rstsr: `ed1f62d`, implemented by commit `182e38d` on branch `261125/blas-lstsq`
  (not pushed); review follow-ups `04f7a12`, `dc2274f`, `2dc1be0` on branch
  `261126/lstsq-review-fixes` (not pushed) — see `IMPLEMENTATION.md`
- environment: 1-thread BLAS (`RSTSR_DEV=1`), OpenBLAS from the system,
  torch conda env for the Python validators (NumPy 2.5.4 / SciPy 1.18.1)

Files: `GRILL-INIT-PROMPT.md`, `GRILL-R1-QUESTIONS.md` (agent),
`GRILL-R1-ANSWERS.md` (user-owned), `DECISIONS.md` (D1–D16), `PLAN.md`
(B0–B6), `IMPLEMENTATION.md` (what landed, deviations, verification).

Driver choice is backed by `../.claude/memory/lstsq-driver-efficiency.md`
(GELSD is the only efficient four-output driver); reference semantics are in
`../.claude/memory/numpy-scipy-lstsq-semantics.md`.

## Results at `182e38d` (2026-11-25, OpenBLAS, 1 thread, `RSTSR_DEV=1`)

| suite | tests | wall |
|--|--|--|
| `tests_driver_impl` | 89 (6 lstsq) | 23.5 s |
| `tests_linalg_func` | 100 (6 lstsq) | 31.3 s |
| `tests_zero_size` | 30 (4 lstsq) | 0.01 s |
| Python validators | 3 files | < 5 s each |

`cargo clippy --all-targets -- -D warnings` clean for both trait crates, `rstsr`
(linalg), and all five device crates; `cargo test -p rstsr --features linalg`
13/13; `cargo doc -p rstsr --no-deps --features linalg` warning-free.

After the review follow-ups (2026-11-26, `IMPLEMENTATION.md` has the full
matrix): OpenBLAS `lapack` and `lapacke` 104 / 89 / 26, MKL 2026.1 lstsq +
solve 15/15 under both configs.

## Gotchas

- Row-major bridge for `M < N`: copy `max(M, N)` rows of `B` in and back
  (see `IMPLEMENTATION.md`); copying `M` rows silently truncates `x`.
- `NRHS = 0` must be padded to one zero column (`DLALSD` returns `info = -4`).
- The `info = -4` NaN check is in netlib master only (`3d90a787f`, after
  3.12.1, unreleased) and unreliable on x86 anyway (`DLANGE('Max')` swallows
  NaN); keep the mapping but assert only "an error" in tests.
- LAPACKE and MKL do link here since 2026-11-26 (OpenBLAS 0.3.34 from conda
  through `libopenblas.so(.0)` symlinks on `LD_LIBRARY_PATH`; `MKLROOT` +
  `MKL_THREADING_LAYER=GNU` for `~/intel/oneapi/mkl/2026.1`). BLIS, AOCL and
  KML stay compile-checked.
- Building "the same matrix in the other layout" from the same byte buffer
  changes the logical matrix: derive it with `to_contig_f(order)` on the
  reference tensor, or re-index explicitly.
