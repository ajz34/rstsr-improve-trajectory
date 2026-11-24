# 2026-11-25 — BLAS/LAPACK linalg surface: what rstsr does not provide

- **Date**: 2026-11-25 (data sampled 2026-11-24 22:06–22:41, Asia/Shanghai).
- **Task**: record, item by item, which linear-algebra functionality rstsr
  provides and which it does not, measured against `numpy.linalg`,
  `scipy.linalg`, `scipy.linalg.blas`, `scipy.linalg.lapack`, and the raw
  BLAS/LAPACK bindings.
- **Status**: read-only survey. No rstsr or rstsr-ffi source was modified; no
  build, test, or benchmark was run.

## Provenance

| item | value |
|--|--|
| rstsr | `main` @ `9e8dac49` (2026-11-24), workspace version 0.10.4, MSRV 1.82, nightly toolchain |
| rstsr-ffi | `main` @ `83efe50` (2026-10-03) |
| netlib bindings | LAPACK 3.12.1 (`rstsr-lapack-ffi` 0.5.1) |
| NumPy | 2.5.4 (installed; sources also under `~/Git-Others/numpy`) |
| SciPy | 1.18.1 (installed; sources also under `~/Git-Others/scipy`) |
| Python | 3.13.14 |
| reproduction | `python3 scripts/collect.py` (reads sources, writes `data/`) |

## Headline counts

| surface | reference size | rstsr |
|--|--|--|
| `numpy.linalg` | 32 names | 24 covered, 2 partial, 5 absent, 1 without an analogue (`LinAlgError`) |
| `scipy.linalg` | 96 functions (plus 14 module attributes, 2 exception classes) | 12 covered, 2 partial, 82 absent |
| `scipy.linalg.blas` | 150 routine names | 144 names present in the netlib Fortran BLAS bindings (6 absent); 5 routine families have a typed driver |
| `scipy.linalg.lapack` | 623 wrappers / 154 routine families (plus 42 `*_lwork` helper families) | 147 families / 479 wrappers present in the bindings (7 families / 24 wrappers absent); 13 translated routines, covering 19 scipy families once real/complex aliases are counted |
| raw BLAS | 150 scipy routines | 175 symbols bound in `rstsr-lapack-ffi::blas` (among the additions: `gemmtr`, `sdsdot`/`dsdot`, the CBLAS `*sub` aliases, `xerbla`) |
| raw LAPACK | upstream `lapack.h` | 1332 symbols bound; 2516 LAPACKE symbols and 154 `lapacke_utils` symbols |

## The four layers

A linear-algebra routine can exist at any of four levels in this workspace:

| layer | location | form |
|--|--|--|
| 1. raw FFI | `rstsr-ffi/*/src/{blas,cblas,lapack,lapacke}/ffi_extern.rs` | `pub fn dgesv_(...)` — `unsafe extern "C"`, raw pointers, `lda`, `ipiv`, workspace, `info` |
| 2. driver | `rstsr-blas-traits/src/{blas2,blas3,lapack_*}/` | `unsafe fn driver_gesv(...)` — one method per routine group on `LapackDriverAPI` / `BlasDriverAPI` |
| 3. interface builder | `rstsr-blas-traits/src/{blas2,blas3,lapack_*}/` | tensor arguments, layout handling, workspace allocation, `info` → `rt::Error` (ADR-0010) |
| 4. linalg entry | `rstsr-linalg-traits/src/` + `rt::linalg::*` | operation-level API (`eigh`, `svd`, `solve_general`, …) |

The driver and builder tiers are translated per routine, one file per LAPACK
routine (ADR-0009). The translation files live once in
`rstsr-blas-traits/driver_impl/` and are symlinked into the five device crates;
`driver_impl/lapacke/` is symlinked into MKL and OpenBLAS only.

## Coverage vocabulary

- **covered** — an rstsr entry performs the same operation; argument-level
  differences are in the `note` column.
- **partial** — an entry exists but only in a restricted form (restriction
  recorded).
- **absent** — no entry.
- **present in bindings** — the symbol exists in `ffi_extern.rs`; it does not
  imply a safe or typed API.

"n-D" in the documents means the two matrix axes are the last two shape axes
under a row-major device and the first two under a column-major one; the
remaining axes are batch.

## Files

| file | content |
|--|--|
| `NUMPY-LINALG.md` | the 32 `numpy.linalg` names, the 19 `rt::linalg` entries, per-backend availability, argument and return conventions |
| `SCIPY-LINALG.md` | the 96 `scipy.linalg` functions with per-function status, plus an argument-level comparison for the 14 overlapping entries |
| `BLAS-RAW.md` | raw BLAS binding counts, the 150 `scipy.linalg.blas` routines against the bindings, the 5 driver families and their builders |
| `LAPACK-RAW.md` | raw LAPACK/LAPACKE binding counts, the 154 `scipy.linalg.lapack` families, the 13 translated routines, and the entry → driver map |
| `data/` | generated inventory tables (`collect.py`) plus the hand-written `numpy_linalg_coverage.tsv` and `scipy_linalg_coverage.tsv` |
| `scripts/collect.py` | regenerates `data/` and validates the hand-written coverage tables against the installed NumPy/SciPy |

## Not covered by this check

- Numerical accuracy or value parity of any entry (no comparison was run).
- Performance of any entry.
- Dtype/promotion/broadcasting conformance beyond the notes recorded in
  `NUMPY-LINALG.md`.
- API stability or intended direction; the documents state what exists at the
  recorded commit.
