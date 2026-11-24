# Raw LAPACK and `scipy.linalg.lapack`

## Binding inventory

| crate | module | symbols |
|--|--|--|
| `rstsr-lapack-ffi` | `lapack` | 1332 |
| | `lapacke` | 2516 |
| | `lapacke_utils` | 154 |
| `rstsr-openblas-ffi` | `lapack` | 1332 |
| | `lapacke` | 2516 |
| `rstsr-mkl-ffi` | `lapack` | 2258 |
| | `lapacke` | 2564 |
| `rstsr-aocl-ffi` | `lapack` | 1332 |
| | `lapacke` | 2516 |
| `rstsr-blis-ffi` | `lapack` | 1332 |
| `rstsr-kml-ffi` | `lapack` | 1332 |

The netlib `lapack` module's 1332 symbols are the s/d/c/z-prefixed routines
(1330 names) plus `lsame_` and `ilaver_`. Each vendor crate is generated from
its own vendored header, so the counts differ (MKL 2026.1 has 2258); the bound
names are unsuffixed (`dgeev_`, not `dgeev_64_`).

## `scipy.linalg.lapack`

SciPy exposes 623 wrappers grouped into 154 routine families, plus 42
`*_lwork` families (120 wrappers). The `*_lwork` wrappers call the routine
with `lwork = -1` to query a workspace size; they have no separate Fortran
symbol, so there is no binding counterpart.

| status | families | wrappers |
|--|--|--|
| present in the netlib `lapack` module | 147 | 479 |
| absent | 7 | 24 |
| `*_lwork` helpers (no Fortran symbol) | 42 | 120 |

The 7 absent families:

| family | wrappers | note |
|--|--|--|
| `gesc2` | 4 | solve using a `getc2` factorization, with scaling |
| `getc2` | 4 | LU with complete pivoting, used by `gesc2` |
| `lartg` | 4 | generate a Givens rotation |
| `lasd4` | 2 | singular value of a rank-one modified diagonal, used by `bdsdc` (s/d only) |
| `pstf2` | 4 | Cholesky with complete pivoting |
| `rot` | 2 | `crot`/`zrot` (complex Givens rotation) |
| `sytf2` | 4 | unblocked Bunch-Kaufman LDL^T factorization |

These routines are absent from the vendored `lapack.h` (the header used to
generate `rstsr-lapack-ffi`), so they cannot be bound from it even though the
LAPACK library exports them.

## The 13 translated routines

Every translated routine has one builder file under `rstsr-blas-traits/src/`
and two driver files under `rstsr-blas-traits/driver_impl/lapack/` and
`rstsr-blas-traits/driver_impl/lapacke/`. The
directories are symlinked into each device crate's `src/driver_impl/`; the
`lapacke` variant exists only for OpenBLAS and MKL (AOCL, BLIS, and KML link
the `lapack` variant only).

| routine | builder | driver impls | complex counterpart | builder options | used by |
|--|--|--|--|--|--|
| GEQRF | `lapack_qr/geqrf.rs` | `lapack/qr/geqrf.rs`, `lapacke/qr/geqrf.rs` | same name | `a` | `qr` |
| ORGQR | `lapack_qr/orgqr.rs` | `.../qr/orgqr.rs` | UNGQR | `a`, `tau`, `k` | `qr` |
| GESV | `lapack_solve/gesv.rs` | `.../solve/gesv.rs` | same name | `a`, `b` | `solve_general` |
| GETRF | `lapack_solve/getrf.rs` | `.../solve/getrf.rs` | same name | `a` | `det`, `slogdet`, `inv` |
| GETRI | `lapack_solve/getri.rs` | `.../solve/getri.rs` | same name | `a`, `ipiv` | `inv` |
| POTRF | `lapack_solve/potrf.rs` | `.../solve/potrf.rs` | same name | `a`, `uplo` | `cholesky` |
| SYSV | `lapack_solve/sysv.rs` | `.../solve/sysv.rs` | CHESV / CSYSV | `a`, `b`, `uplo` | `solve_symmetric` |
| SYEV | `lapack_eigh/syev.rs` | `.../eigh/syev.rs` | CHEEV / ZHEEV | `a`, `jobz`, `uplo` | `eigh`, `eigvalsh` |
| SYEVD | `lapack_eigh/syevd.rs` | `.../eigh/syevd.rs` | CHEEVD / ZHEEVD | `a`, `jobz`, `uplo` | `eigh`, `eigvalsh` |
| SYGV | `lapack_eigh/sygv.rs` | `.../eigh/sygv.rs` | CHEGV / ZHEGV | `a`, `b`, `itype`, `jobz`, `uplo` | `eigh` (generalized) |
| SYGVD | `lapack_eigh/sygvd.rs` | `.../eigh/sygvd.rs` | CHEGVD / ZHEGVD | `a`, `b`, `itype`, `jobz`, `uplo` | `eigh` (generalized) |
| GESVD | `lapack_svd/gesvd.rs` | `.../svd/gesvd.rs` | same name | `a`, `full_matrices`, `compute_uv` | `svd` with `driver = "gesvd"` |
| GESDD | `lapack_svd/gesdd.rs` | `.../svd/gesdd.rs` | same name | `a`, `full_matrices`, `compute_uv` | `svd` (default), `svdvals`, `pinv` |

`SYEV`/`SYEVD`/`SYGV`/`SYGVD`/`ORGQR`/`SYSV` are single files that dispatch to
the real routine for real dtypes and to the Hermitian/unitary counterpart for
complex dtypes (`duplicate_item` tables in the driver file). The Rust builder
names stay `SYEV`, `ORGQR`, `SYSV`, … regardless of dtype.

Each driver file handles both memory orders: column-major input is passed
through, row-major input is transposed into a temporary buffer, and the
workspace query (`lwork = -1`) is resolved before the work call. `info` values
are mapped to `rt::Error` variants by the builder.

## Driver coverage of `scipy.linalg.lapack`

19 of the 154 scipy families are backed by one of the 13 translated routines:
the 13 builder names plus 6 aliases (`syev`→`heev`, `syevd`→`heevd`,
`sygv`→`hegv`, `sygvd`→`hegvd`, `orgqr`→`ungqr`, `sysv`→`hesv`).

```
geqrf  gesdd  gesv  gesvd  getrf  getri  heev  heevd  hegv  hegvd
hesv   orgqr  potrf syev   syevd  sygv  sygvd  sysv   ungqr
```

The other 135 families have no driver; 128 of them are bindable from the
netlib module, 7 are not (table above). Binding a routine does not make it
reachable from `rt::linalg`: the driver, builder, and linalg entry each have to
be added.

## Not covered by the raw module alone

The netlib `lapack` module contains many routines with no builder or driver,
for example the general eigenproblem (`geev`), the generalized non-symmetric
eigenproblem (`ggev`), banded and tridiagonal drivers, the least-squares
drivers (`gels*`), the QZ routines (`gges`/`ggev`), the generalized SVD (`ggsvd`), and the
orthogonal/unitary utility routines. They are callable only as raw `unsafe`
FFI, with the caller supplying `lda`, `ipiv`, workspace, and `info` handling.
