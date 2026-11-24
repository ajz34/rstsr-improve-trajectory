# Raw BLAS and `scipy.linalg.blas`

## Binding inventory

Counts are distinct `pub fn` names in each FFI module
(`rstsr-ffi/<crate>/src/<module>/ffi_extern.rs`). They include vendor
extensions, service functions, and CBLAS internals, so they are not
apples-to-apples across crates.

| crate | module | symbols |
|--|--|--|
| `rstsr-lapack-ffi` | `blas` | 175 |
| | `cblas` | 153 |
| | `lapack` | 1332 |
| | `lapacke` | 2516 |
| | `lapacke_utils` | 154 |
| `rstsr-openblas-ffi` | `blas` | 445 |
| | `cblas` | 233 |
| | `lapack` | 1332 |
| | `lapacke` | 2516 |
| `rstsr-mkl-ffi` | `blas` | 309 |
| | `cblas` | 299 |
| | `lapack` | 2258 |
| | `lapacke` | 2564 |
| | `service` | 44 |
| `rstsr-aocl-ffi` | `blis` | 5691 |
| | `flame` | 15193 |
| | `lapack` | 1332 |
| | `lapacke` | 2516 |
| `rstsr-blis-ffi` | `blis_x86_64` | 2933 |
| | `flame` | 4951 |
| | `lapack` | 1332 |
| `rstsr-kml-ffi` | `blas` | 175 |
| | `kblas` | 217 |
| | `lapack` | 1332 |
| | `service` | 34 |

The AOCL, BLIS, and KML crates bind vendor headers and then alias the vendor
module to the conventional names:

- `rstsr-aocl-ffi`: `pub use blis as blas; pub use blis as cblas;`
- `rstsr-blis-ffi`: `pub use blis_x86_64 as blis; pub use blis as blas; pub use blis as cblas;`
- `rstsr-kml-ffi`: `pub use kblas as cblas;`

MKL, OpenBLAS, and AOCL bind `lapacke`; BLIS has no `lapacke` module
(`rstsr-blis-ffi` has `lapack` and `flame` only).

## What `rstsr-lapack-ffi::blas` contains beyond the 150 SciPy routines

The netlib module has 175 symbols. 144 of the 150 `scipy.linalg.blas` routine
names are present; the remaining 31 symbols are:

- `gemmtr` in all four precisions (`cgemmtr_`, `dgemmtr_`, `sgemmtr_`,
  `zgemmtr_`) — triangular GEMM, present in the LAPACK 3.12.1
  `CBLAS/include/cblas_f77.h`.
- `sdsdot_`, `dsdot_` — mixed-precision dot products.
- CBLAS `*sub` aliases, e.g. `cdotcsub_`, `cdotusub_`, `dasumsub_`,
  `dcabs1sub_`, `ddotsub_`, `dnrm2sub_`, `dsdotsub_`, `dzasumsub_`,
  `dznrm2sub_`, `icamaxsub_`, `idamaxsub_`, `isamaxsub_`, `izamaxsub_`,
  `sasumsub_`, `scabs1sub_`, `scasumsub_`, `scnrm2sub_`, `sdotsub_`,
  `sdsdotsub_`, `snrm2sub_`, `zdotcsub_`, `zdotusub_` (the return-value forms
  CBLAS uses instead of the Fortran subroutine ABI).
- `xerbla_`, `scabs1_`, `dcabs1_`.

## `scipy.linalg.blas`: 150 routine names

| status | count |
|--|--|
| present in the Fortran BLAS bindings | 144 |
| absent from the Fortran BLAS bindings | 6 |
| present under their CBLAS names (`cblas_<name>`) | 140 |
| absent under their CBLAS names | 10 |

The 6 absent Fortran routines:

| routine | note |
|--|--|
| `cspmv`, `cspr`, `csyr` | complex symmetric BLAS-2; not in the 170 `.f` files of `~/Git-Others/lapack/BLAS/SRC` (LAPACK 3.12.1) |
| `zspmv`, `zspr`, `zsyr` | same, double-complex |

SciPy still exposes wrappers for them; the reference BLAS sources no longer
contain them, so a binding generated from the vendored header cannot declare
them.

The 10 routines without a same-name CBLAS function are the 6 above plus
`cdotc`, `cdotu`, `zdotc`, `zdotu`; CBLAS spells these
`cblas_cdotc_sub`, `cblas_cdotu_sub`, `cblas_zdotc_sub`, `cblas_zdotu_sub`,
which are present in the netlib `cblas` module.

## The typed driver tier

Four driver entries on `rstsr-blas-traits/src/{blas2,blas3}/`, covering five
routine families (SYMM and HEMM share `driver_syhemm` through the
`const HERMI: bool` parameter):

| driver | file | families | builder aliases |
|--|--|--|--|
| `driver_gemm` | `blas3/gemm.rs` | GEMM | `GEMM`, `SGEMM`, `DGEMM`, `CGEMM`, `ZGEMM` |
| `driver_gemv` | `blas2/gemv.rs` | GEMV | `GEMV`, `SGEMV`, `DGEMV`, `CGEMV`, `ZGEMV` |
| `driver_syhemm` | `blas3/syhemm.rs` | SYMM, HEMM | `SYHEMM`, `SSYMM`, `DSYMM`, `CSYMM`, `ZSYMM`, `CHEMM`, `ZHEMM` |
| `driver_trsm` | `blas3/trsm.rs` | TRSM | `TRSM`, `STRSM`, `DTRSM`, `CTRSM`, `ZTRSM` |

Driver signatures (raw pointers, `lda`/`ldb`/`ldc` explicit, no strides):

```rust,ignore
unsafe fn driver_gemm(order, transa, transb, m, n, k, alpha: T, a: *const T, lda, b: *const T, ldb, beta: T, c: *mut T, ldc);
unsafe fn driver_gemv(order, transa, m, n, alpha: T, a: *const T, lda, x: *const T, beta: T, y: *mut T);
unsafe fn driver_syhemm(order, side, uplo, m, n, alpha: T, a: *const T, lda, b: *const T, ldb, beta: T, c: *mut T, ldc);
unsafe fn driver_trsm(order, side, uplo, transa, diag, m, n, alpha: T, a: *const T, lda, b: *mut T, ldb);
```

The corresponding builders take 2-D/1-D tensor views: `GEMM` (`a`, `b`,
`c: Option`, `alpha`, `beta`, `transa`, `transb`, `order`), `GEMV` (`a`, `x`,
`y: Option`, `alpha`, `beta`, `transa`, `order`), `SYHEMM` (`a`, `b`,
`c: Option`, `alpha`, `beta`, `side`, `uplo`, `order`, `HERMI`), `TRSM` (`a`,
`b`, `alpha`, `side`, `uplo`, `transa`, `diag`, `order`). They operate on a
single matrix; batching is handled by the device `matmul` implementation.

The driver implementations live once in
`rstsr-blas-traits/driver_impl/cblas/` (4 routine files + 3 `mod.rs`) and are
symlinked into the five device crates at
`crates-device/rstsr-{openblas,mkl,aocl,blis,kml}/src/driver_impl/`.

### Consumers inside the linalg API

| builder | used by |
|--|--|
| `GEMM` | `matmul` (device impl), `pinv` |
| `GEMV` | `matmul` with a vector operand (device impl) |
| `TRSM` | `solve_triangular` |
| `SYHEMM` | no linalg entry; only the explicit builder (tests/benchmarks) |

### What has no typed driver

Only the five families above have a `driver_*` method and builder. BLAS-1
(`axpy`, `copy`, `dot`, `nrm2`, `scal`, `swap`, `rot*`, the index-of-max
routines `isamax`/`idamax`/`icamax`/`izamax`, …) and the remaining BLAS-2/3
routines (`ger`, `syr`, `syr2`, `syrk`, `syr2k`, `trmv`, `trsv`, `trmm`,
`gbmv`, `sbmv`, `spmv`, …) exist only as raw FFI declarations. `vecdot` and
the norm entries are implemented in `rstsr-core` / `rstsr-native-impl` and on
faer, not through BLAS-1.
