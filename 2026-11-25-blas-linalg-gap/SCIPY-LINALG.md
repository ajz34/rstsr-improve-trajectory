# `scipy.linalg` (SciPy 1.18.1) against `rt::linalg`

`scipy.linalg` exposes 112 public names: 96 functions, 14 module attributes
(`basic`, `blas`, `cython_blas`, `cython_lapack`, `decomp`,
`decomp_cholesky`, `decomp_lu`, `decomp_qr`, `decomp_schur`, `decomp_svd`,
`lapack`, `matfuncs`, `misc`, `special_matrices`), and 2 exception classes
(`LinAlgError`, `LinAlgWarning`).

Counts against rstsr `main` @ `9e8dac49`:

| status | count |
|--|--|
| covered | 12 |
| partial | 2 (`norm`, `solve`) |
| absent | 82 |

The 82 absent include 3 dtype-dispatch helpers (`find_best_blas_type`,
`get_blas_funcs`, `get_lapack_funcs`) that have no direct Rust analogue because
the Rust API is generic over `BlasFloat` and the builder types are selected
statically.

## The 96 functions

| SciPy | status | rstsr | note |
|--|--|--|--|
| `bandwidth` | absent | — | returns the matrix bandwidth |
| `block_diag` | absent | — | assembly helper |
| `cdf2rdf` | absent | — | real block-diagonal form of a complex Schur form |
| `cho_factor` | absent | — | POTRF factor object |
| `cho_solve` | absent | — | factor-based Cholesky solve |
| `cho_solve_banded` | absent | — | banded Cholesky solve |
| `cholesky` | covered | `rt::linalg::cholesky` | lower/upper triangle |
| `cholesky_banded` | absent | — | banded Cholesky factorization |
| `circulant` | absent | — | circulant matrix construction |
| `clarkson_woodruff_transform` | absent | — | sketching transform |
| `companion` | absent | — | companion matrix construction |
| `convolution_matrix` | absent | — | convolution matrix construction |
| `coshm` | absent | — | matrix hyperbolic cosine |
| `cosm` | absent | — | matrix cosine |
| `cossin` | absent | — | CS decomposition |
| `det` | covered | `rt::linalg::det` | 0-d tensor for 2-D input |
| `dft` | absent | — | discrete Fourier transform matrix |
| `diagsvd` | absent | — | assemble a matrix from singular values |
| `eig` | covered | `rt::linalg::eig` | faer only; complex output |
| `eig_banded` | absent | — | banded symmetric/Hermitian eigenproblem |
| `eigh` | covered | `rt::linalg::eigh` | BLAS: `EighArgs` (`b`/`uplo`/`eig_type`/`driver`); faer: tuple forms |
| `eigh_tridiagonal` | absent | — | tridiagonal symmetric eigenproblem |
| `eigvals` | covered | `rt::linalg::eigvals` | faer only |
| `eigvals_banded` | absent | — | banded eigenvalues only |
| `eigvalsh` | covered | `rt::linalg::eigvalsh` | faer: standard only; BLAS: generalized supported |
| `eigvalsh_tridiagonal` | absent | — | tridiagonal eigenvalues only |
| `expm` | absent | — | matrix exponential |
| `expm_cond` | absent | — | matrix exponential condition number |
| `expm_frechet` | absent | — | Frechet derivative of the matrix exponential |
| `fiedler` | absent | — | Fiedler matrix construction |
| `fiedler_companion` | absent | — | Fiedler companion matrix |
| `find_best_blas_type` | absent | — | dtype-dispatch helper; Rust uses generics over `BlasFloat` |
| `fractional_matrix_power` | absent | — | matrix power with non-integer exponent |
| `funm` | absent | — | matrix function of a general callable |
| `get_blas_funcs` | absent | — | dtype-dispatch helper; Rust uses the typed BLAS builders |
| `get_lapack_funcs` | absent | — | dtype-dispatch helper; Rust uses the typed LAPACK builders |
| `hadamard` | absent | — | Hadamard matrix construction |
| `hankel` | absent | — | Hankel matrix construction |
| `helmert` | absent | — | Helmert matrix construction |
| `hessenberg` | absent | — | Hessenberg decomposition |
| `hilbert` | absent | — | Hilbert matrix construction |
| `inv` | covered | `rt::linalg::inv` | GETRF + GETRI on BLAS devices |
| `invhilbert` | absent | — | inverse Hilbert matrix |
| `invpascal` | absent | — | inverse Pascal matrix |
| `ishermitian` | absent | — | Hermitian predicate |
| `issymmetric` | absent | — | symmetric predicate |
| `khatri_rao` | absent | — | Khatri-Rao product |
| `ldl` | absent | — | LDL^T factorization |
| `leslie` | absent | — | Leslie matrix construction |
| `logm` | absent | — | matrix logarithm |
| `lstsq` | absent | — | least-squares solve; no gelsd/gelss/gelsy driver |
| `lu` | absent | — | LU factorization as P,L,U |
| `lu_factor` | absent | — | packed LU factor object |
| `lu_solve` | absent | — | factor-based LU solve |
| `matmul_toeplitz` | absent | — | Toeplitz matrix-vector product |
| `matrix_balance` | absent | — | matrix balancing |
| `norm` | partial | `rt::linalg::matrix_norm / rt::linalg::vector_norm` | split by input kind; no single `ord`-dispatching `norm` |
| `null_space` | absent | — | null space from the SVD |
| `ordqz` | absent | — | ordered QZ decomposition |
| `orth` | absent | — | orthonormal basis from the SVD |
| `orthogonal_procrustes` | absent | — | orthogonal Procrustes problem |
| `pascal` | absent | — | Pascal matrix construction |
| `pinv` | covered | `rt::linalg::pinv` | returns `PinvResult { pinv, rank }` |
| `pinvh` | absent | — | Hermitian pseudoinverse |
| `polar` | absent | — | polar decomposition |
| `qr` | covered | `rt::linalg::qr` | `reduced`/`complete` only; no pivoting, no `r`/`raw` mode |
| `qr_delete` | absent | — | column deletion from a QR factorization |
| `qr_insert` | absent | — | column insertion into a QR factorization |
| `qr_multiply` | absent | — | QR with a matrix multiplication |
| `qr_update` | absent | — | QR rank-one update |
| `qz` | absent | — | QZ (generalized Schur) decomposition |
| `rq` | absent | — | RQ decomposition |
| `rsf2csf` | absent | — | real to complex Schur form |
| `schur` | absent | — | Schur decomposition |
| `signm` | absent | — | matrix sign function |
| `sinhm` | absent | — | matrix hyperbolic sine |
| `sinm` | absent | — | matrix sine |
| `solve` | partial | `rt::linalg::solve_general` | no single `assume_a` (`gen`/`sym`/`her`/`pos`) or `transposed` entry; sym/hem covered by `solve_symmetric`, no positive-definite path |
| `solve_banded` | absent | — | banded linear solve |
| `solve_circulant` | absent | — | circulant linear solve |
| `solve_continuous_are` | absent | — | continuous algebraic Riccati equation |
| `solve_continuous_lyapunov` | absent | — | continuous Lyapunov equation |
| `solve_discrete_are` | absent | — | discrete algebraic Riccati equation |
| `solve_discrete_lyapunov` | absent | — | discrete Lyapunov equation |
| `solve_lyapunov` | absent | — | Lyapunov equation |
| `solve_sylvester` | absent | — | Sylvester equation |
| `solve_toeplitz` | absent | — | Toeplitz linear solve |
| `solve_triangular` | covered | `rt::linalg::solve_triangular` | uplo overload; no unit-diagonal flag at the linalg entry |
| `solveh_banded` | absent | — | banded Hermitian positive-definite solve |
| `sqrtm` | absent | — | matrix square root |
| `subspace_angles` | absent | — | principal angles between subspaces |
| `svd` | covered | `rt::linalg::svd` | `full_matrices`; BLAS `driver` takes `gesvd`/`gesdd` |
| `svdvals` | covered | `rt::linalg::svdvals` | singular values only |
| `tanhm` | absent | — | matrix hyperbolic tangent |
| `tanm` | absent | — | matrix tangent |
| `toeplitz` | absent | — | Toeplitz matrix construction |

## The 14 overlapping entries: SciPy arguments against rstsr

| SciPy signature (abridged) | rstsr |
|--|--|
| `cholesky(a, lower=False, overwrite_a=False, check_finite=True)` | `cholesky(&a)` / `cholesky((&a, Lower))`; no overwrite/check flags |
| `det(a, overwrite_a=False, check_finite=True)` | `det(&a)` |
| `eig(a, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)` | `eig(&a)` only: no generalized `b`, no left eigenvectors, no homogeneous output |
| `eigh(a, b=None, *, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)` | `EighArgs` (`b`, `uplo`, `eigvals_only`, `eig_type`, `subset_by_index`/`subset_by_value` accepted but ignored, `driver` = `ev`/`evd`/`gv`/`gvd`); no overwrite or `check_finite` flags |
| `eigvals(a, b=None, overwrite_a=False, overwrite_b=False, check_finite=True, homogeneous_eigvals=False)` | `eigvals(&a)`; no generalized `b`, no homogeneous output |
| `eigvalsh(a, b=None, *, lower=True, overwrite_a=False, overwrite_b=False, type=1, check_finite=True, subset_by_index=None, subset_by_value=None, driver=None)` | `eigvalsh(&a)` / `eigvalsh((&a, uplo))`; generalized and subset/driver only on BLAS devices; faer has the tuple forms only |
| `inv(a, overwrite_a=False, check_finite=True)` | `inv(&a)` |
| `norm(a, ord=None, axis=None, keepdims=False, check_finite=True)` | `matrix_norm(&a)` / `vector_norm(&a)`; `matrix_norm` has no `axis`, `vector_norm` has `axes` |
| `pinv(a, *, atol=None, rtol=None, return_rank=False, check_finite=True)` | `pinv(&a)` / `pinv((&a, atol, rtol))`; rank always returned; faer additionally accepts `(a, rtol)` with a scalar or per-matrix array `rtol` |
| `qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False, check_finite=True)` | `qr(&a)` / `qr((&a, QrMode))`; `Reduced`/`Complete` only; no `lwork`, no pivoting |
| `solve(a, b, lower=False, overwrite_a=False, overwrite_b=False, check_finite=True, assume_a=None, transposed=False)` | `solve_general(&a, &b)`; no `assume_a`/`transposed`/`lower`; `solve_symmetric` covers sym/hem, no `pos` path |
| `solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False, overwrite_b=False, check_finite=True)` | `solve_triangular(&a, &b)` / `((&a, &b, uplo))`; `trans` handled by passing a transposed view; `unit_diagonal` not exposed (the `TRSM` builder has `diag`) |
| `svd(a, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=True, lapack_driver='gesdd')` | `svd(&a)` / `svd((&a, full_matrices))` / `SVDArgs { full_matrices, driver }`; `compute_uv=false` corresponds to `svdvals` |
| `svdvals(a, overwrite_a=False, check_finite=True)` | `svdvals(&a)` |

## SciPy module attributes with no rstsr counterpart

`scipy.linalg` re-exports the `decomp_*` submodules and `basic`, `blas`,
`lapack`, `matfuncs`, `misc`, `special_matrices`. In rstsr the corresponding
names are flat `rt::linalg` entries or absent; there is no submodule split
inside `rt::linalg`. `blas` and `lapack` are covered by `BLAS-RAW.md` and
`LAPACK-RAW.md`.
