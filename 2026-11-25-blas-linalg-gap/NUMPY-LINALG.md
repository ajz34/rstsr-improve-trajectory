# `numpy.linalg` (NumPy 2.5.4) against `rt::linalg`

`numpy.linalg` exposes 32 names (the module also has a `test` helper, excluded
here). The comparison is against rstsr `main` @ `9e8dac49`.

## Where the names live in rstsr

`rt::linalg` is exactly the 19 entries of
`rstsr-linalg-traits::prelude::rstsr_funcs`
(`rstsr/rstsr-linalg-traits/src/prelude.rs`). It is a feature-gated module
(`linalg` feature) of the `rstsr` facade.

The names NumPy's `linalg` re-exports from its core namespace (`matmul`,
`outer`, `vecdot`, `tensordot`, `trace`, `cross`, `matrix_transpose`,
`diagonal`) are *not* in `rt::linalg`; they are free functions at the crate
root (`rt::matmul`, `rt::outer`, …), defined in
`rstsr-core/src/tensor/linalg/` and `rstsr-core/src/tensor/indexing.rs`.

## The 32 names

| NumPy | status | rstsr | note |
|--|--|--|--|
| `LinAlgError` | no analogue | `rt::Error` | errors are an enum variant, not an exception class |
| `cholesky` | covered | `rt::linalg::cholesky` | `(&a)` / `(&a, Upper)`; default triangle follows the device order; BLAS devices call POTRF |
| `cond` | **absent** | — | composable from norms and `inv`, but no entry |
| `cross` | covered | `rt::cross` | core entry, not under `rt::linalg` |
| `det` | covered | `rt::linalg::det` | 2-D input returns a 0-d tensor |
| `diagonal` | covered | `rt::diagonal` | core entry; returns a view |
| `eig` | covered | `rt::linalg::eig` | faer only; complex output for every input |
| `eigh` | covered | `rt::linalg::eigh` | generalized `(a, b, uplo, type)`; `EighArgs` on BLAS only |
| `eigvals` | covered | `rt::linalg::eigvals` | faer only |
| `eigvalsh` | covered | `rt::linalg::eigvalsh` | faer: standard only; BLAS: generalized too |
| `inv` | covered | `rt::linalg::inv` | GETRF + GETRI on BLAS devices |
| `lstsq` | **absent** | — | no gelsd/gelss/gelsy driver |
| `matmul` | covered | `rt::matmul` | core entry; BLAS devices dispatch GEMM/GEMV |
| `matrix_norm` | covered | `rt::linalg::matrix_norm` | faer only |
| `matrix_power` | covered | `rt::linalg::matrix_power` | faer only; negative exponents supported |
| `matrix_rank` | covered | `rt::linalg::matrix_rank` | faer only; returns `Tensor<usize, _, IxD>` |
| `matrix_transpose` | covered | `rt::matrix_transpose` | core entry; returns a view |
| `multi_dot` | **absent** | — | |
| `norm` | partial | `rt::linalg::matrix_norm` / `rt::linalg::vector_norm` | split into two entries; no single `ord`-dispatching `norm` |
| `outer` | covered | `rt::outer` | core entry |
| `pinv` | covered | `rt::linalg::pinv` | returns `PinvResult { pinv, rank }` |
| `qr` | covered | `rt::linalg::qr` | `reduced` / `complete` only; no pivoting |
| `slogdet` | covered | `rt::linalg::slogdet` | returns `SLogDetResult { sign, logabsdet }` |
| `solve` | partial | `rt::linalg::solve_general` | square `A` (same requirement as NumPy); no `assume_a` dispatch |
| `svd` | covered | `rt::linalg::svd` | `full_matrices`; BLAS `driver` takes `gesvd` / `gesdd` |
| `svdvals` | covered | `rt::linalg::svdvals` | BLAS devices call GESDD with `compute_uv = false` |
| `tensordot` | covered | `rt::tensordot` | core entry |
| `tensorinv` | **absent** | — | |
| `tensorsolve` | **absent** | — | |
| `trace` | covered | `rt::trace` | core entry; optional offset |
| `vecdot` | covered | `rt::vecdot` | core entry |
| `vector_norm` | covered | `rt::linalg::vector_norm` | faer only |

Counts: 24 covered, 2 partial (`norm`, `solve`), 5 absent (`cond`, `lstsq`,
`multi_dot`, `tensorinv`, `tensorsolve`), 1 without an analogue
(`LinAlgError`).

## The 19 entries and their backends

| entry | `DeviceFaer` | BLAS devices | BLAS driver |
|--|--|--|--|
| `cholesky` | yes | yes | POTRF |
| `det` | yes | yes | GETRF (via `slogdet`) |
| `eig` | yes | **no** | — |
| `eigh` | yes | yes | SYEV / SYEVD / SYGV / SYGVD |
| `eigvals` | yes | **no** | — |
| `eigvalsh` | yes | yes | SYEV / SYEVD / SYGV / SYGVD |
| `inv` | yes | yes | GETRF + GETRI |
| `matrix_norm` | yes | **no** | — |
| `matrix_power` | yes | **no** | — |
| `matrix_rank` | yes | **no** | — |
| `pinv` | yes | yes | GESDD |
| `qr` | yes | yes | GEQRF + ORGQR |
| `slogdet` | yes | yes | GETRF |
| `solve_general` | yes | yes | GESV |
| `solve_symmetric` | **no** | yes | SYSV |
| `solve_triangular` | yes | yes | TRSM (BLAS-3) |
| `svd` | yes | yes | GESVD / GESDD |
| `svdvals` | yes | yes | GESDD |
| `vector_norm` | yes | **no** | — |

`DeviceFaer`: 18 of 19 (`solve_symmetric` absent). BLAS devices: 13 of 19
(`eig`, `eigvals`, `matrix_norm`, `matrix_power`, `matrix_rank`,
`vector_norm` absent).

## Argument and return conventions

### Call form

Entries are free functions. The input is one parameter — the tensor, or a tuple
of arguments:

```rust,ignore
rt::linalg::eigh(&a);
rt::linalg::eigh((&a, Upper));
rt::linalg::eigh((&a, &b, Lower, 2));
```

Every entry has a fallible `_f` twin returning `rt::Result`.

### Return values

| entry | return |
|--|--|
| `eigh` | `EighResult { eigenvalues, eigenvectors }`; `.into()` gives a tuple |
| `eig` | `EigResult { eigenvalues, eigenvectors }` |
| `svd` | `SVDResult { u, s, vt }` |
| `qr` | `QrResult { q, r }` |
| `slogdet` | `SLogDetResult { sign, logabsdet }` |
| `pinv` | `PinvResult { pinv, rank }` |
| `det` | a tensor; a 2-D input yields a 0-d tensor |
| `matrix_rank` | `Tensor<usize, B, IxD>` |
| `inv`, `cholesky`, `solve_*`, `pinv.pinv` | tensors |

The result structs all implement `From`/`Into` to their natural tuple.

### dtypes

- `eig` / `eigvals` return `Complex<T::Real>` for every input dtype.
- `eigh` / `eigvalsh` / `svdvals` return the real part dtype; `eigh` and `svd`
  return complex vectors for complex input.
- `det` keeps the input dtype; `slogdet` returns `(T, T::Real)`.
- `matrix_rank` returns the index dtype `usize`.

### `eigh` / `eigvalsh`

On BLAS devices the entry is driven by `EighArgs`:

| field | values |
|--|--|
| `a` | required matrix |
| `b` | optional generalized `B` |
| `uplo` | `Lower` / `Upper`; default follows the device order |
| `eigvals_only` | bool; `eigvalsh` sets it |
| `eig_type` | `1`, `2`, `3` (LAPACK `ITYPE`) |
| `subset_by_index` | accepted, **ignored** (destructured as `_` in `ref_impl_eigh_simple_f`) |
| `subset_by_value` | accepted, **ignored** (same) |
| `driver` | `"ev"`, `"evd"`, `"gv"`, `"gvd"`; no `evr` / `evx` |

`DeviceFaer` implements `eigh` only for the tuple forms (no `EighArgs`), so it
has no `driver`, `eigvals_only`, or subset arguments. Its `eigvalsh` handles
the standard problem only; generalized eigenvalues go through
`eigh((&a, &b, uplo, type))`.

### `svd` / `svdvals`

- `SVDArgs { a, full_matrices: Option<bool>, driver: Option<&str> }`.
- `full_matrices` must be `Some` for `svd`; `SVDArgs_` asserts it (the `None`
  form is the `compute_uv = false` path used by `svdvals`).
- `driver`: `"gesvd"` or `"gesdd"`; default `"gesdd"`.
- `svdvals` always calls GESDD with `compute_uv = false`.
- `DeviceFaer` `svd` takes a bool (`full_matrices`) only; no driver choice.

### `qr`

- `QrMode` is `Reduced` (default) or `Complete`; the string forms
  `"reduced"` / `"complete"` parse to it.
- The enum is `#[non_exhaustive]`; no `economic`, `r`, `raw`, or pivoted mode.
- BLAS devices call GEQRF then ORGQR.

### `solve_*`

- `solve_general`: `(a, b)`; `a` must be square (rectangular input is rejected
  rather than solved in the least-squares sense); `b` may be `(M, K)` or
  `(M,)`; batch dimensions of `a` and `b` broadcast.
- `solve_symmetric`: `(a, b)`, `(a, b, uplo)`; SYSV (sym/hem).
- `solve_triangular`: `(a, b)`, `(a, b, uplo)`; TRSM. The `TRSM` builder has a
  `diag` flag, but the `solve_triangular` entry does not expose it.
- There is no `assume_a`, `transposed`, or `lower` flag on `solve_general`.

### Norms

- `matrix_norm`: `ord` (`None` → Frobenius; `Fro`, `Nuc`, `P(±1)`, `P(±2)`,
  `Inf`, `NegInf`) and `keepdims`; no `axis` argument — the two matrix axes are
  fixed (last two row-major / first two column-major).
- `vector_norm`: `ord` (`None` → 2-norm; any real `p`, `Inf`, `NegInf`;
  `Fro`/`Nuc` rejected), `axes` (`AxesIndex`, `None` = all), `keepdims`.
- Both are faer-only.

### `pinv`

- BLAS devices: `(a)`, `(a, atol, rtol)`.
- faer: `(a)`, `(a, atol, rtol)`, and `(a, rtol)` where `rtol` is a
  `PinvRtol` (scalar, per-matrix array, or `None`).
- Default tolerance: `atol = 0`, `rtol = max(M, N) * eps`. NumPy's `pinv`
  default is `rcond = 1e-15`.
- Returns the numerical rank alongside the pseudoinverse.

### `matrix_power`

- `(a, isize)`; `n = 0` returns the identity; `n < 0` inverts the base once,
  then exponentiates by squaring. A singular base with `n < 0` yields
  `inf`/`NaN` rather than an error.

### n-D input

All 19 entries accept n-dimensional input: the matrix is the last two axes
under a row-major device and the first two under a column-major one, the rest
is batch. `eig`, `eigvals`, `matrix_power`, `matrix_rank`, `matrix_norm`, and
`vector_norm` are faer-only, so on BLAS devices only the 13-entry subset
listed above accepts n-D input.
