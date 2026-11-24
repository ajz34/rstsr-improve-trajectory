# Decisions — QR decomposition on the BLAS devices (grill closed 2026-11-24)

Rounds: R1 (scope, layers, tests, docs, delivery), R2 (TensorMut overload,
fingerprints, buffer plan, error contract). Answers to R1 were given on screen
and transcribed at the top of `GRILL-R2-QUESTIONS.md`; R2 all accepted.
Facts gathered at rstsr `c9fcc20` (`main`), rstsr-ffi `83efe50`.

## Scope

- **D1** `rt::linalg::qr` on the BLAS devices implements `QrMode::Reduced` and
  `QrMode::Complete` only, with the faer device's impl set (`Tr`,
  `(Tr, QrMode)`, `(Tr, &str)`) → `QrResult<Q, R>`.
- **D2** NumPy `'r'` / `'raw'` deferred (return arity differs; `Out` is fixed
  per argument type). **D3** column pivoting deferred, as a new function.
- **D4** All five devices through the shared trees + one symlinked
  `linalg_auto_impl/qr.rs`; only OpenBLAS (openmp; `lapack` and `lapacke`
  configs) is run locally. No FFI change (all symbols bound).

## Driver tier (`rstsr-blas-traits`)

- **D5** New family `lapack_qr/` with two builders: `GEQRF_` → `(h, tau)`;
  `ORGQR_` (`a: h`, `tau`, optional `k`) → Q. Composition
  `ref_impl_qr_simple_f` in `ref_impl_blas.rs`.
- **D6** One trait `ORGQRDriverAPI<T>`; vendor impl maps real → `orgqr`,
  complex → `ungqr`. Both traits join `LapackDriverAPI<T>` and preludes.
- **D7** Workspace: plain `lwork` query for all three routines (no
  `iwork`/`rwork`). The `lapacke-translation` §3a table has no row for them
  and is **not** edited (owner: edit only on contradiction).

## Entry tier (`rstsr-linalg-traits/src/blas_impl/qr.rs`)

- **D8** Stacked n-dim input, `StackLayout` walk as in `blas_impl_svd_f`.
- **D9** Input is scratch: immutable → copy; owned / `&mut` → consumed and
  left destroyed. `TensorMut` overloads (bare = reduced, with `QrMode`)
  return `QrResult<Tensor, Tensor>`.
- **D10** Buffers: reduced with `m >= n` runs `ORGQR` in place on `h` (R copied
  out first); otherwise a `zeros` `(m, k')` buffer in the call order holds the
  first `min(n, k')` columns of `h` (zero-padded completion, NumPy behavior).
- **D11** Zero-size synthesized in the builder, never calling the drivers:
  reduced `Q (m, k)`, `R (k, n)`; complete `Q = I_m` if `n == 0 < m`, else
  `Q (m, m)`/`(0, 0)`; `R` zeros.
- **D12** `ndim < 2` → `InvalidLayout`; no squareness/rank check.

## Tests, docs, delivery

- **D13** Driver conformance `driver_impl/f64/{geqrf,orgqr}.rs` (layout
  matrix, tall/wide/square, buffer identity, zero-size); linalg conformance
  `linalg_func/func_{f64,c64}/qr.rs` (reconstruction, orthonormality,
  triangularity, shapes, stack, zero-size, `ndim < 2`).
- **D14** Fingerprints only on the phase-canonicalized form (`Q[:, :k]`,
  `triu(R)[:k, :]` with real-positive diagonal); validator scripts get a
  `# ### qr` section. Complete-Q columns `k..m` are never fingerprinted.
- **D15** No new doc page; builder docstrings per `api-doc-conventions`;
  one rustdoc sentence on `QrMode` about BLAS devices.
- **D16** One PR, rstsr branch `261124/blas-qr`, no auto-commit.
