# QR decomposition on the BLAS devices — outcome (2026-11-24)

Decisions: `DECISIONS.md`. Implementation: rstsr branch `261124/blas-qr`
committed as `fb6f921` (56 paths) on top of `c9fcc20` (`main`); no PR opened.

## What was built

- `rstsr-blas-traits/src/lapack_qr/`: `GEQRF_` (`GEQRFDriverAPI<T>`, → `(h, tau)`)
  and `ORGQR_` (`ORGQRDriverAPI<T>`, `a`/`tau`/`k` → Q in place); registered in
  `lib.rs`, `prelude.rs`, `LapackDriverAPI<T>`.
- `rstsr-blas-traits/driver_impl/{lapack,lapacke}/qr/{geqrf,orgqr}.rs`: shared
  thunks (real → `?orgqr`, complex → `?ungqr`), `lwork` query.
- `rstsr-linalg-traits/src/ref_impl_blas.rs`: `ref_impl_qr_simple_f` +
  `orgqr_2d`; `blas_impl/qr.rs`: stacked entry + `QrAPI` impls (`Tr`,
  `(Tr, QrMode)`, `(Tr, &str)`, `TensorMut` overloads), symlinked as
  `linalg_auto_impl/qr.rs` in all five device crates.
- Tests: `blas_tests_impl/driver_impl/f64/{geqrf,orgqr}.rs`,
  `linalg_func/func_{f64,c64}/qr.rs` (symlinked into the five crates);
  validator scripts extended (`# ### qr` in c64, wide-complete in f64).
- `QrMode` rustdoc: one sentence on BLAS support.
- `rstsr-agents` `7373cf9`: `git-commit-coauthor` registers DeepSeek Harness
  (<service@deepseek.com>).

## Results

| Config | tests_driver_impl | tests_linalg_func |
|---|---|---|
| OpenBLAS 0.3.32 (system, OpenMP), `lapack` | 48/48 | 56/56 |
| OpenBLAS 0.3.30 (conda `rest-user`), `lapacke` | 48/48 | 56/56 |
| MKL 2026.1, QR subset | 16/16 | 6/6 |
| AOCL 5.3.0, QR subset | 16/16 | 6/6 |

Also green: `cargo test -p rstsr-linalg-traits`, facade `cheatsheet_linalg`
(11/11), clippy + rustfmt on the touched crates. BLIS/KML not exercised (no
local library); both consume the same shared trees.

The system OpenBLAS exports no `LAPACKE_*` symbols, so the `lapacke` config
only links against the conda OpenBLAS (owner: lapacke testing may be skipped).

## Notes for the implementer / reviewer

- **Two facts learned from the tests** (both bit the first draft):
  1. `orgqr` with `k` reflectors and `n > k` columns of `a` produces the
     orthonormal **completion** — complete mode with `m > n` passes
     `k = min(m, n)` on a zero-padded `(m, m)` buffer (NumPy `qr_complete`).
  2. R extraction is row `i` from the diagonal **rightwards** (`i..n`), not the
     leading `i + 1` elements.
- Fingerprints of `R` are only portable after making `diag(R)` real-positive;
  `abs(Q[:, :k])` is portable as is. Complete-Q columns `k..m` are never
  fingerprinted (arbitrary completion).
- Cross-crate `duplicate_item` tables need `'a` in the generics column when the
  substituted `Tr` carries a lifetime, and must not carry it for the owned row
  (clippy `extra_unused_lifetimes`).
- `QrMode` is `#[non_exhaustive]`: device-crate matches need a wildcard or
  `rstsr_invalid!` arm.
