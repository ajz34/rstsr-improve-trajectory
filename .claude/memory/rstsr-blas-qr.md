---
name: rstsr-blas-qr
description: "QR decomposition on the BLAS devices (2026-11-24): GEQRF/ORGQR builders, orgqr completion for complete mode (k = min(m,n) reflectors on an (m,m) buffer), canonical-R fingerprints, duplicate_item lifetime column trick; OpenBLAS/MKL/AOCL green, uncommitted on 261124/blas-qr"
metadata:
  type: project
---

2026-11-24. Grill closed in 2 rounds; decisions and outcome in
`2026-11-24-blas-qr/{DECISIONS,README}.md`. Code on rstsr branch
`261124/blas-qr` (off `c9fcc20`), **uncommitted** (50 paths), PR not opened.

- Layers: `rstsr-blas-traits/src/lapack_qr/{geqrf,orgqr}.rs` (builders +
  `GEQRFDriverAPI`/`ORGQRDriverAPI`, in `LapackDriverAPI`), shared thunks
  `driver_impl/{lapack,lapacke}/qr/`, `ref_impl_qr_simple_f` +
  `blas_impl/qr.rs` symlinked into all five device crates.
- `rt::linalg::qr` on BLAS = `QrMode::Reduced`/`Complete` only; NumPy `'r'`/
  `'raw'` and pivoting deferred by the owner (new function for pivoting).
  `TensorMut` overloads consume the input as scratch (owner asked for them).
- **orgqr completion**: `DORGQR(m, n, k)` with `n > k` orthonormally completes
  the extra columns; complete mode `m > n` calls it with `k = min(m,n)` on a
  zero-padded `(m, m)` buffer (NumPy `qr_complete` does the same). `tau` has
  `min(m,n)` entries.
- Test gotchas: R extraction is row `i` from the diagonal rightwards;
  `fingerprint(R)` only after real-positive `diag(R)`; never fingerprint
  complete-Q columns `k..m`; `duplicate_item` puts `'a` in the generics column
  when `Tr` has a lifetime but not for the owned `Tensor` row; cross-crate
  `#[non_exhaustive]` match needs a wildcard arm.
- Verification: OpenBLAS 48+56 (lapack), conda OpenBLAS under `lapacke` 48+56,
  MKL and AOCL QR subsets 16+6; the system OpenBLAS exports no `LAPACKE_*`
  symbols. Owner: lapacke testing may be skipped unless `lapack` misbehaves.
