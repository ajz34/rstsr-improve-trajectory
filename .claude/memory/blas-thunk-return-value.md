---
name: blas-thunk-return-value
description: "rstsr-blas-traits trait impls (driver_impl/lapack) must return the thunk's return value: a RowMajor bridge failure returns -1010/-1011 without writing `info`. Fix (rstsr `c87a99e`, branch `261125/blas-thunk-info`): 24 sites in eigh (8), qr (8), svd (4), solve (4); eig was already fixed; the lapacke tree has no bridge and is unaffected."
metadata:
  type: project
---

Found while adding eig to the `lapacke-translation` skill (2026-11-25).

- `xxx_work` thunks return raw LAPACK `info` for ColMajor, but for RowMajor
  they return `-1010`/`-1011` on an allocation or bridge failure **without
  touching `info`**. A trait impl that ends with `xxx_work(..); info` (or
  reads `work_query` after an ignored query call) reports success after a
  failed bridge. The fixed shape is `let info_query = xxx_work(..)` and
  `let info = xxx_work(..)`, both returned.
- Fixed files (discarded call statements per file): `eigh/{syev,syevd,
  sygv,sygvd}.rs` 2 each; `qr/{geqrf,orgqr}.rs` 4 each; `svd/{gesdd,gesvd}.rs`
  2 each; `solve/getri.rs` 2, `solve/sysv.rs` 2 — 20 query calls bound as
  `info_query`, 4 QR compute calls turned into tail expressions.
- Detection script pattern: in each `impl XxxDriverAPI` block, a line that
  starts `fn_..(` without `let`/`return`/`=` and whose closing paren is
  followed by `;` is a discarded call; a tail expression (no `;`) is fine.
- Verification after the fix: OpenBLAS driver+linalg under `lapack` and
  `lapacke` (48+56 each), MKL and AOCL full driver suites (48 each), clippy
  `-D warnings`, fmt.
