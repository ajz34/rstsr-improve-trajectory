---
name: blas-thunk-return-value
description: "rstsr-blas-traits trait impls (driver_impl/lapack) must return the thunk's return value: a RowMajor bridge failure returns -1010/-1011 without writing `info`. As of 2026-11-25, 23 call sites in eigh (8), qr (8), svd (4), solve (3) still drop it (OOM-only path); eig is fixed; the lapacke tree has no bridge and is unaffected."
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
- Affected files (count of discarded call statements): `eigh/{syev,syevd,
  sygv,sygvd}.rs` 2 each; `qr/{geqrf,orgqr}.rs` 4 each; `svd/{gesdd,gesvd}.rs`
  2 each; `solve/getri.rs` 2, `solve/sysv.rs` 1. The eig real impls were fixed
  in the eig review; the complex eig compute calls were already tail
  expressions.
- Detection script pattern: in each `impl XxxDriverAPI` block, a line that
  starts `fn_..(` without `let`/`return`/`=` and whose closing paren is
  followed by `;` is a discarded call; a tail expression (no `;`) is fine.
- Retrofit is mechanical (10 files) but touches merged code of several
  families: needs the owner's go-ahead and its own branch.
