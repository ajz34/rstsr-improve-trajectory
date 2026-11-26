---
name: rstsr-blas-lstsq-implemented
description: "lstsq on the BLAS devices implemented (branch 261125/blas-lstsq, 2026-11-25): GELSD driver + LstsqPlan entry, 5-device wiring, fingerprint tests green on OpenBLAS; lapacke compile-only on this machine; M<N B-row copy-back pitfall"
metadata:
  type: project
---

2026-11-25. `lstsq` implemented in `/home/a/rstsr_pack/rstsr` on branch
`261125/blas-lstsq` (off `ed1f62d`, working tree only — no auto-commit in
rstsr). Report: `2026-11-25-blas-lstsq/IMPLEMENTATION.md`.

Surface: `rt::linalg::lstsq((&a, &b))` / `(&a, &b, rcond)` (+ `_f` twins),
`LstsqResult { x, residuals, rank, s }` with tuple `From` both ways; LAPACK
`GELSD` only; per-slice batch semantics (`LstsqPlan` in
`rstsr-linalg-traits/src/stack.rs`); residuals `None` from the driver when
`M <= N` or `rank < min(M, N)`, NaN filled by the entry.

Wiring quirk worth remembering: `rstsr-linalg-traits/src/blas_impl/*.rs` are
**source templates compiled inside each device crate** through the
`crates-device/*/src/linalg_auto_impl/<fn>.rs` symlinks — a new entry function
needs 5 symlinks plus 5 `linalg_auto_impl/mod.rs` lines, and
`cargo check -p rstsr-linalg-traits` does not compile it.

Verification: OpenBLAS `tests_driver_impl` 85 / `tests_linalg_func` 97 /
`tests_zero_size` 30 pass; `cargo check --workspace --all-targets`, fmt and
clippy clean. `--features lapacke` is compile-only here (system OpenBLAS has no
`LAPACKE_*` symbols; MKL not linkable), and BLIS/AOCL/KML/MKL test links fail
because their libs are not on the linker path.

Pitfall: for `M < N` `DGELSD` zeroes `B(M+1:N, :)` and puts the length-`N`
solution in `B(1:N, :)`, so a row-major bridge must copy `max(M, N)` rows back —
copying `M` rows silently returns a truncated solution with identical `s`.

Follow-up 2026-11-26: review fixes, in-place consumption and the LAPACKE
write-back are committed on `261126/lstsq-review-fixes`, and `lapacke` and
MKL do run on this machine now — see `rstsr-blas-lstsq-review.md` and
`blas-lapacke-mkl-test-env.md`.
