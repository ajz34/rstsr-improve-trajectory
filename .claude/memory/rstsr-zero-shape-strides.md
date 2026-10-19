---
name: rstsr-zero-shape-strides
description: Zero-size layouts have no stride policy in rstsr; BLAS leading dimensions read raw empty strides and can go illegal (issue #161).
metadata:
  type: project
---

rstsr implements the NumPy *semantic* rule for zero-size layouts (`size()==0` ⇒ both C/F
contiguous and preferred; `check_strides`/`bounds_index`/`is_broadcasted`/`Layout::eq`
already exempt them) but has **no value policy**: static `Ix<N>` computes strides with
`.max(1)` on a zero axis (`[0,18].f() = [1,1]`) while dynamic `IxD` multiplies literally
(`[1,0]`), so the same shape gets different layouts depending on which dimension type
built it.

Consequences (investigated 2026-10-19 against rstsr `c1a6b79`, v0.10.2; report in
`2026-10-19-zero-shape-strides/`):

- `ld_row`/`ld_col` consume raw strides for empty layouts and only special-case dim == 1;
  they can return 0 or 1. OpenBLAS validates `ldb >= max(1,m)` *before* its
  `m==0||n==0` return, so `solve_triangular` with an empty RHS `[18,0]` passes `LDB=1`
  and silently hits `xerbla` (BLAS-3 drivers return `()`); the LAPACK twin
  (`solve_general`/`DGESV`) returns `InvalidValue "illegal argument"`.
- `f_prefer()`/`c_prefer()` are true for empty layouts, which makes
  `overwritable_convert_with_order` skip the normalizing `into_contig_f` path for
  owned/mut tensors. The same empty shape as a *view* is normalized internally and
  works — the failure is representation-dependent.
- NumPy 2.x: documented "no elements ⇒ strides never used, both contiguous"; allocation
  zeroes strides (PR #21477), but reshape/ravel restore computed strides. Its linalg
  boundary never passes raw strides: `ld = max(N,1)` and explicit empty handling
  (`_is_empty_2d`, `pinv` early return, `lstsq` n_rhs padding).

Recommended policy: semantic rule is normative; contiguous constructors emit all-zero
strides for zero-size shapes in both `Ix` and `IxD`; `ld_*` clamp to `max(1, required)`;
and every BLAS/LAPACK wrapper early-returns on empty operands (mandatory, since empty
views can carry arbitrary strides). See [[rstsr-linalg-batching-r1]] for other
device-order/batch conventions in the same layer.
