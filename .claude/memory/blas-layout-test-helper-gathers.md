---
name: blas-layout-test-helper-gathers
description: "BLAS layout tests: a helper that ends in `.into_owned()` compacts padded/strided views, so `*_layout_{f,c}_padded` / `*_non_contig` tests deliver compact matrices (true for svd_layout.rs, geqrf.rs, orgqr.rs; fixed for eig geev.rs); use `into_owned_keep_layout()` and pass `view_mut()`; a 1x1 block is in order whatever its strides."
metadata:
  type: feedback
---

`blas_tests_impl/driver_impl/f64/svd_layout.rs`-style `make_case` builds a
padded or strided buffer, slices the `(m, n)` block and ends with
`.into_owned()`, which **gathers** the view into a compact tensor
(strides `[1, 10]` became `[1, 6]`). Every layout test built on it ran the
driver on plain C/F inputs; the `ld > n` and stride-2 paths were never
exercised.

- Fix (done for `geev.rs`/`ggev.rs`, 2026-11-25): keep the layout with
  `into_owned_keep_layout()` (the owned tensor owns the whole buffer, the
  visible block keeps its strides/offset), assert the strides in the helper,
  and run each case twice: `view()` (driver copies; input must be untouched)
  and `view_mut()` (in place when the layout is in order for the call).
- In-place expectation must come from the tensor, not the case name:
  `t.c_prefer() || t.f_prefer()`. A `1 x 1` block reports **both** whatever
  its strides, so a "non-contig 1x1" goes in place (LAPACK writes the
  rotated `alpha` back for a 1x1 pencil) — not an aliasing bug.
- Padding integrity: snapshot `t.raw()` (whole buffer) and check that the
  visible block changed and every other element is identical
  (`assert_destroyed_in_place` in `driver_impl/f64/geev.rs`).
- LAPACK leaves `n <= 2` inputs intact in GEEV/GGEV, so "destroyed" can only
  be asserted for `n >= 3`.
- Still to do (pre-existing, other families): `svd_layout.rs`, `geqrf.rs`,
  `orgqr.rs` helpers gather the same way.
