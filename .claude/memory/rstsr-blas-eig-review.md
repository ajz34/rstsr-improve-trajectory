---
name: rstsr-blas-eig-review
description: "2026-11-25 two-axis review of the BLAS eig/eigvals feature (commit 23f726f) and its uncommitted fixes: naive num complex division for alpha/beta, static-dim overloads sharing cores with dynamic dims, layout tests that gathered, stale docs; what was fixed, what was left open, and the test-size restructuring that cut the eig subset from 500 s to 23 s."
metadata:
  type: project
---

Branch `261125/blas-eig` (rstsr): feature committed as `23f726f`, the
review fixes are **uncommitted** on top of it (owner: no auto-commit).
Task record: `2026-11-24-blas-eig/README.md` in this repo.

**Confirmed by the review and fixed**
- `w = alpha / beta` used `num`'s naive division (see
  `num-complex-naive-division`) -> local `cdiv` in `ref_impl_blas.rs`.
- Static-dim generalized overloads (`eig.rs`, `eigvals.rs`) share their
  cores with the dynamic-dim forms (`rstsr-static-dim-core-serves-dyn`).
- Layout tests delivered compact inputs (`blas-layout-test-helper-gathers`).
- `ndim < 2` returned `AxisError` instead of `InvalidLayout`; complex GGEV
  thunk `beta` typed as real; compute call's return value dropped; unpaired
  complex eigenvalue read past `n` in release builds.
- Docs: faer handles the standard problem only (`EigSide`, `Homogeneous`,
  generalized are BLAS-only); cheat sheet still named the two-field
  `EigResult`; `eig((a, Left))` named a value the prelude does not export
  (`FlagSide::L`); cheat-sheet rows without their Rust/parity twins.
- Tests restructured per `blas-test-size-policy`.

**Left open (judgement or owner decision)**
- 16-row overload slot table repeated per impl and two ~15-line column
  normalizers (BLAS vs faer bound systems): sibling-endorsed style.
- `svd_layout.rs` / `geqrf.rs` / `orgqr.rs` layout helpers still gather.
- `from_raw_parts_mut(a, n * lda)` nominal overrun: ADR-0009 bridge pattern.
- Pre-existing rustdoc `redundant_explicit_links` in the `pinv` doc.
- Thunk glue duplicated between the `lapack` and `lapacke` trees (by design:
  each tree mirrors its vendor API).

**Review mechanics that worked**: three sub-agents (spec / values,
API+docs, driver conventions) each reporting file:line + quoted code;
every point re-derived in the main session before fixing (two reported
"bugs" were test artefacts; one doc claim was already fixed).
