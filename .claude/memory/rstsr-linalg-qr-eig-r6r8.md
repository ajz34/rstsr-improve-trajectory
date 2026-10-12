---
name: rstsr-linalg-qr-eig-r6r8
description: "qr (R6) + eig/eigvals (R8): the last array-API linalg members — QrMode non_exhaustive enum for the mode string, faer MatRef::qr(), complete-mode R zero-pad + identity empty-Q fix; conformance 1327/22/33 -> 1332/17/33, faer suite 68/68"
metadata:
  type: project
---

2026-10-12. `eig`/`eigvals` merged as `#146`; `qr` on branch `261012/qr` (off
main `6786abe`, post-`#146`; PR RESTGroup/rstsr#147) — commits `4848dc6`
(feature) + `2f63ece` (review fixes). **R6 + R8** of the array-API linalg
convergence; with them the extension has **no absent member left**. Suite
**`1327 / 22 / 33` → `1332 / 17 / 33`** (+5 / −5, 0 regressions; flipped
`test_eig`, `test_eigvals`, `test_qr`, `test_has_names[linalg-qr]`,
`test_extension_func_signature[linalg.qr]`). The residual `linalg` 3
(`test_matrix_rank`/`test_pinv`/`test_svd`) are all one faer SVD
`NoConvergence` gap on large-magnitude `f32`. The other 14 reds are the
out-of-scope `fft` name nodes.

- **faer device only** (like the rest). `qr` adds `QrAPI` +
  `faer_impl/qr.rs`; `eig`/`eigvals` add `EigAPI`/`EigvalsAPI` +
  `faer_impl/{eig,eigvals}.rs`; all in the three `duplicate_item` tables +
  `prelude.rs`.
- **`QrMode`** is the one with a string argument: `#[non_exhaustive] enum
  QrMode { #[default] Reduced, Complete }` with `TryFrom<&str>`/`FromStr`
  (`'reduced'`/`'complete'`), parsed **at the boundary**. `QrAPI` is implemented
  for the bare tensor (= reduced), `(tensor, QrMode)`, and `(tensor, &str)` (the
  `&str` impl parses then delegates — elided lifetimes in the `duplicate_item`
  `Tr` column are fine). `QrResult<Q, R>` converts to/from `(Q, R)`. Only these
  two modes — the array-API has no `'r'`/`'raw'` (those are NumPy-only, different
  arity). The `From<()>`/`From<Option<()>>` impls were added then **removed** as
  dead (no impl consumes them; contrast `NormOrd`, whose equivalents feed the
  norm-args groups).
- **faer 0.24 `MatRef::qr()` is infallible** → `Qr<T>`: reduced =
  `compute_thin_Q()` (M,K) + `R()` (K,N); complete = `compute_Q()` (M,M) + faer's
  thin (K,N) `R` **zero-padded** to (M,N) (`Mat::<T>::zeros(m,n);
  r.submatrix_mut(0,0,k,n).copy_from(qr.R())`). Batched via `map_stack_slices2`
  (both outputs dtype `T`, dim `D`); `T: ComplexField + Num` (`zeros_f` needs
  `Num`).
- **Empty-input bug (found by `/code-review high`, fixed in `2f63ece`):** the
  zero-size guard returned complete-mode `Q` as **zeros**, but NumPy/array-API
  want the **identity completion** for `n==0, m>0` (`test_qr` never checks
  orthonormality, so conformance had stayed green with a wrong value). Fix:
  build `eye_f` and assign it per batch slice via `map_stack_slices` (order-
  agnostic). `eye_f((m,&device))` yields a **dynamic** dim → `.into_dim::<Ix2>()`
  (infallible; no `?`).
- **Fingerprint tests must sign/phase-canonicalize** (Q/R unique only up to a
  unit factor per column): `abs(Q)` over the **first K columns** — complete-mode
  extra columns are an *arbitrary* orthonormal completion, not comparable to
  NumPy's — plus `R` rescaled to a real-positive diagonal; reduced and complete,
  tall and wide, and `test_qr_empty`. `eig` sorts the unordered spectrum. Faer
  suite 64 → **68/68**.
- **Shim**: `linalg.qr(x, /, *, mode='reduced')` → `(Q, R)` namedtuple; `eig` →
  `(eigenvalues, eigenvectors)`; `eigvals` → spectrum. The "qr is absent" prose
  in `linalg.rs`/`api.py` was dropped. Review re-run: conformance unchanged
  **`1332 / 17 / 33`**.

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R6/R8, §0). Related:
[[rstsr-linalg-norms-r5]], [[rstsr-linalg-matrix-rank-r4]],
[[arrayapi-convergence-harness]].
