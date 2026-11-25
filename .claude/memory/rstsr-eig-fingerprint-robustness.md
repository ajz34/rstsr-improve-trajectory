---
name: rstsr-eig-fingerprint-robustness
description: "Eigensolver fingerprints across LAPACK builds (learned 2026-11-25): sorted complex order unstable (ties in real parts) → sort re/im independently; |v| per-column sorted fingerprints; random 1024 pencils ill-conditioned (SPD B); raw GGEV scaling differs MKL vs OpenBLAS → unit-norm first; rstsr test-API quirks (view.raw(), i((.., i, j)) no expansion, ColMajor batch broadcast aligns leading dims)"
metadata:
  type: reference
---

Learned while validating BLAS eig/eigvals (2026-11-25); all bit a draft.

**Fingerprints (validator + Rust tests)**
- Sorted *complex* spectra are not a robust reference when distinct
  eigenvalues share a real part (roundoff flips the order): fingerprint the
  sorted real and imaginary parts independently.
- Never compare two spectra by **sorted index**: sort both and zipping pairs
  the members of a conjugate pair wrongly as soon as a last-bit difference
  flips the order inside the pair (bit the "homogeneous α/β == eigvals"
  checks once the library divided exactly). Pair each expected value with its
  nearest actual value (`assert_same_spectrum_within` in the linalg tests).
- Eigenvector blocks: fingerprint each column of `|v|`, sort the per-column
  values, fingerprint again (order-invariant). Reordering by a canonical
  spectrum order breaks on the same ties.
- A random 1024 pencil `(A, b)` or `(A, b + n·I)` is ill-conditioned: OpenBLAS,
  SciPy's LAPACK and SciPy's `eigvals` agree on real parts (1e-13) but differ
  by up to 33 (raw b) / 0.06 (b + n·I) in imaginary parts and eigenvectors,
  all with residuals ~1e-18. Use an SPD `B = b·bᵀ + n·I`; the complex raw `b`
  was fine. Standard random `A` (DGEEV/ZGEEV) fingerprints were portable.
- **Raw GGEV vector scaling differs between LAPACK builds** (MKL vs OpenBLAS:
  same "largest component ±1" convention, different per-column scaling;
  column norms 1.0–5.9). Driver-tier tests must rescale columns to unit norm
  before fingerprinting; the linalg tier already does.
- Homogeneous `(α, β)`: fingerprint `|α|, |β| / (|α| + |β|)` with rows sorted
  independently (scaling- and order-invariant).
- Python gotcha: `A.reshape(1024,1024)[:256,:256]` is not the leading 256
  block of a 1-D manifest file; use `A[:256*256].reshape(256,256)`. MKL from
  Python needs `MKL_THREADING_LAYER=SEQUENTIAL` (conda's libiomp clashes).

**rstsr test-API quirks**
- `view.raw()` spans the whole storage (a row view returned both rows);
  read elements with `i(k).to_scalar()`.
- `i((.., i, j))` does not expand `..` over all leading axes — one `..` is
  one axis. Spell the axes out or slice stepwise from the last axis.
- `TensorReference` = `TensorAny<DataReference>`; `From<Tensor>` is not
  implemented: bind the owned copy and pass `t.view_mut().into()`.
  `broadcast_to` on a temporary view → E0716; bind the view first.
- Column-major stacks put the matrix axes first (`[n, n, batch…]`); batch
  broadcast aligns *leading* batch dims (a one-axis `b` pairs with the first
  batch axis of `a`); batch iteration is row-major over the batch dims.
- `(&a, &b)` with `b: &Tensor` is a double reference (no impl).
- `duplicate_item`: core tables may elide the lifetime of a sole
  `&TensorAny` operand (clippy `needless_lifetimes`); delegation tables need
  the named lifetime (E0637 — associated `Out` carries it).
