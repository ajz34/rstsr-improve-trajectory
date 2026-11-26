# Grilling init prompt — LU decomposition (`lu`) on the BLAS devices

Recorded 2026-11-26 (Asia/Shanghai). Human prompt (verbatim):

> You are going to review full grilling session log in blas-qr / blas-eig /
> blas-lstsq in rstsr-improve-trajectory. Then you are going to implement lu.
> Note we may require optional argument `permute_l` of scipy.linalg.lu, but
> `p_indices` to always be True so no this argument.
> You are going to plan or discuss with me on this implementation. If
> necessary, can grill me.

Session constraints: none stated. All facts were looked up in the main
session (no sub-agent).

## State at the start of the grill

- rstsr: `68f4891` on `main` (= `origin/main`): stacked linalg #168, gemv
  #167, QR #169, eig #170, thunk-return fix #171 and lstsq #172 are all in;
  the eig and lstsq review fixes are contained in #170 / #172 (diffed the
  fix branches against `main`: no difference in the touched paths). Clean
  tree, workspace version 0.10.4.
- rstsr-ffi: `83efe50` (`main`), clean. rstsr-agents `6375962`, clean.
- rstsr-improve-trajectory: `14f01e3`, clean before this directory.
- **No rstsr file has been edited.** This is a planning session.

## What the prompt already settles

1. Target: `lu` through the shared trees in `rstsr-blas-traits` /
   `rstsr-linalg-traits`, in the line of the three earlier campaigns.
2. The signature follows `scipy.linalg.lu`; the optional `permute_l` is in
   scope; `p_indices` is fixed to `True` (permutation as row indices), so it
   is not an argument.
3. The process is the established one: grill (`GRILL-R<N>-QUESTIONS.md`),
   `DECISIONS.md`, `PLAN.md`, then implementation on a new branch with no
   auto-commit in rstsr.

## Facts checked before round 1 (no user questions asked for these)

- No `lu`, `lu_factor` or `lu_solve` exists anywhere in rstsr, not on the
  faer device either. faer 0.24.4 has `MatRef::partial_piv_lu()`.
- `GETRF` is already translated (builder, trait, both thunk trees); `lu`
  adds no LAPACK routine. Two properties of the existing path matter: the
  builder maps `info > 0` to an error although LAPACK reports a *completed*
  factorization there, and the row-major thunk skips the back-transpose for
  `info != 0`.
- SciPy 1.18.1 `lu` is batched, returns `(P, L, U)` or `(PL, U)`, and never
  fails on a singular matrix. Fingerprints of `P`, `L`, `U` and `PL` agree
  between two OpenBLAS builds to `6e-11` (probe `/tmp/lu_probe/portability.py`).
