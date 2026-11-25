---
name: openblas-amd64v3-zgeev
description: "Ubuntu resolute's amd64v3 OpenBLAS 0.3.32+ds-5 has a broken zgeev (eigenvectors wrong for n>=76, eigenvalues fine; dgeev/zheev fine); the plain amd64 package of the same version is correct. GitHub's ubuntu-latest runner installs the amd64v3 package, so rstsr's tier1-openblas jobs are pinned to ubuntu-24.04 (noble, 0.3.26)."
metadata:
  type: reference
---

Found 2026-11-25 while chasing a CI-only `test_zgeev` failure (rstsr PR #170).

- Ubuntu resolute (26.04) publishes an **`amd64v3`** rebuild of `openblas`
  (`libopenblas0-{pthread,openmp}_0.3.32+ds-5_amd64v3.deb`). Its `zgeev`
  returns eigenvectors violating `A v = λ v` and `A^H v = conj(λ) v`
  (residual 1e-1..1e0) while the eigenvalues are correct. `dgeev` and `zheev`
  are fine, so it is the non-symmetric complex eigenvector path only.
- Threshold: correct for `n <= 75`, wrong from `n = 76`; independent of
  pthread/openmp, of the requested side, and of `OPENBLAS_CORETYPE` (all cores
  fail), so not a single bad BLAS kernel.
- Same version's plain `amd64` package is correct (3e-14), as is noble's
  0.3.26, local 0.3.34, MKL 2026.1 and AOCL 5.3.0.
- GitHub Actions `ubuntu-latest` now installs the `amd64v3` package, which made
  the `tier1-openblas` jobs red with failures that moved between runs and
  sizes. Fix applied: pin those jobs to `runs-on: ubuntu-24.04`
  (rstsr `16519a0`); keep the strict eigen fingerprints, they pass there.
- Reproducer without rstsr: `2026-11-25-openblas-amd64v3-zgeev/repro_zgeev.py`
  (ctypes only); the directory also holds the Launchpad report draft.
- Diagnostic lesson: when a numerical test fails only in CI and its numbers
  move between runs, call the *library* directly with a fixed matrix before
  blaming the wrapper code; the C-level check settled it in one step.
