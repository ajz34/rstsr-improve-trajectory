---
name: openblas-amd64v3-zgeev
description: "Ubuntu resolute's amd64v3 OpenBLAS 0.3.32+ds-5 has a broken zgeev for n>=76: GCC 15.2 PR124802 miscompiles LAPACK 3.12.1 ZLAQR5 in liblapack-dev's liblapack_pic.a, which the openblas package embeds. Upstream OpenBLAS and the plain amd64 package are correct. rstsr's tier1-openblas jobs are pinned to ubuntu-24.04 (noble, 0.3.26)."
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

## Root cause (2026-11-25 follow-up)

- GCC 15.2 wrong-code **PR 124802** (duplicate **PR 124111**, "lapack-3.12.1
  miscompiled since r15-2820") miscompiles LAPACK 3.12.1 `zlaqr5.f` with
  `-O2 -march=x86-64-v3` (SLP-vectorized complex-double FMA pattern); fixed in
  GCC 15.3 / commit `r15-11080` (2026-04-07). `ZLAQR5` is the multishift sweep
  used by `ZHSEQR` for `n > 75`, hence the threshold.
- Ubuntu's `openblas` uses `no-embedded-lapack.patch`: it extracts
  `liblapack-dev`'s `liblapack_pic.a` (amd64v3 pic objects are compiled with
  `-march=x86-64-v3` by gcc 15.2) and links that native code without LTO
  (openblas is on `lto-disabled-list`). LAPACK's own `liblapack.so` is
  LTO-linked and correct.
- Evidence: object substitution identifies `zlaqr5.o`; substituting the
  installed distro object reproduces the failure; `-fno-tree-vectorize`,
  `-fno-tree-slp-vectorize` or `-ffp-contract=off` fix it; the official GCC
  PR124802 testcase aborts with host gcc-15.
- Upstream OpenBLAS 0.3.32 built from source here is correct (it compiles its
  bundled LAPACK with `-fno-tree-vectorize`). Stonking 0.3.33+ds-3ubuntu1
  (gcc 15.3, lapack 3.12.1-8) is correct; resolute 0.3.32+ds-5 and the
  Proposed 0.3.32+ds-5ubuntu0.1 still fail (tested).
- Full report: `local-runs/261125-openblas-zgeev/reports/README.md`
  (Launchpad draft: `launchpad.md`).
