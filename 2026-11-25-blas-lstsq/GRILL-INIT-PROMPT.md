# Grilling init prompt — least squares (`lstsq`) on the BLAS devices

Recorded 2026-11-25 (Asia/Shanghai). Human prompt (verbatim):

> With previous git commits, as well as rstsr-improve-trajectory grill sessions
> (on qr, eig); can you make plan and discuss with me, or if necessary, grill
> with me, on implementing lstsq for BLAS devices?

Session constraints: none stated (subagents allowed). The QR and eig grill
directories were read first as the precedent for scope, driver/entry/test
conventions.

## State at the start of the grill

- rstsr: `ed1f62d` (`main`, #171 merged; QR #169 and eig #170 are in), clean
  tree, workspace version 0.10.4.
- rstsr-ffi: `83efe50`, clean.
- rstsr-improve-trajectory: `fc0915c`; two unrelated files modified by the
  previous session (`2026-11-25-openblas-amd64v3-zgeev/README.md`,
  `.claude/memory/openblas-amd64v3-zgeev.md`) — left untouched.
- **No rstsr file has been edited.** This is a planning session.

## What the prompt already settles

1. Target: `lstsq` on the BLAS devices, through the shared trees in
   `rstsr-blas-traits` / `rstsr-linalg-traits`.
2. The precedent process: grill first (`GRILL-R<N>-QUESTIONS.md`), decisions
   in `DECISIONS.md`, then `PLAN.md`, then implementation on a new branch with
   no auto-commit in rstsr.
3. The earlier sessions' working assumptions carry over: only OpenBLAS
   (OpenMP build) is run locally; the other four devices consume the same
   shared trees and must work in principle.

## Facts checked before round 1 (no user questions asked for these)

- Nothing named `lstsq` / `gels` / `least_squares` exists in rstsr — not even on
  the faer device. NumPy's `lstsq` is one of the five absent `numpy.linalg`
  names (`2026-11-25-blas-linalg-gap/NUMPY-LINALG.md`).
- `linalg.lstsq` is **not** part of the array API standard in any published
  revision: added to the draft 2021-05 (PR #119), removed 2021-08 (PR #240),
  absent from the 2021.12…2025.12 stubs and from array-api-tests. There is no
  conformance target; NumPy/SciPy are the references.
- NumPy 2.5.4: 2-D only, LAPACK `gelsd`, default `rcond = eps * max(M, N)`,
  residuals read out of the GELSD `B` workspace, empty when `rank < N` or
  `M <= N`.
- SciPy 1.18.1: **batched** `(..., M, N)` × `(..., M[, K])`, default
  `cond = eps` (no `max(M, N)` factor), `lapack_driver` gelsd/gelss/gelsy,
  residuals `NaN` where `rank < N`.
- `[sdcz]gelsd`, `[sdcz]gelss`, `[sdcz]gelsy` are bound in all five device FFI
  trees; `[sdcz]gelst` is MKL-only and "GELSY2" does not exist. System OpenBLAS
  0.3.32 exports `dgelsd_`/`dgelss_`/`dgelsy_` (no LAPACKE; the conda 0.3.30
  build has both).
