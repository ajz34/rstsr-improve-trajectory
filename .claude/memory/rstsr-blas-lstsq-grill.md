---
name: rstsr-blas-lstsq-grill
description: "BLAS lstsq grill (2026-11-25): R1 answered, DECISIONS D1-D16 + PLAN B0-B6 written; GELSD-only, SciPy residual shapes, per-slice rank, Python fingerprints; next = implement on 261125/blas-lstsq"
metadata:
  type: project
---

2026-11-25. Grill for `lstsq` on the BLAS devices, directory
`2026-11-25-blas-lstsq/` (`GRILL-INIT-PROMPT.md`, `GRILL-R1-QUESTIONS.md`,
user-owned `GRILL-R1-ANSWERS.md`, `DECISIONS.md`, `PLAN.md`).
rstsr at `ed1f62d`, rstsr-ffi at `83efe50`; planning only, no rstsr file
edited yet.

**R1 answers**: Q1/Q2/Q4/Q6/Q7/Q8/Q9 "follow the recommendation"; Q3 asked for
a fact check of the per-order matrix-axis convention; Q5 asked whether GELSD is
the efficient choice; Q10 requested Python fingerprint tests (2–5 large 2-D
cases per function).

**Settled decisions** (D1–D16 in `DECISIONS.md`): BLAS-only, no faer impl/py
shim; `LstsqResult { x, residuals, rank, s }` + tuple conversions; device-order
stacking (`(..., M, N)` row-major, `(M, N, ...)` col-major; fact-checked
`stack_split_shape` + `device.default_order()`); `rcond: T::Real`, default
`eps*max(M,N)`, pass-through (LAPACK clamps `<=0`/`>=1` to eps), NaN rcond =
error; GELSD only (benchmark: GELSS 10–60x slower, GELSY can't give s/residuals,
SVD+GEMM 1.1–1.9x slower — see `lstsq-driver-efficiency.md`); SciPy residual
shapes/dtype `T::Real` with NaN where rank<n and `(...,0)` for m<=n; per-slice
rank tensor; destructive `a`, `b` always copied into the `max(M,N)`-row
workspace; `K==0` padded to one zero column, `min(M,N)==0` synthesized; error
union `-4`/`-5`/`-7`/`-10` = NaN/Inf input.

**Plan** (B0–B6): branch `261125/blas-lstsq`; `rstsr-blas-traits`
`lapack_lstsq/gelsd.rs` (`GELSD_` builder returns `(x, s, rank,
Option<residual sums>)`) + thunks in the shared `driver_impl/lapack{,e}/lstsq/`;
`rstsr-linalg-traits` `LstsqPlan` in `stack.rs`, `ref_impl_lstsq_f`, 16
`LstsqAPI` impl rows; tests + Python validators; docs notes only (no cheat-sheet
row: the page's test is faer-backed).

Implemented on `261125/blas-lstsq` (see
`rstsr-blas-lstsq-implemented.md` and the task's `IMPLEMENTATION.md`).

Related memory: `numpy-scipy-lstsq-semantics.md`,
`lstsq-driver-efficiency.md`, `rstsr-blas-device-source-symlinks.md`.
