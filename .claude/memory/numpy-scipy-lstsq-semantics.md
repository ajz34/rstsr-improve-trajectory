---
name: numpy-scipy-lstsq-semantics
description: "Reference semantics of lstsq (2026-11-25): not in any published array API revision; NumPy 2.5 is 2-D only with eps*max(M,N), SciPy 1.18 batches with cond=eps and NaN residuals; GELSD residual trick; nrhs=0 broken; info=-4 = NaN in A"
metadata:
  type: reference
---

Verified 2026-11-25 against NumPy 2.5.4 (`numpy/linalg/_linalg.py:2418-2563`,
`umath_linalg.cpp:4090-4200`), SciPy 1.18.1 (`scipy/linalg/_basic.py:1246-1468`,
`_batched_linalg`), LAPACK 3.12 sources, array-api history (PR #119/#240).

- **Not array API**: `linalg.lstsq` was added to the draft 2021-05 (PR #119),
  removed 2021-08 (PR #240) before the first published revision; absent from
  every stub since 2021.12, from the 2025.12 linear-algebra extension (24
  functions), and from array-api-tests. Draft signature `lstsq(x1, x2, /, *,
  rtol=None)`, `rtol = max(M,N)*eps`, 4-tuple, rank `shape(x1)[:-2]`, no
  emptiness rule for residuals.
- **NumPy 2.5.4** `lstsq(a,b,rcond=None)`: 2-D only (`_assert_2d`); 1-D b →
  `(M,1)`; 1-D output squeezed to `(N,)`; `rcond = eps*max(M,N)`; always
  `dgelsd`; `residuals` = squared B-workspace rows `n+1:m` only when
  `rank==N && M>N`, else **empty `(0,)`**; 1-D b residuals `(1,)`; `rank` is a
  0-D `np.int32`; negative rcond passes through; f32 is computed in f64
  (`_commonType`).
- **SciPy 1.18.1** `lstsq(a,b,cond=None,...,lapack_driver=None)`: **batched**
  `a (...,M,N)`, `b (...,M)` or `(...,M,K)`, batch dims broadcast; `cond = eps`
  (**no max factor** — diverges from NumPy on borderline rank); `lapack_driver`
  ∈ {gelsd default, gelss, gelsy}; `residuals` `(...,K)` with **NaN** where
  `rank<N`, `(...,0)` when `M<=N`, 1-D b → batch-shaped (0-D unbatched);
  `rank` 0-D `np.int64`, per-slice array when batched; `s=None` for gelsy;
  `cond<0` rejected. Its **batched zero-size path is not batch-correct** (x
  drops batch axes, n==0 residuals reduce over batch, rank comes back scalar).
- **GELSD facts**: destroys A; B is the solution workspace, `LDB >=
  max(1,max(M,N))`, solution in first N rows; residual sum-of-squares of column
  i = sum of squares of `B(n+1:m,i)` when `m>=n && rank==n` (no extra GEMM;
  both NumPy and SciPy read it there). One `LWORK=-1` query returns optimal
  LWORK **and** min LIWORK in `IWORK(1)`; complex also min LRWORK in
  `RWORK(1)`. `LIWORK = max(1, 3*minmn*nlvl + 11*minmn)`.
- **`nrhs = 0` is broken**: GELSD does not quick-return, and the internal
  `DLALSD` reports `info = -4` (SciPy 1.18.1 fails with "Internal LAPACK
  errors: slice 0 emits lapack info=-4"). NumPy pads to one zero column and
  trims; rank/s are real for `k == 0`.
- `m==0 || n==0` quick-returns rank 0; synthesize without a call (n==0 →
  residuals = column sums of |b|²).
- `info = -4` from GELSD means **NaN/Inf in A** (complex: overflow too), no
  XERBLA. LAPACKE's own NaN checks return -5 (A) / -7 (B) / -10 (rcond).
- `rcond < 0` or `rcond >= 1` both fall back to machine epsilon inside
  `DLALSD` (`dlalsd.f:239-243`).
- FFI: `[sdcz]{gelsd,gelss,gelsy}_` bound in all five device trees and
  MKL; LAPACKE gelsd in openblas+mkl only; `gelst` MKL-only, no LAPACKE, no
  "GELSY2" anywhere.
