---
name: lstsq-driver-efficiency
description: "Which LAPACK least-squares driver to use in rstsr: GELSD is the efficiency choice for the 4-output contract; GELSS 10-60x slower, GELSY 2.4-4.4x faster but solution+rank only, SVD+GEMM 1.1-1.9x slower"
metadata:
  type: reference
---

2026-11-25 measurements (AMD Ryzen 9 9950X3D, 16 cores/32 threads; conda
OpenBLAS 0.3.30, SciPy 1.18.1 `lstsq(..., check_finite=False)`, best of 3,
including the input copies SciPy's wrapper does; ms):

| M x N x K | gelsd | gelss | gelsy | SVD+GEMM |
|--|--|--|--|--|
| 8192x256x1, 1T | 51.1 | 628.5 | 45.2 | 83.2 |
| 8192x256x32, 1T | 57.0 | 710.9 | 47.9 | 84.7 |
| 4096x1024x1, 1T | 398.1 | 28999 | 237.2 | 560.2 |
| 2048x2048x1, 1T | 942.3 | 62636 | 338.3 | 1637.1 |
| 512x2048x1, 1T | 66.5 | 1317.9 | 83.3 | 134.4 |
| 4096x4096x1, 16T | 5734.3 | — | 1306.4 | 6543.8 |
| 2048x2048x1, 16T | 556.7 | — | 137.2 | 872.7 |
| 8192x256x1, 16T | 43.6 | — | 25.9 | 54.6 |

Conclusions for the rstsr `lstsq` design:

- **GELSD** is the only driver that returns all four NumPy outputs (`x`,
  `residuals`, `rank`, `s`) in one call, and it is the fastest such driver.
- **GELSS** (QR + DBDSQR, also 4 outputs) is 10–60x slower: never a candidate.
- **GELSY** (complete orthogonal factorization) is 2.4–4.4x faster on
  square/large-M problems and 1.25x **slower** on 512x2048, but returns only
  `x` and `rank` — no singular values, no residuals. A solution-only fast path
  would need a separate API and is deferred.
- GESDD + threshold + GEMM composition is 1.1–1.9x slower than GELSD and needs
  U/V memory.
- LAPACK drivers scale poorly with threads (2048²: 942→557 ms from 1→16
  threads), and rstsr's per-slice loop is serial: batched performance is
  driver-bound. Per-slice parallelism is a future lever, not part of the first
  implementation.

Reproduce with `/tmp/bench_lstsq.py` style scripts: `scipy.linalg.lstsq` with
`lapack_driver='gelsd'|'gelss'|'gelsy'`, `overwrite_a=True,
overwrite_b=True, check_finite=False`; pass `b` as `(m, k)` (SciPy pads to
`max(m,n)` rows internally — passing `(max(m,n), k)` raises a shape error).
Do not run `gelss` at 4096² (≈30 min).
