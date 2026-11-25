# B0: probe results (2026-11-26)

Source: `src/probe_forms.rs` (crate `exp-cuda-forms`, path dependencies on the rstsr checkout,
no rstsr file edited). Raw outputs: `results/b0_dot.txt`, `results/b0_gemv.txt`,
`results/b0_gemv_batch.txt`, `results/b0_syrk.txt`.

Environment: NVIDIA GeForce RTX 5070 Laptop GPU (8 GiB), CUDA 13.2 toolkit / cuBLAS 13.4, rstsr
`261021/cuda` @ `80e626d`, stock release profile, default flags (f32 mode `CUBLAS_COMPUTE_32F`).
Timing: repeat count fitted to about 0.1 s, median of 5 trials, wall time per call around a
synchronized loop. The copy "ceiling" is `memcpy_dtod` of the same byte count on the same stream
(it exceeds DRAM bandwidth below 32 MiB because both buffers sit in L2). Every part passed a
correctness gate against a host loop before timing; the gates also confirmed that
`cublasCsyrk_v2` is the plain-transpose form and leaves the other triangle untouched.

Chain rows go through `rt::matmul` with a fresh output per call; anchors are one raw call.

## 1. Inner dot: `vecdot` against the bandwidth ceiling

`[k] @ [k]`, `f64`. (Full table with `f32` and all `B` in `results/b0_dot.txt`.)

| log2 k | `vecdot` today (us) | copy ceiling (us) | cuBLAS dot, device mode (us) | two-stage `vecdot`, B = 1024 (us) | today, % of ceiling | two-stage, % of ceiling |
|---|---|---|---|---|---|---|
| 20 | 323 | 8.7 | 13.3 | 13.7 | 3 % | 64 % |
| 21 | 715 | 24.0 | 23.0 | 23.9 | 3 % | 100 % |
| 22 | 2 723 | 198 | 187 | 185 | 7 % | 107 % |
| 23 | 5 441 | 398 | 368 | 374 | 7 % | 106 % |
| 24 | 10 862 | 795 | 732 | 730 | 7 % | 109 % |
| 25 | 21 800 | 1 590 | 1 463 | 1 457 | 7 % | 109 % |
| 26 | 43 447 | 3 190 | 2 915 | 2 910 | 7 % | 110 % |

`f32` is the same picture: today 3–8 %, two-stage 107–109 % for `k >= 2^22`.

- **Today's kernel fails D5** (75 % of the ceiling): 7 %. The cause is in
  `kernels/reduce_plan.rs`: the plan puts one block per output tile, and a rank-0 output is one
  tile, so the whole reduction runs on one SM.
- **A two-stage composition of the same kernel passes**: partials over a `[B, k / B]` view
  (`B` outputs = `B` blocks), then one reduction of the `B` partials. Any `B` in 256..4096 gives
  101–110 % for `k >= 2^22`; large `B` degrades (65 536: 85–108 %). It is 15x faster than today
  and **equal to the cuBLAS dot** in device pointer mode, so the own kernel stays (D3): the cuBLAS
  dot is not significantly faster, and it would need the handle-wide pointer-mode switch.
- Below `2^22` the two-stage form is 13–24 us, near the cuBLAS dot (13–23 us); launch-bound.
- The output of a 1-D dot is **rank 0** (`shape []`): `vecdot_from` needs a rank-0 output.

### 1b. The same dot as a `1 x k @ k x 1` GEMM on the chain (round 3, Q15)

`results/b0_dot_gemm.txt`. The operands reshaped to `[1, k]` and `[k, 1]`, output `[1, 1]`,
`rt::matmul` with one driver enabled.

| dtype | log2 k | chain: Lt | chain: Ex | chain: cuBLAS | two-stage `vecdot` | copy ceiling |
|---|---|---|---|---|---|---|
| f64 | 20 | 13.6 | 13.6 | 13.7 | 14.0 | 8.9 |
| f64 | 22 | 190 | 190 | 192 | 191 | 203 |
| f64 | 24 | 754 | 754 | 753 | 748 | 810 |
| f64 | 26 | 2 996 | 3 003 | 3 007 | 2 990 | 3 245 |
| f32 | 20 | 7.3 | 7.6 | 7.6 | 9.7 | 4.9 |
| f32 | 22 | 60.2 | 60.0 | 60.1 | 21.0 | 24.9 |
| f32 | 24 | 400 | 396 | 396 | 373 | 403 |
| f32 | 26 | 1 566 | 1 574 | 1 581 | 1 502 | 1 625 |

- The GEMM route reaches 92–97 % of the ceiling for `k >= 2^22` and equals the two-stage `vecdot`
  within 5 % (one exception at `f32`, `k = 2^22`, where the two-stage form is 3x faster). Lt,
  `cublasEx` and the classic gemm are identical: cuBLAS runs its own multi-block reduction for
  `m = n = 1`.
- So the inner dot needs **no driver code at all**: the rule table's rank-1 case can be lifted to
  a `1 x k @ k x 1` problem in `resolve_geom` (a size-1 axis inserted on each operand and on the
  output) and sent down the chain. `alpha`/`beta`, the driver flags and mixed dtypes come for
  free; the two-stage `vecdot` and the epilogue kernel are not needed.

## 2. gemv: the chain against raw `?gemv`

`chain / gemv`: below 1 means the chain is faster.

| dtype | cell | matvec | vecmat |
|---|---|---|---|
| f64 | 1024 x 1024 | 1.00 | 1.00 |
| f64 | 4096 x 4096 | 1.00 | 1.00 |
| f64 | 8192 x 8192 | 1.00 | 1.00 |
| f64 | 65536 x 64 | 1.00 | 1.00 |
| f64 | 256 x 452 676 | 0.97 | 0.97 |
| f64 | 452 676 x 256 | 1.00 | 1.00 |
| f32 | 1024 x 1024 | **0.67** (5.6 vs 3.7 us) | 0.91 |
| f32 | the five larger cells | 0.99–1.02 | 0.99–1.00 |

Per driver: Lt, `cublasEx` and the classic gemm give the same time on every cell (within 1 %).
Batched (`f64`, 64 x 1024 x 1024): `cublasDgemvStridedBatched` 1 460 us, the chain 1 455 us, for
both the broadcast rule (`[p, m, k] @ [k]`) and the narrow GEMM (`[p, m, k] @ [p, k, 1]`).

- **cuBLAS gemm already reaches its gemv kernels** for `m == 1` / `n == 1`, batched or not. The
  only exception is one launch-bound `f32` cell (1.9 us absolute). **D9 is not met: gemv is not
  implemented.** (Not measured: `c32`/`c64`; no reason to expect a different picture.)

## 3. syrk: `?syrk` + a triangle copy against the chain

`chain / (syrk upper + copy of n^2 / 2 elements)`: above 1 means syrk is faster.

| n | k | f64 | f32 | c32 (k = n) |
|---|---|---|---|---|
| 256 | n / 8 | 0.45 | 1.27 | |
| 256 | n | 0.97 | 0.70 | 1.76 |
| 256 | 8 n | 1.00 | 0.53 | |
| 512 | n / 8 | 1.88 | 0.63 | |
| 512 | n | 2.02 | 0.69 | 2.29 |
| 512 | 8 n | 1.98 | 0.58 | |
| 1024 | n / 8 | 1.83 | 0.61 | |
| 1024 | n | 1.90 | 0.96 | 2.82 |
| 1024 | 8 n | 1.85 | 1.07 | |
| 2048 | n / 8 | 1.76 | 1.21 | |
| 2048 | n | 1.87 | 1.77 | 2.69 |
| 2048 | 8 n | 1.78 | 2.57 | |
| 4096 | n / 8 | 1.87 | 1.71 | |
| 4096 | n | 1.90 | 1.82 | 2.55 |
| 4096 | 8 n | 1.88 | 2.96 | |

- The gain is the expected halving of the flops: **1.8–2.0x for `f64`** and **1.7–3.0x for
  `c32`** once the problem is large enough; the triangle copy is 0.1–1 % of the time.
- Two syrk calls cost exactly twice one call, i.e. the same as gemm: the mirror is what makes
  the win (D7).
- Where syrk loses, it is cuBLAS's own small-problem kernels: `f32` for `n < 2048` (0.53–1.07),
  `f64` at `n = 256` (0.45–1.00). **A size rule by output type** keeps every win and drops every
  loss: `f64` from `n >= 512`, `f32` from `n >= 2048`, complex at every measured size.
- `c64` was not timed (same kernels as `f64` by expectation): it takes the `f64` rule, and B3's
  test matrix includes a timing-free correctness cell only.

## Consequences for the plan

| form | gate | outcome |
|---|---|---|
| inner dot | D5: >= 75 % of the ceiling | **implement** as a `1 x k @ k x 1` GEMM on the chain (§1b; a two-stage `vecdot` would do as well) |
| gemv | D9: >= 10 % | **drop**: no gain |
| syrk | D9: >= 10 % | **implement** with a size rule per output type |

Outside this plan, found on the way: every CUDA reduction to a rank-0 output (a whole-tensor
`sum`, a norm) has the same one-block plan as today's dot. That is a property of the reduce
kernel, not of matmul, and is recorded as a follow-up for the owner.
