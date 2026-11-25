# Decisions: CUDA matmul special forms (grill closed 2026-11-26)

Rounds: `GRILL-R1-QUESTIONS.md`, `GRILL-R2-QUESTIONS.md` (answers transcribed at the top of
the next round). rstsr `261021/cuda` @ `80e626d`. Facts: `GRILL-INIT-PROMPT.md`,
`RESEARCH-cublas.md`.

## Shape of the feature

- **D1 (R1/Q1)** Problems are classified into a **form** (`Dot`, `Gemv`, `Syrk`, `Gemm`) after the
  geometry is resolved and before the chain is walked. No new `GemmDriver` variant, no new
  `MatMulDrivers` bit; the gemm chain is the fallback when a form routine declines.
  `trace_matmul` names the routine. Disabling the classic driver turns gemv and syrk off.
- **D2 (R1/Q7)** Probe, numbers and patches live in this directory; rstsr work on `261021/cuda`,
  one commit per batch, each on an explicit go.

## Inner dot

- **D3 (R1/Q2, R2/Q11)** `[k] @ [k]` runs on the own kernel (`classify_reduce` +
  `vecdot_launch`) with a one-element epilogue for `alpha` and `beta * c`. The naive GEMM kernel
  (one thread for a `1 x 1 x k` tile) is not used. The cuBLAS dots are anchors of B0, timed
  blocking (host result) and in device pointer mode on a private handle; they replace the own
  kernel only if significantly faster, which is not expected. If they did: the mode switch would
  live inside the handle lock, and `cublasSetPointerMode_v2` does not exist on a CUDA 12 runtime
  (the CI pin).
- **D4 (R1/Q3)** The dot ignores `MatMulDrivers`: it is a reduction, not a GEMM driver.
- **D5 (R1/Q2, R2/Q10)** The dot is gated on **bandwidth**, not flops: B0 measures `vecdot` on
  `k` in `2^20..2^26` (`f32`, `f64`) against the device-to-device copy ceiling and
  `cublasDotEx`. Acceptance: at least 75 % of the copy ceiling for `k >= 2^22`; below that the
  kernel is tuned (or replaced, a new round) before the dot is wired. Small `k` is launch-bound
  and not gated.

## gemv

- **D6 (R1/Q4, R2/Q8, R2/Q12)** The `Gemv` form: `m == 1` or `n == 1`, `k > 1` (`k == 1` is an
  outer product, stays gemm), unbatched **and affine-batched** (`cublas{S,D,C,Z}gemvStridedBatched`,
  a broadcast operand is a step of 0). Typed `f32`/`f64`/`c32`/`c64` only; free vector strides
  through `incx`/`incy` (no copy); the matrix needs one unit-stride axis. Everything else falls
  to the gemm chain. Half gemv is out: not bound, "no new FFI" stands.

## syrk

- **D7 (R1/Q5, R2/Q13)** The `Syrk` form follows the BLAS devices' matmul rule
  (`rstsr-openblas/src/matmul.rs`): `beta == 0`, the three types equal, `a` and `b` the same
  pointer at their layout offsets, and `b`'s layout equal to `a`'s with the axes reversed. One
  `cublas{S,D,C,Z}syrk_v2` call fills a triangle, then a mirror kernel copies it to the other.
  `beta != 0` (mirroring is wrong for a non-symmetric old `c`), batched, `a @ a.h()` and half
  types fall to the gemm chain. `cublasCsyrk` is the plain-transpose form (confirmed by a test in
  B0).
- **D8 (R2/Q9)** Batched `a @ a.t()` follows batched gemm (cuBLAS 13.2 has no batched syrk).

## gemv: withdrawn by the probe (2026-11-26)

- **D6 is withdrawn.** B0 (`B0-REPORT.md` §2): the chain equals raw `?gemv` within 3 % on all 22
  large cells (Lt, `cublasEx` and classic gemm identical), and batched is identical; the only gap
  is one launch-bound `f32` cell of 1.9 us. D9's bar is not met, so no gemv routine is written.

## Gate

- **D9 (R1/Q6)** B0 measures before any driver code. A form is implemented only if at least 10 %
  faster than the chain's choice on the cells where it applies, and not slower elsewhere. The
  inner dot is implemented whatever the numbers (a functionality gap), subject to D5.
  *Outcome: gemv dropped; syrk implemented with a size rule (D11); dot implemented (D12).*

## Deferred (no interface cost)

- **D10 (R2/Q9)** Gapped (non-affine) batches of gemm or gemv, through
  `cublasGemmBatchedEx` / `cublasGemmGroupedBatchedEx` / `cublas{t}gemvBatched` with device
  arrays of pointers. The entry is chosen inside `prepare`; the forms do not touch it.

## Out of scope

Half gemv, batched or half syrk, `herk`, `syrkx`, a tensor-level `syrk`/`dot` API, cuTENSOR.

## Execution order

1. B0 probe (this directory): dot bandwidth, gemv, syrk. **Next.**
2. Review gate: D5 and D9 decide B2 and B3.
3. B1 inner dot -> B2 gemv -> B3 syrk -> B4 matrix, bench, docs, each on an explicit go.

## After the probe (grill closed 2026-11-26, round 4)

- **D4 is replaced (R4/Q19)** The inner dot follows `MatMulDrivers` like any GEMM: it is a GEMM.
- **D11 (R3/Q16, R4/Q20, R4/Q21)** The `Syrk` form is taken for `f64` and complex outputs at any
  size (one cell on record: `f64`, `n = 256, k = 32`, 0.45, about 17 us absolute), and for `f32`
  outputs only from `n >= 2048` (below, cuBLAS's `f32` syrk kernels lose 0.53–0.96; the
  maintainer's first proposal, `max(k, n) >= 1024`, was measured to lose on five cells). The
  thresholds are constants of the classic driver.
  - **Revised in B3 (2026-11-26); superseded by the next item** (`B3-REPORT.md`): B0 had
    measured square operands only. The sweep over `(n, k)` shows the `f32` form losing up to 2.3x
    on thin operands, and `f64`/`c64` losing below `n = 256` (the mirror launch). Implemented:
    `n >= 512` for every type, and `f32` only for `n in 2048..=4096` and `k >= 1024`, the region
    with no measured loss.
  - **Settled by the maintainer (2026-11-26, after B3)**: no size gate at all. A flag on
    `CudaRuntimeFlags`, `MatMulUseSyrk { Allow, Scientific, Disallow }`, default `Scientific`
    (`f64`, `c32`, `c64`); `Allow` adds `f32`; `Disallow` never. Implemented in B3.
- **D12 (R3/Q15, R4/Q19)** The inner dot is the rank-1 case of the rule table **lifted to a
  `1 x k @ k x 1` GEMM** in `resolve_geom` (a size-1 axis inserted on each operand and on the
  rank-0 output) and sent down the chain. No `Dot` form, scratch or epilogue kernel. B0 §1b: the
  chain reaches 92–97 % of the copy ceiling for `k >= 2^22`, equal to a two-stage `vecdot`
  within 5 %. The gain is for the types the cuBLAS drivers serve (`f32`, `f64`, `c32`, `c64`),
  which are the important ones (maintainer); on the naive driver the dot is one thread.
- **D13 (R3/Q18, maintainer)** Other rank-0 CUDA reductions stay as they are; the inner dot is
  handled in matmul only. Noted in the README, no work planned.

## Execution order

1. B0 probe: done (`B0-REPORT.md`).
2. B1 inner dot as a GEMM (`geom.rs`, host tests, dot rows in the tests) -- on the maintainer's go.
3. B2 syrk (CPU condition, gates of D11, one `?syrk_v2` call plus the mirror kernel, trace,
   tests) -- on the go.
4. B3 matrix, bench, docs, README numbers -- on the go.
