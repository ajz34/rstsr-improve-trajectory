# Plan: CUDA matmul special forms (inner dot, gemv, syrk)

Status: **grill closed 2026-11-26** (`DECISIONS.md`, D1–D13); B0 measured (`B0-REPORT.md`).
Outcome: gemv dropped; the inner dot is a `1 x k @ k x 1` GEMM on the chain; syrk has an `f32`
size gate (as planned; see the final state below).
**Final state (2026-11-26)**: all three batches are committed on rstsr `261021/cuda`, on top of
`80e626d`: `bc4c8d9` (scale alignment fix), `a7c65aa` (inner dot), `196e348` (syrk), `dae8198`
(`MatMulUseSyrk`), `63b694c` (matrix and bench). Deviations from the plan text below, in order
of appearance: there is **no size gate** any more (D11 was revised by the maintainer after the
B3 sweep); the output types are chosen by the flag `MatMulUseSyrk { Allow, Scientific, Disallow }`
on `CudaRuntimeFlags` (default `Scientific`); the trace names the routine `cublas (syrk)`; the
dot and syrk tests live in `tests/test_matmul_forms.rs` and as rows of the layout matrix.
Facts: `GRILL-INIT-PROMPT.md`, `RESEARCH-cublas.md`.

## Problem

Three shapes reach the CUDA matmul chain today and get less than they could:

| shape | today | could |
|---|---|---|
| `[k] @ [k]` (inner dot, rank-0 output) | `UnImplemented` from `resolve_geom` | the existing `vecdot` kernel |
| `[m, k] @ [k]`, `[k] @ [k, n]` (gemv) | cuBLASLt gemm with `n == 1` / `m == 1` | `cublas?gemv_v2` (if measured faster) |
| `a @ a.t()` (Gram / density matrix) | full gemm, `n^2 k` flops | `cublas?syrk_v2` + a triangle mirror, `n^2 k / 2` flops (if measured faster) |

The inner dot is a functionality gap; gemv and syrk are performance questions that only a
measurement can settle (cuBLAS may already route skinny gemm to gemv-like kernels, and may implement
syrk through gemm-like kernels).

## Design (settled)

### No form for the dot

`resolve_geom` lifts the rule table's rank-1 case to a `1 x k @ k x 1` problem: a size-1 axis is
inserted on each operand and on the rank-0 output, and the problem goes down the chain like any
GEMM. `alpha`, `beta`, the driver flags and the trace come from the chain; mixed dtypes are served
by the naive driver as for any GEMM. Measured (B0 §1b): 92–97 % of the copy ceiling for
`k >= 2^22`, equal to the two-stage `vecdot` within 5 %.

### The `Syrk` form

No new `GemmDriver` variant and no new `MatMulDrivers` bit. Condition as the BLAS devices
(`rstsr-openblas/src/matmul.rs`), on the caller's layouts: `beta == 0`, the three types equal,
`a` and `b` the same pointer at their offsets, `b`'s layout equal to `a`'s with the axes
reversed, unbatched, and the size gate of D11 (`f64`/complex: any `n`; `f32`: `n >= 2048`). When
it holds, the classic driver is asked first with `cublas{S,D,C,Z}syrk_v2`; a small mirror kernel
copies the filled triangle across the diagonal with the output's own strides. When the form does
not hold, or the routine declines (layout cuBLAS cannot express, status), the gemm chain runs
unchanged. `cublasCsyrk` is the plain-transpose form (confirmed in B0). `trace_matmul` names the
routine (`dsyrk`), and disabling the classic driver sends every syrk case to the chain.

## Batches (each rstsr commit on an explicit go)

| batch | content | gate |
|---|---|---|
| **B0** probe | done: `B0-REPORT.md` | reviewed |
| **B1** inner dot | lift the rank-1 case in `geom.rs` to `1 x k @ k x 1`; host tests; dot rows in the device tests | go |
| **B2** syrk | `Syrk` form, size gate (D11), classic `?syrk_v2`, mirror kernel, trace; tests | go |
| **B3** matrix, bench, docs | form rows in the layout matrix test (both orders, serving types, fall-through cases on both sides of the `f32` gate), `bench_matmul` rows, module docs table, README numbers here | go |

gemv was planned as a batch and is dropped (B0, section 2).

## Tests (B1–B3)

- `tests/test_matmul_forms.rs`:
  - dot: `alpha`, `beta` with a caller output, strided views, both device orders,
    `f32`/`f64`/`c32`/`c64`, each serving driver forced, mixed dtype on the naive driver, a
    rank-0 output next to the `[1]`/`[1, 1]` shapes, large `k` on the multi-block path;
  - syrk: `a @ a.t()` fresh, sub-matrix views `a[:n, :k]`, both orders, `f32`/`f64`/`c32`/`c64`,
    `f32` sizes just below and just above 2048, `beta != 0` stays gemm and is correct,
    `a @ a.h()` stays gemm and is correct, batched stays gemm, a `b` that is not exactly the
    reversed layout of `a` is not taken.
- With the classic driver disabled the gemm chain gives the same answers; `trace_matmul` names
  the routine.
- Gates: `cargo test -p rstsr-cuda --features faer -- --test-threads=1` (nvcc 13.2), the
  `CUDARC_CUDA_VERSION=12060` build, clippy and `cargo doc` of `rstsr-cuda`.

## Out of scope

**gemv** (measured: cuBLAS gemm already reaches its gemv kernels, B0 section 2), batched syrk
(no cuBLAS entry; stays gemm), half syrk (no entry), `herk` (`a @ a.h()` is not detectable from
layouts: `conj` is materialised), `syrkx`, a tensor-level `syrk`/`dot` API, cuTENSOR.

**Found by the probe, for the owner (Q18):** every CUDA reduction to a rank-0 output (a
whole-tensor `sum`, a norm) has the same one-block plan as today's dot, i.e. runs on one SM.
A two-stage rank-0 plan in the reduce kernel would fix all of them at once.

Deferred with no interface cost (Q9): **non-affine batches** (a gapped multi-axis batch part of
gemm) through `cublasGemmBatchedEx` / `cublasGemmGroupedBatchedEx` /
`cublas{t}gemvBatched` with device arrays of pointers; today they fall to the tiled and naive
kernels. The choice of entry is made inside `prepare`, so it can be added later untouched by the
forms.
