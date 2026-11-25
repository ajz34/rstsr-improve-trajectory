# B2 — `src/matmul/` skeleton, the move, the naive driver, the flags refactor

Committed in rstsr as `c75aecd` on `261021/cuda` (the owner's go given after review). The same patch
is kept as `patches/b2-matmul-skeleton.patch`.

| | |
|---|---|
| rstsr | `261021/cuda` @ `c75aecd` |
| GPU | NVIDIA GeForce RTX 5070 Laptop 8 GiB, sm_120, CUDA 13.2, debug build for tests, `--release` for timing |
| raw output | `results/b2_full_suite.txt`, `results/b2_driver_check.txt`, `results/b2_bench_{cublas,tiled,naive}.txt`, `results/b2_trace.txt` |

## What was done

| plan item | result |
|---|---|
| 1. `src/matmul/` layout | `mod.rs` (41 lines: docs, `mod`, re-exports), `mode.rs`, `flags.rs`, `geom.rs`, `cublas_args.rs`, `inner.rs`, `drivers/{mod,cublas,tiled,naive}.rs`. Not created yet: `tensor.rs`, `drivers/cublas_lt.rs`, `drivers/cublas_ex.rs` (empty files would only be noise; they arrive with their batches). |
| 2. the move | `kernels/gemm.rs` geometry -> `geom.rs`; `kernels/cublas_gemm.rs` argument derivation -> `cublas_args.rs`; the typed cuBLAS launch -> `drivers/cublas.rs`; the tiled kernel -> `drivers/tiled.rs` (kernel source and template substitutions byte-for-byte unchanged, D30 respected). `kernels/{gemm,cublas_gemm}.rs` deleted. All host tests moved with their code. |
| 3. naive driver | `drivers/naive.rs`: one thread per output element, loop over `k`, per-axis batch strides, templated `@TA@ @TB@ @TC@ @TACC@`. Serves every `CudaArith` type plus `f16`/`bf16`, and every type triple including mixed operands. |
| 4. selection | `drivers/mod.rs`: `GemmDriver` enum (public, `#[non_exhaustive]`), the candidate table (`candidates::<TA,TB,TC>`), the ordered dispatch with fall-through, the error naming the drivers and the triple. `inner.rs`: `matmul_with_mode_inner` (mode is the last argument) and the `DeviceMatMulAPI` impls. |
| 5. flags | `CudaRuntimeFlags` is `#[non_exhaustive]` with `compute_type_32f/64f/16f`, `fixedpoint_mantissa_bits`, `matmul_drivers`, `trace_matmul`. `MatMulDrivers` bit set (`CUBLAS_LT`, `CUBLAS_EX`, `CUBLAS`, `TILED`, `NAIVE`; `all`, `none`, `contains`, `union`, `minus`, `BitOr`, `Sub`). The three op-behaviour environment variables are gone. `DeviceCuda::with_flags` added. `ComputeType32F` re-exported from `flags::`. |
| 6. ports | `tests/test_matmul_tiled.rs` forces the tiled driver through the flags; the check test and the timing example (kept here in `src/`) take the driver as a parameter. |
| 7. verification | below |

New in the rstsr tree, beyond the plan's list: `tests/test_matmul_naive.rs` (8 tests), a regression
suite for the new driver and the chain. The B1 check stays only here (`src/`).

## Design points worth knowing

- **Driver contract.** A driver takes a resolved `Problem` and returns `Ran` or `Declined(reason)`.
  A decline never enqueues anything, so fall-through cannot change the answer. Real failures
  (NVRTC, launch, ordering events) are errors.
- **Geometry is driver-free.** `resolve_geom` no longer decides `lane_axis` or refuses a non-affine
  batch: it returns the batch parts (per-axis shape and strides) of all three operands, and each
  driver states what it cannot express. `GemmGeom::caller()` gives the same addresses in the
  caller's roles, which is what the naive kernel needs under `ColMajor`.
- **Single-type drivers through a run-time type check.** `DeviceMatMulAPI` is now generic over
  `(TA, TB, TC)`; the cuBLAS and tiled drivers check the triple with `TypeId` and re-type the
  buffers (`CudaBuffer::view_as`), the fast-path shape `rules/code-concepts.md` names. No public
  bound was widened.
- **Naive arithmetic.** Each operand is cast to `@TACC@` (`f32` for a half output, else the output
  type), the `k` products are multiplied and added in `@TACC@`, and the result is rounded to the
  output type once. For every type but half this is exactly the D11 rule; for half it is the
  "round once" behaviour Q21 argued for (a product rounded to `f16` first would lose it).
- **`beta == 0`** is decided on the host (`read_c` flag), not by an in-kernel comparison, so the
  same kernel serves real, half and complex outputs and never reads an uninitialised `c`.

## Verification

**Whole crate, `--features faer`, `--test-threads=1`** (`results/b2_full_suite.txt`): 201 tests pass.

| target | tests |
|---|---|
| `--lib` (host: geometry, cuBLAS arguments, driver table, bit set, cast rule) | 82 |
| `test_kernels` | 55 |
| `test_ops_wave` | 26 |
| `test_device_basic` (incl. the ported `flags_resolution` and `with_flags`) | 13 |
| `test_pinned` | 12 |
| `test_matmul_naive` (new) | 8 |
| `test_matmul_tiled` (ported to the flags) | 5 |

`cargo clippy --all-targets --features faer`: no warning in `rstsr-cuda` (three pre-existing ones in
`rstsr-core`). `cargo check --all-features` and `cargo doc --no-deps`: clean.

**Layout matrix, one module per `(dtype, driver)`** (`src/test_matmul_tiled_check.rs`,
`results/b2_driver_check.txt`): 72 tests, 360 cases, every accepted one exact.

| module | cases | ok | refused |
|---|---|---|---|
| tiled `f64`, tiled `f32` | 45 each | 35 each | 10 each |
| naive `f64`, `f32`, `i32`, `i64`, `c64`, `c32` | 45 each | 45 each | 0 |

The tiled refusals are the same ones as in B1 (16 `REFUSED` lines over the matrix — no unit-stride
axis, non-affine batch — plus the 4 cases of the explicit refusal test), so the move is
behaviour-preserving. Naive answers every one of them. Through the default chain the same cases
also pass (`test_matmul_naive`).

**`test_matmul_naive`** covers: plain shapes; the declined layouts through the default chain and
with naive forced; a forced driver that cannot serve -> error naming the driver, the triple and the
enable set (empty set likewise); `i32`, `u8`, mixed `f32 x f64 -> f64`; `f16 x f16 -> f32` through
`matmul_from`; complex `c64` (batched, transposed, stepped operand) and `c32`; `matmul_from` into stepped and reversed output views with and without `beta`; both
default orders on a batched problem.

**Timing** (`src/bench_matmul_check.rs`, 10 reps, wall time per call, `results/b2_bench_*.txt`):

| shape | cuBLAS | tiled | naive |
|---|---|---|---|
| f32 512³ | 9.4 TF/s | 4.3 TF/s | 1.0 TF/s |
| f32 2048³ | 15.8 TF/s | 5.1 TF/s | 0.6 TF/s |
| f32 256³ x128 | 9.1 TF/s | 4.8 TF/s | 1.1 TF/s |
| f64 2048³ | 326 GF/s | 384 GF/s | 367 GF/s |
| f64 256³ x128 | 337 GF/s | 392 GF/s | 354 GF/s |

The cuBLAS and tiled columns match B1 (same kernels, same numbers): the chain adds nothing
measurable. Naive is 10-25x behind cuBLAS in f32 (DRAM-bound, as expected for a fallback) and level
with both in f64, where this GeForce part is bound by its 1/64-rate fp64 pipe, not by memory.

**Trace** (`results/b2_trace.txt`, `src/trace_matmul_check.rs`): `trace_matmul` prints one line per
attempt, e.g.

```text
[rstsr-cuda matmul] double x double -> double 20x16x12 batch 1 mode Default: tiled -> declined (cannot serve: an operand has no unit-stride axis (strides of the left operand))
[rstsr-cuda matmul] double x double -> double 20x16x12 batch 1 mode Default: naive -> ran
```

and the tiled-only failure reads `cuda matmul: no enabled driver can serve double x double -> double
(20x16 @ 16x12, batch 1); enabled MatMulDrivers(tiled); declined: tiled: cannot serve: ...`.

## Bugs found on the way

- **Naive under `ColMajor` used the swapped extents** (`geom.m`/`geom.n` are the row-major
  roles): the first batch slice came out right by the symmetric enumeration and the others did
  not. Fixed by giving the caller view its own `(m, k, n)`; pinned by `naive_both_orders` and the
  `caller_view_addresses_the_callers_elements` host test.
- The CPU serial device **refuses a strided `c`** in `matmul_from` and **panics on integer
  overflow** (debug builds) where the device wraps. Both are pre-existing CPU behaviours; the new
  tests route around them, and the overflow difference is now listed under the crate's divergences.

## For the owner

1. **`#[non_exhaustive] CudaRuntimeFlags`** (per round 2, Q16) means no struct literal outside
   the crate: `let mut f = CudaRuntimeFlags::default(); f.matmul_drivers = ..;
   device.with_flags(f)`. If the literal matters to you, the attribute is one line to drop.
2. **New in-tree test file** `tests/test_matmul_naive.rs`: not in the plan's list; say so and it
   goes (the matrix here would still cover the driver).
3. **`DeviceMatMulAPI` is generic over the triple now**, so integer and mixed-operand `matmul`
   compile and run on naive already. `DeviceExtMatMulAPI` itself (`rt::ext_matmul`) is still B5.
   Complex matmul did not compile on CUDA before B2 (the impl was bounded by `CublasScalar`); it
   now runs on naive and is in both suites, still without an accelerated driver (B5).
4. **Inner dot** (`[k] @ [k]`) is still refused by name for every driver. Naive could serve it
   trivially (it is a `1 x 1` GEMM there); kept as is until you say otherwise.
5. **Mode keys**: `CudaMatMulMode` is implemented for `f32`, `f64`, `c32`, `c64`, `f16`, `bf16`
   (integers: `Mode = ()`), keyed by the output type, so `bf16 -> bf16` reads the 16F field while
   the D18 table lists 32F modes for it. To settle with B5.
6. `fixedpoint_mantissa_bits` defaults to `0` = "library default"; the real range is B3's.
7. `MatMulDrivers::all()` includes the `cublasLt` and `cublasEx` bits although no driver sits
   behind them yet; they are never candidates, so the bits are inert until B3/B4.
8. `trace_matmul` replaces `RSTSR_CUDA_DEBUG_GEMM` but prints less: no pointer bases or cuBLAS
   argument dump. Say if that once-per-process dump is worth keeping as a second flag.
9. Rustfmt was applied to the new files only; the crate's existing files are not fmt-clean (11
   diffs in the old `flags.rs` and `gemm.rs` alone), so the moved tiled code is reformatted
   relative to its old home while its kernel text is unchanged.
