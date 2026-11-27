---
name: cutensor-probe-facts
description: 2026-11-27 empirical cuTENSOR 2.7.0 probe on the RTX 5070 (sm_120) - stride-0 accepted (inputs correct, output garbage), duplicate mode = sum-of-strides diagonal, opC CONJ silently ignored, alignment unchecked, mixed R32xC32 fails only at plan creation, GEMM-shaped bit-identical to cuBLAS, c64 1.25x faster than Zgemm, one-time 28 ms lazy init, plan cache saves nothing; cudarc cutensor feature facts and the libcutensor.so symlink requirement
metadata:
  type: reference
---

Probed 2026-11-27 (C/C++, report + sources in `2026-11-27-cuda-einsum-cutensor/probe/PROBE-REPORT.md`).
Library: pip `nvidia-cutensor` 2.7.0 (CUDA-13 build, soname `libcutensor.so.2`, only heavy dep
`libcublasLt.so.13`) at `/home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/`;
cuTENSOR is NOT installed system-wide. 2.4.1 (conda pkg cache) behaves identically except where noted.

## Where 2.7.0 differs from its own documentation

- **Stride 0 is accepted** (descriptor docs: "must be larger than zero"). Inputs: correct broadcast,
  any size, batched. **Output with stride 0: accepted and garbage** (write race). 2.4.1 rejects
  stride 0 (NOT_SUPPORTED). Never rely on it: copy stride-0 inputs, refuse stride-0 outputs.
- **A mode listed twice in one tensor is accepted** and addresses the diagonal (offset = SUM of the
  two strides). Same in 2.4.1. Undocumented; prefer an explicit diagonal *view* (same arithmetic).
- **opC = CONJ is silently accepted and ignored** (D = A*B + beta*C). Same in 2.4.1. opA/opB CONJ
  works (c64/c32, strided); CONJ on a real type -> CreateContraction NOT_SUPPORTED.
- **alignmentRequirement is not checked** against pointers or strides (256 declared for 8-byte
  pointers: correct results, also on cuTENSOR's own kernels). Only rule: alignment >= element size.
- **Scalar pointer type is never checked** (wrong type -> silently wrong, e.g. all zeros).
  SCALAR_TYPE attribute is read-only (Set to a real type -> INVALID_VALUE).
- **R32 x C32 -> C32 and R64 x C64 -> R64** pass CreateContraction and fail only in
  **CreatePlan** (NOT_SUPPORTED). Treat NOT_SUPPORTED at any planning stage as a decline.
- Negative stride / extent 0 -> INVALID_VALUE at descriptor creation.
- Every D mode must occur in A or B (INVALID_VALUE otherwise).

## What works as documented

- Arbitrary positive strides (odd strides, every mode non-contiguous, offset views), permuted modes,
  C aliasing D (beta accumulate), rank-0 operands and rank-0 output, broadcast by omitting extent-1
  modes, rank-8/rank-12 operands, f16/bf16 with COMPUTE_DESC_32F and f32 scalars (align 2 ok),
  mixed R64 x C64 -> C64 (both orders, CONJ allowed), TF32 (rel 3.3e-4) and 3XTF32 (2.4e-7).

## Timing (RTX 5070 Laptop, 36 SM, f64 at 0.33 TFLOP/s)

- Host planning: **one-time ~28 ms lazy init of the cuBLAS path** inside the first
  `cutensorEstimateWorkspaceSize` that maps to cuBLAS (64^3 triggers it, 8^3 does not); afterwards
  first plan 20-50 us (non-GEMM 0.2 ms), repeated identical problem 12-16 us; the plan cache saves
  nothing measurable for GEMM-shaped problems (CACHE_MODE_NONE identical).
- Device: f64/f32 GEMM-shaped cuTENSOR == cuBLAS **bit-identical**, ratio 1.00; **c64 1024^3
  cuTENSOR 1.25x faster than cublasZgemm** (0.42 vs 0.34 TFLOP/s). Non-GEMM contractions run at the
  full f64 GEMM rate (0.31-0.35 TFLOP/s) and need their 32-36 MiB workspace.

## cudarc 0.19.10 cutensor facts

- Feature `cutensor` = `cutensor-02006` (2.6 header; 2.3-2.6 supported): `cutensor::sys` bindgen
  fns + `cutensor::result` thin Result wrappers; no safe layer (`CuTensor`/`TensorDescriptor` exist
  only in an `ignore`d doc example).
- The compute descriptors (`CUTENSOR_COMPUTE_DESC_32F/64F/TF32/...`) are exported **const globals**
  (`nm -D` shows `D` symbols), absent from cudarc; read them via the public
  `cudarc::cutensor::sys::culib()` (`libloading::Library`) with `get::<*const cutensorComputeDescriptor_t>`.
- `fallback-dynamic-loading` dlopens by name: candidates `libcutensor.so`, `libcutensor64.so`,
  `libcutensor64_13*.so`, `libcutensor.so.{13,12,11,10,9,1}` - the pip dir has only
  `libcutensor.so.2`, so a `libcutensor.so` symlink (and `LD_LIBRARY_PATH` to its dir) is REQUIRED
  locally. Linking a C probe needs both `libcutensor.so` (link) and `libcutensor.so.2` (SONAME at run).
- `cudaDataType_t`/`cutensorOperator_t` in `cutensor::sys` are separate copies from cublas's.
