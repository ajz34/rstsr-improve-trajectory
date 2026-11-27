# Grilling: cuTENSOR-backed `tensordot` / `einsum` on `DeviceCuda`

Started 2026-11-27. Target: the `rstsr-cuda` crate of the `rstsr` repo (read at
`261127/cuda-prelude-f-convention` @ `937ce04`; the CUDA work lives on `261021/cuda`).

## The initial prompt (verbatim)

> You are going to see if tensordot and einsum functions can use cuTENSOR to accelarate.
> The dtypes of concern is f64/f32/c64/c32, f16/bf16 optional. Recovering CPU's capability is already good.
> Different to BLAS devices that tensordot use matmul to be performed with, we use cuTENSOR or einsum anyway in cuda device, if possible.
> You are going to see how to do this, discuss with me. At last probably we can write a new plan in rstsr-improve-trajectory.
> Please browse the current crate status and see how existing einsum is done.
> The possible API shape is `rt::cuda::einsum` and `rt::cuda::tensordot`, and all related implementations at `rstsr-cuda/src/einsum`.
> You need to at least notify me that if cutensor can support arbitary strides of tensor. An additional problem you need to do fact check is that how cutensor handles complex conjugate.

## Fact base gathered before round 1 (agent-authored)

### How einsum/tensordot exist today

- `einsum` exists **only** in the TBLIS plugin (`rstsr-tblis`, CPU, `Raw = Vec<T>`): `rt::tblis::einsum(subscripts, operands: impl TensorViewListAPI<T, B>, optimize: impl PathOptimizer, memory_limit)` (+ `_f`, `einsum_with_output[_f]`, `einsum_from[_f]` on `TensorAny<_, _, _, IxD>`). The parser and the N-ary contraction are TBLIS's; `opt-einsum-path` 0.2 only supplies the path optimizer. All operands share one dtype `T: TblisFloatAPI`. `rt::tblis::tensordot` = `tensordot_to_einsum_str` (rstsr-common) + `einsum`.
- `tensordot` core entries (`rt::tensordot[_f|_from|_from_f]`, methods) sit on `DeviceTensordotAPI<TA, TB, TC, DA, DB, DC>` (uninit output, normalized axes, no conj). CPU devices: `tensordot_gemm_layouts` (view-only single GEMM) else a naive kernel (rayon: `tensordot_naive_cpu_rayon`).
- `DeviceCuda` implements `DeviceTensordotAPI<T, T, T, _>` as the **view-only tier**: `tensordot_gemm_layouts` -> `matmul_uninit` (cuBLAS chain); if the three blocks do not collapse to 2-D it refuses with `UnImplemented` ("would need a packing pass"). `DeviceExtTensordotAPI` (mixed dtypes) is not implemented. `crates-device/rstsr-cuda/src/linalg.rs`.
- CUDA matmul chain (precedent for drivers, flags, resources): `src/matmul/{mod,doc.md,mode,flags,geom,cublas_args,inner,tensor}.rs`, `drivers/{cublas,cublas_ex,cublas_lt,syrk,tiled,naive}.rs`; `CudaRuntimeFlags.matmul_drivers` bit set; `CudaOrdinalObj::cublaslt()` (per-ordinal handle, `OnceLock<Option<_>>`, created after `context().bind_to_thread()`); `CudaStreamObj::cublaslt_workspace()` (32 MiB per stream, lazy); `TensorCudaMatMulAPI` for the CUDA-specific entries; `prelude::{rstsr_traits, rstsr_structs, rstsr_funcs}` exported by the facade as `rt::cuda::*`.
- Tests: hardware tests skip silently without a GPU (`require_cuda!`), copy-and-compare against the CPU device (`host_nd`, exact integer-valued data or `assert_close`).

### cuTENSOR facts (primary sources: header 2.7.0, API reference, user guide, release notes)

- **Library**: not part of the CUDA toolkit; a separate download. cudarc 0.19.10 has a `cutensor` feature (= `cutensor-02006`, bindings from the 2.6 header; 2.3-2.6 supported) exposing `cutensor::sys` (bindgen fns, incl. `cutensorCreateContraction`, `cutensorCreateContractionTrinary`, `cutensorCreateReduction`, `cutensorCreatePermutation`, `cutensorContract`, `cutensorReduce`, `cutensorPermute`, plan/plan-preference/workspace functions, `cutensorGetVersion`, `cutensorGetErrorString`) and `cutensor::result` (thin `Result` wrappers). There is **no safe layer** (`CuTensor`/`TensorDescriptor` only in an `ignore`d doc example).
- **Not in cudarc**: the compute descriptors. Since 2.0 they are *exported const globals* (`CUTENSOR_EXTERN const cutensorComputeDescriptor_t CUTENSOR_COMPUTE_DESC_64F;` in `cutensor/types.h`; `nm -D` shows them as `D` symbols), so bindgen skipped them. Reachable at runtime through the already-public `cudarc::cutensor::sys::culib()` (`libloading::Library`) with `get::<*const cutensorComputeDescriptor_t>(b"CUTENSOR_COMPUTE_DESC_64F\0")`.
- **Loading**: with cudarc's `fallback-dynamic-loading` (what rstsr-cuda uses) the `cutensor` feature needs no header at build time; `libcutensor.so` is dlopened by name at first use (`libcutensor.so`, `libcutensor64.so`, ... candidates) and the call panics if absent (the crate's existing `catch_unwind` probe pattern applies).
- **Strides** (`cutensorCreateTensorDescriptor`): `stride[i]` is the displacement in elements of mode i; "If stride is NULL, a packed generalized column-major layout is assumed"; "**Each stride must be larger than zero**; to be precise, a stride of zero can be achieved by omitting this mode entirely ... cuTENSOR will then automatically infer that the [omitted] mode should be broadcasted". `extent` "must be larger than zero". `alignmentRequirement` is the base-pointer alignment in bytes (cudaMalloc: 256). Modes may appear in any order; up to 64 modes; "each mode may appear in each tensor at most once"; modes in A/B must appear in D. Summary: **arbitrary positive strides yes (no packing needed, sub-tensors and transposed views fine); no stride 0 (broadcast via omitting an extent-1 mode), no negative strides, no zero extents.**
- **Conjugate**: `cutensorOperator_t` has `CUTENSOR_OP_CONJ = 9`. `cutensorCreateContraction` takes `opA`, `opB`, `opC` "applied to each element ... before it is further processed"; the API reference restricts only `opC`: "Currently only CUTENSOR_OP_IDENTITY is supported". Release notes 2.8.0: "Fixed cutensorCreateContraction() to correctly reject a non-identity opC ... instead of being silently accepted and subsequently producing wrong results" (so 2.7.0 and older accept a non-identity `opC` and compute garbage). User guide "Supported Unary Operators": for `CUDA_C_32F/64F/16F`, `IDENTITY` or `CONJ` (the table is written for elementwise ops; the contraction table lists the same operator type for opA/opB). The block-sparse (beta) contraction explicitly has "no conjugate support" and IDENTITY-only opA/opB/opC. Summary: **conjugation is a per-operand unary flag on A and/or B of a dense contraction; no conjugate on C; CONJ is complex-only (real operands are expected to be rejected); conj-transpose = CONJ plus the mode order, nothing special.**
- **Contraction type table** (2.7.0): A, B, D all `R_32F` (compute 32F/TF32/3XTF32/16BF/16F; tensor cores Ampere+/Volta+ for the reduced ones), all `R_64F` (compute 64F: tensor cores "Ampere+"; or 32F: "No"), all `C_32F` (32F "No", TF32/3XTF32 Ampere+), all `C_64F` (64F Ampere+, 32F No), `R_16F`x`R_16F`->`R_16F` (32F compute), `R_16BF` likewise, **mixed only `R_64F x C_64F -> C_64F` and `C_64F x R_64F -> C_64F`** (no `R_32F x C_32F`). No integer contraction. Scalar type (`alpha`/`beta`): `R_16F/16BF/32F` output -> `float`; `R_64F` -> `double`; `C_32F` -> `cuFloatComplex`; `C_64F` -> `cuDoubleComplex`.
- **Execution contract**: plan-based API (descriptor -> operation descriptor -> plan preference -> workspace estimate -> plan -> `cutensorContract` on a stream); "The active CUDA device must match the CUDA device that was active at the time at which the plan was created"; workspace optional, 256-byte aligned, "recommended ... e.g. 128 MB"; `descD` must equal `descC`; D may alias C (beta accumulation); A/B must not overlap D; plan cache is per handle and on by default (`CUTENSOR_DISABLE_PLAN_CACHE`); `CUTENSOR_ALGO_DEFAULT` / `DEFAULT_PATIENT` / GETT / TGETT / TTGT; incremental autotuning optional; handle APIs are "non-blocking, no reentrant, and thread-safe" (the plan cache is shared across threads of one handle).
- **Local machine**: cuTENSOR is **not** installed system-wide. Available copies: pip `nvidia-cutensor` **2.7.0** (CUDA-13 build, only heavy dep `libcublasLt.so.13`) at `/home/a/miniconda3/envs/torch-gpu4pyscf/lib/python3.11/site-packages/cutensor/{include,lib/libcutensor.so.2}` (active conda env; needs its `lib/` on `LD_LIBRARY_PATH`); conda 2.4.1 in `/scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-*`; 2.0.1 in an old pyscf env. GPU: RTX 5070 Laptop, sm_120, 8 GB, driver 595.99.02; CUDA 13.2 (`nvcc` 13.2).
- Empirical probe (strides, broadcast, alignment, rank-0, CONJ variants, scalar types, planning cost, cuTENSOR-vs-cuBLAS timing): running in `probe/`, report lands in `probe/PROBE-REPORT.md`; results are folded into round 2.

### Probe 1 results (done 2026-11-27; `probe/PROBE-REPORT.md`, sources and raw outputs beside it)

Where cuTENSOR 2.7.0 differs from its own documentation (2.4.1 identical unless noted):

- **Stride 0 is accepted** (docs: must be > 0). Inputs broadcast correctly at any size; a stride-0
  **output** is accepted too and yields garbage (write race). 2.4.1 rejects stride 0. Decision
  consequence: copy stride-0 inputs (never rely on it), refuse stride-0 outputs (never silent).
- **A mode listed twice in one tensor is accepted** and addresses the diagonal (offset = sum of the
  two strides; 2.4.1 same). Undocumented; the explicit diagonal *view* does the same arithmetic.
- **opC = CONJ is silently accepted and ignored** (D = A*B + beta*C; 2.4.1 same). opA/opB CONJ are
  correct for c64/c32, strided included; CONJ on a real type is NOT_SUPPORTED.
- **alignmentRequirement is not checked** (256 declared for 8-byte pointers: correct); only rule
  alignment >= element size. **Scalar pointer types are never checked** (wrong type -> zeros).
- **R32 x C32 -> C32** and **R64 x C64 -> R64** pass `cutensorCreateContraction` and fail only in
  **`cutensorCreatePlan`** (NOT_SUPPORTED): a decline can arrive at any planning stage.
- Negative stride / extent 0 -> INVALID_VALUE at descriptor creation; every D mode must occur in
  A or B.

As documented: arbitrary positive strides, permuted modes, C aliasing D, rank-0 operands and
output, broadcast by omitting extent-1 modes, rank-8/12 operands, f16/bf16 (32F compute, f32
scalars), mixed R64 x C64 -> C64, TF32 (rel 3.3e-4), 3XTF32 (2.4e-7).

Timing (RTX 5070 Laptop, 36 SM): host planning has a **one-time ~28 ms lazy init** (first problem
that maps to cuBLAS), then 20-50 us per new plan (0.2 ms non-GEMM), 12-16 us on a repeat; the plan
cache saves nothing measurable. Device: GEMM-shaped f64/f32 **bit-identical to cuBLAS** (ratio
1.00); **c64 1024^3 1.25x faster than `cublasZgemm`**; non-GEMM contractions run at the full f64
GEMM rate (0.31-0.35 TFLOP/s) and need their 32-36 MiB workspace.

Loading fact for the Rust side: cudarc's candidate names (`libcutensor.so`, `libcutensor64*.so`,
`libcutensor.so.{13,12,11,10,9,1}`) do not include the pip package's `libcutensor.so.2`, so a
`libcutensor.so` symlink in a directory on `LD_LIBRARY_PATH` is required locally.
