# Grill: CUDA matmul implementation (drivers + compute modes)

Started 2026-11-25. Grills the design of the matmul implementation of
`rstsr-cuda` (rstsr `261021/cuda` @ `9c7d49a`).

## Initial prompt (user, verbatim)

You are going to plan and discuss with me on matmul implementation of cuda device.

The CUDA matmul is somehow different to conventional CPU, in that
- There are multiple drivers. We will use the following ways in priority: cuBLASLt, cuBLASEx, cuBLAS, naive implementation / native implementation.
- There will have different modes. We have defined `ComputeType32F`, but there are other situations. We will probably need to introduce more compute types. These compute types binds to certain output type (not input type), like F64 also require one.
- Usual DeviceMatMulAPI do not pass compute modes. You should create an internal matmul function that handles different modes, and DeviceMatMulAPI uses the mode that parsed from `&device`.
- There may possibly have some kind of matmul that F16 x F16 = F32. In this case, `DeviceMatMulAPI` and internal matmul operation should be properly defined that `c_f32.matmul_from(&a_f16, &b_f16, 1.0, 0.0)` should work, but `&a % &b` / `rt::matmul(&a, &b)` on two F16 tensors may still give F16 output. See that if this is possible.

Some notes:
- You are suggested to create a new directory on matmul implmentation, not to increase src/ folder pressure.
- You are not suggested to spawn subagents in this session (due to current LLM provider status, not saying that subagent is not suitable for this task).
- Follow rstsr-improve-trajectory, record this in new directory in that repo.

## Facts gathered before round 1 (agent, read from the repos)

### Current code (`rstsr/crates-device/rstsr-cuda`)

- `DeviceMatMulAPI<T, T, T, DA, DB, DC>` is one impl in `src/linalg.rs:87`, bound
  `T: CublasScalar + num::Zero` (f32/f64 only). Both `matmul` and `matmul_uninit` call
  `kernels::gemm::gemm_launch`, which resolves the geometry (`resolve_geom`, shared) and then tries
  the typed cuBLAS `cublas<T>gemm[_strided_batched]` (`kernels/cublas_gemm.rs`), falling through to
  the tiled kernel (`kernels/gemm.rs`, single `@T@` template, `BM=BN=64`).
- `RSTSR_CUDA_MATMUL=tiled` forces the tiled kernel; `RSTSR_CUDA_DEBUG_GEMM=1` prints geometry.
- `ComputeType32F { Default, Pedantic, Fast16F, Fast16BF, FastTF32 }` (`src/flags.rs`) lives in
  `CudaRuntimeFlags.compute_type_32f`; it is **never consumed** by any launch.
  `DeviceCuda::flags()` resolves `Option<CudaRuntimeFlags>` override against the global default.
- cuBLAS handle: one per stream (`CudaStreamObj::cublas_slot`), mutex-locked inside `launch`.
- `DeviceTensordotAPI<T, T, T, ..>` is bound to `DeviceMatMulAPI<T, T, T, Ix2, Ix2, Ix2>`.

### Core contract (`rstsr-core`)

- `DeviceMatMulAPI<TA, TB, TC, ..>` is already generic over three dtypes; `alpha: TC, beta: TC`.
- `rt::matmul` / `&a % &b` require `TA: Mul<TB, Output = TC>`. `half` implements
  `Mul<f16> for f16 -> f16` (same for `bf16`), so two f16 operands infer `TC = f16`.
- `matmul_from`, `matmul_from_f`, `matmul_with_output` have **no** `Mul` bound: `TC` is the output
  tensor's type, so `c_f32.matmul_from(&a_f16, &b_f16, ..)` only needs
  `DeviceMatMulAPI<f16, f16, f32, ..>` to exist. The requested behaviour is expressible with no
  core change (inference ambiguity is not possible: `DC` is fixed by `c`).
- `DeviceExtMatMulAPI<TA, TB, TC, ..>` (promotion, mixed operand dtypes) exists in core; CUDA has none.

### Bindings (cudarc 0.19.10, already the sole CUDA dependency)

- `cublas::result::gemm_ex` (safe) and `sys::cublasGemmEx` / `cublasGemmStridedBatchedEx`,
  `cublasSetMathMode`, `cublasSetEmulationStrategy` are available.
- `cublaslt` is behind the cargo feature `cublaslt` (not enabled now); it carries raw `sys` bindings
  (`cublasLtMatmul`, `..AlgoGetHeuristic`, `..AlgoInit`, `..PreferenceCreate/SetAttribute`,
  `..MatmulDescSetAttribute`, `..MatrixLayoutSetAttribute`) plus a thin `result`/`safe` layer.

### Hardware matrix (2026-11-24, RTX 5070 Laptop sm_120, cuBLAS 13.4; `../2026-11-24-cuda-gemm-functionality`)

- Accepted: f32, f64, c32, c64, f16, bf16, i8 -> i32, e4m3 (Lt only); f16/bf16 -> f32 output.
- A and B must share a dtype (f16+bf16, f32+f64 rejected).
- `bf16 -> bf16` needs 32F accumulate. TF32 works (2.6e-4). BF16x9 accepted but silently falls back
  to native f32 on sm_120.
- FP64 fixed-point emulation engages **only** through the classic handle
  (`cublasSetMathMode` + emulation strategy/mantissa knobs), not through Lt on this GPU.
