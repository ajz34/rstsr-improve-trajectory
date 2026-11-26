---
name: cudarc-cublas-version-gating
 description: cudarc 0.19.10 cuBLAS bindings are gated per CUDA version in src/cublas/sys/mod.rs - cublasSetMathMode/GetMathMode from cuda-12090, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH from cuda-13000; CUDARC_CUDA_VERSION is spelled "13000" (5 chars)
metadata:
  type: reference
---

Read in `~/.cargo/registry/src/*/cudarc-0.19.10/` on 2026-11-26 (check the blocks above a symbol, the file has
many `#[cfg(any(feature = "cuda-..."))]` sections):

- `cublasSetMathMode` / `cublasGetMathMode`: `cuda-12090`, `cuda-13000`..`cuda-13040` only (around line 5429).
- `cublasMath_t` constants: `CUBLAS_TF32_TENSOR_OP_MATH` in all recent blocks; the `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH = 8`
  block (around line 221) is `cuda-13000`+ only.
- No binding for `cublasSetFixedPointEmulationMaxMantissaBitCount` in this version.
- cudarc's build script accepts `$CUDARC_CUDA_VERSION` only as `format!("{major}0{minor}0")`, e.g. `12060`,
  `13000`; an unknown value panics. It enables exactly one `cuda-XXXXX` feature.
- A dependency's cargo features are invisible to `cfg`, hence rstsr-cuda's `build.rs` cfgs
  (`cublas_math_mode`, `cublas_fixedpoint_math`) derived from the same variable or `nvcc --version`.
