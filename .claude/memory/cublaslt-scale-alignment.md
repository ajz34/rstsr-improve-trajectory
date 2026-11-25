---
name: cublaslt-scale-alignment
 description: cuBLASLt loads 16-byte c64 alpha/beta with aligned SIMD loads; a Complex<f64> stack local is 16-aligned only by chance, so pass scales through #[repr(C, align(16))] storage (rstsr-cuda AlignedScale); crash is a host SIGSEGV inside libcublasLt, build-profile dependent
metadata:
  type: project
---

cuBLASLt reads a `C_64F` scale (`Complex<f64>`, 16 bytes, Rust alignment 8) with `movdqa`, a
16-byte aligned load (seen on cuBLAS 13.4, RTX 5070 Laptop, 2026-11-26). A plain stack local
`&alpha` is 16-aligned only by chance: a release build and one debug test passed, another debug
caller segfaulted on the host inside `cublasLtMatmul` (no status is returned).

- rstsr-cuda: all three cuBLAS drivers pass `alpha`/`beta` through `AlignedScale<T>`
  (`#[repr(C, align(16))]`, `matmul/drivers/mod.rs`); host test `scales_are_aligned_for_cublas`.
- Diagnosing such a crash: a standalone reproduction can pass in release and fail in debug; the
  cuBLASLt log (`CUBLASLT_LOG_LEVEL=5`) shows the descriptors are valid; `gdb -batch -ex run -ex
  "x/6i $rip"` gives the faulting instruction.
- Record: `2026-11-26-cuda-matmul-forms/B1-REPORT.md`.
