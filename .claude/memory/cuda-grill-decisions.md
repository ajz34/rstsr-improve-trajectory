---
name: cuda-grill-decisions
description: Early maintainer decisions for the 2026-10-21 CUDA backend grill (cudarc as sole dependency; doc set structure)
metadata:
  type: project
---

Decisions stated by the maintainer on 2026-10-21 (session start of the
`2026-10-21-cuda-grill` task), before the comprehension docs were written:

- **cudarc is the preferred sole external CUDA dependency** for a future
  rstsr CUDA backend, "if possible". candle, dfdx, oxicuda, cubecl,
  CUTLASS are references/precedents only, not dependencies. Anything
  missing from cudarc would go through the project's own FFI repo pattern
  (rstsr-ffi) rather than pulling another crate.
- Max 5 subagents in this session's fan-outs.
- Doc set being produced in `2026-10-21-cuda-grill/`:
  01 rstsr backend architecture, 02 cupy architecture vs numpy,
  03 rust GPU ecosystem (cudarc-centric), 04 dfmp2 workload gap analysis,
  plus a README index/synthesis written by the lead agent.

Doc set COMPLETED 2026-10-21 (01: 656 lines, 02: 448, 03: 248, 04: 417,
README synthesis). Headline results:
- **Raw verdict YES**: `DeviceRawAPI::Raw` is an associated type
  (rstsr-core/src/storage/device.rs:9-11); rstsr-cuda can set
  `Raw = CudaBuffer<T>` and reuse core tensor types wholesale.
- **Four real restrictions**: R1/R2 host-slice element access
  (iter/Index gated on `Raw = Vec<T>`; blocks test parity, 8/85 files);
  R10 `BlasDriverBaseAPI` pins `Raw = Vec<T>` (whole BLAS/LAPACK driver
  tier unimplementable for CUDA); R12/R13 host-closure ops
  (map/custom-reduce); R15 fancy indexing is O(n) D2H+H2D round-trip.
- **Eager is enough**: cupy is eager-host/async-device, no lazy engine;
  only hazard is stream-ordered Drop into a pool (arena-per-stream).
- **NVRTC lean endorsed** (doc 03): flexibility + debugging win, PTX on
  par, GEMM ceiling is cuBLAS/Lt anyway; cost is engineering (kernel
  generation, typed launchers, disk cache, pool allocator). cudarc-only
  viable; Lt safe layer hard-wires FAST_TF32 → use result layer for
  strict-f32.
- **TF32**: per-call compute type via cublasLt, `Arc<CudaDeviceState>` +
  `AtomicU8 math_mode` + RAII guard; NOT in same_device; cupy's env-var
  global is the anti-pattern.
- **dfmp2 port**: computational surface nearly complete (sum_with_dtype
  = torch fold, PR #110; zero-copy GEMM reshape trick; slice_mut+assign
  assembly); API blockers: astype (G-007), real change_device (Display
  needs it), stream/sync, memory_info, math-mode.
Next step proposed: grill rounds on the 12 topics in README §4.

Machine facts relevant to CUDA experiments (verified 2026-10-21):
RTX 5070 Laptop GPU, 8 GB VRAM, compute capability 12.0 (Blackwell),
driver 595.91.07, CUDA toolkit 13.2 at /usr/local/cuda. Local cudarc
checkout (~/Git-Others/cudarc) is version 0.19.10 @ ec1cac7 (supports
CUDA 13.0-13.4, sm_120 via cuda-13020 feature). NVRTC defaults to
compute_75 — always pass --gpu-architecture at runtime. rstsr master at
task start: 1c3bc11. See [[rstsr-bench-context]] for the CPU side.
