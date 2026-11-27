---
name: cutensor-elementwise-facts
description: 2026-11-27 cuTENSOR 2.7.0 elementwise API facts - no unary entry, float/complex only, complex limited to identity/conj/add/mul, negative stride and zero extent rejected, narrow "one width step, same family" promotion (binary entry both directions, trinary narrowing only), C=nullptr invalid even at gamma=0, rank-0 rejected, descriptors carry no offset; plan setup ~50 us, execute ~2 us; perf parity with the hand kernel except both-transposed (cuTENSOR 1.8x), fused a*b+c (1.5x), L2-resident small tensors (hand 8x)
metadata:
  type: reference
---

Probed 2026-11-27 on the RTX 5070 Laptop (sm_120), cuTENSOR 2.7.0 pip build; sources, logs and the
report are in `2026-11-27-cuda-cutensor-elementwise/` (`probe/facts.cpp` groups g1-g4, `probe/bench.cpp`,
`bench/src/main.rs`, README tables).

## API shape

- Two entries, **no unary**: binary `D = opAC(α·opA(A), γ·opC(C))`, trinary
  `D = opABC(opAB(α·opA(A), β·opB(B)), γ·opC(C))`. `descC` must equal `descD`; the pointers may be
  equal (in-place) or different (out-of-place `D = a*b+c` verified).
- Unary = binary with `opC=IDENTITY, γ=0`, but **C must be a valid pointer even at γ=0**
  (`nullptr` → `INVALID_VALUE` at execute).
- General binary with two free layouts = trinary with `γ=0` (the binary entry pins C to D's layout).
- Rank-0 operand → `INVALID_VALUE` at execute; rank-1 extent-1 broadcasts fine (use it as a scalar).
- Scalars are host-typed by storage: R32F for f16/bf16/f32, R64F for f64, C32F/C64F for complex.
- **Descriptors have no offset**: windowed/sliced tensors pass `base + off*sizeof(T)`; a 16-byte
  misaligned base works at no measurable cost.

## Types / ops

- Storage: R16F, R16BF, R32F, R64F, C32F, C64F only. **No int/bool** (R32I → create INTERNAL_ERROR,
  R8U creates but plan NOT_SUPPORTED). Compute must equal the widest storage width.
- Real (f16/bf16/f32/f64): IDENTITY, NEG, ABS, EXP, SQRT, RCP, CEIL, SIN, ADD, MUL, MAX, MIN. CONJ
  is invalid for real.
- Complex (c32/c64): **only IDENTITY, CONJ, ADD, MUL**; NEG/ABS/EXP/SQRT/RCP/CEIL/SIN/MAX/MIN all
  `INVALID_VALUE`. Any complex math beyond +/-/* must stay in the hand kernel.

## Promotion (the answer to "does it promote?")

Small fixed subset, everything else is NOT_SUPPORTED: same family (real or complex), at most one
width step (f16↔f32, f32↔f64, c32↔c64), compute = the wider. f16↔f64, bf16↔f32, real↔complex all
fail. In the trinary entry `typeA == typeB` is required, and A/B→C/D only **narrows** (f64 into f32
OK, f32 into f64 NOT_SUPPORTED); the binary entry accepts both directions (so `d(f64)=a(f32)+c(f64)`
verifies). No integer promotion anywhere (i32→f32 and f32→i32 both fail).

## Views

- OK: contiguous, transposed input(s), window (offset + padded stride), stride-2 slice, broadcast by
  omitting a mode, broadcast by stride 0, transposed/windowed/step output, rank-6, in-place A=D.
- **Rejected at descriptor creation**: negative stride (flip) and any zero extent → INVALID_VALUE.
  Flips and empty tensors need a materialized copy.

## Timing (RTX 5070 Laptop, cuTENSOR vs rstsr hand kernel, preallocated, median of 7×50)

- Setup per distinct problem: ~44-72 us for descriptors+pref+estimate+plan (CreatePlan alone 11-16 us;
  the internal plan cache saves nothing). Execute call host-side ~2.0 us vs rstsr ~3.0 us.
- f64 2048²: add/mul/copy parity (±3%), exp rstsr 1.3x faster, A.T/B.T cuTENSOR 1.2x, both-transposed
  cuTENSOR 1.8x, step/window/bcast/out.T parity. Fused a*b+c cuTENSOR 1.5x (rstsr needs 2 kernels).
- f32: contiguous parity (rstsr ~8% faster), A.T cuTENSOR 1.3x, fused muladd cuTENSOR 1.7x.
  c64/c32: contiguous parity, A.T cuTENSOR 1.2x.
- Sizes: 100×100 cuTENSOR 1.2x faster (lower launch overhead); 1000×1001 the hand kernel is **8x**
  faster (24 MB working set stays in L2 at 2.6 TB/s while cuTENSOR runs at HBM ~330 GB/s); odd
  2047×2053 / 4093×4099 the hand kernel is 1.1-1.2x faster; 4096² parity.
- f16/bf16 have no hand-kernel baseline (`CudaArith` excludes half); cuTENSOR add f16 0.112 ms,
  bf16 0.030 ms for 2048².
