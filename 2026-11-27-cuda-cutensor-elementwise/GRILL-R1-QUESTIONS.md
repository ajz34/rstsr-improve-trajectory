# R1 — cuTENSOR elementwise vs the rstsr-cuda hand kernel

2026-11-27. Scope round: no benchmark has run yet. This file records the fact base I built before
asking, the proposed matrix, and the open decisions. Everything here is a proposal; please correct
freely.

## Environment (fixed for this experiment)

| item | value |
|---|---|
| GPU | NVIDIA RTX 5070 Laptop, 8 GiB, sm_120; driver 595.99.02 |
| CUDA / nvcc | 13.2 / 13.2.78 |
| cuTENSOR | pip `nvidia-cutensor` 2.7.0 (CUDA-13 build) under the `torch-gpu4pyscf` conda env; not system-wide. The wheel ships only `libcutensor.so.2`, so cudarc's `fallback-dynamic-loading` needs a `libcutensor.so` symlink in this directory |
| rstsr | working tree `261127/cuda-prelude-f-convention` @ `937ce04` (clean) |
| hand kernel | `crates-device/rstsr-cuda/src/kernels/elementwise.rs` — NVRTC-generated CUDA C, one program per (dtype tuple, expression); `_f` flat entry, `_p` 64-bit magic-divmod entry, `_pn` 32-bit entry, rank-generic fallback; 4 elements/thread; compiled on first use, cached in memory and on disk |

## The cuTENSOR elementwise API (read from the 2.7.0 header)

There is **no unary entry point**. Two entry points only:

- **binary**: `D = opAC( α·opA(A), γ·opC(C) )` with **`descC == descD`** — C is both the second
  operand and the destination; C and D may alias.
- **trinary**: `D = opABC( opAB( α·opA(A), β·opB(B) ), γ·opC(C) )` with **`descC == descD`**.
- `A` and `B` must not overlap `D`; a zero scalar means that tensor is not read.

What this means for the rstsr op families:

| rstsr op | cuTENSOR mapping |
|---|---|
| unary `d = f(a)` | binary, `γ = 0`, A free |
| binary `d = a + b` with two arbitrary layouts | **trinary**, `γ = 0` (the binary form would force B into the output layout) |
| in-place accumulate `c = c + a` / `c = a*b + c` | natural: C = D = c (trinary with `opAB = MUL`, `opABC = ADD`) |
| out-of-place `d = a*b + c` | needs c pre-copied into d first (one extra pass) unless c's buffer is the destination |
| `where(cond, x, y)` | **not expressible** — no select/compare operator, and no bool storage type |
| `clip(a, lo, hi)` | **two passes** (one bound per pass); no single-call form |
| `a + 2.0`, `clip(a, lo, hi)` with scalar bounds | no scalar operand — needs a rank-0 device tensor (or an extra pass) |
| in-place unary `a = f(a)` | **not expressible** (A may not overlap D); needs a copy |

Operator set: 27 unary (`identity, sqrt, relu, conj, rcp, sigmoid, tanh, exp, log, abs, neg, sin,
cos, tan, sinh, cosh, asin, acos, atan, asinh, acosh, atanh, ceil, floor, mish, swish, soft_plus,
soft_sign`) and 4 binary (`add, mul, max, min`). No select, no comparisons, no `pow`, `atan2`,
`remainder`, `hypot`, `copysign`, `logaddexp`, no bitwise/integer arithmetic, no bool.

## dtype rules (documented; the empirical matrix is part of this experiment)

The header's supported-combination tables, given as `typeA | typeB(trinary only) | typeC(=typeD) |
descCompute`:

| typeA | typeB | typeC/D | compute |
|---|---|---|---|
| R16F | R16F | R16F | 16F or 32F |
| R16BF | R16BF | R16BF | 16BF or 32F |
| R32F | R32F | R32F | 32F |
| R64F | R64F | R64F | 64F |
| C32F | C32F | C32F | 32F |
| C64F | C64F | C64F | 64F |
| R32F | R32F | **R16F** | 32F |
| R64F | R64F | **R32F** | 64F |
| C64F | C64F | **C32F** | 64F |

Reading: every row has `typeA == typeB`; `typeC` may be **narrower** than the inputs, never wider;
the compute type is the input width. So there is no promotion in the rstsr sense
(`f32 + f64 -> f64`, `i32 + f32 -> f64`, scalar promotion) — the only flexibility is
*compute-wide, store-narrow*. Integer and bool types are absent from the table.

The probe will confirm each cell by status code and, where accepted, by value (not just
acceptance): documented tables in cuTENSOR have been wrong before (stride-0 output, `opC = CONJ`).

## Views (prior probe, same repo `2026-11-27-cuda-einsum-cutensor`)

- arbitrary positive strides, offset views, omitted modes (broadcast): accepted and correct;
- stride 0 on an **output**: accepted but garbage (write race) — must be refused/copied;
- negative strides: rejected at descriptor creation (`INVALID_VALUE`) — flips need a copy;
- `alignmentRequirement` is not checked, so offset views work;
- plan creation: one-time ~28 ms lazy init when the cuBLAS path is first touched, then 20-50 µs
  per new plan for contraction-shaped problems; elementwise plan cost is unmeasured.

So transposed / sub-window / step-sliced map directly to modes+strides; `flip` does not.

## Proposed experiment (defaults, to be adjusted by the answers)

1. **Facts probe** (this directory, Rust + cudarc `cutensor` sys, or C++ if simpler): the dtype
   matrix above, unary-via-binary, trinary-via-γ=0, rank-0 scalar tensors, null C pointer with
   γ=0, negative/zero strides, plan cost with and without JIT, and value checks for every accepted
   cell.
2. **Bench crate** (Rust, path dep on `rstsr` + `rstsr-cuda`, `cudarc` with `cutensor`); both paths
   share the same `CudaBuffer`s and `Layout`s:
   - *kernel-only*: preallocated output, `DeviceCuda` op-trait call vs `cutensorElementwise*Execute`;
   - *full op*: `rt::add` / `rt::where` vs a thin cuTENSOR adapter that allocates its output;
   - *overheads*: descriptor + plan creation (cold), plan-cache hit, host-side execute cost, stream
     ordering (`order_input`/`order_output` events the hand kernel pays per launch);
   - correctness vs the hand kernel before timing; median of repeated CUDA-event samples.
3. **Report**: per-cell table, bandwidth as % of a device copy baseline, verdict per family, and a
   short "could this be integrated" section (plan cache keyed by dtype + mode set + shape +
   strides; JIT first-call cost; unsupported ops).

Default matrix: ops `{neg, exp, sqrt, sin, abs, conj(c64)}` unary, `{add, mul, sub, div, max, min}`
binary, `{where, clip, c += a*b}` ternary; layouts `{contig, B transposed, both transposed,
sub-window, step-sliced, B broadcast, flip (facts only), transposed output}`; sizes `{1e4, 2048²,
4096²}` elements (2-D arena for the view cases, one 3-D swapaxes cell); dtypes `{f32, f64, c32, c64}`
for perf, `{f16, bf16, i32, i64, u8, bool}` for facts only.

## R1 questions

**Q1 — What does "ternary" mean here, and how hard should we chase it?**
I read it as rstsr's 3-input family (`where`, `clip`; `op_mutd_*`). cuTENSOR can express neither in
one call. Options:
- **(a) [proposed]** Document `where` as unexpressible; benchmark `clip` as the 2-pass best effort
  against the 1-pass kernel; skip a `where` emulation.
- (b) Also benchmark a float `where` emulation (`y + c*(x-y)`, 2-3 passes; semantics differ for
  NaN/bool) to show the size of the gap.
- (c) Drop ternary from the perf matrix; report only the expressibility analysis.
- If you meant something else by "ternary" (e.g. the fused `a*b+c` form rather than `where`/`clip`),
  say so — the matrix changes.

**Q2 — dtype coverage for the perf matrix.**
- **(a) [proposed]** `f32, f64, c32, c64` perf; `f16, bf16, i32, i64, u8, bool` in the facts probe
  only (they are absent from cuTENSOR's documented table, so they are support questions, not perf
  cells).
- (b) Add `f16/bf16` perf cells.
- (c) `f64/c64` only (the electronic-structure dtypes).

**Q3 — size regime.** Plan/launch overhead dominates below ~1e6 elements and vanishes above ~1e7.
- **(a) [proposed]** Sweep `1e4 / 2048² / 4096²`; report the crossover.
- (b) Large only (≥ 1e6): "which is faster in steady state".
- (c) Custom — give me the shapes/sizes you actually care about (e.g. grid tensors, AO tensors),
  and I will center the matrix there.

**Q4 — integration depth.**
- **(a) [proposed]** Device-level comparison + plan-cache/overhead model; no rstsr source edits.
  Answers "is the cuTENSOR kernel faster" and "what would an integration pay".
- (b) Also write a trial `DeviceCuda` elementwise adapter (descriptor + plan cache) and measure it
  through `rt::add`/`rt::clip`; answers "would rstsr actually get faster", more work.
- (c) (a) now, (b) only if (a) shows a win.

## Notes

- All code, logs and reports stay in this directory; `rstsr` is not edited.
- Answers: please edit `GRILL-R1-ANSWERS.md` (I will not create or touch it), or answer on screen.
