# DECISIONS — cuTENSOR elementwise vs the rstsr-cuda hand kernel

Decision log for this experiment. R1 answers arrived on screen (2026-11-27); transcribed here by
the agent as background (the user-owned `GRILL-R1-ANSWERS.md` is not created by the agent).

## D1 — "ternary" means the fused three-operand form, not `where`/`clip`

> user: "Well probably point 4. If some options that cuTENSOR cannot express, just not express that.
> We want to check the efficiency and functionality of cuTENSOR."

- The perf matrix's ternary cells are the **fused two-op form**
  `D = opABC(opAB(a,b), C)` with `C = D` (e.g. `c += a*b`, `min(a*b, c)`, `max(a+b, c)`), plus the
  out-of-place variant that needs the C operand pre-copied into D.
- `where` and `clip` are still in the **functionality** table: they are rstsr's 3-input ops, and the
  report states exactly why they are not expressible (`where`) or need two passes (`clip`). No
  emulation benchmark is run for them.
- The same "express it or state it is unsupported" rule applies to every other op family
  (comparisons, `pow`, `atan2`, `%`, bitwise, bool, in-place unary, ...).

## D2 — dtypes

> user: "Add f16/bf16 perf cells."

- Perf: `f32`, `f64`, `c32`, `c64` against the hand kernel; `f16`, `bf16` timed on the cuTENSOR
  side. The current hand kernel has no half arithmetic (`CudaArith` excludes `f16`/`bf16`; the
  dtype module calls it "a later wave"), so the half cells have no same-dtype baseline; they are
  reported against the hand kernel's `f32` cell and as a capability gap.
- Functionality: additionally `i32`, `i64`, `u8`, and `bool` (bool is not a cuTENSOR type at all).

## D3 — sizes

> user: "Add some non-2^n (odd number) cases."

- Power-of-two anchor sizes: `100x100`, `2048x2048`, `4096x4096`.
- Odd sizes: `1000x1001`, `2047x2053`, `4093x4099` (near 1e6, 4.2e6, 16.8e6 elements).
- The report calls out any size effect separately from the layout effect.

## D4 — investigation only, no adapter

> user: "Well cuTENSOR code can be C. This is actually and simply some investigation before real
> rstsr implementation, so know what is and what extent is good enough, not doing too further."

- The cuTENSOR side is a standalone C++ probe (as in `2026-11-27-cuda-einsum-cutensor`); the
  baseline is a small Rust bench crate in this directory that depends on rstsr by path.
- No rstsr source edit, no `DeviceCuda` adapter, no PR. The report ends with what an integration
  would have to do and which subset could pay off.

## D5 — scope of the view matrix

- `contig`, `B transposed`, `A transposed`, `sub-window` (offset, row stride > width), `step-sliced`
  (stride 2), `B broadcast`, `transposed output`.
- `negative stride` (flip) and `stride-0 output` are functionality cells only: the prior probe
  already showed cuTENSOR rejects the former and mis-writes the latter, and the hand kernel handles
  both; the point is to confirm for elementwise, not to time them.
- Correctness is checked (CPU reference) for every accepted cell before its timing is trusted.
