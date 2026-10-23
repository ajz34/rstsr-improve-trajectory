---
name: cuda-kernel-nvrtc-grill
description: 2026-10-23 kernel/dtype grill CLOSED (4 rounds, D1-D12): NVRTC PTX+runtime arch, CudaDType monomorphization-dispatch, u8-wise builder launch, 3-tier specialization ladder, ordinal-only same_device; PLAN B1-B7 awaits go
metadata:
  type: project
---

Grill closed 2026-10-23 (4 rounds;
`2026-10-21-cuda-decisions/02-kernel-nvrtc/`, DECISIONS.md D1–D12
canonical, PLAN.md has P1–P5 + B1–B7). Baseline rstsr `261021/cuda` @
`cdadd12`. Related: [[cuda-grill-decisions]], [[cuda-device-basic-grill-closed]]
(whose strict stream comparison this round revised).

Headline decisions:

- **Scope (D1)**: NVRTC infra + DEFER-path kernels (strided assign/cast,
  fill/full/ones, format gather, to_contig, arange/linspace) +
  elementwise f32/f64 kernels. Reductions/GEMM/RNG/solver/graphs/
  allocator out.
- **Launcher (D2)**: cudarc builder as pure mechanics, **u8-wise
  marshaling** (`&CudaSlice<u8>` pushes are trait-free; scalars as
  `[u8;N]`/u64/usize via built-in impls) — zero cudarc dtype-trait
  dependence, so future FP4/INT4 types need only our own trait impl
  (owner's readiness concern; orphan rule blocks cudarc-trait
  back-fill anyway). Raw `result::launch_kernel` = documented
  escalation only.
- **Compile (D3)**: PTX via safe layer; arch `compute_<cc>` derived at
  runtime from `compute_capability()` (NVRTC defaults compute_75);
  strict library defaults (`--std=c++17`, no fast-math, `-lineinfo`
  debug); user knobs later via `CudaRuntimeFlags` → raw option
  strings. cudarc quirk: `use_fast_math` field lowers to
  `--fmad=true` — avoid.
- **Cache (D4)**: per-ordinal in-memory `HashMap<KernelKey,
  Arc<CudaFunction>>` in `CudaOrdinalObj` + on-disk PTX cache day one
  (`RSTSR_CUDA_CACHE_DIR`, `RSTSR_CUDA_CACHE=off`).
- **DType (D5)**: local `trait CudaDType { const CNAME: &'static str }`,
  16 types (no i128/u128); kernel families generic over it;
  **monomorphization IS the dispatch** — no runtime dtype tag anywhere
  (owner's "frequent TypeId dispatch" concern resolved: zero per-call
  runtime dispatch). Missing bound = compile error (omit policy).
- **Assign casts (D6)**: keep `TA: DTypeCastAPI<TC>` bound; cast
  kernels per (TA,TC) pair lazily; extend rstsr-dtype-traits with
  half↔{f32,f64} (currently compiles on NO device) — dtype-traits PR
  lands FIRST; complex→real stays out; multi-dtype compute ops
  deferred by policy.
- **Coverage split (D7)**: layout-only kernels all 16 dtypes; math
  f32/f64 first, half via float-promotion wrappers, complex later
  (own prelude, no cuComplex).
- **Streams (D8, revises device-basic)**: `same_device` drops stream
  clause entirely — ordinal + flags-as-stored + order; ops event-wait
  foreign-stream input buffers; cross-ordinal stays explicit
  `change_device` copy; async ops, sync only at host reads.
- **Indexing (D9, owner-shaped)**: 3-tier ladder — T0 runtime-stride
  flat floor (always correct, rank-agnostic) / T1 inner-contiguous-run
  partially-strided (nonzero offset OK; owner explicitly wants this) /
  T2 full-contig; launch-time tier selection; no CuPy dim-squash in
  v1; efficiency effort concentrated on elementwise.
- **Delivery (D12)**: ALL work on same branch `261021/cuda`, **no PR
  until owner instructs**; crate-local copy-and-compare tests.

Facts discovered (agent fan-out, five reports): cudarc 0.19.10 has no
`CudaDevice` (contexts+streams), no module/PTX cache, PTX-only safe
NVRTC, no name-expression support, slice pushes need no `DeviceRepr`;
rstsr-core has NO runtime dtype registry (TypeId fast paths over
generic fallback is the whole idiom); `DTypeCastAPI` gaps = no
half↔numeric/complex→real/usize↔isize; device-basic B6 allocator
NEVER implemented (status log wrong); DEFER'd strided-assign path has
2 latent bugs (no-op layout_gather, absolute-index layout_scatter) —
fixed wholesale by kernels (D11).

Ride-along ledger: dtype-traits `defaults = ["half"]` misspelled key
(empty real defaults); `src/tmp.rs` untracked cargo-expand artifact
(owner to remove); half↔float cast PR sequencing.

Implementation awaits owner go (PLAN.md B1–B7, pre-flight P1–P5 first).
