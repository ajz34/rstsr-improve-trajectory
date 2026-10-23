# DECISIONS — `02-kernel-nvrtc`: kernel layer + dtype policy for rstsr-cuda

Canonical record of the kernel/dtype grill, closed 2026-10-23 after
four rounds (R1–R4; question/answer files in this directory). Baseline:
rstsr branch `261021/cuda` @ `cdadd12` (device-basic slice, D1–D8 of
../01-device-basic/DECISIONS.md). Owner directives carried in: cudarc
sole external CUDA dep; no cudarc trait adoption; multi-dtype compute
ops may be deferred but assignment-side casts wanted; unpublished crate
— reverts/breaking changes acceptable.

## D1 — Scope

This round implements: (a) NVRTC infrastructure (compile, cache,
launcher, source prelude); (b) kernels retiring all registered `//
DEFER:` naive paths — strided copy/cast assign, fill/full/ones, format
gather, non-contig `to_contig` — plus `arange`/`linspace` device
impls; (c) elementwise unary/binary/compare float kernels. Out: 
reductions, GEMM/BLAS, RNG, cuSOLVER, CUDA graphs, DLPack bridge, NCCL,
allocator framework (`CudaAlloc` — noted unimplemented from
device-basic; own round later).

## D2 — Launcher: cudarc builder as pure mechanics, u8-wise

Use cudarc's `stream.launch_builder(&func).arg(..)` — but with zero
dependence on cudarc's dtype-trait layer:

- Buffers push as `&CudaSlice<u8>` (the builder's slice push has **no**
  `DeviceRepr` bound; our storage is already byte-erased), keeping the
  automatic cross-stream event waits when cudarc tracking is on.
- Scalars (fill values, sizes, strides) push as `[u8; N]` byte arrays /
  `u64` / `usize` — covered by cudarc's built-in impls alone
  (`[T; N] where T: DeviceRepr`).
- Rationale: kernel args never carry the tensor dtype as a Rust type;
  future special dtypes (FP4/INT4/float8) then need only our own
  `CudaDType` impl, never a cudarc change (orphan rule also forbids
  us back-filling cudarc's traits for third-party types).
- Raw `driver::result::launch_kernel` with own `Vec<*mut c_void>`
  packing stays the documented escalation (needed for packed `extra`
  params / launch attributes / clusters, which are sys-only); unused
  in v1.

## D3 — Compile: PTX via safe layer, runtime arch, strict defaults

- PTX through `nvrtc::compile_ptx_with_opts`; driver-side JIT at
  `load_module` (driver keeps its own JIT cache). CUBIN (sys-only
  `nvrtcGetCUBIN`) only if measured load latency demands.
- `--gpu-architecture=compute_<major><minor>` derived at **runtime**
  from `CudaContext::compute_capability()` at first kernel use per
  ordinal (NVRTC defaults to `compute_75`; never omit). Arch spelling
  verified by pre-flight (P1). Not a compile-time choice; no
  hard-written per-machine values.
- Entry points plain `extern "C"` names (cudarc has no name-expression
  machinery; templated sources instantiate per dtype in generated
  source instead).
- Options: library-fixed strict defaults v1 — `--std=c++17`, no
  fast-math, `-lineinfo` in debug builds. User knobs arrive
  additively later through `CudaRuntimeFlags` (env-seeded global +
  per-device override + launch-time resolution, the pattern already
  shipped for `compute_type_32f`), mapping flags → raw NVRTC option
  strings. Avoid cudarc's `CompileOptions::use_fast_math` (lowers to
  `--fmad=true`, not `--use_fast_math`); pass raw strings.

## D4 — Two-tier kernel cache, day one

- In-memory: per-ordinal `Mutex<HashMap<KernelKey, Arc<CudaFunction>>>`
  in `CudaOrdinalObj` (slot reserved in its doc comment).
  `KernelKey = (family, dtype C-names, variant bits, arch,
  options-hash)`.
- On-disk PTX cache (CuPy lesson): key = hash(source + options + arch
  + nvrtc version + payload-format version); location
  `$XDG_CACHE_HOME`/`~/.cache/rstsr/cuda`; env knobs
  `RSTSR_CUDA_CACHE_DIR`, `RSTSR_CUDA_CACHE=off`.

## D5 — DType architecture: local `CudaDType`, monomorphization as
dispatch, no runtime tag

```rust
pub trait CudaDType: Clone + Send + Sync + 'static {
    const CNAME: &'static str; // CUDA C scalar name
}
```

- Impl'd for 16 scalars: bool, i8/i16/i32/i64/isize, u8/u16/u32/u64/
  usize, f32, f64, f16, bf16, c32, c64. i128/u128 excluded (no
  portable CUDA C type); they keep host round-trip paths.
- Kernel families generic over `CudaDType`; kernel source built from
  `T::CNAME`; cache keyed by the same string. **The Rust
  monomorphization IS the dtype dispatch** — no runtime dtype
  enum/tag anywhere; `TypeId` only where the house fast-path idiom
  already uses it (memcpy-class paths), never for kernel selection.
- Unsupported dtype ⇒ missing trait bound ⇒ compile error at call
  site (same "omit" policy as host-closure ops).
- This answers the owner's "frequent dispatched dtype" concern: zero
  per-call runtime dtype dispatch in the design.

## D6 — Assignment conversions

- CUDA `OpAssignAPI`/`OpAssignArbitaryAPI` keep bound
  `TA: DTypeCastAPI<TC>` (same table as CPU) + both ends `CudaDType`.
- Cast kernels generated per (TA, TC) pair lazily: C-cast template for
  numeric→numeric; special templates bool↔T (`x != 0`, `T(1)/T(0)`)
  and primitive→complex (`T(x, 0)`); c32↔c64. Ends outside
  `CudaDType` (i128/u128) → host round-trip fallback.
- Extend `rstsr-dtype-traits` `DTypeCastAPI` with half↔{f32,f64}
  (gated on `half`): today f16 cross-dtype assignment compiles on no
  device. **Sequencing: dtype-traits PR lands first**, kernels start
  on in-table pairs; approach validated empirically ("first try
  implement and see if this way works").
- Complex→real casts stay out (separate semantic discussion).
- Multi-dtype compute ops (e.g. mixed f32/f64 operands) remain
  deferred by policy — assignment casts are the round's dtype surface.

## D7 — v1 dtype coverage split

- Layout-only kernels (copy/fill/cast/gather/to_contig): all 16
  `CudaDType` members day one.
- Math kernels (unary/binary/compare): f32/f64 first; half next via
  float-promotion wrappers in kernel source; complex later with a
  minimal own complex prelude (no cuComplex reliance).
- Bool condition arrays for `where`/mask kernels from day one
  (tensor-layer indexing requires `DeviceAPI<bool>`).

## D8 — Streams: ordinal-only `same_device` + op-level event ordering

(revises device-basic D2's strict stream comparison, per owner
directive; unpublished crate, breaking change accepted)

- `same_device` compares **ordinal only** + flags-as-stored + order;
  the stream clause drops entirely. Any two tensors on the same GPU
  are same-device.
- Stream is an execution detail the op handles: launch on the input
  device value's execution stream; every input buffer whose allocation
  stream differs gets event record (on buffer's stream) → wait (on
  launch stream) via a small `result::event` helper — for ALL
  foreign-stream inputs, not just relabeled ones. Output allocates on
  the launch stream.
- Cross-ordinal remains an explicit `change_device`/`into_device` copy;
  never an implicit peer-copy fallback inside ops. `DeviceMismatch`
  names both devices.
- Ops enqueue-and-return (async); the only host sync points are host
  reads: `get_index`/`set_index`, `to_scalar`, `to_cpu_vec`,
  `Display` (format gather = one batched D2H + single sync),
  `memory_info`. Per-op `synchronize` from the interim paths is
  dropped.

## D9 — Kernel indexing: three-tier specialization ladder

(owner amendments in R4: both runtime and specialization wanted;
partially-strided explicitly in scope)

- **T0 generic**: runtime-stride flat kernel — shape/strides/offset/
  rank as runtime arrays, flat grid-stride loop decomposing linear →
  multi-index. Always compiled, any rank/layout/offset; the
  correctness floor. Rank-agnostic (IxD-ready; escapes rank
  monomorphization per doc01 R20; signed-stride math per R21).
- **T1 inner-contiguous run (partially strided)**: 2-level
  decomposition — outer grid-stride over the non-contiguous part,
  inner linear loop over the trailing contiguous run; nonzero offset
  as a kernel arg. Covers row/column slices, broadcast rows, padded
  rows. (House precedent: CPU serial contiguity-switch walk,
  `CONTIG_SWITCH` in rstsr-native-impl `op_with_func`.)
- **T2 fully contiguous**: C/F-order whole-buffer linear path;
  nonzero offset by pointer arithmetic at launch.
- Launch-time selection inspects layouts, picks the highest applicable
  tier. Dtype/rank remain runtime in v1 — no CuPy dim-squash/
  canonicalization machinery yet (future refinement if measurements
  justify; doc03 §4.6 gap (d) becomes "measure which specializations
  pay"). Efficiency effort concentrated on elementwise families;
  copy/fill/cast/gather get T0+T2 only.

## D10 — Module granularity + source layout

Rust string templates per kernel family + shared prelude
(`rstsr_cuda_common.cuh` as a const: index math helpers, dtype
aliases, order/flag plumbing). One NVRTC program per specialization
(family × dtypes × variant) loaded lazily — not one fat module per
family. Launch geometry (grid-stride, block ≈256) is family-internal.

## D11 — Correctness debt: fix-forward

The interim strided-assign path's defects (no-op `layout_gather`
ignoring source strides/offset; `layout_scatter` absolute-indexing a
`lc.size()` dst) are fixed wholesale by the kernel replacement with
new strided/offset-source tests vs CPU — no patching of interim code.

## D12 — Tests + delivery

- Crate-local tests (extend `test_device_basic.rs` / sibling
  `test_kernels.rs`): per kernel family, CUDA result → D2H → compare
  vs `DeviceCpuSerial` (copy-and-compare parity). Shared `core_func`
  wiring stays deferred (blocked on core R1 `get/set` migration);
  parity harness/CI tier is a later round.
- All work on the **same branch `261021/cuda`**; **no PR** until the
  owner instructs (owner directive, R2/R3).

## Pre-flight gates (P1–P5, run before rstsr-cuda edits)

1. NVRTC compile of prelude + one cast kernel via safe layer; arch
   spelling (`compute_120` vs `sm_120`) and `--std=c++17` accepted.
2. First-use latency: cold compile+load / warm / disk-cache hit.
3. Builder u8-marshal probe: `&CudaSlice<u8>`, `[u8; 16]`, u64,
   usize — no `DeviceRepr` beyond built-ins.
4. Two-stream event record/wait ordering probe on one ordinal.
5. Disk-cache layout probe: hash key, read/write, version
   invalidation.

## Ride-along side-ledger (recorded, not this grill's decisions)

- `rstsr-dtype-traits/Cargo.toml`: `defaults = ["half"]` misspelled
  key → real default features empty; fix rides with the half↔float
  cast PR.
- `rstsr-dtype-traits/src/tmp.rs`: untracked ~4830-line `cargo expand`
  artifact in the owner's tree — owner to remove (main-repo rule: ask
  before acting).
- Device-basic status log claims B1–B9 implemented; B6 (allocator)
  actually absent — recorded here so the record is honest; allocator
  round will own it.

## Deferred with intent (later rounds, rough order)

1. Reductions (M3) with `sum_with_dtype` f64-accumulator semantics.
2. GEMM via cuBLAS/Lt (M4) + `CudaRuntimeFlags` consumption of
   `compute_type_32f`; linalg tier (R10).
3. Half/complex math-kernel coverage beyond D7's first wave.
4. CUBIN output, `cuLaunchKernelEx` attributes if needed.
5. CuPy-style dim-squash/shape-class specialization (if measured to
   pay).
6. Allocator framework grill (`CudaAlloc` etc.).
7. Transfer round (events/pinned/async_h2d), DLPack bridge, NCCL,
   CUDA graphs.
8. Core `get/set` migration enabling shared `core_func` bodies;
   parity harness + CI tier round.

## Changelog of the debate

- R1: 10 questions; owner settled 8, redirected Q2 (cudarc traits →
  FP4/INT4 readiness concern) and Q3 (compile options → who
  controls). R2 re-proposals accepted.
- R2: async ops, test placement settled; owner: same branch, no PR.
  New directives: streams may differ input/output (reopen Q8),
  unpublished crate = breaking changes fine.
- R3: `same_device` ordinal-only ("you can try that"), pre-flight
  list, `CudaDType` name, completion record.
- R4: indexing ladder — owner wanted both runtime floor and
  specializations for elementwise, with partially-strided (contiguous
  axis + nonzero offset) explicitly in scope. Grill closed.
