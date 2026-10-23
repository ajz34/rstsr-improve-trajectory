# GRILL R2 — `02-kernel-nvrtc`: launcher re-proposal + follow-ups

## Agent-authored background — R1 answers as received on screen (2026-10-23)

Transcribed from the owner's on-screen replies to GRILL-R1-QUESTIONS.md
(this file is the agent's record; the answers file remains user-owned
and untouched):

- **Q1 (scope): Okay.** — infra + DEFER retirement + elementwise float
  kernels + arange/linspace; reductions/GEMM/RNG/solver/graphs/allocator
  out.
- **Q2 (cudarc traits): clarified, not answered.** Owner: "Well I mean
  that we may or may not use special types like FP4/INT4 in future,
  which I'm not sure if cudarc implements. We may be prepared to those
  special types that cudarc does not implement. Based on this
  preparation, you may re-propose your thought on this question." →
  re-proposed below as R2-Q1.
- **Q3 (compile target/arch/options): clarified, partly answered.**
  Owner: "I want to make sure if this is hard-written on rstsr, or the
  user can change or the library will adapt at compile time. If these
  compile options/arch are for local development, okay." → factual
  resolution in background below; remaining policy split is R2-Q2.
- **Q4 (cache): Okay.** Both tiers from day one.
- **Q5 (CudaDType, no runtime tag): Okay and related to my Q2
  comments.** (I.e. the trait design stands, but must be consistent
  with the FP4/INT4-readiness concern.)
- **Q6 (assign casts): "Well I think you are going to first try
  implement and see if this way works."** — empirical validation
  approved; sequencing note below.
- **Q7–Q10: Okay.**

Settled by R1: scope (Q1), cache design (Q4), dtype architecture
(Q5: local `CudaDType`, 16 types, monomorphization-as-dispatch, no
runtime tag), assign-cast direction (Q6, validate empirically), v1
dtype split (Q7), event-wait cross-stream helper (Q8), fix-forward on
strided-assign defects (Q9), per-specialization lazy modules (Q10).

## New facts bearing on the re-proposal (agent-sourced)

- **Kernel args never carry the tensor dtype as a Rust type.** A kernel
  parameter list is: device pointers, lengths/strides (u64/usize), and
  occasionally one scalar *value* (e.g. `fill`). The element dtype
  lives in the compiled kernel source (`T::CNAME`), not in the arg
  types. So the dtype-trait question decouples from the launch question
  almost entirely — the only dtype-typed arg is a fill-style scalar,
  which can be marshaled as raw bytes (`to_ne_bytes`).
- **Orphan rule**: rstsr-cuda cannot `impl cudarc::DeviceRepr` for
  `half::f16` or any third-party type (foreign trait + foreign type).
  cudarc ships those impls behind its own cargo features (`f16`, and
  `f8`/`f4` for the float8/float4 crates) — our current cudarc dep does
  NOT enable them. Waiting on cudarc's trait coverage for every future
  special dtype is exactly the lock-in the owner flagged.
- **Builder mechanics available without any dtype trait**: cudarc's
  `LaunchArgs` pushes `&CudaSlice<T>` **with no `DeviceRepr` bound**
  (any T; our buffers are `CudaSlice<u8>` internally), and scalar
  pushes support `[T; N] where T: DeviceRepr` — so `[u8; 2/4/8/16]`
  byte arrays (f16 … c128 fill scalars) are covered by the built-in u8
  impls alone. Slice pushes also carry automatic cross-stream event
  waits when cudarc tracking is on (overlaps Q8's helper).
- **Raw path** (`driver::result::launch_kernel` + own
  `Vec<*mut c_void>`) remains the escalation if the builder ever blocks
  us (packed `extra` params, launch attributes are sys-only anyway).
- **Q3 fact**: arch is not compile-time anywhere in the design — it is
  read at runtime from the device (`compute_capability()` at first
  kernel use per ordinal) and enters the compile options and the cache
  key; a process touching two GPUs compiles/loads per-arch entries
  automatically. What IS library-chosen today: the option set itself
  (`--std=c++17`, strict numerics, `-lineinfo` in debug).
- **cudarc quirk to avoid**: `CompileOptions::use_fast_math` lowers to
  `--fmad=true`, not `--use_fast_math`; when fast-math is ever wired,
  pass raw option strings instead of that field.
- **Ride-along fixes noticed** (not this grill's decisions, recorded for
  the implementation PR): `rstsr-dtype-traits/Cargo.toml` has
  `defaults = ["half"]` — misspelled key, real default features are
  empty; and `rstsr-dtype-traits/src/tmp.rs` is an untracked ~4830-line
  `cargo expand` artifact in the owner's tree (main-repo rule: left for
  the owner to remove).
- **Sequencing (Q6)**: half↔{f32,f64} cross-cast kernels require the
  `DTypeCastAPI` additions in `rstsr-dtype-traits` first — that lands as
  its own small reviewed PR; kernel work starts with pairs already in
  the table.

---

## Questions

### R2-Q1 — Launcher approach (re-proposal for R1-Q2)

Use cudarc's launch **builder as pure mechanics**, with everything
marshaled u8-wise: buffers pushed as `&CudaSlice<u8>` (no dtype trait
involved), scalars pushed as `[u8; N]` byte arrays / `u64` / `usize`
(built-in impls only). Zero cudarc dtype-trait dependence on our path —
future FP4/INT4/eight-float types need only a `CudaDType` impl of ours
(CNAME + size), never a cudarc change. Automatic slice event-waits kept.
Escalation path to raw `result::launch_kernel` documented but unused in
v1.

➡️ Recommend yes — this is the preparation the owner asked for: special
dtypes are our trait's problem alone, while we still don't rebuild the
launch wheel.

### R2-Q2 — Compile options policy (resolution of R1-Q3)

Arch: runtime-derived per device (already the design; nothing
hard-written, nothing user-facing). Options: v1 hardcodes strict
library defaults inside rstsr-cuda; user control arrives **additively
later** through `CudaRuntimeFlags` (the env-seeded global + per-device
override + launch-time resolution pattern already shipped for
`compute_type_32f`), mapping flags → raw NVRTC option strings. No user
knob ships in this round.

➡️ Recommend yes (library defaults now, additive flags later).

### R2-Q3 — Op asynchrony after kernels

Once kernels replace the DEFER paths, drop the per-operation
`synchronize` that today's interim code performs after every copy/fill
(assign fast path, fill_naive, h2d_into…). Ops become enqueue-and-return
on the device's execution stream; host reads remain the only sync
points (`get_index`/`set_index`/`to_scalar`/`to_cpu_vec`/`Display`/
`memory_info`), matching the model already documented in the crate
(lib.rs) and the cuda-grill docs. The format gather becomes one batched
D2H + single sync.

➡️ Recommend yes (async ops, sync at host reads only).

### R2-Q4 — Kernel test placement this round

Crate-local tests only (extend `test_device_basic.rs` or a sibling
`test_kernels.rs`): per kernel family, generate CUDA result → D2H →
compare against `DeviceCpuSerial` reference (the copy-and-compare
parity shape). Shared `core_func` body wiring stays deferred (blocked
on the R1 `get/set` migration per rules/testing.md); a later round
decides the parity harness and CI tier.

➡️ Recommend yes.

### R2-Q5 — Branch / PR logistics

Current state: device-basic sits un-PR'd on `261021/cuda` @ `cdadd12`.
Plan: open **PR1 (device-basic) now** for review; this round's work
goes on a **new branch stacked on `261021/cuda`** (e.g.
`261023/cuda-kernels`), its own PR after PR1 merges (or rebased,
maintainer's call at review time). Alternative: keep growing
`261021/cuda` into one larger CUDA PR.

➡️ Recommend the split (PR1 now, stacked branch for kernels) — review
gates stay small; the plan's PR2 (facade wiring) follows independently.
