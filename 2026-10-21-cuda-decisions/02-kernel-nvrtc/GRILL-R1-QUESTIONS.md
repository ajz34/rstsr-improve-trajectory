# GRILL R1 — `02-kernel-nvrtc`: kernel layer + dtype policy

Round 1 of the kernel/dtype grill. Facts below are agent-sourced
(2026-10-23, five read-only reports: cudarc 0.19.10 census; rstsr-core
dispatch idioms; rstsr-cuda crate inventory; rstsr-dtype-traits inventory;
cuda-grill doc digest). Answers to `GRILL-R1-ANSWERS.md` (user-owned).

## Agent-authored background (facts the questions lean on)

**cudarc 0.19.10 surface** (checkout `ec1cac7`):

- NVRTC safe layer: `compile_ptx_with_opts(src, CompileOptions)` → PTX
  only (CUBIN is sys-only). `CompileOptions` covers `ftz/prec_sqrt/
  prec_div/fmad/use_fast_math/maxrregcount/include_paths/options/arch`,
  but `arch: Option<&'static str>` passes verbatim (no derivation from
  `compute_capability()`), and `use_fast_math: Some(true)` lowers to
  `--fmad=true` (quirk). No name-expression/lowered-name support outside
  `sys`; compile log only on error.
- Modules: `CudaContext::load_module(Ptx) -> Arc<CudaModule>` (cfg
  `nvrtc`), `load_function(name) -> CudaFunction`. `cuModuleLoadData`
  only (driver-side PTX JIT, no JIT options, no module cache anywhere in
  cudarc). `CudaModule` is per-context.
- Launch: safe builder `stream.launch_builder(&func).arg(..).launch(cfg)`
  — args pushed as `&T: DeviceRepr` scalars or `&CudaSlice/&CudaView`
  (slice pushes carry automatic cross-stream event waits when cudarc's
  tracking is on); NO tuple launch; no way to push a raw packed buffer
  (`args` is `pub(super)`). Raw path: `driver::result::launch_kernel(func,
  grid, block, smem, stream, &mut [*mut c_void])` — plain pointer array;
  the `extra` packed-buffer mechanism is hardcoded NULL (sys-level
  `cuLaunchKernelEx` exists). `DeviceRepr`/`ValidAsZeroBits` are public,
  unsealed, method-less — external `unsafe impl` is the sanctioned
  extension point, and for plain C scalars (f32/f64/i64/u64/usize/bool/
  half/…) the impls already exist.
- f16/bf16: cargo feature `f16` on cudarc gives `DeviceRepr`/"__half"
  mapping; header paths + arch are the caller's job.

**rstsr-core house idiom** (branch `261021/cuda` @ `cdadd12`):

- No runtime dtype registry exists anywhere in core/dtype-traits. Fast
  paths = `TypeId`-guard early returns re-typing raw buffers, with the
  fully generic `T` impl as the always-present fallback
  (`impl_uninit_dispatch!` ×7 BLAS/faer sites; `for_each_fast_dtype!`
  17-dtype list in `device_cpu_serial/set.rs`).
- `DTypeCastAPI<T> { fn into_cast(self) -> T }` (rstsr-dtype-traits
  promotion.rs): fixed compile-time table — 10×10 primitive `as`-matrix,
  bool↔14, primitive→complex, c32↔c64. Gaps: **no half↔numeric, no
  complex→real, no usize↔isize, no bool↔i128/u128**.
- Kernel contract: device ops write into `&mut [MaybeUninit<TC>]` and
  must write EVERY element before `assume_init_impl`; no zero-fill
  reliance. Flattened visit order must follow `self.default_order()`
  (reshape.rs flips it explicitly).
- CUDA today: no launch/module/cache code at all (greenfield);
  `DeviceAPI<T>` bound is just `Clone+Send+Sync+'static`; byte-erased
  storage (`CudaSlice<u8>`) already covers all dtypes; only dtype-list
  code is per-dtype formatting impls.

**rstsr-cuda state** (crate @ `cdadd12`):

- 14 `// DEFER:` sites: strided assign/assign_arbitary (D2H→CPU→H2D),
  fill/full/ones (host gen + blocking H2D, geometric memcpy doubling),
  format gather (per-element D2H+sync), h2d_into, d2h/h2d_scalar.
  `arange`/`linspace` traits not implemented at all yet.
- Latent defects in the DEFER'd strided-assign interim path:
  `layout_gather` (assignment.rs:139) ignores source strides/offset
  (returns raw buffer), and `layout_scatter` indexes a `lc.size()`-long
  dst with absolute `offset + Σ idx·stride` — strided/offset sources
  silently copy raw order or trip a length assert. Reachable via
  `change_layout_f`, `reshape(copy)`, `tile`, `repeat`, `roll`, …
- **B6 (allocator) is NOT implemented** despite the 01-device-basic
  status log saying "B1–B9 implemented": no `CudaAlloc`, no
  `empty_cache`, no OOM-retry, no `RSTSR_CUDA_ALLOC_CONF`. Allocation is
  bare `stream.alloc::<u8>()` over cudarc `malloc_async`. Plan text, not
  code, is the intent record.
- Two streams per op: buffers carry allocation stream (copies/frees),
  device value carries execution stream (future launches) — D4 split.
  Same-ordinal `change_device` relabel keeps the buffer on its old
  stream while the device value points at the new one (cross-stream
  hazard for kernels).
- Flags: `CudaRuntimeFlags { compute_type_32f }` exists with env seed +
  launch-time resolution; nothing NVRTC-related (`fast_math`, ftz) yet.

**Doc-set positions** (2026-10-21-cuda-grill): NVRTC endorsed, cudarc
sole dep, gaps fillable via result/sys; eager-async settled; strict FP32
default; f32/f64 first, half next; doc01 §5 recommends local scalar
bound + dispatch inside the kernel launcher; module granularity and the
v1 dtype tag question deliberately left open (this grill).

**Side-finding** (rstsr-dtype-traits): `Cargo.toml` has `defaults =
["half"]` — misspelled key, so the crate's real default features are
empty (all consumers compensate by declaring `features = ["half"]`).
Also `src/tmp.rs` is an untracked 4830-line `cargo expand` artifact that
pollutes greps.

---

## Questions

### Q1 — Round scope

Does this round's implementation slice = (a) NVRTC infra (compile,
cache, launcher, source prelude), (b) the kernels that retire the
registered DEFER paths (strided copy/cast assign, fill/full/ones,
format gather, to_contig), plus `arange`/`linspace` device impls,
(c) elementwise unary/binary/compare float kernels (doc03 M2), while
reductions (M3), GEMM/BLAS (M4), RNG (M5), cuSOLVER/graphs (M6) are
later rounds, and the allocator framework (B6, currently unimplemented)
stays out of this grill?

➡️ Recommend yes: infra + DEFER retirement + elementwise float kernels;
the rest later. Allocator gets its own round (owner noted needing the
CPU use-case first).

### Q2 — What "no cudarc traits" covers

The directive "we are not going to use cudarc's provided trait": which
reading?

- (a) No cudarc **dtype/type-mapping** machinery (CudaTypeName, any
  device-scalar abstraction) — our own dtype trait — but the **launch
  builder API** (`PushKernelArg`/`DeviceRepr` as the mechanics of
  `.arg()`) is used as-is. Cost: none extra; plain C scalars already
  have `DeviceRepr`; pushing `&CudaSlice` needs no trait of ours.
  Benefit: automatic cross-stream event waits from cudarc's tracking.
- (b) No cudarc trait **types** on our path at all: raw
  `driver::result::launch_kernel` with our own `Vec<*mut c_void>`
  packing. Cost: we own cross-stream ordering (manual event waits via
  `result::event`), a bit more unsafe code. Benefit: zero dependence on
  cudarc's arg trait layer ever.

➡️ Recommend (a): the builder is cudarc's API surface, not a dtype
design; "our own dtype traits" is honored by Q5. Fall back to (b) only
if you meant the builder itself.

### Q3 — Compile target, arch, options

Compile to **PTX** through the safe layer with `--gpu-architecture=
compute_<major><minor>` derived from `compute_capability()` (static
table for common caps, leak-or-const for the rest), accepting
driver-side JIT at module load (the driver keeps its own JIT cache).
Kernel entry points are plain `extern "C"` names (no name-expression
machinery). Compile options are fixed strict defaults v1
(`--std=c++++17`, no fast-math; `-lineinfo` in debug builds);
`CudaRuntimeFlags` consumption (ftz/prec-div/…) stays additive-later.
CUBIN via sys (`nvrtcGetCUBIN`) only if measured load latency demands
it. Validate the arch string spelling on the local 13.2 toolkit in a
pre-flight step.

➡️ Recommend yes.

### Q4 — Kernel cache

Per-ordinal in-memory cache `Mutex<HashMap<KernelKey, Arc<CudaFunction>>>`
living in `CudaOrdinalObj` (its doc comment already reserves "kernel
cache"), `KernelKey = (family, dtype C-names, variant bits, arch,
options-hash)`. Plus an **on-disk PTX cache** from day one (CuPy
lesson): key = hash(source + options + arch + nvrtc version + payload
format version), location `$XDG_CACHE_HOME/rstsr/cuda` (default
`~/.cache/rstsr/cuda`), env overrides `RSTSR_CUDA_CACHE_DIR` /
`RSTSR_CUDA_CACHE=off`.

➡️ Recommend yes to both; disk cache is a file write keyed by a hash,
not worth deferring.

### Q5 — DType architecture: local trait, no runtime tag

One local trait in rstsr-cuda, proposed `CudaDType` (name open):

```rust
pub trait CudaDType: Clone + Send + Sync + 'static {
    const CNAME: &'static str;   // CUDA C scalar name
    // byte size derivable via size_of; zero/one only if ever needed
}
```

implemented for the 16 storable scalars (bool, i8..i64/isize, u8..u64/
usize, f32, f64, f16, bf16, c32, c64; i128/u128 excluded — no portable
CUDA C type). Kernel families are **generic over `CudaDType`**; the Rust
monomorphization IS the dtype dispatch (kernel source built from
`T::CNAME`, cache keyed by the same string) — **no runtime dtype enum or
tag is introduced**, and `TypeId` chains appear only where the house
fast-path idiom already uses them (memcpy-class paths), never for
kernel selection. Unsupported dtype ⇒ missing bound ⇒ compile error at
the call site (the same "omit" policy as R12 host-closure ops).

➡️ Recommend yes. This answers your "frequently dispatched dtype"
doubt: with a bound-driven design there is no per-call runtime dtype
dispatch at all — dispatch is monomorphization + a hash-map lookup by
string key.

### Q6 — Assignment conversions (the `assign_arbitary` dtype question)

Keep the public CUDA bound `TA: DTypeCastAPI<TC>` (same as CPU) but add
both-ends-`CudaDType`; the cast kernel is generated per (TA, TC) pair on
first use (lazy compile + cache): plain C-cast template for
numeric→numeric, special templates for bool↔T (`x != 0` / `T(1)/T(0)`)
and primitive→complex (`T(x, 0)`), c32↔c64. Pairs outside `CudaDType`
(i128/u128 ends) fall back to the host round-trip path rather than
erroring. And: extend `rstsr-dtype-traits`' `DTypeCastAPI` table with
**half↔{f32,f64}** (the gap that makes f16 assignment not compile on ANY
device today; gated on the `half` feature); complex→real casts stay out
(separate semantic discussion, numpy-divergence territory).

➡️ Recommend yes: kernels for the in-table pairs + the half↔float
addition to promotion.rs. Note this touches `rstsr-dtype-traits`, i.e.
a core-workspace PR rides along with the CUDA branch.

### Q7 — v1 kernel dtype coverage split

Copy/fill/cast/gather kernels (layout-only, no math): **all 16
`CudaDType` members day one** (they are dtype-templated, near-zero
marginal design cost). Math kernels (unary transcendental/binary
arithmetic/compare): **f32/f64 first**; half next via float-promotion
wrappers in kernel source (`__half2float` round-trips); complex later
with a minimal complex-ops prelude (NVRTC has no bundled cuComplex we
can rely on; CuPy ships its own header). `where`/mask kernels take bool
conditions from day one (tensor layer requires `DeviceAPI<bool>` for
indexing).

➡️ Recommend yes.

### Q8 — Cross-stream launch discipline

Kernels launch on the **device value's execution stream** (D4). At op
entry, any input buffer whose allocation stream differs gets an
explicit ordering: record event on the buffer's stream, `stream_wait_-
event` on the launch stream (small helper over `result::event`; this
is what cudarc's builder would do automatically under reading (a) of
Q2). The alternative — error out on foreign-stream buffers — is
simpler but breaks same-ordinal relabeled tensors, which the device-
basic round deliberately allows.

➡️ Recommend the event-wait helper (~15 lines) over the error rule.

### Q9 — Fix-forward for the strided-assign defects

The interim strided-assign path's two defects (no-op `layout_gather`,
absolute-index `layout_scatter`) get fixed **wholesale by the kernel
replacement** (Q1b), with crate tests added for strided/offset sources
vs CPU reference — no patching of the interim code on the way by.

➡️ Recommend yes.

### Q10 — Module granularity + source layout

Kernel sources are Rust string templates per family (shared prelude
header `rstsr_cuda_common.cuh` as a const: index math helpers, dtype
aliases, order/flag plumbing), one **NVRTC program per specialization**
(family × dtypes × variant) loaded lazily — not one fat module per
family (long compiles, unused dtype instantiations). Launch geometry
(grid-stride 1-D, block ≈256) is a family-internal implementation
detail, not policy.

➡️ Recommend yes (per-specialization modules, lazy).
