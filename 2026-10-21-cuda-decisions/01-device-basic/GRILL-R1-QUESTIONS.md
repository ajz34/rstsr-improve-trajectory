# GRILL R1 — device and data representation for `rstsr-cuda`

Scope of this round: the initialization decisions listed in `GRILL-INIT-PROMPT.md` —
the `DeviceCuda` object, the `DeviceRawAPI::Raw` buffer type, the split of state
between them, flags/shared state, streams placement, and the Rust equivalent of
Python's `with`. Kernel strategy (NVRTC), the BLAS/linalg tier, the dtype set,
and the test plan are deliberately out of scope (later rounds; grill topics 4,
6, 8, 10-11 of `2026-10-21-cuda-grill/README.md` §4).

Not re-challenged here (already settled in the grill docs, per instruction):
cudarc as sole external CUDA dependency; NVRTC over cubecl; eager-async
execution; the necessity of *some* caching/stream-ordered allocation; dynamic
TF32 (not env var); crate placement `crates-device/rstsr-cuda` + `rt::cuda::`
member-crate wiring; row-major default.

Background gathered 2026-10-21 by four read-only extraction agents (reports in
this session's transcript; rstsr @ `1c3bc11`, cudarc 0.19.10 @ `ec1cac7`,
candle 0.11.0, dfdx 0.13.0, oxicuda 0.1.3, pytorch shallow clone, cupy @
`8f7a35d`). Citations below use the agents' verified `file:line` forms.

---

## 0. New facts that correct the grill-doc baseline

These came out of the extraction and change (or sharpen) what docs 01/03/04
assume. Each is verified against source; challenge any you doubt.

- **F1 — `Raw: Clone` is value semantics, so `CudaBuffer::clone` must be a deep
  D2D copy.** `Tensor::clone` → `to_owned()` → `view().into_owned()` →
  `DataCloneAPI for DataRef` → `C::clone` (`tensor/ownership_conversion.rs:356-366`,
  `:378-383`; `storage/data.rs:373-392`). CPU `Tensor::clone` deep-copies the
  `Vec` the same way. Doc 04 §3's recommendation "make Raw a cheaply clonable
  handle (Arc-backed), not a bare CudaSlice" is **wrong for rstsr**: a shallow
  `Raw::clone` would make two *owned* tensors alias one buffer. The
  quadratic-cost worry behind that recommendation is also overstated: the
  `Raw: Clone` *bounds* on `to_cpu_vec`/`Debug` are satisfiable without those
  paths ever calling `clone` (they can D2H directly from the pointer).
  cudarc's `CudaSlice: Clone` is already deep (`clone_dtod`,
  `driver/safe/core.rs:909-920`) but panics on driver error — see Q1.
- **F2 — `Raw` cannot literally be `CudaSlice<T>`; it must be our own struct.**
  Two independent reasons: (a) `uninit_impl` needs
  `DeviceRawAPI<MaybeUninit<T>>::Raw`, but `CudaSlice<MaybeUninit<T>>` is
  unconstructible — cudarc's typed alloc requires `T: DeviceRepr`, and
  `MaybeUninit` does not implement it (`core.rs:1088-1134`, `:1623`). (b) The
  tensor layer transmutes `&mut Raw<T> ↔ &mut Raw<MaybeUninit<T>>` at ~10
  sites (`tensor/operators/op_with_func.rs:44-48`, `tensor/map_elementwise.rs:102-106`,
  `op_unary_common.rs:235-241`, `op_binary_arithmetic.rs:734-739`, linalg sites)
  and `assume_init_impl` is hand-written per device (CPU literally transmutes
  the `Vec`, `device_cpu_serial/creation.rs:44-58`). So `Raw<T>` must be
  layout-identical to `Raw<MaybeUninit<T>>` — a struct whose fields do not
  depend on `T` (plus `PhantomData`) satisfies this trivially.
- **F3 — `unique*` truncation is an inherent-method contract, not a trait.**
  The tensor layer allocates n-sized scratch, calls the device op with
  `raw_mut()`, and `assume_init`s **trusting the raw length** the device left
  (`operators/set.rs:7-15`; `tensor/set.rs:146-189`). CPU impls call
  `values.truncate(u)`. So `CudaBuffer` needs a mutable *logical* length
  decoupled from the underlying block size. No core change needed — the
  `truncate` is our own inherent method.
- **F4 — cudarc's `CudaSlice` free is already stream-ordered** where the device
  supports memory pools: Drop → `cuMemFreeAsync` on the slice's own stream
  (`core.rs:855-874`); without pool support it is a blocking stream sync +
  `cuMemFree`. There is no Rust-side caching allocator in cudarc, but the
  driver-mempool API is fully available at the `result` layer and unused:
  `mem_pool::{create, set_attribute, trim_to, alloc_async}` (`driver/result.rs:1546-1636`).
  Setting a release threshold on a pool gives torch-style "cache stays warm"
  behavior with zero Rust allocator code.
- **F5 — multi-stream mode is implicit, context-global state.** Creating any
  second owned stream increments `num_streams` and switches the *whole context*
  into event tracking: every subsequent kernel launch inserts per-argument
  event waits/records (`core.rs:40-46`, `:446-468`; `launch.rs:100-166`). A
  design that never creates a second stream never pays this; `new_stream()`
  once + `unsafe disable_event_tracking()` (contract: single stream) keeps
  launches lean.
- **F6 — `CudaBlas` binds its stream at construction** (`set_stream` is
  `unsafe fn (&mut self)`, one handle per stream); `CudaBlasLT` takes the
  stream per call. Handle placement follows stream placement.
- **F7 — "current stream" placements in the wild**: torch/cupy = thread-local
  per-device slot + RAII/context-manager save-restore; every launch queries
  the ambient stream (`c10/cuda/CUDAStream.cpp:166-168, 387-402`;
  `CUDALoops.cuh:331`). candle = one stream inside the device value, no
  switching API. dfdx = two hardcoded internal streams with manual
  `wait_for`/`set_stream` dances (evidence ad-hoc multi-stream without a guard
  abstraction gets messy). oxicuda = explicit `&Stream` parameter everywhere,
  no ambient. None of the three Rust projects has both a pool *and* user
  streams.
- **F8 — cross-stream reuse safety, mechanically**: cupy = one arena *per
  stream*, blocks never migrate, no events; torch = one pool per device, each
  block stamped with its allocation stream, same-stream-only reuse by
  construction, cross-stream use requires explicit `recordStream` → events
  processed at the head of every malloc (`CUDACachingAllocator.cpp:2708-2719`,
  `:4166-4168`, `:4847-4888`). The driver mempool (F4) gives cupy-like
  stream-partitioned reuse semantics *by driver construction*.
- **F9 — contexts are physical; our software state is not.** Two
  `CudaContext::new(0)` calls return two `Arc`s over the *same* retained
  primary context. Without a registry, two `DeviceCuda::new(0)` handles would
  share the GPU but diverge in software state (flags, pool accounting, kernel
  cache) — on an 8 GB card two independent caches fight each other.
- **F10 — non-blocking streams do not synchronize with the legacy null
  stream.** `new_stream()` creates `CU_STREAM_NON_BLOCKING`; a plain
  synchronous `cudaMemcpy` D2H is only ordered w.r.t. the null stream, so
  reading data produced on our non-blocking stream through it is a data race.
  Every D2H/H2D must be issued on the owning stream (`memcpy_*_async` on
  `buffer.stream()`) and synchronized before host use. This is a hard contract
  on all host-read paths (Q7).

---

## 1. The sketch I will defend (spine for the questions)

```rust
// device value: cheap, Clone, per-tensor
pub struct DeviceCuda {
    inner: Arc<CudaDeviceInner>,   // canonical per ordinal (registry, Q3)
    default_order: FlagOrder,      // per-instance snapshot, CPU-consistent
}

// shared, interior-mutable state for one physical GPU
pub struct CudaDeviceInner {
    ctx: Arc<CudaContext>,                 // primary ctx, ordinal
    stream: Arc<CudaStream>,               // v1: the single non-blocking stream (Q4)
    // lazy, per-stream singletons (Q4/F6):
    blas: Mutex<Option<Arc<CudaBlas>>>,    // bound to `stream` at creation
    blaslt: Mutex<Option<Arc<CudaBlasLT>>>,
    math_mode: AtomicU8,                   // shared flags (Q5); NOT in same_device
    // later: kernel/module cache (NVRTC round), pinned staging arena, allocator cfg
}

// raw storage: opaque to core, T-independent layout (F2/F3)
pub struct CudaBuffer<T> {
    slice: CudaSlice<u8>,        // owning alloc: ptr + capacity + stream (D2D-cloneable)
    len: usize,                  // logical element count (unique* truncates this)
    _m: PhantomData<*const T>,   // !Send/!Sync by default; manual impls (Q9)
}
```

---

## Q1. `Raw` = `CudaBuffer<T>`: shape, clone, truncate, pointer access

**Decision**: the exact `Raw` struct and its inherent API.

The u8-erased `CudaSlice<u8>` + separate logical `len` (sketch above) is my
recommendation, because:

- It is the only shape that satisfies F2 without leaving cudarc: allocation is
  `stream.alloc::<u8>(len * size_of::<T>())`; the struct's fields are
  T-independent, so the `&mut Raw<T> ↔ &mut Raw<MaybeUninit<T>>` transmutes
  and the hand-written `assume_init_impl` (rebuild the struct, keep `slice`,
  copy `len`) are sound.
- `DeviceRepr` stops mattering at the Raw boundary entirely — no dtype can be
  excluded from *storage* because cudarc lacks a `DeviceRepr` impl (f16/bf16/
  complex are kernel-round concerns, not storage concerns).
- Drop = `CudaSlice<u8>`'s stream-ordered free (F4); clone = deep D2D copy via
  `memcpy_dtod` on the owning stream (F1).
- `truncate(u)` = `self.len = u.min(self.len)`; kernels and `Storage::len`
  see the logical length; the block keeps its capacity (F3).

Inherent API needed by our own device impls (all `pub(crate)` except where
noted): `len`, `truncate`, `device_ptr() -> *const T`-erased device address
(kernels need ptr + layout offset arithmetic; this is also what
`get_index_ptr` returns — documented "never dereference on host", answering
R7 without a core change), `stream()`.

**Open points to challenge:**

1. **`Clone` panics on driver error.** `CudaSlice::clone` is
   `try_clone().unwrap()`. For `CudaBuffer` I would implement `Clone` by the
   same deep copy but panic through our error mapping (backtrace-preserving
   `rstsr_unwrap` style), and not add a `try_clone` public twin — matches
   `Vec` (clone aborts on OOM, never `Result`). Acceptable, or do you want
   clone to be fallible-panicking with a documented `try_clone` escape?
2. **Zero-size allocations**: `CudaStream::null::<u8>()` (0-byte sentinel) for
   `len == 0` vs a real 0-byte alloc. I say `null` — but then `truncate`-style
   ops on empty buffers never touch the driver. Confirm.
3. **Should `CudaBuffer` keep the `Arc<CudaStream>` directly** (flattening
   `slice.stream`) instead of going through `CudaSlice`? That requires taking
   ownership of the free ourselves (`leak` + `upgrade_device_ptr` +
   custom Drop), which is exactly what we *must* do anyway if/when a custom
   Rust pool replaces the driver pool (a pool backref field would then be
   added). Since `Raw` is opaque to core, evolving the internals later is
   non-breaking — so I recommend **not** doing that now (v1 = wrap
   `CudaSlice<u8>`, simplest correct thing). Challenge accepted if you want
   the pool-ready shape from day one.

## Q2. Allocator for v1: driver mempool vs own Rust pool; injectable?

**Decision**: v1 allocation strategy and whether to abstract it now.

Options:

- **(a) Driver mempool, Rust side zero**: allocations via
  `stream.alloc` (`cuMemAllocAsync` — used automatically when
  `MEMORY_POOLS_SUPPORTED`, which every GPU we target has), frees via
  `CudaSlice` Drop (`cuMemFreeAsync`). Stream-partitioned reuse by driver
  construction (F8-equivalent). Optionally set
  `CU_MEM_POOL_ATTR_RELEASE_THRESHOLD` on the device's pool (via the unused
  `result::mem_pool` bindings) so freed blocks stay cached instead of being
  returned to the OS — this is the single knob that turns cudarc's raw
  alloc/free into a torch-like caching allocator.
- **(b) Own Rust pool now** (dfdx `TensorCache`-style exact-size free list, or
  full cupy arena): full control of accounting/OOM policy, but must solve
  cross-stream reuse (stream-keyed arenas or torch-style recordStream events)
  ourselves — and every Rust precedent we surveyed either ignores the problem
  (oxicuda's pool ignores its `stream` parameter!) or disables the cache by
  default (dfdx).
- **(c) v1 = (a) + a private `trait CudaAlloc` seam**, custom pool later.**

**My recommendation: (c) with the trait private (not public API).** The driver
pool gives stream-ordered caching with zero code; the 8 GB pressure valve is
the release threshold + `mem_get_info` exposure, not a hand-rolled arena. If
profiling later shows driver-pool deficiencies (fragmentation at our size mix,
`cuMemAllocAsync` latency), the trait already exists to slot a Rust pool in,
and `Raw` internals can grow a pool backref then (Q1.3) without any core or
facade break.

**Challenges:**
1. Default release threshold: torch keeps everything cached (threshold =
  total); cupy default also caches. On your 8 GB laptop with ~3.5 GB/task
  workloads, cached-free means peak memory looks "leaked" until trim. I
  propose threshold = unlimited (torch-like) + explicit
  `rt::cuda::empty_cache(&device)` (→ `trim_to(0)`), and document it. Or do
  you want a default cap (fraction of device memory)?
2. Do we need OOM-retry policy in v1 (torch frees + GC + retries), or is a
  typed `MemoryError` enough until the workload demands more? I say: error
  only; the pool makes retry trivial to add later.
3. Env knob (`RSTSR_CUDA_ALLOC_CONF`-like) — yagnos for v1, or do you want
  threshold configurable via env from the start (workload sets
  `PYTORCH_CUDA_ALLOC_CONF` today)? My lean: constructor/method knob only, no
  env parsing in v1.

## Q3. `DeviceCuda` shape: state split, and the registry question

**Decision**: what lives in the device value vs `Raw` vs shared inner; whether
`DeviceCuda::new(0)` twice gives one state or two.

My split rule (this answers your note 1):

- **`Raw` carries per-allocation facts only**: pointer, capacity, logical len,
  owning stream (needed for stream-ordered free and every copy — F4, F10).
  Never policy, never flags, never handles.
- **The device value carries identity + snapshot semantics**: `default_order`
  (per-instance; `set_default_order(&mut self)` mutates the instance only —
  exactly the CPU precedent; nothing in the tensor layer calls it at op time,
  verified).
- **`Arc<CudaDeviceInner>` carries everything shared that ops read at launch
  time from the input tensor's device**: context, stream(s), BLAS handles,
  flags (`math_mode`), later kernel/module cache and pinned arena. This is the
  `Arc<ThreadPool>` precedent of `DeviceCpuRayon`, generalized.

**The registry question (F9) — my main challenge in this round.** Two
`DeviceCuda::new(0)` with separate `Inner`s share the physical context but
diverge in math_mode, caches, and pool accounting (two caches fighting over 8
GB; two cublas workspaces). Torch and cupy both canonicalize per device id
(per-device allocator, per-device pools). Options:

- **(a) Registry**: process-global `OnceLock<Mutex<HashMap<ordinal, Weak<Inner>>>>`;
  `new(ordinal)` fetch-or-create. `DeviceCuda` remains a cheap cloneable
  handle. Yes, this is *global state* — but a read-mostly identity cache, not
  global *policy*; all mutable policy stays inside the per-GPU `Inner`. This
  is my recommendation: it makes `math_mode` semantics match torch's
  per-device flag (minus TLS), keeps one pool per GPU, and dedups handles.
- **(b) Instance state, no registry** (doc 04's implicit shape): simpler, no
  global at all; but ad-hoc `DeviceCuda::new(0)` in two places silently forks
  GPU state, and your multi-GPU workload (one device per worker) only works
  because each worker constructs exactly one device — a contract nobody
  enforces.
- **(c) Instance state + registry only for heavyweight resources** (pool,
  kernel cache), flags per-instance. Worst of both: mode divergence remains.

Sub-decisions bundled here:

1. **`Default` (R8)**: `DeviceAPI: Default` forces it. Options: (i)
   `default()` = `new(0)` unwrapping init errors (panics without CUDA — CPU
   `Default` never fails); (ii) lazy `Inner`: `default()` always succeeds,
   first actual CUDA call initializes and can fail through the normal
   `Result`. I recommend **(ii)** — generic code doing `B::default()` on a
   CUDA-less machine then gets an error at first op, not a panic at
   construction; cost is a `OnceLock`-guarded lazy field in the handle. Your
   call: is the extra lazy machinery worth it vs a documented panic?
2. **`same_device` predicate (R22)**: my proposal: `ordinal && default_order`
   — no stream (constant per device in v1), no math_mode (a precision knob
   must not fragment `same_device` — the workload toggles TF32 across calls on
   the same tensors; already agreed in doc 04 §2), no pool/handle identity.
   Note the CPU precedent *does* compare `default_order` (and `DeviceFaer`
   compares `num_threads` too) — I keep order in, drop the threads analogue
   (none exists). Confirm, and confirm CUDA devices may carry
   `set_default_order` at all (or should CUDA pin RowMajor always, making the
   field vestigial but harmless?).
3. `DeviceCuda: Debug` (tensor `Debug` needs it): derive through Inner with
   ordinal + flags summary, never pointer values. Trivial; noting for
   completeness.

## Q4. Streams in v1: one stream per device, or user-visible streams now?

**Decision**: the v1 stream model (this answers your note 3; README grill
topic 9 is the later full round).

Options:

- **(a) v1 = single non-blocking stream per device, invisible** (candle
  model): `Inner.stream` created once via `new_stream()`, every `CudaBuffer`
  carries it, every kernel/copy/BLAS call runs on it, `rt::cuda::synchronize`
  + `device.synchronize()` wraps `stream.synchronize()`.
- **(b) User-visible streams from day one** (torch/cupy ambient TLS +
  guard): the honest full model, but it drags in Q6's ambient machinery, F5's
  event-tracking overhead (second stream flips the context-wide mode), per-
  stream BLAS handles (F6), and the cross-stream allocator question (F8) —
  all before any workload needs them.
- **(c) Streams visible but not ambient** (oxicuda model): ops take explicit
  streams — does not fit rstsr's trait surface (device ops have no stream
  parameter; adding one is a framework change).

**My recommendation: (a), with two deliberate choices inside it:**

1. Create the stream with `new_stream()` (`CU_STREAM_NON_BLOCKING` — never the
   legacy null stream; avoids implicit sync with other libraries), then call
   `unsafe disable_event_tracking()` with the documented contract "this device
   uses exactly one stream; the allocator is stream-ordered; no cross-stream
   hazard exists". This keeps launches free of per-argument event traffic
   (F5). Alternative: keep tracking on (safer, ~2 event ops per kernel
   argument) — I lean disable + contract; challenge me if you want tracking
   as a belt-and-braces default.
2. Design so the later multi-stream round changes **no v1 shape**:
   `CudaBuffer` already carries its stream (free stays correct); `Inner` grows
   a stream table; `same_device` then needs a stream policy decision
   (strict-reject mixed-stream operands vs auto-join) — that is exactly the
   later grill round.

**Also bundled**: `rt::cuda::synchronize(&device)` (and
`Event` exposure?) — I say v1 exposes `synchronize` only; events wait for the
stream round. `memory_info`/`device_count`/`is_available` (workload gaps 26-27)
land here too as free functions over the registry/context — no design
controversy, just confirming they are in this milestone.

## Q5. Flags: which, where, and the None/default/global question

**Decision**: the flag mechanism (this answers your note 2; the *semantics* of
TF32 per op is the later math-mode round).

Mechanism (mostly pre-agreed in grill README §2, sharpening only):

- Flags live in `CudaDeviceInner` as atomics (`math_mode: AtomicU8`),
  mutated via `DeviceCuda::set_math_mode(&self, ...)` — **interior
  mutability, not `&mut self`** (a `&DeviceCuda` recovered from any tensor
  must be able to flip it; ops read the flag from the input tensor's device
  at launch time, so it must be shared state, not a snapshot — doc 04 §2's
  analysis, which the extraction confirms: ops clone the input's device).
- **No process-global, no TLS, no env var.** "None/default" flags are simply
  the enum's default variant read at use time — no `Option`/lazy-init state
  anywhere.
- **Not part of `same_device`** (Q3.2).

**Challenges:**

1. **v1 flag inventory**: I propose exactly one flag — `math_mode`
   (Strict f32 / TF32; f64 and reduced-precision-f16 policies can be
   `#[non_exhaustive]` additions). Every other candidate (fast-math for NVRTC
   compile options, deterministic mode, pool threshold) either belongs to a
   later subsystem or is a construction-time setting, not a runtime flag.
   Keep the inventory minimal and additive. Agree?
2. **The thread-interference caveat, stated honestly**: a shared-per-GPU
   atomic flag is *coarser* than torch's thread-local global. Two threads
   sharing one GPU with overlapping `with_math_mode` guards interfere; torch
   would not. The registry (Q3a) makes this "per-GPU semantics", which is
   defensible and matches your multi-GPU workload (one worker per GPU), but it
   is a real divergence from torch worth recording in the divergence doc.
   Accept, or do you want per-*thread* mode (TLS) — which then fights the
   per-tensor device model (a tensor's device has no thread)? My answer:
   accept per-GPU; per-call `matmul_with_args`-style overrides are the escape
   hatch if ever needed.

## Q6. Python's `with` → what in Rust? (your note 4)

**Decision**: the scoped-override API shape, for flags now and (later)
streams.

Candidates:

- **(a) RAII guard**: `let _g = rt::cuda::with_math_mode(&device, Mode::Tf32);`
  ... scope end restores. Restore-on-panic for free; zero closure nesting; the
  classic Rust translation of `with`.
- **(b) Closure-scoped**: `rt::cuda::with_math_mode(&device, Mode::Tf32, || { ... })`
  — cannot leak (no guard to `mem::forget`), but forces closure style:
  early returns, borrowck on moved data, and interop with `?` inside all get
  awkward. (Our own `with_blas_num_threads` precedent is exactly this shape —
  and is documented as a race hazard.)
- **(c) Explicit context object threaded through** — no ambient anything; heaviest
  ergonomics; does not fit the existing `rt::` signatures at all.

**My recommendation: (a) for flags, and defer the stream-guard question to the
stream round.** Specifically:

```rust
pub struct MathModeGuard { device: DeviceCuda, prev: MathMode } // restore on Drop
pub fn with_math_mode(device: &DeviceCuda, mode: MathMode) -> MathModeGuard;
```

- The guard stores the *previous value*, not the mode — nested guards compose
  correctly.
- `mem::forget(guard)` leaks an override (mode stays); that is the standard,
  documented RAII caveat — same class as `MutexGuard`. I do not think this
  justifies closure-scoping.
- Note the guard mutates *shared* state (Q5.2): it is a scoped override of
  the GPU's flag, not of "this thread's view". Document, don't fight.
- When streams arrive (later round): if we adopt an ambient current-stream
  (torch model), the guard becomes the familiar save/restore pair — and Q4's
  v1 answer means we need none of it now. If we stay stream-in-device (candle
  model), a stream "guard" is instead `device.use_stream(s)` producing a
  *different device handle* whose tensors carry that stream — no ambient state
  at all. I flag this fork now because Q4(a) does not preclude either.

## Q7. Host-read paths: infallible trait methods, stream ordering, Display

**Decision**: how the host-sync surface is implemented given F10; mostly
confirming warts rather than choosing.

- `get_index`/`set_index` are **infallible in the trait** (`-> T`, R6). Our
  impl: stream-ordered D2H of one element on the buffer's stream + sync, then
  panic through our error mapping on driver failure (documented). A fallible
  `get_index_f` core twin is a nice-to-have core change, not a v1 blocker —
  propose it upstream separately. OK?
- `to_cpu_vec`/`into_cpu_vec`: D2H on the buffer's stream + sync; never
  `Raw::clone` (the `Raw: Clone` bound is just satisfied, not exercised — F1).
- `to_scalar` routes through `get_index` (verified) — one element, same
  treatment.
- `Display` needs `DeviceChangeAPI<DeviceCpuSerial>` (Q8) and prints a host
  `Vec` — full D2H per print, user-triggered, fine. `Debug` additionally
  exercises `to_cpu_vec`.
- `outof_cpu_vec(Vec<T>)` is **blocking by contract** (it consumes and drops
  the host buffer); H2D on the stream + sync (or sync-free with the Vec's
  memory guaranteed valid until return — it is, by ownership; but the *drop*
  must happen after the copy completes → still sync, or stage through a
  pinned buffer + event. v1: plain sync; the async pinned-staging API is the
  stream round). Confirm v1-plain-sync.

## Q8. CUDA ↔ CPU `DeviceChangeAPI`: both directions in the crate

**Decision**: minor, but it *is* representation-adjacent and a workload
blocker (gaps 3, 6).

Proposal: `rstsr-cuda` implements `DeviceChangeAPI` for
`DeviceCuda → DeviceCpu{Serial,Faer,...}`-side and
`DeviceCpu… → DeviceCuda` (coherence allows both: local type in `BOut`
position). For CPU→CUDA: `Repr = ReprTo = DataOwned<CudaBuffer<T>>` — both
directions are copies (no zero-copy across address spaces; `DataRef`
host-borrow → H2D copy inside the impl). Also CUDA→CUDA (`same_device` false
case: different ordinal) via `memcpy_peer` — include in v1 or error? I say
include (cudarc's `memcpy_dtod` already takes the peer path cross-context,
`core.rs:1767-1790`); it is ~20 lines. A generic core impl for
`DeviceCpuSerial → any BOut` (R5's optional core change) stays out of v1.

## Q9. Send/Sync + naming confirmations (small)

- `CudaBuffer<T>`: manual `unsafe impl Send/Sync` (fields are `u64` ptr +
  `Arc<CudaStream>` + usize + `PhantomData<*const T>`; the `PhantomData`
  poison is deliberate — like cudarc, assert no host aliasing; the *data* is
  only touched via device ops). Tensors then cross worker threads (workload
  row 30). `DeviceCuda`: auto (`Arc<Inner>` all-Sync). Confirm you're happy
  with the manual-impl + safety-comment pattern rather than `PhantomData<T>`
  gymnastics.
- Names: `rstsr-cuda` crate, `DeviceCuda`, `CudaBuffer<T>`, `rt::cuda::*`
  facade namespace, `MathMode`/`MathModeGuard`. Any renames now (cheap) vs
  never (public API)?

---

## Summary of my positions (the spine, one line each)

| Q | Position |
|---|---|
| Q1 | `Raw = CudaBuffer<T> { CudaSlice<u8>, len, PhantomData }`; deep D2D `Clone`; inherent `truncate`; `device_ptr()` |
| Q2 | v1 driver mempool (`cuMemAllocAsync` + release threshold), private alloc trait seam; `empty_cache`; no env knob |
| Q3 | Device value = identity+order; `Arc<Inner>` = shared policy; **registry per ordinal**; lazy `Default` = ordinal 0; `same_device` = ordinal + order |
| Q4 | v1 single non-blocking stream per device, event tracking disabled under a documented single-stream contract; streams round deferred |
| Q5 | One v1 flag: `math_mode: AtomicU8` in Inner, `set_math_mode(&self)`; no global/TLS/env; not in `same_device`; per-GPU (not per-thread) semantics accepted |
| Q6 | RAII guard storing previous value (`with_math_mode`); no closure-scoping; stream guard deferred to the stream round |
| Q7 | All host reads stream-ordered via buffer's stream; infallible trait methods panic via error mapping; `outof_cpu_vec` blocking in v1 |
| Q8 | Both transfer directions in the crate; copies (no cross-space zero-copy); peer CUDA→CUDA included |
| Q9 | Manual `unsafe Send/Sync` with safety comments; names as proposed |
