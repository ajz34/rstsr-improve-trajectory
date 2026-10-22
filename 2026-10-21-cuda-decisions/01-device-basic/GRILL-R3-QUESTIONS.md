# GRILL R3 — device and data representation for `rstsr-cuda`

Absorbs the R2 answers (received on screen 2026-10-22; transcribed as §0 —
agent-authored summary, answers files remain user-owned). Two decisions
remain open by your own instruction ("still challenge me"): the flags model
(§2) and the stream model (§3). §4 re-grounds the dissolved guard question;
§5 is the ledger.

---

## 0. Background — R2 answers transcribed (agent-authored)

| Item | Your answer |
|---|---|
| Q1 | Defends flags-light-and-compared; comparison applies to light **and** stream. |
| Q1-S1 | "Let it fail": device pre-built + mode set before upload; re-tag tensors via `to_device`. Proposes `static flags_default: Arc<Mutex<CudaRuntimeFlags>>` (global, safely mutable) + `flags_override: Option<CudaRuntimeFlags>` per device (comparable, deep clone); None inherits the global; `with_` deferred or global-level only; maybe `device.get_flag(&str)`-style accessors; set via `set_flags_override` / mutating the global. Not final — challenge again. |
| Q2 | Cross-stream operands: disagree, or agree only with a **warning**. `to_device` should return view (or Cow view) so stream change is explicit; LHS-only stream poses user-side design issues. Prefers `let mut d2 = d.clone(); d2.set_stream(s2);` over `with_stream` value-transform. `outof_cpu_vec` stays naive (device stream); wants a separate `rt::cuda::async_h2d`-style API for true async transfers. Proposes `CudaOrdinalObj { default_stream, override_stream: Arc<Option<..>> }` with priority **shared override > device stream > shared default**, `with_stream` mutating the shared override; computation (not creation) honors it. Not final — challenge again. |
| Q3 | Needs re-grounding on Q1's outcome. |
| Q4 | OK on §5.1+§5.2: private `CudaAlloc`; CUDA allocation its own. |
| Q5 | OK; formatting decoupled from core device/storage definition. |
| Q6 | OK: `change_device` like `change_shape` — view→view, owned→owned without clone, preserve lifetime/ownership; cloned owned as fallback. |

Settled this round from your answers: private `CudaAlloc` + driver-mempool
defaults (§5.2 of R2); formatting by visible-gather (technical, decoupled);
lifetime-preserving `change_device`; buffer's allocation stream owns
free/reuse while the device value's stream owns execution.

---

## 2. R3-Q1 — Flags: your model formalized; I concede S1 — with four sub-challenges

### 2.1 The concession, and why your model is actually stronger

Your design, stated precisely:

```rust
static CUDA_FLAGS_DEFAULT: RwLock<CudaRuntimeFlags> = /* parsed once from env, safely mutable */;

pub struct DeviceCuda {
    ordinal: Arc<CudaOrdinalObj>,      // heavy identity: ctx, default stream, handles, alloc
    stream:  Arc<CudaStreamObj>,       // execution stream (§3)
    flags:   Option<CudaRuntimeFlags>, // None = track global; Some(x) = pinned
    default_order: FlagOrder,
}
```

with **launch-time resolution**: an op reads
`flags.or(global_default)` from the input tensor's device when the op
consumes flags (GEMM compute type, initially).

I withdraw the "must be live-atomic" claim. Your model dominates it on two
axes my R2 design lacked:

1. **The torch-style global toggle works with zero re-tagging** — flip
   `CUDA_FLAGS_DEFAULT`, every `None` device's *next op* flips (launch-time
   resolution, not creation-time snapshot — this is the load-bearing
   detail; a creation-time snapshot would strand existing tensors and kill
   the workflow). That is the DF-MP2 pattern (phases hoisted around regions).
2. **Mixed modes in one region are a loud error, not silent arithmetic
   drift** — `Some(A)` vs `Some(B)` (or vs `None`) fails `same_device`.
   My live-atomic design could never offer per-thread/per-region mode
   isolation on one GPU; yours does, via two device values.

And the cost I feared (re-tag churn) is bounded: re-tag = same-ordinal
`to_device` = view relabel (cheap), once per entry tensor per phase.

### 2.2 Sub-challenges (these four decide the final form)

**(a) Compare the representation, not the effective value — recommend, confirm.**
`same_device` should compare `Option<CudaRuntimeFlags>` *as stored*. The
alternative — resolve `None` through the global and compare effective flags
— makes `same_device`'s result depend on mutable global state: `Some(Strict)`
vs `None` passes today, fails after someone flips the global. A comparison
that changes over time is a debugging nightmare. Consequence of the
representation rule, to accept explicitly: `Some(Strict)` and `None`-under-
global-Strict are *different devices* even though they compute identically —
mixing them errors; the fix is a one-line re-tag. Confirm you accept that
false-positive-by-design.

**(b) Flags gate all multi-tensor ops, not just flag-consuming ones.**
`same_device` is checked by every multi-operand op (add, matmul, assign,
`+=`…). So tensors on `Some(Tf32)` and `Some(Strict)` devices cannot even be
*added* — flags are irrelevant to `add`, yet the device contexts differ and
we error. This is the consistent price of "flags are identity" (same as
`default_order` today). The alternative — compare flags only in ops that
consume them — is not expressible (`same_device` is one device-level
predicate). Accept, or does this change your answer?

**(c) No stringly-typed accessors.** `device.get_flag(&str)` invites
runtime typos and unordered evolution. Typed accessors instead:
`device.flags() -> CudaRuntimeFlags` (resolved), `device.flags_override()
-> Option<..>`, `device.math_mode() -> MathMode` (the common one), plus
`set_flags_override(&mut self, Option<CudaRuntimeFlags>)` — your mutating-
method style, `set_default_order` precedent. Stringly keys belong only to
the env parser (`RSTSR_CUDA_ALLOC_CONF`-style, one-time).

**(d) The global's lock shape.** Reads happen at every flag-consuming launch
(GEMM); `Mutex` serializes those reads across all threads. Use
`RwLock<CudaRuntimeFlags>` (Flags is small `Copy`; read = lock+copy), or an
`ArcSwap`-style eventually-consistent `Arc` if you want lock-free reads —
but that's a dependency or hand-rolled atomic pointer; `RwLock` is enough
for v-now. Also: `static` wants the plain lock type, not `Arc<Mutex>` (the
Arc buys nothing in a static). `CudaRuntimeFlags`: small struct, `Copy +
PartialEq + Debug`, fields additive (`math_mode: MathMode` today;
`fast_math: bool` when NVRTC lands; …), `MathMode` `#[non_exhaustive]`.

Also confirm: global setter `rt::cuda::set_default_flags(&CudaRuntimeFlags)`
(+ getter); **guard deferred** per your answer — a `with_default_flags`
RAII guard over the global is additive later and nothing now depends on it.
Env: the global's initial value parsed once from env at first device
creation (§5.2 R2 scheme: env → global default → per-device override).

---

## 3. R3-Q2 — Streams: one contradiction to resolve, one proposal to challenge

### 3.1 The contradiction in your answer

You said both:

- (i) "comparison should be applied to both light and **stream**" — i.e.
  `same_device` compares the stream; mixed-stream operands → `DeviceMismatch`
  **error**; and
- (ii) cross-stream operands: "agree but … log some **warning**".

Both cannot hold: if `same_device` compares streams, mixed-stream operands
never reach the kernel — there is nothing to warn about. Pick one:

- **Option A — strict (compare; error on mix).** One device value = one
  execution context `(ordinal, stream, flags, order)`. Changing stream =
  `set_stream` on a (cloned) device value + re-tag operands via same-ordinal
  `to_device` (view). Cross-stream producer/consumer = user re-tags the
  entry tensors to the consumer's device; safety of consuming a buffer
  allocated on another stream is handled automatically by cudarc's event
  tracking (launch waits on the buffer's recorded write event; Drop waits
  its events — tracking stays ON, §3.3).
- **Option B — lenient (ignore; warn).** `same_device` ignores stream; ops
  launch on the first input's stream with a warning when operand streams
  differ. More forgiving, but the warning is a runtime log you cannot act
  on structurally, and "which stream ran my op" becomes implicit — the very
  thing your `to_device`-explicitness argument rejects.

**My recommendation: A**, on three grounds: (1) it is the same philosophy
you defended for flags — explicit contexts, loud errors on mixing; (2) it
kills the LHS-implicitness you objected to ("stream only on LHS may pose
design issue") more thoroughly than a warning does; (3) reversibility
asymmetry — tightening B into A later is a breaking change (legal programs
start erroring); loosening A into B later merely turns errors into warnings,
which almost never breaks real code. By your own hard-to-revert rule, A now
is the safe direction. Note the price, symmetric to flags (b): tensors on
two streams of one ordinal cannot be operands of the *same* op without a
re-tag, even where streams are semantically irrelevant — because their
device contexts differ.

### 3.2 Challenging the shared `override_stream` (priority chain)

Your proposal: `CudaOrdinalObj { default_stream, override_stream }`,
`with_stream` flips the shared override; priority
**shared override > device stream > default**; computation honors it,
creation does not.

**The core problem: a shared override is a cross-thread hazard, exactly the
`with_blas_num_threads` anti-pattern (R24).** The override lives in the
per-ordinal object, so it is visible to *every thread* using that GPU.
Thread A opens `with_stream(s1)` for a region; thread B, mid-loop on the
same GPU, silently jumps to s1. Nothing breaks *correctness* (event
tracking), but execution placement becomes nondeterministic — for a
multi-worker-per-GPU or mixed-library program this is the documented
cautionary precedent, not a model. The torch answer is TLS (per-thread
current stream); the cupy answer is TLS; your R1 answer even said "TLS mode
is straightforward in this scheme" — but the sketch above is shared, not
TLS, and TLS keyed by ordinal is real machinery
(`thread_local { RefCell<HashMap<ordinal, Option<Stream>>> }`).

**My counter-proposal: defer the override entirely.**

- Everything the override gives you — "run this region on stream s without
  touching each tensor" — is already expressible under Option A: build
  `let mut d2 = d.clone(); d2.set_stream(&s);` once per region, re-tag the
  few entry tensors once (`a.to_device(&d2)` views), and *every derived
  tensor inherits d2* (ops clone the first input's device). One re-tag per
  entry tensor per region ≈ the ergonomics of one `with` per region, and it
  is explicit at every tensor's creation site.
- Your workload needs none of it: DF-MP2 is single-stream per device; the
  side-stream idea was measured and abandoned in the source file
  (non-pinned `as_tensor` host-blocks; doc 04 row 2).
- Reversibility: adding a TLS override *later* is purely additive (a
  per-thread mask consulted before launch; no published signature or
  `same_device` rule changes). Publishing a shared override now and
  reshaping it into TLS later is a behavior change for every concurrent
  user. Defer is the strictly safer direction by your rule.
- `with_stream` as a *name* stays reserved for that future guard so the
  eventual API reads naturally.

If you still want an override now, I need your ruling on shared-vs-TLS, and
my position is TLS-or-nothing (shared is the one variant I would veto).

### 3.3 What stands regardless (recap + your R2 points adopted)

- `CudaStreamObj`: our wrapper (creation with priority, `wait`/`join`,
  `synchronize`; identity equality = handle+context, used by `same_device`
  under Option A). Events (`CudaEvent` exposure) deferred to the
  transfer/overlap round — `wait`/`join` suffice for manual ordering.
- `set_stream(&mut self, &Arc<CudaStreamObj>)` — your mutating style;
  `DeviceCuda::new(ordinal)` starts on `ordinal_obj.default_stream`
  (non-blocking, never the legacy null stream).
- Buffer vs device stream invariant (settled R2): `CudaBuffer`'s allocation
  stream (inside `CudaSlice`) owns free/reuse ordering; the device value's
  stream owns kernel launch. Re-tagging never moves data.
- **Event tracking ON permanently** — even under Option A the Drop path
  needs it (a buffer allocated on s1 can be dropped while the device value
  points at s2; the Drop-time event waits are what keep the pool free
  safe). `disable_event_tracking` remains an `unsafe` opt-out lever,
  documented "single stream per ordinal, you promise".
- `outof_cpu_vec`: blocking, on the device value's stream (naive path) —
  your answer. True async transfers = separate explicit API
  (`rt::cuda::async_h2d`/`async_dtoh` taking a stream + pinned staging;
  design with the pinned-memory machinery in the transfer round — flagged
  in the ledger, not forgotten).
- cuBLAS handles: per ordinal object behind a mutex, our own FFI over
  cudarc `sys`/`result` (your F6 stance), `cublasSetStream(<op stream>)`
  before each enqueue.

---

## 4. R3-Q3 — Guard question, re-grounded (dissolves)

With flags as `Option` + global (§2), there is no live per-GPU flag state,
so the R2 §4 guard construction is moot:

- No device-level `with_math_mode` (would fight the representation
  semantics; a pinned `Some(x)` is set by `set_flags_override`, not scoped).
- A *global* `with_default_flags` RAII guard remains possible later
  (previous-value restore over `CUDA_FLAGS_DEFAULT`) — additive, deferred
  per your answer. Until then: direct `set_default_flags` + flip back.
- The future stream guard = the deferred TLS override (§3.2), same
  previous-value shape. One guard pattern, two knobs, both later.

Nothing else remains of Q3; confirm and it closes.

---

## 5. Ledger

**Settled (cumulative, this grill):**

1. `Raw = CudaBuffer<T> { CudaSlice<u8>, len: usize, PhantomData<*const T> }`;
   deep-D2D `Clone` (panics via error mapping); inherent `truncate` (logical
   len, capacity untouched); `device_ptr()` accessor; empty buffer via
   null-slice behind one constructor (revertible to real 0-byte alloc);
   manual `unsafe Send/Sync` + safety comments.
2. Device split: `DeviceCuda { Arc<CudaOrdinalObj>, stream, flags, default_order }`
   with per-ordinal registry (fetch-or-create), `Default` = panicking
   `new(0)`, `Display` = `cuda:0`, `Debug` detailed.
3. Flags: `Option<CudaRuntimeFlags>` per device + global
   `CUDA_FLAGS_DEFAULT` (`RwLock`, env-seeded, safely mutable);
   launch-time resolution; representation comparison in `same_device`
   *(pending §2a/2b confirmations)*; typed accessors; `set_flags_override`;
   guard deferred. *(§2 sub-challenges open.)*
4. Streams: device-value stream + `set_stream` mutation + strict
   `same_device` stream comparison *(pending §3.1 ruling)*; shared
   override_stream **challenged — recommend defer** *(§3.2 open)*; events
   deferred; event tracking ON; async transfer API = later round.
5. Allocator: private `CudaAlloc` seam; driver mempool
   (`cuMemAllocAsync`/`FreeAsync`), unlimited release threshold by default,
   `empty_cache` trim valve, OOM = trim-everything → retry once → typed
   `MemoryError`; `RSTSR_CUDA_ALLOC_CONF` → global config → per-device
   override. Core `DeviceAllocatorAPI` = separate future grill (needs a
   concrete CPU use-case).
6. Host reads: `DeviceToCpuAPI` = stream-ordered D2H + sync;
   `get/set_index` sync-then-panic-on-error; `outof_cpu_vec` blocking.
7. Formatting: CUDA implements `DeviceFormatAPI` directly; visible-gather
   under print limits (contract now, fast path with kernels); shared
   visible-index helper alongside `format_tensor_cpu_serial`.
8. `change_device`: lifetime-preserving (view→view, owned→owned, no clone
   on same-ordinal relabel; copies only cross-space/ordinal; peer
   CUDA→CUDA included).

**Open (need R3 answers):**

1. §2(a): representation-vs-effective flag comparison (I recommend
   representation; accept the `Some(x)`≠`None`-under-x false positive?).
2. §2(b): flags gate *all* multi-tensor ops — accept?
3. §2(c)(d): typed accessors OK? `RwLock` global OK? Names
   (`CudaRuntimeFlags`, `CUDA_FLAGS_DEFAULT`, `MathMode`) OK?
4. §3.1: Option A (strict stream comparison) vs B (warn) — ruling?
5. §3.2: shared `override_stream` — defer (my recommendation), TLS now, or
   shared now (I veto shared)?
