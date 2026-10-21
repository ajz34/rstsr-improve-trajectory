---
name: cuda-device-basic-grill-r1
description: 2026-10-21 rstsr-cuda device/storage grill R1 questions issued; four background extraction reports (core traits, cudarc, Rust precedents, torch allocator/streams); key corrections to grill docs
metadata:
  type: project
---

Grill round R1 for rstsr-cuda device + data representation issued 2026-10-21
(`2026-10-21-cuda-decisions/01-device-basic/GRILL-R1-QUESTIONS.md`); answers
pending in `GRILL-R1-ANSWERS.md` (user-owned). Related: [[cuda-grill-decisions]].

Extraction facts that CORRECT the 2026-10-21-cuda-grill docs:

- `Raw: Clone` is load-bearing VALUE semantics (`Tensor::clone` → `DataRef::into_owned`
  → `C::clone`): doc 04 §3's "cheaply clonable Arc handle" recommendation is WRONG
  for rstsr — `CudaBuffer::clone` must be a deep D2D copy.
- `Raw` cannot be `CudaSlice<T>`: `CudaSlice<MaybeUninit<T>>` unconstructible
  (`MaybeUninit: !DeviceRepr`), and tensor layer transmutes `&mut Raw<T> ↔
  &mut Raw<MaybeUninit<T>>` at ~10 sites → Raw must be our own T-independent-layout
  struct `CudaBuffer<T> { CudaSlice<u8>, len, PhantomData }`.
- `unique*` truncation is an inherent-method contract (tensor layer `assume_init`s
  trusting raw length; CPU calls `Vec::truncate`) → CudaBuffer needs mutable logical
  `len` decoupled from block capacity.
- cudarc 0.19.10: `CudaSlice` Drop = `cuMemFreeAsync` (stream-ordered) when
  MEMORY_POOLS_SUPPORTED; `result::mem_pool::*` (create/set_attribute/trim_to)
  exists UNUSED → driver mempool + release threshold = caching allocator with zero
  Rust code (recommended v1 over own pool). Multi-stream mode is implicit
  context-global state (any 2nd owned stream → per-launch event tracking for all
  slices); v1 single-stream design avoids it.
- `CudaBlas` binds stream at construction (`set_stream` unsafe &mut); `CudaBlasLT`
  per-call stream.
- torch current-stream = thread-local per-device slot + RAII guard; torch allocator
  = one pool/device, blocks stamped with allocation stream, cross-stream reuse
  needs explicit recordStream+events; cupy = arena-per-stream. Driver mempool gives
  cupy-like stream partitioning by construction.
- Non-blocking stream + plain synchronous memcpy = data race: every H2D/D2H must be
  issued on the owning stream then synced.

R1 positions (mine, awaiting challenge answers): CudaBuffer as above; v1 driver
mempool + private alloc trait seam; DeviceCuda = Arc<Inner> + per-instance order,
REGISTRY per ordinal (fetch-or-create Inner), lazy Default = ordinal 0;
same_device = ordinal + order (no math_mode, no stream in v1); single non-blocking
stream per device with event tracking disabled under documented single-stream
contract; single v1 flag math_mode AtomicU8 in Inner via set_math_mode(&self) +
RAII guard storing previous value (per-GPU semantics, diverging from torch TLS);
host reads stream-ordered then panic via error mapping; outof_cpu_vec blocking v1;
DeviceChangeAPI both directions in crate incl. peer CUDA→CUDA; manual unsafe
Send/Sync on CudaBuffer.
