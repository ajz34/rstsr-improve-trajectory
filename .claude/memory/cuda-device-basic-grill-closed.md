---
name: cuda-device-basic-grill-closed
description: 2026-10-22 rstsr-cuda device+storage grill CLOSED after 3 rounds; DECISIONS.md D1-D8 (CudaBuffer, DeviceCuda ordinal/stream/flags split, strict same_device, driver mempool); supersedes R1 positions
metadata:
  type: project
---

Grill closed 2026-10-22 (3 rounds; `2026-10-21-cuda-decisions/01-device-basic/`,
DECISIONS.md is canonical). Related: [[cuda-grill-decisions]],
[[cuda-device-basic-grill-r1]] (R1 facts F1-F10 still valid; R1 *positions*
superseded).

Final design (D1-D8):

- `CudaBuffer<T> { CudaSlice<u8>, len: usize, PhantomData }`; deep-D2D Clone;
  inherent `truncate` (unique* contract); null-slice empty behind one ctor;
  manual unsafe Send/Sync; no pool internals yet.
- `DeviceCuda { Arc<CudaOrdinalObj>, stream: Arc<CudaStreamObj>, flags:
  Option<CudaRuntimeFlags>, default_order }`; registry per ordinal
  (Weak map, fetch-or-create); Default = panicking new(0); Display `cuda:0`.
- Flags: `Option` per device + global `CUDA_FLAGS_DEFAULT: RwLock` (env
  seeded, safely mutable); **launch-time resolution** (op reads
  flags.or(global) from input tensor's device — torch-style global toggle
  works with zero re-tagging); `same_device` compares flags AS STORED
  (Some(x)≠None accepted) and flags gate ALL multi-tensor ops. OWNER OVERRODE
  my live-atomic design (his model is stronger: loud mixed-mode errors).
- Streams user-visible from day one: `set_stream(&mut self)` mutation (not
  `with_stream` transform); STRICT `same_device` stream comparison (Option A;
  mixed-stream = DeviceMismatch; explicit re-tag via same-ordinal to_device
  view); shared `override_stream` WITHDRAWN (R24 hazard, my challenge
  accepted) — TLS override maybe later, additive; event tracking stays ON
  (Drop safety); buffer's allocation stream owns free, device's stream owns
  launch.
- Allocator: private `CudaAlloc` trait, driver mempool default, unlimited
  release threshold, `empty_cache` trim valve, OOM = trim→retry→MemoryError,
  `RSTSR_CUDA_ALLOC_CONF` env → global → per-device. Core DeviceAllocatorAPI
  = separate future grill (owner needs CPU use-case first).
- #164 verified: DeviceToCpuAPI/DeviceFormatAPI capability traits exist;
  Raw:Clone off host reads; CUDA printing = visible-gather (never full D2H).
- change_device lifetime-preserving (view→view, owned→owned; relabel
  same-ordinal, peer copy cross-ordinal).
- cudarc stance: substrate for maintenance/cuda-gates/CudaSlice only; BLAS
  through our own FFI over sys/result (owner directive).
- Deferred with intent: NVRTC round, GEMM/math-mode consumption, transfer
  round (events/pinned/async_h2d), guard sugar, tests, graphs/DLPack/NCCL.

Next grills likely: `02-kernel-nvrtc`, `02-allocator-framework`, tests.
