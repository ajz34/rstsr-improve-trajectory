---
name: cuda-pinned-host-buffer-grill
description: 2026-11-23 grill CLOSED (5 rounds) on a pinned host tensor type for rstsr-cuda - DECISIONS.md D1-D10 (DeviceCudaPinned + PinnedBuffer + extension traits, async pinned<->CUDA edges, one event per buffer, no creation traits, no pool)
metadata:
  type: project
---

Task dir `2026-11-23-pinned-host-buffer/` (DECISIONS.md canonical). Baseline
rstsr `261021/cuda` @ `6d17d04`, cudarc 0.19.10.

Key decisions:
- **DeviceCudaPinned { ordinal, base: DeviceCpuSerial }** in `rstsr-cuda`,
  `Raw = PinnedBuffer<T>` (byte-erased `PinnedHostSlice<u8>`, `T: Copy` gate);
  no core change. Neither `DeviceCpuSerial` nor `DeviceCuda` can take a second
  `Raw` (coherence, one associated type per trait impl).
- No op traits / creation traits on the pinned device: 142 core tensor-level
  conveniences are hard-bound to `Raw = Vec<T>`; CPU compute only via
  `host_view`/`host_view_mut` -> `DeviceCpuSerial` views through a fabricated
  `ManuallyDrop<Vec<T>>` (`asarray(&[T])` pattern; a `TensorView` holds
  `&Vec<T>`, not a slice).
- **Inherent impls on `TensorAny` are impossible from rstsr-cuda (E0116)** ->
  tensor-level methods must be local extension traits
  (`TensorPinnedHostViewAPI`, `CudaSyncAPI`); owner prefers functions over
  traits unless necessary (it is, here).
- Pinned <-> CUDA `DeviceChangeAPI` edges non-blocking by default; one
  `CU_EVENT_BLOCKING_SYNC` event per buffer (cudarc model), every host access
  syncs first; D2H must record exec->alloc event for the source buffer
  (foreign-stream read-lifetime hole). `into_device` from pinned is
  effectively blocking (drop syncs).
- Flags default `0` (not cudarc's WRITECOMBINED); no pool (event-gated
  caching host allocator later). Stream API: `new_stream_with_ordinal`,
  `new_stream(&self)`, `with_stream`, public `CudaEvent`, `synchronize`.
- Reference facts: CuPy = raw untyped pinned buffer only (no pinned ndarray);
  PyTorch = normal CPU tensor, pinned-ness invisible (pointer query).

Owner naming preference: `DeviceCudaPinned` (pinned first, backend second).
Next: PLAN.md + implementation on the `261021/cuda` branch, bench in a new
trajectory dir; execution awaits user go.
