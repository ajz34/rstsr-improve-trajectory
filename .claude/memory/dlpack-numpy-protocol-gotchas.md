---
name: dlpack-numpy-protocol-gotchas
description: DLPack/NumPy protocol facts verified 2026-10-03 (numpy 2.5): from_dlpack call pattern, element strides vs byte_offset, capsule naming, dtype/device accept tables, remaining rstsr dtype gaps.
metadata:
  type: reference
---

Verified against NumPy v2.5.2 source + NumPy 2.5.1 probes; full evidence in
`2026-10-03-rust-numpy-review/notes/numpy-dlpack.md` and `notes/dlpack-v13-spec.md`.

- **NumPy as consumer**: `np.from_dlpack(x)` calls
  `x.__dlpack__(dl_device=None, copy=None, max_version=(1, 0))`; on `TypeError` it retries
  with a bare `x.__dlpack__()` — a producer can be called twice. Bare `PyCapsule`s are
  rejected; `__dlpack_device__` is never called.
- **NumPy as producer**: `a.__dlpack__()` with no arguments emits a **legacy** capsule;
  `max_version=(1, 0)` emits `dltensor_versioned` with version (1, 0). Read-only arrays
  cannot be exported as legacy (`BufferError`). Too-high `max_version` is clamped, never
  rejected.
- **Layout**: DLPack strides are in **elements**, `byte_offset` in bytes. NumPy always emits
  `byte_offset = 0`, non-NULL `data` (even for 0-d/size-0), and does no alignment checks in
  either direction. Non-C-contiguous exports require strides that are multiples of itemsize
  (size-1 dims are exempt and may be truncated to 0).
- **Dtype accept table** (both directions): bool (8), int/uint (8/16/32/64), float
  (16/32/64), complex (64/128); `lanes` must be 1; native byte order only. Thus rstsr's
  `i128/u128` and `bf16` cannot be handed to stock NumPy; datetime64/timedelta64, strings,
  object and structured dtypes cannot cross DLPack at all. NumPy 2.5+ offers
  `np.dtypes.register_dlpack_dtype` as a user escape hatch.
- **Device**: NumPy imports only kDLCPU/kDLCUDAHost/kDLROCMHost/kDLCUDAManaged (device_id
  ignored/round-tripped); as a producer it reports only CPU. `np.from_dlpack(..., device=)`
  accepts only `"cpu"`.
- **Ownership**: on success NumPy renames the capsule to `used_dltensor[_versioned]` and calls
  the deleter once via an internal base capsule; on failure it drops the capsule un-renamed
  (producer destructor/deleter runs). Legacy imports are always non-writeable; versioned
  imports honor the `READ_ONLY` flag.

See [[rstsr-numpy-interop-review]] for what rstsr intends to do with this.
