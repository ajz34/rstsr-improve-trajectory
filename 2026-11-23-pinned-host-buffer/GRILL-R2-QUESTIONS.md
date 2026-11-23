# GRILL R2 — the device type for pinned tensors

## Agent-authored background: answers received on screen (transcribed)

- **Q1**: (A) — `PinnedBuffer` + its device live in `rstsr-cuda`.
- **Q2**: reopened by the user. "Is it possible to use `DeviceCpuSerial` for the
  device, but implement only `DeviceRawAPI<Raw = PinnedBuffer<T>>`, and does
  that offend the orphan rule? If `DeviceCpuSerial` is not a good idea, do you
  prefer `DeviceCuda` or a new device type `DeviceCpuPinned`?"
- Q3–Q13: not yet answered; they wait on this round.

## Facts (agent-sourced, `261021/cuda` @ `6d17d04`)

- F13. `DeviceRawAPI<T>` has exactly one associated type `Raw`. The impl
  `impl<T> DeviceRawAPI<T> for DeviceCpuSerial { type Raw = Vec<T> }`
  exists (`rstsr-core/src/device_cpu_serial/device.rs:24-26`). A second impl
  for the same type would be a **coherence conflict** (E0119), not an orphan
  rule problem — the orphan rule is irrelevant, the impl would be in
  `rstsr-core` anyway. Same for `DeviceCuda` (`type Raw = CudaBuffer<T>`,
  `rstsr-cuda/src/device.rs:158-160`).
- F14. Every `DeviceCpuSerial` op/creation/storage impl is written concretely
  over `Vec<T>` (`&mut Vec<TC>`, `&Vec<TA>`, ...), and the `DeviceCpuSerial`
  struct is `{ default_order: FlagOrder }` with a derive `Default`. Giving it
  a pinned mode would mean either a `Raw` enum (`Vec | Pinned`) through all
  those impls, or a context field — i.e. a CUDA handle inside `rstsr-core`.
- F15. Precedent for a thin wrapper device: `pub struct DeviceFaer { base:
  DeviceCpuRayon }` (`device_faer/device.rs:6`), `DeviceCpu` is an alias
  chosen by feature (`lib.rs:45-49`).

---

❓ **Q2'** - **Device type**: neither `DeviceCpuSerial` nor `DeviceCuda` can
carry a second `Raw` (F13). Create the new type
`DeviceCpuPinned { ordinal: Arc<CudaOrdinalObj>, base: DeviceCpuSerial }` in
`rstsr-cuda`, with `default_order`/`same_device` delegated to `base` (F15
pattern), `Default` = ordinal 0. The embedded `base` lets `host_view()` drop
its `device: B` argument and default to the embedded `DeviceCpuSerial`; an
`host_view_on::<B>(device)` generic form can come later if `DeviceFaer` (rayon)
host compute is wanted.

➡️ Yes: new type `DeviceCpuPinned` wrapping `DeviceCpuSerial`; R1 Q3–Q13
otherwise stand, with `host_view` argument-free.
