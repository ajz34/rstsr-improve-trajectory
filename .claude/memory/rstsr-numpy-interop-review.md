---
name: rstsr-numpy-interop-review
description: 2026-10-03 consensus review of rstsr↔NumPy interop (rust-numpy code literature review; DLPack chosen) — led to the rstsr-cpu-dlpack crate, see [[rstsr-cpu-dlpack-implementation]].
metadata:
  type: project
---

On 2026-10-03 a full review of rust-numpy (v0.29.0+8 `da6bf5be`), NumPy 2.5 DLPack support,
DLPack v1.3 and the rstsr-side gaps was written to `2026-10-03-rust-numpy-review/`
(REVIEW.md + questions/answers/response/decisions + notes/ + experiments/). Status after R1 (same day):
**Q1 (c) both directions; Q2 (c) DLPack primary + copy fallback; Q3 (a) new workspace crate with
`rstsr-cpu-*` naming; Q7/Q10(a)/Q12 accepted; Q4 open.** After R2 (same day, `DECISIONS-R2.md`):
**the heavy rust-numpy/ndarray layer is deferred; the deliverable is the pure `rstsr-cpu-dlpack`
crate (dlpack-ffi + rstsr-core, no pyo3); the Python-side `PyCapsule` holder stays a ~30-line
documented shim, because PyCapsule is CPython-only.** Follow-up the same day: the design was written
and the maintainer redirected implementation into the rstsr repo — the crate now exists at
`crates-interop/rstsr-cpu-dlpack` on branch `261003/rstsr-cpu-dlpack` (see
[[rstsr-cpu-dlpack-implementation]]); the Python host harness and the design stay in the task dir.

Key durable facts from it:
- `np.from_dlpack` requires an object with `__dlpack__` (bare capsules rejected); NumPy calls
  the producer with `dl_device/copy/max_version=(1,0)` and silently retries a bare call on
  `TypeError` (producer may be invoked twice); default `max_version=None` yields a *legacy*
  capsule.
- DLPack strides are in **elements** (same unit as rstsr), `byte_offset` in bytes; NumPy
  exports `byte_offset=0` and refuses non-native byte order.
- rstsr docs previously dropped `from_dlpack` as "not possible for another language" — true
  only for a pure-Rust array-API namespace; a pyo3 bridge crate is the workaround, kept out
  of `rstsr-core`.
- Zero-copy *views* of foreign buffers are already expressible in rstsr (`DataRef` +
  non-owning-`Vec` pattern used by `asarray`); *owning* one needs a storage repr carrying the
  owner — and that repr can be defined in the bridge crate with public API only, **no
  rstsr-core change** (V2; see [[rstsr-external-storage-repr]]).
- The in-repo "owning a foreign allocation is UB" note (`device_faer/conversion.rs:85-94`) is
  **faer-specific** (64-B over-alignment + padded row capacity ⇒ `Vec` dealloc-layout
  mismatch; verified against faer 0.22.6); it does not forbid NumPy-buffer adoption (V1).
- Import has three rungs, not two: read-only zero-copy view / **consumer-owned copy via
  `__dlpack__(copy=True)`** (NumPy copies before export and sets `IS_COPIED` = "solely owned
  by the consumer") / rstsr-side copy (V4).
- Declared workspace MSRV 1.82 is stale under default features: faer 0.22.6 requires 1.84, so
  pyo3 0.29's 1.83 raises nothing (V3).

Links: [[dlpack-numpy-protocol-gotchas]], [[rstsr-tensor-extraction-quirks]].
