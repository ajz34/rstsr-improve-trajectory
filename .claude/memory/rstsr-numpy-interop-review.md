---
name: rstsr-numpy-interop-review
description: 2026-10-03 consensus review of rstsr↔NumPy interop (rust-numpy code literature review; DLPack as the likely mechanism); decisions pending in 2026-10-03-rust-numpy-review/QUESTIONS.md.
metadata:
  type: project
---

On 2026-10-03 a full review of rust-numpy (v0.29.0+8 `da6bf5be`), NumPy 2.5 DLPack support,
DLPack v1.3 and the rstsr-side gaps was written to `2026-10-03-rust-numpy-review/`
(REVIEW.md + QUESTIONS.md + notes/ + experiments/). Status: **awaiting maintainer consensus
on Q1–Q4** before any design or implementation work; no repository outside the task dir was
touched.

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
  non-owning-`Vec` pattern used by `asarray`); *owning* a foreign allocation is documented UB
  and would need a new storage variant.

Links: [[dlpack-numpy-protocol-gotchas]], [[rstsr-tensor-extraction-quirks]].
