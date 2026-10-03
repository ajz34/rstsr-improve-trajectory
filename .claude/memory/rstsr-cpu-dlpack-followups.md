---
name: rstsr-cpu-dlpack-followups
description: Post-review outcome for rstsr-cpu-dlpack (2026-10-03) — zero-copy adoption rejected (too unsafe); basic-indexed view export implemented (with core Clone for DataArc/TensorArc); decision log.
metadata:
  type: project
---

The two post-implementation review threads for `rstsr-cpu-dlpack` are **closed** (2026-10-03);
full records in `2026-10-03-rstsr-cpu-dlpack/FOLLOWUPS.md`.

- **Zero-copy adoption** (`from_dlpack_*_adopt_f` via `Vec::from_raw_parts`) — **rejected by the
  maintainer** ("decided not implement, at least currently; this is too unsafe"); no code. The
  four unverifiable facts (base vs interior pointer, capacity, allocator identity, deleter
  semantics) stay in FOLLOWUPS.md §1 as the rationale.
- **View export** — **implemented**: `to_dlpack_shared_view[_f](base, view)` (same-root `ptr::eq`
  check, layout validated through `new_f`, owner cloned into the export, `READ_ONLY`, repeatable;
  `data = base + view offset`), with the `DlpackSharedBaseAPI<T>` seam implemented for
  `TensorDlpackShared` and `TensorArc`. Deliberately **not** implemented for `TensorDlpack`
  (foreign import): its owner cannot be cloned (double deleter) — foreign-buffer views stay
  copy-only (`to_dlpack_copy`).
- Enabled by core additions — see [[rstsr-tensorarc-clone]] (`Clone for DataArc`/`TensorArc`,
  zero-copy on data, COW via `Arc::make_mut`; plus the `DataArc::into_owned` shared-buffer fix).
- Q1 flags stay advisory ([[dlpack-numpy-protocol-gotchas]]); no other threads remain open.

Context: [[rstsr-cpu-dlpack-implementation]], [[rstsr-numpy-interop-review]].
