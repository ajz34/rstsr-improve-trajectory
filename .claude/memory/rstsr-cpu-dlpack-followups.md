---
name: rstsr-cpu-dlpack-followups
description: Post-review open threads for rstsr-cpu-dlpack (2026-10-03) — unsafe zero-copy adoption (decision pending) and shared-view export design (deferred), with the decision log and where to resume.
metadata:
  type: project
---

Follow-ups from the 2026-10-03 post-implementation review conversation are recorded in
`2026-10-03-rstsr-cpu-dlpack/FOLLOWUPS.md`; **nothing implemented yet**.

- **Zero-copy adoption** (`TensorDlpack → Tensor` via `Vec::from_raw_parts`, unsafe): blocked by
  base-pointer / capacity / allocator-identity / deleter-semantics being unverifiable from the
  DLPack struct; `IS_COPIED` is the only runtime gate. Options: (a) unsafe entry only,
  (b) + adoptable export flavor, (c) document only. **Pending decision.**
- **View export** (`to_dlpack_shared_view_f(shared, view)`): design ready; everything already
  compiles (`shared.i(...)` yields a `TensorView` over the root buffer). Same-root `ptr::eq` check
  + `new_f` bounds validation + Arc-clone keepalive + `READ_ONLY`; repeatable, no `unsafe`.
  **Deferred** until other aspects are handled. Residual: `TensorArc` bases need a core `DataArc`
  clone / arc accessor first.
- Dropped: unsafe owner-less borrow export. Flags stay advisory (`READ_ONLY` ignored by torch;
  NumPy keys writeability off `READ_ONLY` alone) — see [[dlpack-numpy-protocol-gotchas]].

Context: [[rstsr-cpu-dlpack-implementation]], [[rstsr-numpy-interop-review]].
