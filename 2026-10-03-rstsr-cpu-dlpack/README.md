# `rstsr-cpu-dlpack` — design + prototype

- **Date**: 2026-10-03
- **Task**: build the pure `rstsr-cpu-dlpack` crate (DLPack ⇄ rstsr tensors, CPU devices, no pyo3)
  that `DECISIONS-R2.md` selected as the rstsr↔NumPy interchange deliverable — design first, then a
  prototype in this directory; the crate moves to the rstsr workspace later under the normal review
  policy (patch + porting notes only from here).
- **Status**: design written ([DESIGN.md](./DESIGN.md)); prototype in progress under `prototype/`.
- **Inputs**: [`../2026-10-03-rust-numpy-review/DECISIONS-R2.md`](../2026-10-03-rust-numpy-review/DECISIONS-R2.md)
  (scope, assumptions A1–A4) and `RESPONSE-discussion-R1.md` (verifications V1–V5).
- **Pinned sources**: rstsr `f179c46` (v0.9.0); `dlpack-ffi` 1.3.0; numpy v2.5.2; DLPack v1.3.

## Files

| File | What it is |
|---|---|
| [DESIGN.md](./DESIGN.md) | the design: scope, module layout, API surface, the `DataDlpack` repr and its safety contract, flags/ownership policy, dtype and device tables, validation checklist, error taxonomy, Python boundary, test plan, open questions |
| `prototype/` | the working prototype: the crate, a `cdylib` demo host, and the Python end-to-end suite |
| `PORTING-NOTES.md` | what changes when the crate moves into the rstsr workspace (written at handoff) |
| `rstsr-cpu-dlpack.patch` | the crate as a patch against the rstsr workspace (written at handoff) |

## Headline design points

1. **One repr serves both directions**: `DataDlpack<C, O>` = a fabricated `Vec<T>` span plus a
   keep-alive owner (`O`). Import uses a foreign owner (pointer + deleter, called exactly once on
   drop); the export share uses an `Arc` owner.
2. **rstsr's `DataArc` is not cloneable** (no `Clone`, no inner-`Arc` accessor —
   `storage/data.rs:34,257-268`), so a `TensorArc` cannot be duplicated; shared/zero-copy export
   therefore goes through the bridge crate's own `Arc` owner. Owning rstsr-side sharing has no
   supported spelling today.
3. **No mutable zero-copy in v1**, for a concrete reason: `DataMutAPI::raw_mut` exposes
   `&mut Vec<T>`, and a fabricated span must never be resized. A bounded mutable accessor would have
   to exist in rstsr first (open question Q3).
4. **Export flags**: move/copy exports set `IS_COPIED` (sole ownership transfers; NumPy shows a
   writeable array); shared exports set `READ_ONLY`. This deliberately reads A1's "read-only" as
   being about aliased zero-copy views (open question Q1).
5. **Import is read-only and owns the producer's lifetime** — the deleter travels inside the tensor,
   so dropping the last Rust handle releases the NumPy buffer; imported tensors are `!Send`/`!Sync`
   by default (a NumPy deleter is `Py_DECREF`, GIL-bound).
