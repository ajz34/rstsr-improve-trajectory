# `rstsr-cpu-dlpack` — design + implementation

- **Date**: 2026-10-03
- **Task**: build the pure `rstsr-cpu-dlpack` crate (DLPack ⇄ rstsr tensors, CPU devices, no pyo3)
  that `DECISIONS-R2.md` selected as the rstsr↔NumPy interchange deliverable.
- **Status**: design written ([DESIGN.md](./DESIGN.md)); **crate implemented and green in the rstsr
  repo on branch `261003/rstsr-cpu-dlpack`** (`crates-interop/rstsr-cpu-dlpack`) — the maintainer
  redirected implementation into the rstsr repo ("why not directly implement in rstsr repo? Just
  create a new branch on that"). The Python end-to-end harness stays here, under `prototype/`.
  The crate is committed in the rstsr repo as `49a9c45`; the view-export change of
  [FOLLOWUPS.md](./FOLLOWUPS.md) §2 is the branch's current working tree. Both post-implementation
  review threads are closed there: zero-copy adoption was **rejected** ("too unsafe") and the
  basic-indexed view export was **implemented** (plus the core `Clone for DataArc`/`TensorArc`
  additions it needed).
- **Inputs**: [`../2026-10-03-rust-numpy-review/DECISIONS-R2.md`](../2026-10-03-rust-numpy-review/DECISIONS-R2.md)
  (scope, assumptions A1–A4) and `RESPONSE-discussion-R1.md` (verifications V1–V5).
- **Pinned sources**: rstsr `f179c46` (v0.9.0); `dlpack-ffi` 1.3.0 (crates.io, published 2026-10-03);
  numpy v2.5.2; DLPack v1.3.

## Specimens

| Where | What |
|---|---|
| `../../../rstsr/crates-interop/rstsr-cpu-dlpack` (branch `261003/rstsr-cpu-dlpack`) | the crate: `src/` + `tests/` (18 tests) + workspace wiring in `rstsr/Cargo.toml` |
| [DESIGN.md](./DESIGN.md) | the design: API surface, the `DataDlpack` repr and its safety contract, flags/ownership policy, validation checklist, error taxonomy, test plan, open questions, prototype results |
| [FOLLOWUPS.md](./FOLLOWUPS.md) | post-review threads, now closed: zero-copy adoption **rejected** (too unsafe), shared-view export **implemented** (design + what shipped + core additions), decision log |
| `prototype/` | the harness: `demo-ffi/` (cdylib host with a Rust-written `PyCapsule` destructor), `python/` (reference capsule holder + end-to-end suite + captured output), `rstsr-cpu-dlpack/` (archival snapshot of the crate at move time) |
| `prototype/README.md` | how to run the Rust tests and the Python end-to-end suite |

## Verified end-to-end (details in DESIGN.md §12)

- `cargo test -p rstsr-cpu-dlpack`: 18 tests + doctest green; `cargo +nightly miri test` green
  (no UB, no leaks) — the check that matters for the fabricated-span repr and deleter discipline.
- Python 3.13 + NumPy 2.5.1, real `PyCapsule`s: 8 cases / 64 checks green — zero-copy shared
  export (read-only, repeatable), move export with a NumPy→Rust mutation round trip,
  `__dlpack__(copy=True)` deep copy, imports of strided/reversed/transposed/f32/i64/i32 arrays with
  pointer equality, producer lifetime (weakref dies when the import is freed), legacy capsule
  import, seven malignant-tensor rejections, 2000 unconsumed capsules freed by the Rust destructor.

## Headline design points

1. **One repr serves both directions**: `DataDlpack<C, O>` = a fabricated `Vec<T>` span plus a
   keep-alive owner (`O`). Import uses a foreign owner (pointer + deleter, called exactly once on
   drop); the export share uses an `Arc<Vec<T>>` owner.
2. **rstsr's `DataArc` is not cloneable** (no `Clone`, no inner-`Arc` accessor —
   `storage/data.rs:34,257-268`), so a `TensorArc` cannot be duplicated; shared/zero-copy export
   therefore goes through the bridge crate's own `Arc` owner. Owning rstsr-side sharing has no
   supported spelling today.
3. **No mutable zero-copy in v1**: `DataMutAPI::raw_mut` exposes `&mut Vec<T>`, and a fabricated
   span must never be resized — a bounded mutable accessor would have to exist in rstsr first.
4. **Export flags**: move/copy exports set `IS_COPIED` (sole ownership transfers; NumPy shows a
   writeable array); shared exports set `READ_ONLY` (the design's open question Q1).
5. **Import is read-only and owns the producer's lifetime** — the deleter travels inside the tensor,
   so dropping the last Rust handle releases the NumPy buffer; imported tensors are `!Send`/`!Sync`
   by default (a NumPy deleter is `Py_DECREF`, GIL-bound).
