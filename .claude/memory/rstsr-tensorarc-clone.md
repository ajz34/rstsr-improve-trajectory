---
name: rstsr-tensorarc-clone
description: rstsr-core gained `Clone for DataArc`/`TensorArc` (2026-10-03) — zero-copy on data, copy-on-write via Arc::make_mut; `DataArc::into_owned` now clones instead of panicking when shared.
metadata:
  type: reference
---

Added 2026-10-03 in the rstsr repo (branch `261003/rstsr-cpu-dlpack`, working tree on top of
`f179c46`) so that `rstsr-cpu-dlpack` can export views of a `TensorArc`
([[rstsr-cpu-dlpack-followups]]). Minor-semver trait impls, sanctioned by the maintainer.

- `impl<C> Clone for DataArc<C>` (`rstsr-core/src/storage/data.rs`): `Arc::clone` — zero-copy.
- `impl Clone for TensorArc<T, B, D>` (`rstsr-core/src/tensor/ownership_conversion.rs`, beside the
  `Tensor`/`TensorCow` impls): storage clone (Arc bump) + layout clone — zero-copy on **data**,
  layout copied.
- **Soundness**: the only safe mutation path is `DataMutAPI::raw_mut` = `Arc::make_mut`, which
  copies a shared buffer before handing out a `&mut` — clones cannot alias mutably through safe
  code. `DataForceMutAPI::force_mut` (unsafe) keeps its caller-uniqueness contract. External
  `Arc` handles were already obtainable via the public `From<Arc<C>>`, so sharing was never a
  type-level guarantee.
- **Related fix**: `DataArc::into_owned` was `Arc::try_unwrap(..).ok().unwrap()` → panicked when
  the buffer was shared (reachable through `into_owned_keep_layout` / `into_cow`, both documented
  as "moved if possible, or fully cloned"); it now falls back to cloning the data.
- Verified by: `storage::data` unit tests (zero-copy clone, COW detach, shared `into_owned`
  fallback), a `TensorArc`-clone doctest in `ownership_conversion.rs`, and the bridge's
  `tensor_arc_view_export_shares_core_buffer_and_detaches_on_write` integration test (all green,
  also under miri).
