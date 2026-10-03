---
name: rstsr-external-storage-repr
description: An out-of-crate crate can define its own rstsr storage repr (e.g. a foreign-owner buffer) using public API only — no rstsr-core change; with the exact bounds, the traps, and the faer UB-note scope.
metadata:
  type: reference
---

Verified 2026-10-03 at rstsr `f179c46` (v0.9.0) for the NumPy-interop design
(`2026-10-03-rust-numpy-review/RESPONSE-discussion-R1.md` V2).

- **Public, non-sealed surface**: `Storage<R, T, B>` + `pub fn new`
  (`rstsr-core/src/storage/device.rs:14, :81`); `DeviceRawAPI<T>::Raw` (`:9`);
  `DataAPI`/`DataCloneAPI`/`DataMutAPI`/`DataForceMutAPI` (`storage/data.rs:285-320`);
  `pub type TensorAny<R,T,B,D>` (`tensorbase.rs:98`); validated safe constructor
  `TensorAny::new_f` (`check_strides` + `bounds_index ≤ storage.len()`, `tensorbase.rs:195-208`).
- **Core impls are generic over `R`** (~80 sites), e.g. `impl<R,T,B,D> TensorViewAPI for
  TensorAny<R,T,B,D> where R: DataAPI<Data = B::Raw>` (`ownership_conversion.rs:662`, mutable twin
  `:725`); op dispatch goes through those traits (`tensor/operators/op_with_func.rs:9`). So an
  externally defined repr participates in views/manipulation/ops; the orphan rule is irrelevant
  because no new core impl is needed.
- **Every device has `type Raw = Vec<T>`** (serial, faer, openblas/mkl/blis/aocl/kml), so
  `B: DeviceAPI<T, Raw = Vec<T>>` is a complete bound today; a repr must expose `&Vec<T>` (the
  fabricated-`Vec` + `ManuallyDrop` + external-owner pattern; contract in
  `DataRef::from_manually_drop`, `storage/data.rs:87-100`).
- **Traps**: the fabricated `Vec` must never be dropped/resized (its base may be an interior
  pointer after negative-stride normalisation); do not implement `DataCloneAPI` unless a deep copy
  of the *span* is intended (it clones the whole raw buffer, cf. `DeviceCpuSerial::to_cpu_vec`);
  `DataForceMutAPI` = the mutable-view gate (opt-in, `unsafe` contract); `into_raw_f` only exists
  for `DataOwned` while the layout-driven extraction `to_raw_f` is generic over `R`.
- **The in-repo UB note is faer-specific**: `device_faer/conversion.rs:85-94` (faer over-aligns to
  64 B and pads `row_capacity`; verified in faer 0.22.6 `src/mat/matown.rs:9-13, 83-85`). The
  transferable rule: never let a *deallocating* `Vec<T>` own foreign memory — NumPy adoption is
  not affected.

Related: [[rstsr-numpy-interop-review]], [[dlpack-numpy-protocol-gotchas]].
