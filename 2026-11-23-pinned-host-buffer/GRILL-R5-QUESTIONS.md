# GRILL R5 — how tensor-level methods attach from `rstsr-cuda`

## Agent-authored background: answers received on screen (transcribed)

- **Q20**: okay. **Q21**: okay. **Q23**: okay.
- **Q22**: "if in future some other GPU devices are to be added, is it better
  to add to a core trait? Otherwise I prefer usual functions instead of trait,
  unless trait generic is necessary."

### Fact (agent-sourced) — this corrects a detail in R3/R4

- F17. Rust forbids inherent `impl` blocks on a type defined in another crate
  (E0116). `TensorAny<R, T, B, D>` is defined in `rstsr-core`, so
  **`rstsr-cuda` cannot write
  `impl<R, T, D> TensorAny<R, T, DeviceCudaPinned, D> { fn host_view(..) }`**
  — not even specialised to its own device type. The same holds for
  `Tensor::synchronize()` on `TensorAny<R, T, DeviceCuda, D>`. The only forms
  available from `rstsr-cuda` are: (a) a *local extension trait* implemented
  for the foreign type (allowed: the trait is local), giving method syntax
  after import; (b) free functions taking `&TensorAny<..>`; (c) a trait in
  `rstsr-core` (blanket-impl'd for `TensorAny`, like `TensorDeviceChangeAPI`).
- F18. House style: every tensor-level capability in core is a trait with a
  blanket impl for `TensorAny`/`&TensorAny`/`&mut TensorAny`
  (`TensorViewAPI`, `TensorDeviceChangeAPI`, ...). `rstsr-cuda` today only
  implements *core* traits for `DeviceCuda`; it has no tensor-level extension
  trait yet, so these would be the first two.

---

❓ **Q24** - **`host_view` / `host_view_mut` must also be a trait (F17)**: the
R1/R3 "inherent method on `TensorAny<.., DeviceCudaPinned, ..>`" is not
legal from `rstsr-cuda`. Proposed: local extension trait

```rust
pub trait TensorPinnedHostViewAPI<'a, T, D: DimAPI> {
    fn host_view(&'a self) -> TensorView<'a, T, DeviceCpuSerial, D>;
    fn host_view_mut(&'a mut self) -> TensorViewMut<'a, T, DeviceCpuSerial, D>;
    fn host_view_on<B>(&'a self, device: &B) -> TensorView<'a, T, B, D> where B: DeviceAPI<T, Raw = Vec<T>>;
}
impl<'a, R, T: Copy, D> TensorPinnedHostViewAPI<'a, T, D> for TensorAny<R, T, DeviceCudaPinned, D>
    where R: DataAPI<Data = PinnedBuffer<T>> { .. }
```

exported from the crate prelude, so `t.host_view()` reads as a method.

➡️ Yes.

---

❓ **Q25** - **`synchronize` form (Q22)**: not a core trait now (one async
backend; core has no notion of readiness), and not an inherent method (F17).
Between a local extension trait and free functions: the extension trait keeps
`tensor.synchronize()` / `device.synchronize()` method syntax, mirrors
`DeviceCuda::synchronize()` that already exists, and needs no new naming
namespace; free functions would be the only non-method entry points in the
crate.

```rust
pub trait CudaSyncAPI { fn synchronize_f(&self) -> Result<()>; fn synchronize(&self) { .. } }
impl CudaSyncAPI for DeviceCuda { /* exec stream */ }
impl CudaSyncAPI for DeviceCudaPinned { /* no-op: nothing pending on the device itself */ }
impl<R, T, D> CudaSyncAPI for TensorAny<R, T, DeviceCuda, D> { /* exec stream, then buffer alloc stream */ }
impl<R, T, D> CudaSyncAPI for TensorAny<R, T, DeviceCudaPinned, D> { /* buffer event */ }
```

When a second GPU backend lands, the trait moves to `rstsr-core` next to
`TensorDeviceChangeAPI` (additive, non-breaking for callers already importing
it from the prelude).

➡️ Yes: local extension trait, promote to core later.
