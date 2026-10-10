# PLAN — `tensordot` in rstsr-core (CPU-serial + rayon-auto-impl)

Status: **design settled (grilling R1–R3 closed); implementation not started.**
Awaiting owner GO. Base: rstsr `#133` `d7056ba`, branch `261010/tensordot`
(workspace `rstsr-local-workspace`). Core only — the TBLIS plugin is untouched.

## 1. Objective & acceptance

Implement `rt::tensordot` (array-API `tensordot(x1, x2, /, *, axes=2)`) with a
naive but correct CPU-serial device kernel and a hand-written rayon twin; add a
view-only GEMM fast path at the device layer. Performance is explicitly *not* the
bar; **array-API / NumPy parity** is.

Acceptance (`2026-10-10-arrayapi-linalg/CHECKLIST.md` §R3):
`test_has_names[linear_algebra-tensordot]`,
`test_signatures.py::test_func_signature[tensordot]`,
`test_linalg.py::test_tensordot`, the `linalg-*` name node — plus rstsr's own
NumPy-parity tests, a `doc_draft` test, and the docstring.

## 2. Settled decisions (R1–R3)

| # | Decision |
|---|---|
| scope | hybrid **L0 (naive device op) + L1 (view-only GEMM)**, L2 staged/blocked deferred |
| GEMM location | **device layer** — the device op decides GEMM-vs-naive; tensor layer = thin wrapper |
| copy rule | **never materialize a copy to enable GEMM**; non-view reshape ⇒ naive |
| GEMM delegation | call the device's own `self.matmul(...)` (`DeviceMatMulAPI`) |
| axes type | reuse `AxesPairIndex<isize>` |
| axes forms | NumPy superset: `None`→2, `int n`, `(A,B)` (each int or seq); bare collection ⇒ rstsr same-axes extension |
| default | `None`→2 and `()`→2 |
| output | free-A axes (orig order) ++ free-B axes (orig order); full contraction → 0-d; **no broadcast**; **no conjugation** |
| dtype promotion | **applied in the device impl only**; tensor + device-trait layers stay dtype-agnostic |
| process | two commits (core, then shim), split by files; **no rstsr commits until manual check** |

## 3. Public surface

- free fns: `rt::tensordot(a, b, axes)`, `rt::tensordot_f`, `rt::tensordot_from(c, a, b, axes)`,
  `rt::tensordot_from_f` (vecdot-parity `_from` family; skip `_with_output`).
- methods on `TensorAny`: `a.tensordot(b, axes)` / `_f`; `c.tensordot_from(a, b, axes)` / `_f`.
- inputs `impl TensorViewAPI<Type = T, Backend = B, Dim = D>`; `axes: impl TryInto<AxesPairIndex<isize>, Error: Into<Error>>`.
- exports: add names to `rstsr-core/src/prelude.rs` linalg block (`rstsr_funcs`); export
  `DeviceTensordotAPI` through `operators/exports` (device-implementor surface, like vecdot).

## 4. Files (create / edit)

Create:
- `rstsr-core/src/operators/linalg.rs` — add `pub trait DeviceTensordotAPI<…>`.
- `rstsr-core/src/device_cpu_serial/linalg/tensordot.rs` — impl for `DeviceCpuSerial`.
- `rstsr-native-impl/src/cpu_serial/tensordot.rs` — naive kernel.
- `rstsr-native-impl/src/cpu_rayon/tensordot.rs` — rayon twin.
- `rstsr-core/src/feature_rayon/auto_impl/tensordot.rs` — `impl … for DeviceRayonAutoImpl`.
- `rstsr-core/src/tensor/linalg/tensordot.rs` — tensor layer.

Register (edit): `device_cpu_serial/linalg/mod.rs`, `feature_rayon/auto_impl/mod.rs`,
`tensor/linalg/mod.rs`, `rstsr-native-impl/src/{cpu_serial,cpu_rayon}/mod.rs`,
`rstsr-native-impl/src/prelude_dev.rs`, `rstsr-core/src/prelude.rs`.

Symlink (do not copy): `device_faer/rayon_auto_impl/tensordot.rs` and the five
`crates-device/*/src/rayon_auto_impl/tensordot.rs` → `../../feature_rayon/auto_impl/tensordot.rs`.

Note: the GEMM path uses the device's `DeviceMatMulAPI`, which **every** device already
implements (BLAS/faer/naive) — so the shared `auto_impl` file needs no per-device edit.

## 5. Tensor layer (`tensor/linalg/tensordot.rs`)

Mirror `vecdot_f` (`vecdot.rs:196-268`), with two differences: **no free-axis broadcast**,
and the axes expansion for `Val`.

1. device check (`same_device`).
2. resolve `axes` (`AxesPairIndex`):
   - `None` → `Val(2)` (default).
   - `Val(n)`: require `0 ≤ n ≤ min(ndimA, ndimB)`; `axes_a = (ndimA-n..ndimA)`, `axes_b = (0..n)`
     (both ascending, paired index-by-index).
   - `Pair(a,b)`: `normalize_axes_index(a, ndimA, false, false)` / `(b, ndimB, …)`; lengths equal.
3. `(la_ctr, la_free) = a.layout().dim_split_axes(&axes_a)` (returns `(axes, rest)`); same for b.
4. assert `la_ctr.shape() == lb_ctr.shape()` (contracted sizes match).
5. output shape = `la_free.shape() ++ lb_free.shape()`; allocate fresh contiguous output in the
   device default order, via `uninit_impl`; call `device.tensordot(…)`; wrap in `Tensor::new_f`.
   (No `broadcast_layout`, unlike vecdot.)

## 6. Device-op contract

```rust
pub trait DeviceTensordotAPI<TA, TB, TC, DA, DB, DC>
where DA: DimAPI, DB: DimAPI, DC: DimAPI,
      Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn tensordot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw, lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw, la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw, lb: &Layout<DB>,
        axes_a: &[isize], axes_b: &[isize],
    ) -> Result<()>;
}
```

The tensor layer is generic over `TC` and carries **only** device-trait bounds — it never
computes the output dtype. The device impl owns the element/promotion bound (e.g.
`TA: DTypePromoteAPI<TB, Output = TC>`, or `TA: Mul<TB, Output = TC>` for the same-dtype
subset); either way the promotion decision lives **here**, not upstream.

## 7. Device impl = GEMM trigger + naive fallback

```
split: (la_ctr, la_free) = la.dim_split_axes(axes_a);  (lb_ctr, lb_free) = lb.dim_split_axes(axes_b)
K = prod(la_ctr.shape()); M = prod(la_free.shape()); N = prod(lb_free.shape())

// canonical orders (layout-only; nothing is moved):
lA_canon = Layout(free_a.shape ++ ctr_a.shape, free_a.stride ++ ctr_a.stride, off)   // [M…, K…]
lB_canon = Layout(ctr_b.shape ++ free_b.shape, ctr_b.stride ++ free_b.stride, off)   // [K…, N…]
a2 = layout_reshapeable(&lA_canon, &[M, K], order)   // Ok(Some) ⇒ copy-free view
b2 = layout_reshapeable(&lB_canon, &[K, N], order)
c2 = layout_reshapeable(&lc,       &[M, N], order)

if let (Some(a2), Some(b2), Some(c2)) = (a2, b2, c2) {
    return self.matmul_uninit(c, &c2, a, &a2, b, &b2, one);   // real GEMM on BLAS/faer
}
naive fallback (below)
```

- `layout_reshapeable` (`rstsr-common/src/layout/reshape.rs`) is the copy-free-reshape test
  (NumPy `_attempt_nocopy_reshape`); it returns `Ok(None)` when data would be copied.
- `k ≥ 2`: merging the contracted axes into a single `K` is only valid when they are
  mergeable; `layout_reshapeable` enforces that automatically (returns `None` otherwise) ⇒
  naive fallback. Staged one-axis-at-a-time GEMM is **L2, deferred**.
- `axes = 0` (outer): `k=0`, `K=1`; the same code path gives a `(M,1)×(1,N)` GEMM when the
  inputs are contiguous — no special case.

## 8. Naive kernel + rayon twin

- serial (`cpu_serial/tensordot.rs`): split contracted/free; iterate the output (free)
  multi-index; sequential reduction over the contracted multi-index (single general loop,
  any `k`, no staging); no `ext_conj`.
- rayon (`cpu_rayon/tensordot.rs`): copy `vecdot_naive_cpu_rayon` — `PARALLEL_SWITCH = 512`
  serial fallback + `Layout`-dispatch parallel loop over output elements + `AtomicPtr`-hoisted
  output pointer.
- bounds on the device impl: `TA: Clone + ExtNum, TB: Clone, TC: Clone + Zero` (+`Send+Sync`
  in the rayon twin); no conjugation.

## 9. Tests & docs

- `tests/core_func/linalg/test_tensordot.rs` — NumPy-parity: `axes=0/1/2`, `(A,B)` pairs
  (incl. negative axes), paired multi-axis, 0-d output, mismatched-sizes error,
  out-of-range / duplicate errors, dtype cases. Translated per skill `core-test`.
- `tests/doc_draft/linalg/test_tensordot.rs` — doc examples.
- docstring — Overloads Table + row/col-major notice (skill `api-doc-conventions`).
- conformance: the four §R3 nodes, run via skill `rstsr-faer-py-tests` after the shim commit.

## 10. Commits

1. `rstsr-core`: op trait + serial/rayon kernels + auto_impl + symlinks + tensor layer +
   prelude + tests + docs.
2. `rstsr-faer-py` shim: `crates-interop/rstsr-faer-py/src/linalg.rs` + `python/rstsr_faer/api.py`
   (expose top-level `tensordot`, add to `_LinalgNamespace`).

Both held for manual check; **no push to rstsr**.

## 11. Open confirmations

- Promotion mechanism in the device impl: full `DTypePromoteAPI` lattice (mixed dtypes) vs
  same-dtype `TA: Mul<TB, Output=TC>` now with the lattice later. (Array-API `tensordot` value
  tests are same-dtype; mixed-dtype is the separate `arrayapi-linalg` R2/G-009 item.)
- `ExtNum` vs a weaker `Zero + Mul + Add` bound (no conjugation needed).
- `PARALLEL_SWITCH = 512` (inherited from vecdot) — tune/drop as measurement dictates.
