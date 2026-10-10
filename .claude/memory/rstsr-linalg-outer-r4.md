---
name: rstsr-linalg-outer-r4
description: "outer/ext_outer (array-API linalg R4): dedicated DeviceOuterAPI family, outer on all devices via feature_rayon auto_impl symlinks, ext_outer cpu_serial+faer, 1-D-only, conformance 1303/46/33"
metadata:
  type: project
---

2026-10-10, branch `261010/linalg-outer` (off post-`#137` main `f03c3ca`), not
yet pushed. First **R4** member of the array-API linalg convergence
([[rstsr-ext-linalg-r2]] is R2). Suite `1300 / 49 / 33` → **`1303 / 46 / 33`**,
0 regressions.

- **Owner decision (asked, since the checklist's R4 note said "outer = broadcast
  mul + reshape")**: implement as a *real device-op family*, not composition —
  mirroring `matmul`/`vecdot`. Also: `outer` for **all** devices, `ext_outer`
  for **cpu_serial + faer** only.
- **Shape**: `DeviceOuterAPI<TA,TB,TC>` in `operators/linalg.rs` and
  `DeviceExtOuterAPI<TA,TB,TC>` in `operators/ext_linalg.rs`. These two traits
  are **not generic over dims** (unlike the vecdot/tensordot traits): the
  signature is fixed at `Layout<Ix1>`/`Layout<Ix1>` → `Layout<Ix2>`, which
  enforces the 1-D contract at the device layer.
- **Kernel** (`rstsr-native-impl/src/cpu_{serial,rayon}/outer.rs` +
  `.../ext_linalg/outer.rs`): builds the two broadcast operand layouts by hand
  (`Layout::new([n,m], [sa,0], off)` — `Layout::new` accepts zero strides, its
  `check_strides` runs with `skip_zero = true`), then
  `translate_to_col_major(..., TensorIterOrder::K)` +
  `layout_col_major_dim_dispatch_3` (serial) /
  `layout_col_major_dim_dispatch_par_3` with an `AtomicPtr`-hoisted output
  (rayon, `PARALLEL_SWITCH = 512` on `n*m`). The ext twin swaps the inner
  product for `promote_pair`.
- **Device coverage = the family's convention, and it is the interesting bit**:
  same-dtype `outer` is implemented (a) explicitly for `DeviceCpuSerial` in
  `device_cpu_serial/linalg/outer.rs`, and (b) for **`DeviceRayonAutoImpl`** in
  `feature_rayon/auto_impl/outer.rs`. Because `DeviceRayonAutoImpl` is a
  *per-crate alias of that crate's own device* (`device_faer/device.rs:10`
  `pub(crate) use self::DeviceFaer as DeviceRayonAutoImpl;`), one shared file
  plus **one symlink per crate** (`device_faer/rayon_auto_impl/outer.rs` +
  the five BLAS crates' `src/rayon_auto_impl/outer.rs`) gives `outer` to faer
  and every BLAS device for free. `ext_outer` instead goes explicit
  (`device_cpu_serial/ext_linalg/`, `device_faer/ext_linalg/`) so the BLAS
  devices do **not** get it — the same asymmetry as ext_matmul/vecdot/tensordot.
- **Contract**: 1-D only (array-API `linalg.outer`; the suite comments "outer
  does not work on stacks"), enforced at runtime in `op_refa_refb_outer`
  (`rstsr_assert!` → `InvalidValue`) and by `to_dim::<Ix1>()`. **No
  conjugation** (unlike `vecdot`). Result is `Tensor<TC, B, Ix2>` — rank-2, so
  the shim narrows it with `into_dim::<IxD>()`.
- **The result's memory arrangement follows the input's `device.default_order()`**
  (owner's rule), not `TensorIterOrder::default()` — the latter is `K`, so a
  `match TensorIterOrder::default()` with a `_` arm silently means "always C"
  (the first cut had exactly that bug; `vecdot`'s identical-looking match is
  safe only because its `_` arm calls `get_layout_for_binary_op(..., order)`).
  Fixed in `aa9e345` with a layout test. **Generalizable:** never use
  `TensorIterOrder::default()` to pick a fresh result's contiguity.
- **NumPy provenance is `linalg/tests/test_linalg.py::TestOuter`** (the array-API
  shaped `np.linalg.outer`: 1-D only, raises for a non-vector) — the only value
  source for `outer`; `np.outer` (top-level) instead flattens, and that
  distinction now lives in `numpy_differences.md`. NumPy holds `TestOuter`'s
  checks as **class-body asserts**, so `sync_numpy.py` gained a class-body index
  (empty method name) to hash it.
- **`test_outer` DOES grade binary promotion**: `two_mutual_arrays(dtypes=
  dh.real_dtypes, …)` + `ph.assert_dtype(..., in_dtype=[x1,x2])` with no
  `expected`, which resolves to `dh.result_type(x1.dtype, x2.dtype)`. So the
  shim must route `linalg.outer` through `ext_outer`; the same-dtype entry would
  fail the first mixed draw (G-009). (This was the answer to the owner's
  pre-implementation question.)
- **Shim** (`crates-interop/rstsr-faer-py`): `op_ext_outer` + `linalg_outer`
  pyfunction via `dispatch_bin_promote_arith!`, registered in `lib.rs`; Python
  `api.py` gains the native import, a positional-only `def outer(x1, x2, /)`,
  and `_LinalgNamespace.outer`; the module prose drops `outer` from the absent
  list. `outer` is `xp.linalg`-only (not top-level), so `__all__` is unchanged.
- **Flipped (3)**: `test_linalg.py::test_outer`,
  `test_has_names[linalg-outer]`,
  `test_extension_func_signature[linalg.outer]`.
- **Gates**: 272 doctests (+2), 178 lib (+9), 592 entry-row (+6); clippy/rustdoc
  clean; `cargo check -p rstsr-openblas --all-targets` clean (this compiles the
  `core_func` test tree the BLAS crates symlink, so it is the real "does the
  BLAS column build" check) plus plain checks on the other four BLAS crates;
  lib tests re-run clean under `--no-default-features --features col_major`.

- **`outer_from` / `outer_from_f`** (follow-up, owner-requested; no
  `ext_outer_from`): the output-providing variant, mirroring `vecdot_from` —
  `c: impl TensorViewMutAPI<Dim = **Ix2**>` (the device op is pinned to
  `Layout<Ix2>`, so unlike `vecdot_from`'s generic `DC` the rank is fixed),
  broadcast gate + shape check, then `transmute::<&mut Raw<TC>, &mut
  Raw<MaybeUninit<TC>>>(c.raw_mut())`. **No `alpha`/`beta`** — `outer` has no
  accumulate form, so `c` is fully overwritten (contrast `matmul_from`). Method
  form `c.outer_from(&a, &b)` needs its own `impl<R,T,B> TensorAny<R,T,B,Ix2>`
  block (the generic `D` block can't reach a pinned `Ix2` receiver).

- **`trace` / `trace_with_dtype`** (R4, second member): pure composition,
  `diagonal` + `sum_axes`, no device op. **Default axes follow the device
  default order** (owner's rule): last two under `RowMajor`, first two under
  `ColMajor`; explicit axes are order-independent. `diagonal` appends the
  diagonal *last*, so `sum_axes(-1)` is right for any axes. Signature pairs with
  the reductions:
  `trace(x, offset) -> Tensor<B::TOut, B, IxD>` + `trace_with_dtype::<TOut>`
  (the array-API `dtype=`). `offset: impl Into<DiagonalArgs>` reuses the local
  type (`()`, `None`, int, `(offset, a1, a2)`) — no new `*Args`, no orphan
  issue. Only new bound: `B: OpSumAPI<T, D::SmallerOne>` (+ `DimSmallerOneAPI`).
  Out-of-range offset → empty diagonal → trace `0` (the `la.size()==0` branch).
  Provenance `_core/tests/test_numeric.py::TestNonarrayArgs::test_trace` (L349).
- **`matrix_power` deferred, not dropped** (owner's clarification): it is out
  of scope for **rstsr-core** only — `rstsr::linalg` (`rstsr-linalg-traits`)
  may implement it later, which is where the real `inv` lives. Reason it cannot
  be core-side: array-API tests `n` in −10..10 and `n < 0` needs the **matrix
  inverse** (`rt::inv` in rstsr-core is element-wise; `rstsr-linalg-traits` has
  the real one but *depends on* rstsr-core, so the dependency cannot point the
  other way).

See `2026-10-10-arrayapi-linalg/CHECKLIST.md` (R4). Related:
[[rstsr-ext-linalg-r2]], [[rstsr-linalg-batching-r1]],
[[arrayapi-convergence-harness]], [[rstsr-faer-py-wrapper-only]].
