---
name: rstsr-static-dim-core-serves-dyn
description: "rstsr overload cores `(TrA, TrB)` with `DA: DimAPI + DimSmallerOneAPI` also match the dynamic-dim overloads (DA = DB = IxD); `into_dim::<DA::…>()` unwraps, so a result whose rank depends on the data panics in the `*_f` entries; guard with `layout().shape().const_ndim()`."
metadata:
  type: project
---

In `rstsr-linalg-traits/src/blas_impl/*.rs` (symlinked as
`linalg_auto_impl/` in every device crate) there is no separate
dynamic-dim overload: `(TensorView<…, IxD>, TensorView<…, IxD>)` is the
same `impl … for (TrA, TrB)` with `DA = DB = IxD`, because `IxD` satisfies
`DimAPI + DimSmallerOneAPI` (`SmallerOne = IxD`).

- A rank rule that only applies to static dims must therefore test the dim
  kind at runtime: `t.layout().shape().const_ndim()` is `Some(n)` for `Ix<n>`
  and `None` for `IxD`. A plain `a.ndim() >= b.ndim()` assert in such a core
  also rejects the legitimate dynamic-dim broadcast (done once, 2026-11-25,
  caught by the new `test_eig_static_dim_b_with_more_batch_axes`).
- `into_dim::<D>()` unwraps; in the `*_f` (fallible) entries convert a data
  dependent rank problem into an `Err` before it (`check_static_generalized_ranks`
  in `ref_impl_blas.rs` is the precedent, shared by `eig.rs`/`eigvals.rs`
  through `rstsr_linalg_traits::prelude_dev::*` — files compiled inside the
  device crates cannot use `crate::` paths into linalg-traits).
- To select the static overload in a test, pass `Tensor<_, Ix2>` views and
  annotate the result type (`Result<Tensor<_, Ix1>, rt::Error>`); the dynamic
  call is `a.view().to_dyn()`. `Error` is `rt::Error` (`rstsr::prelude::rt`).
