# Postponed: `norm` family -> future `rstsr-sci-traits`

Owner decision (2026-10-05, end of PR2 review): the `norm` family does not
belong in rstsr-core; it should be implemented in `rstsr-sci-traits` instead.
PR2 was trimmed to custom reduce only. This directory preserves the full
working prototype as reference for that future wave.

## What is here

- `pr2-exploration.diff` - the complete uncommitted PR2 diff against
  rstsr main f736987 at snapshot time: custom reduce (which LANDED in PR2)
  PLUS the whole norm implementation (which was reverted). Norm parts:
  `NormOrd` region + `norm` rt region + module docs in
  `rstsr-core/src/tensor/reduction.rs`, `OpNormAPI` in
  `rstsr-core/src/operators/reduction.rs`, device impls (serial + rayon
  symlink), norm exports in `prelude.rs`, faer runtime test, docs.
- `test_norm.core.rs` / `test_norm.doc_draft.rs` - the entry-binary test file
  (numpy-transcribed + custom regression, all passing at snapshot) and its
  doc_draft twin.

## Design that was settled (grilled through D7/D13/D14)

- Enforced positional `ord` on every function: `norm(x, ord)`,
  `norm_axes(x, ord, axes)`, `norm_with_args(x, ord, args: impl
  Into<ReduceArgs>)`; no `NormArgs` struct (owner rejected it; reuse
  `ReduceArgs`).
- `NormOrd` enum (One/Two/Inf/NegInf/Zero/NegOne/NegTwo/P(f64)/Fro) with
  `From` f64/f32/ints and `TryFrom<&str>`/`<String>` for
  `"l1"/"L1"/"l2"/"L2"/"fro"`; `nuc` EXCLUDED (owner: exclude rather than
  raise UnImplemented; SVD-dependent).
- NumPy vector/matrix convention (owner retracted an attempt to flatten):
  1 axis = vector, 2 axes = matrix with axis-tuple order significant for
  ±1/±Inf (sum axis[0]/axis[1] then max/min the other), >2 axes InvalidValue,
  fro matrix-only (single-axis fro InvalidValue), vector ords with 2-tuple
  axis InvalidValue. Matrix 2/-2 raise UnImplemented (SVD).
- TOut = T::Real (complex -> real). Zero-size: 2-norm/inf-norm of empty = 0;
  matrix empty max stage = 0 (NumPy `max(initial=0)`); empty NegInf = inf
  (divergence: NumPy raises).
- NumPy `ord=None` mapping (rank-dependent): 2-D -> `"fro"`, 1-D -> `2.0`,
  n-D raveled -> `l2_norm`.

## Implementation learnings (likely reusable in sci-traits)

- rustc E0391: bounds like `B: OpMaxAPI<B::TOut, IxD>` (trait WITH an
  associated type, argument = projection of the same self type) cycle bound
  elaboration; routing stage 2 back through `OpNormAPI` with `T = B::TOut`
  hits the same wall. Workaround: dedicated device method
  `OpNormAPI::norm_matrix_cmp_axes(raw, la, sum_axis, cmp_axis, is_max)`
  owning the two-stage, axis adjustment device-side.
- `feature_rayon/auto_impl/reduction.rs` is symlinked into
  `device_faer/rayon_auto_impl/` - one source, two cfg contexts; rayon
  kernels take `Option<&ThreadPool>`.
- clippy redundant_closure suggests `R::<T>::zero` fn items that do NOT
  coerce to `Fn()` under generic associated-type projections - bind closures
  to `let` per match arm.
- `ComplexFloat::Real: Float` is sealed via num's blanket impls; qualify
  `Float::is_nan/recip/powf` at call sites (both traits in scope -> ambiguity).
