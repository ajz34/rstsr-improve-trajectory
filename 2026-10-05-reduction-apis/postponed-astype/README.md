# Postponed: tensor astype / into_astype (PR1 review outcome)

Owner review of PR1 (2026-10-05) postponed the casting implementation: the
macro-enumerated cross-type impl matrix (`impl_tensor_cast!`, 183 pairs
mirroring `rstsr-dtype-traits/src/promotion.rs`'s `DTypeCastAPI` table) was
judged too heavy for now. Design stays as decided (grill D6):

- `astype[_f](&x) -> TensorCow<'a, TOut, B, D>`: view iff `T == TOut`
  (blanket identity impl, numpy `astype(copy=False)` semantics), else
  contiguous convert-copy via `OpAssignArbitaryAPI` + `DTypeCastAPI`.
- `into_astype[_f]`: consuming variant (`for<'x> TensorCastAPI` HRTB dispatch).
- Free-fn partial turbofish is impossible (5 generics) - the METHOD forms
  `x.astype::<f64>()` / `x.into_astype::<f64>()` are the ergonomic spellings.

## Reapply

1. Restore the two files: `dtype_conversion.rs` ->
   `rstsr-core/src/tensor/dtype_conversion.rs`, `test_astype.rs` ->
   `rstsr-core/tests/core_func/reduction/test_astype.rs`.
2. Apply `tracked-edits.patch` (module wiring, prelude, parity-table row,
   test mod entry, numpy_differences entry). NOTE: the patch predates other
   PR1 edits in the same files - re-apply by hand if it conflicts.
3. Re-add the two astype lines in the `test_with_args_and_dtype_faer` inline
   test (`rstsr-core/src/tensor/reduction.rs`).

Alternative designs to consider before reviving (owner hint: "the too much
impl_tensor_cast seems not good"): a sealed `TensorCastAPI` with the matrix
kept out of public API; or dispatch through a single generic `From`-style
fn with runtime dtype enums once rstsr has runtime dtype objects.
