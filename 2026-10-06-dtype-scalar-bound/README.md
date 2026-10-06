# Dtype scalar bound: replacing `num::Num` with an explicit marker (`DTypeScalarAPI`)

**Status:** family-wide rollout **implemented** on rstsr branch
`261006/dtype-scalar-bound` (commit 2aa1c55, 2026-10-06, maintainer-approved);
based on main after PR #114 (`rt::where`) merged. Remaining open questions
3/5/6 below stay open.

**Context:** rstsr branch `261006/rt-where` (`rt::where` element-wise select).
The code review flagged that the scalar overloads bound `TY: num::Num`
(`op_where.rs`), contradicting the earlier design note ("no arithmetic bound;
Clone-level; bool/ints/floats/complex/halfs"). Maintainer initially deferred
("if too difficult, ignore"), then requested an experimental fix to see the
shape of the change.

## Problem analysis

`num::Num` on scalar overloads is **not an arithmetic requirement** - no
`Num` method is ever called on the scalar. Its real role is a **dispatch
discriminator**: scalar overloads (`TensorOpAPI<TB> for &TensorAny`) must not
unify when a tensor reference is passed (`TY := &TensorAny`), or every
tensor-tensor call becomes E0283-ambiguous. Stable Rust has no negative
bounds / specialization, so the discriminator must be a trait that tensor
wrapper types cannot implement.

Candidate replacements that **do not work**:

- `TY: DTypePromoteAPI<TY>` (self-promotion) - defeated by the reflexive
  blanket `impl<T> DTypePromoteAPI<T> for T` (`rstsr-dtype-traits/src/promotion.rs:41`);
  tensor references get it for free.
- `DTypeCastAPI` - same reflexive blanket (`promotion.rs:56`).
- `ExtNum` - no bool impl (same gap as `num::Num`).
- `DeviceAPI<T>` membership - `impl<T> DeviceAPI<T> for DeviceCpuSerial {}`
  is an unconstrained blanket; gates nothing.

So a new **explicit marker** is required: implemented per dtype, never via a
blanket.

## Experiment (committed on `261006/rt-where`)

- New `rstsr-dtype-traits/src/scalar.rs`: `DTypeScalarAPI:
  Sized + Send + Sync + Clone + 'static`, explicit impls for the
  promotion-matrix dtype set (bool, u8-u64/usize, i8-i64/isize, f32/f64,
  Complex&lt;f32&gt;/Complex&lt;f64&gt;, feature-gated f16/bf16).
- Wired through `rstsr-dtype-traits/src/lib.rs` and the `prelude_dev`
  whitelist (dtype traits are re-exported by name, not glob).
- `op_where.rs`: `num::Num` -> `DTypeScalarAPI` on both scalar overloads.
- Tests: `rt::r#where(&c, &x_bool, false)` / `rt::r#where(&c, true, &y_bool)`
  (previously E0277; NumPy accepts bool scalars).

Verification: 353/353 entry_row_cpu, 353/353 with rayon, doctests 197/0,
clippy clean, fmt clean, col_major compiles. No dispatch ambiguity anywhere -
the marker prunes the scalar candidate exactly where `num::Num` did.

## Rollout evaluation (analysis only, not implemented)

Surveyed every `num::Num` site in rstsr-core (4 files). The device kernels
only ever promote (or `From`-convert) and then operate on the resulting type,
so the per-op benefit is whether the promoted-type traits exist for
bool-involved combinations:

| Site | Ops | Bool scalar outcome after swap | Verdict |
|---|---|---|---|
| `op_binary_common.rs` (4 impls) | comparisons (`equal`/`not_equal`/`greater`/`greater_equal`/`less`/`less_equal`) | works both ways (`bool: PartialEq/PartialOrd`; promote bool x numeric -> numeric) | **clear benefit** |
| same | `maximum`/`minimum` | vs numeric works (`ExtReal` on promoted int/float); bool x bool fails (`ExtReal` has no bool impl) | **benefit** (gap noted below) |
| same | `atan2`/`copysign`/`hypot`/`nextafter`/`log_add_exp`/`floor_divide` | works vs float tensors (`DTypeIntoFloatAPI`/`ExtReal` on promoted float); bool x bool fails | partial, harmless |
| same | `pow` | device impl bypasses promotion (`TA: Pow<TB>`); no `Pow<bool>` exists anywhere - only the error message changes (and gets slightly less direct) | no benefit, cannot opt out (shared duplicate impl) |
| `op_binary_arithmetic.rs` (4 impls) | `add`/`sub`/`mul`/`div` etc., scalar path `T: From<TB>` | works: std has `From<bool>` for all integer types **and** f32/f64 -> `rt::add(&i32_t, true)` == `arr + 1` (NumPy weak-scalar result); bool x bool correctly stays rejected (no `Add<bool>`) | **benefit** |
| `op_binary_assign.rs` (1 impl) | `*_assign(&mut t, scalar)` | never unlocks: direct `TA: Op<TB>` (std ops), no `From` conversion, no `AddAssign<bool>`; today's `num::Num` rejection is the cleaner error | keep `num::Num` |
| `device_faer/matmul_impl.rs` | matmul kernels | BLAS domain, out of elementwise scope | skip |

NumPy-consistency notes: `np.greater(mask, False)`, `np.maximum(mask, 0)`,
`np.add(arr, True)` are all idiomatic NumPy and all unlock under the swap.
NumPy's `True + True == 2` (bool x bool arithmetic promoting to int) would
need a promotion-table change (bool x bool -> int for arithmetic ops only) -
a deeper design question, out of scope for a bound swap; rstsr's strong
typing rejects bool x bool arithmetic today and would continue to.

## Open questions for the future discussion

1. Adopt the rollout for the 8 scalar impls in `op_binary_common.rs` +
   `op_binary_arithmetic.rs`? (Mechanical; pure widening - every `num::Num`
   type is `DTypeScalarAPI`, so no existing code changes behavior.)
2. Keep `num::Num` on `op_binary_assign.rs` (recommended: no unlock, better
   errors) or swap for one-uniform-bound?
3. Add `ExtReal for bool` (would enable `maximum(mask, mask)`-shaped calls;
   `ext_max`/`ext_min`/`ext_floor_divide` need defining for bool first)?
4. Accept the `pow` error-message regression that comes with the shared
   duplicate impl block?
5. When `clip` is implemented (planned sibling of `where`): bound its scalar
   min/max on `DTypeScalarAPI` from the start.
6. Long term: NumPy-weak-scalar bool x bool -> int arithmetic is impossible
   under the current promotion table; decide whether that divergence is
   acceptable (it is consistent with rstsr's strong-typing house rule).

## Outcome (implementation, 2026-10-06)

Questions 1 and 2 resolved as recommended: the 8 generic scalar impls
(`op_binary_common.rs` 4 + `op_binary_arithmetic.rs` 4) swapped to
`DTypeScalarAPI`; `op_binary_assign.rs` keeps `num::Num`. rstsr commit 2aa1c55.

Implementation notes discovered while doing it:

- The marker had to be widened to stay a strict superset of `num::Num`:
  `i128`/`u128` and `Complex<f16>`/`Complex<bf16>` added (the latter two hold
  `num::Num` only because rstsr's `half` dependency enables its `num-traits`
  feature). Without them, e.g. `add(&i128_tensor, 3_i128)` (reflexive
  `From<i128> for i128`) would have regressed.
- `f64: From<i128>` does **not** exist in std - lossless `From` conversions
  stop at i32/u32 (the earlier table's "f64: From<i128>" implication was
  wrong). So wide-integer scalars only ever matter against same-width
  tensors; nothing else changes for them.
- Scalar-on-the-left arithmetic (`add(true, &i32_tensor)`, NumPy-legal) is
  **not** unlocked by this swap: those overloads are concrete per-dtype impls
  (`impl_arithmetic_scalar_lhs_all!`), not `num::Num`-generic. Widening them
  is separate work if ever wanted. Scalar-on-the-left comparisons/min-max
  (promote-based) *are* widened by the generic swap.
- rstsr's `assert_equal` cannot verify `i128` tensors (`DTypeCastAPI<f64>`
  unimplemented); i128 parity tests compare via `to_vec()`.

Tests added: `test_scalar_bool` in `custom_add`, `custom_comparison`,
`custom_maximum_minimum` (bool scalar arithmetic via `From<bool>`, bool
comparisons, bool min/max, i128 non-regression, std-operator spelling
`&a + true`). Verification: entry_row_cpu 364/364 (faer-free, rayon, and
faer-default builds), doctests 197/0, clippy/fmt clean, col_major compiles.

## Reference

- rstsr branch `261006/rt-where`: b0054c3 (feature), 06e3b58 (review fixes),
  e960119 (this marker bound, rstsr-side).
- Key files: `rstsr-dtype-traits/src/{scalar.rs,promotion.rs,ext_real.rs}`,
  `rstsr-core/src/tensor/operators/op_where.rs`,
  `rstsr-core/src/device_cpu_serial/operators/op_ternary_common.rs`
  (device kernels for the 15-op family).
