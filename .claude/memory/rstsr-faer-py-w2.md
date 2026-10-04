---
name: rstsr-faer-py-w2
description: W2 elementwise+dunder pass landed (901/399/82); pair-dispatch macro pattern, the ~6 min compile-time cost, and the uncommitted rstsr-side positive
metadata:
  type: project
---

W2 (elementwise surface + operator dunders, 2026-10-05) landed shim-side and
took the rstsr_faer.api red map from 320/980/82 to **901/399/82** of 1382
(stamp `20261005-015407`). Details in
`2026-10-04-rstsr-faer-py/reports/SUMMARY-w2.md`; register G-052..G-057.

**How the bindings work now** (read before touching
`crates-interop/rstsr-faer-py`): output dtypes are expressed with rstsr's own
traits (`T::FloatType`, `<T as DTypePromoteAPI<U>>::Res`), never a shim-side
table; dispatch macros live in `any_tensor.rs`
(`dispatch_t_into_float!`, `dispatch_t_no_complex!`,
`dispatch_t_real_float_same!`, `dispatch_t_numeric_same!`,
`dispatch_t_float_complex_same!`, `dispatch_bin_promote!`,
`dispatch_bin_promote_eq!`, `dispatch_bin_int_bool_self!`,
`dispatch_bin_int_self!`, `dispatch_bin_bool_self!`). Lifting a
dtype-changing result needs the `any_of` **free function**, not a closure
(`|r| r.into_any()` fails method resolution in macro arms). Mixed-dtype arms
are generated from `rstsr-dtype-traits/src/promotion.rs` and are directional:
only the (result, input) order exists (e.g. `(i16, i8)`, never `(i8, i16)`).

**Cost:** the pair-dispatch expansion takes the shim's release build from
~30 s incremental to **~6 min** (166 eq arms + 113 promote arms × op).
Batch shim changes before rebuilding; `maturin build` from the crate dir.

**rstsr-side change is uncommitted on purpose:** `rt::positive`/`positive_f`
(`rstsr-core/src/tensor/operators/op_unary_arithmetic.rs` + prelude + the
fulfillment-table rows) sits in the rstsr working tree awaiting owner review
(owner: "you can add positive function at rust-side ... you would not git
commit on rstsr"). Rust-side **fixes** found during W2 (signbit inverted,
integer-dtype preservation for ceil/floor/trunc/round/conj, pow dtypes,
complex special cases, maximum/minimum NaN, remainder semantics) are
register entries only — see [[rstsr-faer-py-wrapper-only]].
