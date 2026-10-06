---
name: rstsr-dtype-scalar-bound
description: num::Num on scalar overloads is a dispatch discriminator (not arithmetic); DTypeScalarAPI marker replaces it on rt-where; rollout verdicts per op family.
metadata:
  type: project
---

`num::Num` on rstsr scalar overloads (op_binary_common/arithmetic/assign, op_where) is a **dispatch discriminator**: it prunes the scalar impl when a tensor reference is passed (else E0283 ambiguity on every tensor-tensor call). No Num method is ever called. Self-promotion cannot discriminate: `promotion.rs:41` has a reflexive blanket `impl<T> DTypePromoteAPI<T> for T` (same for DTypeCastAPI at :56); `DeviceAPI<T> for DeviceCpuSerial` is an unconstrained blanket; `ExtNum` lacks bool. Stable Rust has no negative bounds → an explicit per-dtype marker is the only shape. Compiler-proven (2026-10-06): `impl<T: num::Num> DTypeScalarAPI for T` + `impl DTypeScalarAPI for bool` is E0119 — coherence must assume upstream (std) may add `Add/Sub/Mul/Div/Rem for bool`, which could make `bool: num::Num` provable; `negative_impls` doesn't help (same orphan rules).

Done: `DTypeScalarAPI` marker (`rstsr-dtype-traits/src/scalar.rs`, promotion-matrix dtype set incl. bool, feature-gated halves) replaces num::Num on rt::where's scalar overloads (branch 261006/rt-where, e960119); bool scalars accepted, no ambiguity, all suites green.

Rollout verdicts (details + open questions in `2026-10-06-dtype-scalar-bound/README.md`):
- benefit: comparisons (PartialEq/Ord on promoted), maximum/minimum vs numeric, arithmetic (`From<bool>` exists in std for all ints AND f32/f64 → `add(&i32_t, true)` == arr+1, NumPy-consistent)
- never unlocks: `*_assign` (direct std Op<TB>, no From; keep num::Num — cleaner errors), `pow` (`Pow<bool>` nowhere; shared duplicate impl forces it along)
- gaps: `ExtReal` has no bool (bool×bool max/min/floor_divide fail); bool×bool→int arithmetic (NumPy True+True==2) needs a promotion-table change, out of scope
- when implementing `clip`: bound scalar min/max on DTypeScalarAPI from the start. Related: [[rstsr-tensor-extraction-quirks]].
