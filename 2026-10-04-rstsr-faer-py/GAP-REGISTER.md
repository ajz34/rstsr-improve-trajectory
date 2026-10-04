# Gap register — rstsr-faer-py

Every divergence of `rstsr_faer.api` (DeviceFaer) from the Python array API
standard 2025.12, as graded by array-api-tests @ `6c0b59f`.

**Wrapper-only rule (owner directive, 2026-10-04):** the shim crate carries
*no algorithms* — neither in its Rust nor its Python layer. It marshals,
validates shapes/dtypes, and calls rstsr. Where rstsr lacks a capability,
the divergence is registered and waits for a rust-side fix; a shim-side
algorithm may be written only with the owner's explicit per-case permission.
(The mask/fancy __getitem__ gathers were briefly shim-side and were REVERTED
on the owner's ruling — G-038/G-039 record the capability requests.)

Columns:
category (`rust-side` = needs an rstsr-core/traits change, fix only on user
permission; `shim-side` = binding work; `suite` = instrument behavior),
fulfillment-table row (`rstsr-core/src/docs/array_api_standard.md`), suite
outcome, issue (filled at review points, opened only on user go).

## Format

| id | area | category | table row | evidence (suite) | note | issue |

## Entries (v0 — pre-first-run expectations)

| id | area | category | table row | evidence | note |
|---|---|---|---|---|---|
| G-001 | sort / argsort | rust-side | D | pending run | known faer-impl gap |
| G-002 | roll | rust-side | D | pending run | known gap |
| G-003 | repeat / tile | rust-side | D | pending run | known gap |
| G-004 | QR (linalg) | rust-side | D | pending run | faer impl lacks QR trait |
| G-005 | slogdet, solve_symmetric | rust-side | D | pending run | faer impl lacks; BLAS impls have |
| G-006 | `__pos__` / `__ifloordiv__` / `__ipow__` | rust-side | D | pending run | fulfillment-table D |
| G-007 | astype / tensor-level dtype cast | rust-side | ? | pending run | no astype/into_dtype API in rstsr 0.9.0; shim casts element-wise via DTypeCastAPI in Rust (lossy semantics ungraded) |
| G-008 | runtime dtype introspection / result_type / can_cast | rust-side | ? | pending run | rstsr dtype is a static type param; promotion exists only as associated types (DTypePromoteAPI), no token-level query |
| G-009 | cross-dtype arithmetic (add etc.) | rust-side | ? | pending run | tensor-tensor arithmetic impls are same-dtype only; comparisons/maximum/minimum/pow are promotion-capable in rstsr |
| G-010 | matmul (`%` / `@`) | rust-side | C(%) | pending run | table's C-status column; suite grades `xp.matmul` |
| G-011 | DLPack exchange (`__dlpack__`, from_dlpack) | shim-side | planned | S2 | PyCapsule over rt::dlpack; import copy-only |
| G-012 | nonzero / where / unique_* / searchsorted / take | rust-side? | ? | pending run | unconfirmed surface; first run decides |
| G-013 | `u64` tolist/item values > i64::MAX | shim-side | — | pending | PyScalar carrier is i64; edge ungraded by suite |
| G-014 | astype(copy=False) aliasing | shim-side | — | pending | handle model deep-copies; no shared-storage alias (TensorArc) in enum yet |
| G-015 | capabilities().max ndim | ? | ? | pending run | placeholder 8; verify what rstsr supports before S3 |
| G-016 | zeros/ones/full for bool | rust-side | ? | compile: `rt::zeros_f` tuple impls gate on `T: num::Num`; bool is not `Num` | shim assembles bool fills via `asarray(Vec<bool>)` — value-exact; a `bool`-capable creation path in rstsr-core would remove the workaround |
| G-017 | `all`/`any` on non-bool dtypes; isnan/isfinite/isinf on int/bool | rust-side | ? | compile: `OpAllAPI`/`OpAnyAPI` impl'd for bool tensors only; `TensorIsNanAPI`/`IsFiniteAPI`/`IsInfAPI` impl'd for float/complex only | spec requires all/any on every dtype (truthiness) and the predicates on every numeric dtype (constant results for ints). Shim derives truthiness via `x != 0` and emits constant arrays — value-exact |
| G-018 | element cast matrix | rust-side | ? | `DTypeCastAPI` impls cover only a subset of pairs (e.g. i8→bool/f32 yes, i8→i16 no) | shim carries its own full primitive-cast table (same `as` semantics rstsr uses; complex→real = real part; Rust float→int saturates vs numpy UB wrap — spec-undefined region) |
| G-019 | tensor-level dtype conversion | rust-side | ? | no astype/into_dtype/change_dtype anywhere in rstsr 0.9.0 | shim's astype = gather elements → cast → asarray (fresh C-contig result) |
| G-020 | `rt::empty_f` is `unsafe` | rust-side | — | signature is unsafe fn (uninitialized memory) | shim routes `xp.empty` through zeros; contents unspecified by the standard |

Compile-cycle notes (S1):
- pyo3 0.29.3 + maturin 1.15; `Py<T>: Clone` requires pyo3 feature `py-clone`.
- rstsr reductions/predicates returning scalars return `B::TOut` associated
  types; whole-array reductions return scalars, not 0-d tensors.
- `rt::equal_f` etc. accept mixed dtypes at the device level (DTypePromoteAPI);
  the shim deliberately keeps same-dtype-only for S1 and records mixed as the
  G-009 gap.

## Entries v1 (after the first suite run, 2026-10-04, chunked stamp 20261004-183548)

Suite result: **255 passed / 1040 failed / 87 skipped of 1382** (19 chunks;
stamp `20261004-194249` — corrected figures, see SUMMARY-s1 correction: the
first merge silently dropped the `test_creation_functions` chunk, whose
process died in the G-030 arange loop) — details in `reports/SUMMARY-s1.md`.
New register entries; "evidence" names the dominant failure class observed:

| id | area | category | table row | evidence | note |
|---|---|---|---|---|---|
| G-021 | `__getitem__` basic indexing | shim-side | blank | all 155 operator tests fail at hypothesis draw: strategy does `result[i]` | highest-leverage item; unblocks data generation suite-wide |
| G-022 | elementwise surface (acos…signbit, pow, maximum/minimum, floor_divide, remainder, …) | shim-side | Y rows | 241 special-case tests resolve `func=None` | `rt::` has all; mechanical via dispatch macros |
| G-023 | `xp.linalg` namespace (matmul, matrix_transpose, vecdot, tensordot) | shim-side | C(%) | `AttributeError: no attribute 'linalg'` (23) | table's `%` column is invisible to Python; suite grades `xp.matmul` |
| G-024 | manipulation/creation remainder | shim-side | mixed | concat/stack/broadcast_*/expand_dims/squeeze/moveaxis/flip/eye/linspace/tril/triu/meshgrid/unstack | `rt::` has most; eye/linspace/tril/triu need checking |
| G-025 | `xp.newaxis` sentinel | shim-side | — | test_constants 4/5 | trivial constant once indexing exists |
| G-026 | in-place / reflected operator dunders (`__ifloordiv__` etc.), `__pos__` | rust-side | D | test_array_object 0/27 | fulfillment-table D; `__mod__` maps to `rt::rem` |
| G-027 | result_type / can_cast / isdtype | rust-side | ? | 16 data_type_functions failures | needs token-level promotion query (G-008); decide shim strategy at S3 review |
| G-028 | axes-supporting reductions & stats (sum/max/min/mean/…, argmax/min) | shim-side | ? | statistical 0/9 | whole-array first via reduction macros; axes via `*_axes` rt fns |
| G-029 | searching/set (where, nonzero, unique_*, searchsorted, isin, count_nonzero) | rust-side? | ? | searching 0/8, set 0/6 | data-dependent shapes; capabilities declares False — resolve claim at S3 |
| G-030 | `rt::arange` infinite loop on step-away-from-end | rust-side | — | probe: `arange(0, 4151497946, step=-129734311.0)` balloons to OOM (found 2026-10-04 post-S1, probe log in reports/) | **RESOLVED rust-side** (rstsr `f94a3cd`, branch `261004/rstsr-faer-py`): `arange_by_partial_ord_cpu_serial` is now direction-aware (sign-mismatch ⇒ empty, matching numpy); the loop guard was the single root of G-030 *and* G-031. Shim guard removed (`881f071`); suite `test_arange` passes (stamp 20261004-214808: 303/997/82, only this test changed) |

Remaining v0 entries (G-001…G-015) keep their status; G-010/G-011 resolve in
S2/S3 as planned. Discrepancy watch for S4: fulfillment-table rows marked Y
that appear in G-022/G-024 evidence = stale table entries (report-only diff
proposed at close).

## Entries v2 (post-S2 wheel, 2026-10-04, chunked stamp `20261004-205656`, NO_EXPLAIN)

Suite result: **302 passed / 998 failed / 82 skipped of 1382** (wheel = branch
`261004/rstsr-faer-py` @ `a52bf4f`, S1 surface + S2 DLPack). Failure-class
census over the 998: 522 marshalling-reject (TypeError), 253 wrong-value,
136 missing-attr, 58 unexpected exception, 26 blocked on missing `xp.astype`,
3 explicit declines. New entries:

| id | area | category | table row | evidence | note |
|---|---|---|---|---|---|
| G-031 | arange narrow-int downward range | rust-side | — | hand-verified: `arange(0, -2, step=-1)` → `[]` on int8/int16; int64/float64 correct (suite: test_arange, `prod(out.shape)=0`) | **RESOLVED** by the same change as G-030 (rstsr `f94a3cd`): i8/i16/u8/u16 always took the upward-only generic loop; it is now direction-aware. Watch-item: unsigned dtypes with negative steps cannot represent the step in `T` at all (wrap/overflow) — ungraded by the suite so far |
| G-032 | finfo/iinfo reject complex dtypes | shim-side | ? | `xp.finfo(complex64)` → ValueError "only real floating-point dtypes are allowed" (4 data_type failures) | spec: finfo accepts complex floating (real-part semantics); shim's own Rust raises |
| G-033 | capabilities key `max dimensions` | shim-side | — | test_inspection: expects `"max dimensions"` (2024.12 rename), shim emits `"max ndim"` | one-line Python fix |
| G-034 | spec name `permute_dims` vs shim `permute_axes` | shim-side | — | "permute_dims is not defined" (all manipulation + has_names) | **fixed** (renamed in api.py, 2026-10-04) |
| G-035 | advanced indexing mixed with slices | shim-side (algorithm; needs permission) | — | `x[idx, :]` raises NotImplementedError | suite's arrays_and_ints tests use ints+arrays only, so ungraded; gather would need the same machinery as G-036's fancy path |
| G-036 | handle-model aliasing: indexing/astype results are copies, never views | shim-side | — | ops.rs getitem_int comment cites this id | spec permits copies; shared-storage views would need a TensorArc repr in the handle enum |
| G-037 | `where` absent from rstsr | rust-side | ? | test_getitem verification calls `xp.where` (1st blocker of test_getitem/test_setitem) | elementwise ternary; rstsr has no primitive — **owner ruled: do not implement shim-side**; candidate for the rust-side batch |
| G-038 | boolean-mask indexing (x[mask] getitem + setitem) | rust-side | ? | `x[bool_array]` / `x[mask]=v` raise NotImplementedError (reverted shim gather) | rstsr has only per-axis bool_select; needs whole-tensor mask gather/scatter or a nonzero primitive; candidate for the rust-side batch |
| G-039 | integer-array (fancy) indexing with broadcasting | rust-side | ? | `x[int_array, ...]` raises NotImplementedError (reverted shim gather) | needs multi-array broadcast gather (index_select is per-axis only); pairs with G-035 (array+slices mixing); candidate for the rust-side batch |

Note on scoping: `test_has_names`/`test_signatures` grade extension names
(linalg-*, fft-*) unconditionally — `--disable-extension` only skips the
extension test *files*, not the name checks. Scoping them out would require
task-side SKIPS_FILE entries (never upstream edits); default plan keeps them
in scope and implements linalg (fft: open decision, see
FAILURE-REDUCTION-PLAN.md wave W7).

## Entries v3 (post-W0/W1 audit of the 980-failure corpus, 2026-10-04 stamp `20261004-233924`)

Method: the full per-test table (`harness/reports/COMPLIANCE-FULL-20261004-233924.csv`,
generated by `harness/compliance_table.py`) plus targeted repros on the
installed wheel. Census and fix-layer roll-up in
`reports/STATUS-2026-10-05-arrayapi-compliance.md`. The entries below are
new findings from that audit (bugs and semantics), not a re-listing of the
v0/v1 wave items:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-040 | `isnan/isfinite/isinf` on 0-d non-float arrays return shape `(1,)` | shim-side | `xp.isfinite(xp.asarray(0, dtype=xp.uint8)).shape == (1,)`; suite `out.shape=(1,), but should be ()` ×3 | the int/bool constant-result fallback (G-017 workaround) builds a shape-(1,) array; must mirror the input shape. Removed if G-017 is fixed rust-side |
| G-041 | `all`/`any` ignore `keepdims=True` | shim-side | `xp.all(xp.asarray([False]), keepdims=True).shape == ()`, expected `(1,)` | also `any`; bind `*_axes` with keepdims |
| G-042 | `sum` lacks `axis=` and `dtype=` | shim-side | `sum(x, axis=0)` → NotImplementedError; `sum(x, dtype=None)` → TypeError | `rt::sum_axes` exists; `dtype` acceptance + accumulation semantics = G-043 |
| G-043 | integer reductions keep the input dtype; spec uses the default integer accumulator | rust-side | suite `out.dtype=uint8, but should be uint64 [sum(uint8)]` | affects sum/prod (mean/std/var promotion to float to verify when bound) |
| G-044 | `negative`/unary minus refused for unsigned dtypes | rust-side | `negative: not defined for bool/unsigned dtypes` on `uint8`; suite grades uint8 and numpy wraps | spec expects defined semantics; gates `__neg__` (bool exclusion is spec-correct) |
| G-045 | assigning into an empty (size-0) tensor raises InvalidLayout | rust-side | `Array(shape=(0,0))[:, :] = v` → `cannot assign to broadcasted tensor` | zero-size dims give stride 0, read as broadcast by the write gate; shim-side special-casing barred by the wrapper-only rule |
| G-046 | `astype` signature lacks `device=` | shim-side | `Argument 'device' missing from signature` | spec 2025.12 astype is `(x, dtype, /, *, copy=True, device=None)` |
| G-047 | `reshape` declares `shape` positional-only | shim-side | `shape is a pos-only argument, but should be a pos or kw argument` ×4 | suite signs off on pos-or-kw |
| G-048 | namespace-info `devices()` returns a list; `default_device()` absent | shim-side | `isinstance(out.devices(), tuple)` (2025.12); `hasattr(out, "default_device")` | Python-layer only |
| G-049 | `dtypes()` takes no `kind` argument | shim-side | `dtypes(kind=...)` TypeError; `test_array_namespace_info_dtypes` | needs the kind vocabulary incl. `signed/unsigned integer` |
| G-050 | `default_dtypes()` keys wrong | shim-side | returns `real/integral/complex`; spec keys `"real floating"/"complex floating"/"integral"` + `"indexing"` | Python-layer only |
| G-051 | `capabilities()["boolean indexing"] == True` while masking raises | shim-side | audit probe: `capabilities()` returns True; `x[mask]` is G-038 | latent overclaim (not the cause of any current failure): must be False until G-038 lands. `"data-dependent shapes": False` is correct today; `"max dimensions": 8` is a placeholder (rstsr has no cap) |

Cross-dtype operands (G-009 family) remain shim-declined: 10 unexpected
exceptions + 14 promotion declines in the census. Note for the rust-side
batch: the tensor-level binary ops are generic over the two input dtypes and
the device comparison kernels are bounded on Rust traits that hold within a
dtype family, so mixed pairs need either promotion wiring or pair dispatch
— not a shim change.

## Entries v4 (W2 elementwise + operator pass, 2026-10-05)

W2 landed shim-side (bindings only): the full elementwise surface and the
operator-dunder set, including mixed-dtype pair dispatch for the
`DTypePromoteAPI`-bound ops. Red map: **901 passed / 399 failed / 82 skipped
of 1382** (stamp `20261005-015407`), up from 320/980/82. Census and full
table in `reports/SUMMARY-w2.md` /
`harness/reports/COMPLIANCE-FULL-20261005-015407.csv`.

New divergences confirmed during the pass:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-052 | dtype-preserving unary kernels for integers | rust-side | `rt::{ceil,floor,trunc,round}_f` promote integer inputs to float64 (`TOut = T::FloatType`); the spec requires the input dtype (suite grades integers for ceil/floor/trunc/round) | shim declines integer inputs per owner ruling 2026-10-05 ("decline now, fix later"). Also `conj` (integers routed through the into-float block) |
| G-053 | `pow` dtype coverage | rust-side | integer bases: `num::Pow` needs an unsigned exponent type (`i8: Pow<i8>` undefined); complex: `num_complex` has no `Complex<T>: Pow<Complex<T>>` (documented upstream limitation) | shim serves float32/float64 same-dtype only; mixed pairs stay G-009 |
| G-054 | `rt::signbit` semantics inverted | rust-side | `signbit(-2.0)` returns False, `signbit(1.0)` returns True (kernel writes `is_positive()`) | wrong values are worse than a decline: the shim raises with this register reference; suite probe `_has_functional_signbit()` consequently reports "unavailable" (only affects zero-sign checks) |
| G-055 | complex transcendental coverage and special-value propagation | rust-side | `expm1(complex)` has no kernel (`Float`-bound impl only); num-based kernels return `nan+nanj` where the spec requires specific values (acos/acosh/asinh/atanh/cosh/sinh/tanh/sqrt special cases, ~35 tests); accuracy diverges at large magnitudes in complex64 (`acos(7281-1j)`, `asinh(-2731+1j)`, `tan(1+45j)` vs numpy/cmath) | rust-side numerical work; register first, no shim workaround |
| G-056 | elementwise `maximum`/`minimum` do not propagate NaN | rust-side | `maximum(NaN, 0.0)` returns 0.0, spec requires NaN (2 tests); consistent with the 2026-10-02 compliance directive for min/max | rust-side fix |
| G-057 | `remainder` signed zero / infinite divisor | rust-side | `remainder(-0.0, 2.0)` → -0.0 (spec: +0), `remainder(1.0, -inf)` → 1.0 (spec: -inf); rstsr's float `%` is fmod-style, the spec's `remainder` follows the divisor's sign (Python `%` semantics) | 8 tests across remainder/`__mod__`/`__imod__` |

Mixed-dtype status after W2 (G-009 family refinements):
- **served by shim pair dispatch** (rstsr device kernels bound on `DTypePromoteAPI`, real dtypes): `maximum minimum floor_divide atan2 copysign hypot nextafter logaddexp` + comparisons (`equal`/`not_equal` additionally across bool and complex; ordering comparisons are real-only per spec).
- **still declined, registered rust-side**: `add subtract multiply divide remainder` (Rust `Add`/`Sub`/... bounds are same-type only), `bitwise_*`/shifts, `pow` mixed, and **mixed-kind Python scalars** (e.g. float scalar + int array).
- compile-time cost measured: shim release build 30 s → ~6 min from the pair-dispatch expansion (166 eq arms + 113 promote arms); worth revisiting if the crate is ever built per-commit in CI.
