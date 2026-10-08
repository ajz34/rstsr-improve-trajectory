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
`DTypePromoteAPI`-bound ops. Red map: **902 passed / 398 failed / 82 skipped
of 1382** (stamp `20261005-020258`, re-verified `20261005-094602` on the
reworked `positive` wheel), up from 320/980/82. Committed on branch
`261004/rstsr-faer-py`: `1546e5e` (rust-side `positive`) + `799c4b3` (shim).
Census and full table in `reports/SUMMARY-w2.md` /
`harness/reports/COMPLIANCE-FULL-20261005-094602.csv`.

New divergences confirmed during the pass:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-052 | dtype-preserving unary kernels for integers | rust-side | `rt::{ceil,floor,trunc,round}_f` promote integer inputs to float64 (`TOut = T::FloatType`); the spec requires the input dtype (suite grades integers for ceil/floor/trunc/round) | shim declines integer inputs per owner ruling 2026-10-05 ("decline now, fix later"). Also `conj` (integers routed through the into-float block) |
| G-053 | `pow` dtype coverage | rust-side | integer bases: `num::Pow` needs an unsigned exponent type (`i8: Pow<i8>` undefined); complex: `num_complex` has no `Complex<T>: Pow<Complex<T>>` (documented upstream limitation) | shim serves float32/float64 same-dtype only; mixed pairs stay G-009 |
| G-054 | `rt::signbit` semantics inverted | rust-side | `signbit(-2.0)` returns False, `signbit(1.0)` returns True (kernel writes `is_positive()`) | wrong values are worse than a decline: the shim raises with this register reference; suite probe `_has_functional_signbit()` consequently reports "unavailable" (only affects zero-sign checks) |
| G-055 | complex transcendental coverage and special-value propagation | rust-side | `expm1(complex)` has no kernel (`Float`-bound impl only); num-based kernels return `nan+nanj` where the spec requires specific values (acos/acosh/asinh/atanh/cosh/sinh/tanh/sqrt special cases, ~35 tests); accuracy diverges at large magnitudes in complex64 (`acos(7281-1j)`, `asinh(-2731+1j)`, `tan(1+45j)` vs numpy/cmath) | **RESOLVED** (v11, rstsr PR #122 squash `b2e22ab`): `rstsr-dtype-traits::c99_complex` (C99 Annex G port) + one `ExtComplexFloat` method per function, both device op tables re-pointed; all 53 suite nodes flipped. See v11 for G-075/G-076 and the "oracle is the spec stubs, not numpy" note |
| G-056 | elementwise `maximum`/`minimum` do not propagate NaN | rust-side | `maximum(NaN, 0.0)` returns 0.0, spec requires NaN (2 tests); consistent with the 2026-10-02 compliance directive for min/max | rust-side fix |
| G-057 | `remainder` signed zero / infinite divisor | rust-side | `remainder(-0.0, 2.0)` → -0.0 (spec: +0), `remainder(1.0, -inf)` → 1.0 (spec: -inf); rstsr's float `%` is fmod-style, the spec's `remainder` follows the divisor's sign (Python `%` semantics) | 8 tests across remainder/`__mod__`/`__imod__` |
| G-058 | `log1p` has no device kernel | rust-side | `TensorLog1pAPI`/`OpLog1pAPI` are declared, but every device impl file carries only `// TODO: log1p` (`device_cpu_serial/operators/op_binary_common.rs`, `feature_rayon/auto_impl/op_binary_common.rs`) — `rt::log1p_f` cannot be instantiated for `DeviceFaer` | 20 suite failures (name neither bound nor bindable); shim side is trivial once a kernel exists (float-only, like `expm1`) |

Mixed-dtype status after W2 (G-009 family refinements):
- **served by shim pair dispatch** (rstsr device kernels bound on `DTypePromoteAPI`, real dtypes): `maximum minimum floor_divide atan2 copysign hypot nextafter logaddexp` + comparisons (`equal`/`not_equal` additionally across bool and complex; ordering comparisons are real-only per spec).
- **still declined, registered rust-side**: `add subtract multiply divide remainder` (Rust `Add`/`Sub`/... bounds are same-type only), `bitwise_*`/shifts, `pow` mixed, and **mixed-kind Python scalars** (e.g. float scalar + int array).
- compile-time cost measured: shim release build 30 s → ~6 min from the pair-dispatch expansion (166 eq arms + 113 promote arms); worth revisiting if the crate is ever built per-commit in CI.

## Entries v5 (W3 statistical wave, 2026-10-06)

Owner directive for this wave: rstsr-faer-py stays wrapper-only; bugs and
missing algorithms are fixed on the rust side (this supersedes the W0-W2
"register-only" stance for the duration of the task). Wheel = rstsr branch
`261005/faer-py-stats` (now PR #112: 7356137 shim stats + 36e0b7c round
fix + ed22e4d round gated behind the std feature after the no-std CI
failure; all 13 checks green, not merging without owner go).

Suite result: **934 passed / 366 failed / 82 skipped of 1382** (stamp
`20261006-002157`, NO_EXPLAIN, 19/19 chunks), up from 902/398/82. Flips:
32 failed->passed, 1 latent wrong-value surfaced (G-059). Full table
`harness/reports/COMPLIANCE-FULL-20261006-002157.csv`; narrative
`reports/SUMMARY-w3.md`.

RESOLVED in this wave (registered earlier):

| id | area | resolution |
|---|---|---|
| G-041 | all/any ignore keepdims | bound through `op_all_axes`/`op_any_axes` (truthiness cast + `all/any_with_args_f`, ReduceArgs axes+keepdims); `test_all`/`test_any` pass |
| G-042 | sum lacks axis=/dtype= | `sum`/`prod`/`max`/`min`/`mean`/`std`/`var` bound over `*_with_args_f` (axes + keepdims); `dtype=` via cast-then-reduce marshalling |
| G-043 | integer reductions keep input dtype | accumulation rule (spec 2025.12: signed->int64, unsigned->uint64, floats/complex keep) applied Python-side by casting with the existing astype path BEFORE the same-dtype reduction — the order the standard itself recommends; u8 sum now yields uint64 |

New entries:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-059 | `round` was ties-away, not ties-to-even | rust-side — **FIXED** | first surfaced by this wave's run: `test_special_cases[round(modf(i)[0]==0.5) -> ROUND_HALF_EVEN]` flipped passed->failed because the derandomized example set newly drew an exact halfway value; the kernel called `f64::round` (half away from zero), the spec requires the even neighbor | fixed on `261005/faer-py-stats`: `round_ties_even_f` in `rstsr-native-impl::scalar_math` behind a std feature gate (intrinsic arm + no_std arithmetic arm) after the plain `f64::round_ties_even` broke the no-std CI builds; regression test `core_func::math::test_unary_math::custom_math_basic::test_round_ties_to_even`; archived in `numpy_differences_resolved.md`. +1 suite test |

Note on the shim's dtype policy: no promotion table was added to the Python
layer; the accumulation dtype default is the spec's documented argument
default for sum/prod/cumulative_* (mirrors `_dtype_or_default`), and the
cast reuses the shim's registered astype path (G-007/G-018 machinery).
`median` is NOT in the 2025.12 statistical surface (checked the stubs:
cumulative_sum/prod, max, mean, min, prod, std, sum, var) — no rust-side
median work is needed for conformance.

## Entries v6 (W4 creation & manipulation complement, 2026-10-06)

Owner directive carries over: rstsr-faer-py wrapper-only; bugs fixed
rust-side. Wheel = rstsr branch `261006/faer-py-creation-manip` (worktree
`tmp/faer-py-w4`, base main `2ce2afd` = merged #112); the branch later grew
W5 + the review round and was merged as PR #113 (squash `c0ea36b`).

Suite result: **984 passed / 316 failed / 82 skipped of 1382** (stamp
`20261006-121144`), up from 934/366/82 — 50 flips / 0 regressions. Flips
and blocked checks in `reports/SUMMARY-w4.md`; table
`harness/reports/COMPLIANCE-FULL-20261006-121144.csv`.

RESOLVED (partial): G-024 creation & manipulation complement — creation
complete; manipulation complete except `repeat`/`roll`/`tile`
(G-002/G-003, no rstsr primitive).

New entries:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-060 | `triu`/`tril` k outside the matrix | rust-side — **FIXED** (`4bd2ab5`) | `rt::triu(ones((3, 3)), 2)` panicked: `j_end = max(i + k, 0)` was not clamped to `ncol`; a huge \|k\| wrapped in tril (zeroed the wrong range) | row clamp + saturating `i + k`; regression tests `custom_tril_triu::{test_k_outside_row_bounds, test_extreme_k}`; archived in `numpy_differences_resolved.md` |
| G-061 | `linspace` endpoint + serial drift | rust-side — **FIXED** (`6fcabe6`) | `linspace(0, 6.4913965932284536e16, 25)[-1]` one ulp short of `stop` (`start + (n-1)*step` rounds to a neighbor); `linspace(2, 10, 100)[-1] = 9.999999999999996` (the serial kernel accumulated `v += step`) | both kernels now compute `y[i] = start + i * step` and assign the endpoint directly; test `custom_linspace::test_endpoint_exact`; NumPy parity is float64-only |
| G-062 | `eye`/`tril`/`triu` bool dtype | rust-side | declines: eye through the name dispatch (no bool arm), tril/triu through `dispatch_t_numeric_same!` ("not defined for bool dtype") | kernels are `Num`-bound (bool has no `Num` impl); outside the suite's generators (`hh.numeric_dtypes` excludes bool); needs a bool-capable creation/tri path (G-016 family) |
| G-063 | `linspace` non-floating dtype | rust-side (policy) | declines with a TypeError naming the `ComplexFloat` bound (float32/float64/complex64/complex128 only) | the spec says the dtype "should be a floating-point data type" — non-float output is not required; the suite draws `real_floating_dtypes` only |

## Entries v7 (W5 searching & indexing + branch review, 2026-10-06)

Suite result: **996 passed / 304 failed / 82 skipped of 1382** (stamp
`20261006-131221`, re-verified `20261006-132433`) — 12 flips / 0
regressions; the review round's fixes then changed no test outcome
(test-level 0-flip diff `20261006-132433` → `20261006-144402`). Tables
`harness/reports/COMPLIANCE-FULL-20261006-{131221,144402}.csv`; narrative
`reports/SUMMARY-w5.md`.

RESOLVED (partial): G-029 searching/set — `argmax`, `argmin`,
`count_nonzero` bound (`test_searching_functions` now passes them;
`where`/`nonzero`/`searchsorted` remain), and `take` bound from the
indexing side. Still absent: `where` (G-037), `nonzero`, `searchsorted`,
`take_along_axis`, `isin`, `unique_*`, `sort`/`argsort`.

New entries — all surfaced by the `/code-review max` round on the branch;
fixes in `4bce438`, each with a core regression test and a
`numpy_differences_resolved.md` entry:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-064 | `Layout::diagonal` super-diagonal gate | rust-side — **FIXED** | `eye(2, 4, k=2)` all-zero; `eye(3, 1, k=2)` a bogus layout-overflow error | the super-diagonal range was gated on `d1` (rows) instead of `d2` (cols); gate is `(0..d2)` now; consumers eye/diag/diagonal |
| G-065 | `squeeze` mixed negative axis list | rust-side — **FIXED** | `(-4, 0)` accepted: only the descending-sort head was checked, so the invalid `-1` survived and addressed a real axis | every mapped axis is validated; test `custom_squeeze_mixed_axes` |
| G-066 | `take`/`index_select` empty indices | rust-side — **FIXED** | empty indices on a zero-length axis raised IndexError (`indices.iter().max().unwrap_or(&0)` tested the sentinel `0`) | returns the empty selection; test `custom_indexing_take` |
| G-067 | argmax/argmin empty output | rust-side — **FIXED** | `argmax(zeros((2, 0)), axis=0)` raised before the axes split; the guard tested the total size | guard applies to the split *reduced* axes: empty output is legal, an empty reduced axis still raises; test `custom_arg_empty` |
| G-068 | tril/triu rank-1 error class | rust-side — **FIXED** | rank-1 input surfaced as a bare AxisError → IndexError in the array-API wrapper | kernels assert `ndim >= 2` up front (InvalidLayout → ValueError); test `custom_tril_triu::test_ndim1_error` |

Also in the review commit: `meshgrid()` with zero vectors now returns `()`
at the binding (legal input per 2025.12), and the shim-side checks the
review found were relocated into the kernels, so `api.py` is
wrapper-only again. PR #113 merged by the owner (squash `c0ea36b`) after
13/13 CI checks passed on the first run.

## Entries v8 (the `where` slice of the searching wave, 2026-10-06)

Suite result: **1014 passed / 286 failed / 82 skipped of 1382** — 18 flips /
0 regressions. Canonical FRESH stamp `20261006-180651` (hypothesis DB deleted,
19/19 chunks); warm-DB re-run `20261006-180325` gave identical totals and a
test-level 0-flip diff. Tables
`harness/reports/COMPLIANCE-FULL-20261006-180651.csv`; narrative
`reports/SUMMARY-where.md`.

Wheel = rstsr branch `261006/faer-py-where` (worktree `tmp/faer-py-where`,
base main `1b09497` = merged #115); PR #116 (13/13 checks green first run)
merged as squash `05cf6ce`.

RESOLVED:

| id | area | resolution |
|---|---|---|
| G-037 | `where` absent | bound over rstsr's `rt::where_f` (core landed as PR #114 `076270b`): a 169-arm ternary dispatch derived from rstsr's promotion matrix (no shim-side table), bool-only condition, `_operands` weak-scalar marshalling. Flips 18 tests — the 12 blocked on `where` (test_where, test_where_with_scalars, has_names/signature, and the verification paths of getitem/asarray_arrays/eye/tril/triu/linspace/unstack/positive) plus 5 `test_nan_propagation[mean/prod/std/sum/var]` generators that build their input with `xp.where` |

New entries:

| id | area | category | evidence | note |
|---|---|---|---|---|
| G-069 | `DTypePromoteAPI<i16> for i8` missing | rust-side — **FIXED** | the where dispatch table surfaced it: every `DTypePromoteAPI`-bound op (maximum/minimum/floor_divide/atan2/copysign/hypot/nextafter/logaddexp, comparisons, where) declined int8×int16 while its mirror row int16×i8 existed; the suite draws both orderings | present since the 2025-09-29 promotion commit `ce977a1`, where a section comment swallowed the impl line (`// internal typeimpl_promotion_asable!(i8, i16, …)`); restored with flags mirroring the i16×i8 row, plus an inline `#[cfg(test)]` module in `promotion.rs` (compile-time completeness over the 13×13 matrix, `Res` checks, restored-row values) |
| G-070 | `where` declined-case registry | shim-side (policy) | non-bool condition → TypeError; both-scalars → TypeError; complex Python scalar vs real array → TypeError | the first two are spec-aligned (condition "should" be boolean; "at least one of x1 and x2 must be an array"); the third is the pre-existing G-009 weak-scalar limit shared by every binary op, not `where`-specific |

Residual red around the new surface (unchanged): `test_nan_propagation[max]`
and `[min]` (reduction NaN propagation, G-056 family — verified independent of
where: `xp.max([1.0, nan]) → 2.0` while sum/mean/prod/std/var propagate);
`nonzero`/`searchsorted`/`sort` still absent (G-029 remainder).

## Entries v9 (current-failure census, 2026-10-06)

No new divergences surfaced by the census; it re-measured the merged `where`
state after the local workspace was relocated to
`~/rstsr_pack/rstsr-local-workspace` (the common local workspace — the wheel
is rebuilt there from scratch). Fresh chunked stamp `20261006-183350`
(FRESH, NO_EXPLAIN, 19/19 chunks): **1014 / 286 / 82 of 1382**, test-for-test
0-flip vs the canonical `20261006-180651`. Full partition of the 286 in
`reports/STATUS-2026-10-06-failure-census.md`; roll-up: missing surface 114
(61 names) · rust-side semantics/kernels 156 · shim-side fixes 16.

Status changes:

| id | change |
|---|---|
| G-013 | upgraded from "ungraded edge" to **graded**: `bitwise_invert` on uint64 is value-correct in the tensor, but scalar extraction / `tolist` wraps through the i64 PyScalar carrier (`~[1] → -2`, `0xFFFFFFFFFFFFFFFE → -1`); `test_bitwise_invert` grades it (2 tests). Shim-side fix (u64-capable carrier). |

## Entries v10 (W6-8 manip/sort/set binding wave, stage 8 of 2026-10-06-manip-sort-set)

Suite stamp `20261006-235845` (NO_EXPLAIN, CHUNKED, warm DB) against branch
`261006/manip-sort-set` @ stage-8 shims + G-072 fix: **1059 / 241 / 82 of
1382** — +45 flips, 0 regressions vs census `20261006-183350` (1014 / 286 /
82), test-for-test (44 binding flips + `test_tile` from the G-072 fix): 14
`test_has_names` + 14 `test_signatures` +
17 runtime (repeat, roll, searchsorted+scalars, nonzero, isin+scalars,
unique_*×4, sort, argsort, take_along_axis, diff, diff_append_prepend,
tile).
`test_concat`/`test_stack` remain red
(pre-existing G-009 joins).

New gaps surfaced:

| id | surface | class | observed behavior | disposition |
|---|---|---|---|---|
| G-071 | `default_dtypes()["indexing"]` key | shim-side — **FIXED** (stage 8) | `searchsorted`'s dtype assertion raised `KeyError: 'indexing'`: the 2025.12 `DefaultDataTypes` TypedDict requires an `"indexing"` key that the `_NamespaceInfo.default_dtypes` dict did not carry (only real/integral/complex) | added `"indexing": int64` — consistent with every index output going through `idx_lift`; 2 suite tests (`searchsorted`, `searchsorted_with_scalars`) |
| G-072 | `tile` with a zero-size input axis | rust-side — **FIXED** (stage 8 review round, 2026-10-07) | `tile(zeros((0, 3)), (1, 2))` panics in `assign_arbitary_uninit_cpu_serial` (assignment.rs:53): the contiguous fast path slices `c[offset_c..offset_c+size]` with a nonzero `offset_c` against a length-0 source slice — the per-block source is empty but the destination block offset is still applied to an (empty) contig destination | fixed in `rstsr-native-impl` `cpu_serial/assignment.rs` + `cpu_rayon/assignment.rs`: early `Ok(())` return when `lc.size() == 0` in the contig fast path (empty dest writes nothing; offsets may point past empty storages); regression test `test_zero_size_input_axis` in `test_tile.rs`; `test_tile` green (suite 1059/241/82) |
| G-073 | 0-d integer Array as a scalar index | shim-side — **FIXED** (stage 8) | suite's reference computation `a_1d[i_1d[j]]` uses a 0-d int Array as index; the `__getitem__` gap gate declined it as integer-array indexing (G-039) | 0-d integral arrays marshal through `__index__` (NumPy scalar-index semantics); genuine mask / multi-element array keys still decline under G-038/G-039; `test_take_along_axis` |

Stage-8 notes for existing gaps:

- G-009 (cross-dtype) grew an intentional, spec-table-driven exception:
  `isin` promotes mixed *integer* pairs to their common int dtype through the
  existing astype path (`_common_int_dtype` mirrors the standard's promotion
  table — value-preserving for every pair the suite draws); mixed
  non-integer pairs still decline. `test_isin` draws promotable int pairs
  (e.g. uint8×uint16), so the same-dtype-only kernel needed the marshal.
- The 16 `test_signatures` flips include `sort`/`argsort`/`repeat`/`roll`/
  `tile`/`searchsorted`/`nonzero`/`isin`/`unique_*`×4/`take_along_axis`/
  `diff` — pos-only `/` and kw-only `*` shapes match the 2025.12 stubs;
  `roll.shift` is pos-or-keyword with default `None` per the stub.
- `repeat`/`roll`/`tile`/`searchsorted`/`sort`/`argsort`/`nonzero`/`diff`/
  `take_along_axis`/`unique_values`/`unique_counts`/`unique_inverse`/
  `unique_all`/`isin` are now **bound** (G-001/G-002/G-003/G-012 fulfilled
  rust-side by stages 1–7; this stage only marshals). `unique_all` is the
  first namedtuple-precedent return (`collections.namedtuple` built in
  api.py); `unique_counts`/`unique_inverse` are namedtuples too (the spec
  text requires `.values`/`.counts` attribute access, and the suite's
  `hasattr` checks grade it).
- Pre-existing all/any fix bundled with this stage (G-074): `op_all_axes`/
  `op_any_axes` built their truthiness zero as a shape-`(1,)` tensor, which
  broadcast-lifted 0-d inputs to `(1,)` before the reduction —
  `all(0-d, keepdims=True)` returned `(1,)` instead of `()`. The zero is
  0-d now; `test_all`/`test_any` pass again (they had passed at census only
  because hypothesis had not yet drawn the 0-d keepdims example; the DB
  grew during this stage's runs).

## Entries v11 (C1 — complex transcendentals, 2026-10-07, rstsr PR #122)

Suite stamp `20261007-153214` (NO_EXPLAIN, CHUNKED, `FRESH=1`) against rstsr
`main` @ `b2e22ab` (squash of branch `261007/complex-transcendentals`):
**1144 / 156 / 82 of 1382** — +53 flips, 0 regressions, node set identical
(1382) versus the previous stamp `20261007-144313` (1091 / 209 / 82). The 53 =
48 `test_special_cases::test_unary` + the `test_acos` / `test_asin` /
`test_asinh` / `test_tan` / `test_tanh` elementwise nodes.

RESOLVED in this wave:

| id | surface | class | observed behavior | disposition |
|---|---|---|---|---|
| G-055 | complex elementary functions (`sqrt`, `cosh`, `sinh`, `tanh`, `tan`, `acos`, `asin`, `acosh`, `asinh`, `atanh`) | rust-side — **FIXED** (`b2e22ab`) | the `num-complex` formulas return `nan + nanj` where the standard prescribes a specific `±inf` / `±0` / `π` combination (its `FIXME #1284`), and `acos`/`asin`/`asinh` lose accuracy at large magnitude | new `rstsr-dtype-traits` module `c99_complex` carries the C99 Annex G routines (inverse trig via the Hull–Fairgrieve–Tang crossover); `ExtComplexFloat` gained one method per function — the real `f32`/`f64` impls delegate to libm, so only complex inputs take the new path — and both device op tables were re-pointed |
| G-075 | `expm1(±0 ± 0i)` | rust-side — **FIXED** (`b2e22ab`) | returned the sign of the input zeros (`-0 + 0j` for a `-0` real part); the standard fixes `+0 + 0j` | zero-input guard in `ext_exp_m1` |
| G-076 | `tanh(±inf + iy)`, `y` finite | rust-side — **FIXED** (`b2e22ab`) | took its imaginary zero from `sin(y)*cos(y)`, whose sign flips at a floating-point zero crossing; the standard fixes `+0` for finite `y` (it leaves the sign open only for `y = ±inf` / `NaN`) | the zero now follows `y` |

G-075/G-076 are **draw-dependent** nodes: they looked green in earlier stamps
only because the warm hypothesis DB had drawn `+0` inputs; a `FRESH=1` run
exposed them. Both are deterministic now.

Method note — **the oracle is the spec stubs, not numpy.** The special-case
table that `test_special_cases.py` grades against is parsed from the *stub
docstrings* (`array-api/src/array_api_stubs/_2025_12/elementwise_functions.py`).
Two things this wave confirmed:

- numpy on this machine calls the **system** complex math (glibc), not the
  vendored `npy_math_complex.c.src` fallbacks it ships — the two disagree in
  the NaN corners (`cacosh(0+NaN i)`, `ccosh(0+i∞)`, `ctanh(0+i∞)`). Probe with
  a *runtime* `-fno-builtin` C program: constant folding of `INFINITY`/`NAN`
  literals yields a third, wrong answer.
- the stub table is **stricter than numpy** in exactly the G-075/G-076
  zero-sign cases; numpy passes them by draw luck. rstsr now passes them
  deterministically, i.e. it is ahead of numpy on that axis.

Known deviation (documented in the code): numpy's `ccosh`/`csinh` middle branch
(`710 < |x| < 1455`) uses `frexp`/`ldexp` so an exactly-zero `cos y` yields `0`
rather than `NaN`; `libm`/`num` expose no `frexp`, so that branch is merged
into `exp(|x|)·0.5` — identical except when the trig factor is exactly zero,
which generated floats do not reach.

## Entries v12 (C4 — floored `remainder`, 2026-10-08, rstsr PR #123)

Suite stamp `20261008-040224` (NO_EXPLAIN, CHUNKED, `FRESH=1`) against rstsr
`main` @ `8a09076` (squash of branch `261008/rem-remainder`; the same-session
`main` @ `53ae4f2` baseline wheel re-measured 1144 / 156 / 82): **1156 / 144 /
82 of 1382** — +12 flips, 0 regressions, node set identical (1382) versus the
baseline stamp. The 12 = 4 special cases × {`remainder`, `__mod__`, `__imod__`}
(`test_binary` twice + `test_iop`).

RESOLVED in this wave:

| id | surface | class | observed behavior | disposition |
|---|---|---|---|---|
| G-057 | `remainder` signed zero / infinite divisor | rust-side — **FIXED** (`8a09076`) | `rt::rem` was Rust's `%` — fmod-style, sign of the *dividend* — so `remainder(-0.0, 2.0)` → `-0.0` (spec `+0`) and `remainder(±finite, ∓inf)` → the dividend (spec the divisor); the stub prescribes the divisor's sign (Python `%`) | `ExtNum::ext_rem` (array-API floored remainder) + a specialised `OpRemAPI` that keeps the `Rem` bound and dispatches the float dtypes by `TypeId` to `ext_rem`; both device op tables |

Corrections and caveats:

- the v4 count for G-057 ("8 tests") undercounted — the chunked suite grades
  **12** nodes.
- integer `rt::rem` is **unchanged** (still Rust `%`, sign of the dividend): the
  device dispatch is float-only by design, so the array-API floored guarantee
  holds for the floating dtypes the standard exercises; the operator module doc
  records the split.

## Entries v13 (C5 + C7 — `signbit` and `pow`, 2026-10-08)

Suite stamp `20261008-045659` (NO_EXPLAIN, CHUNKED, `FRESH=1`) against rstsr
`main` @ `8a09076` (the C4 squash) plus the C5/C7 branch
`261008/signbit-pow` @ `bacc07b` (PR pending): **1170 / 130 / 82 of 1382** —
+14 flips, 0 regressions, node set identical (1382) versus the baseline stamp.
The 14 = the 9 `signbit` nodes (`test_signbit` + 8
`test_special_cases::test_unary[signbit(±0/±inf/±NaN)]`) and the 5 `test_pow`
params (`pow`/`__pow__`/`__ipow__` × array/array, array/scalar).

RESOLVED in this wave:

| id | surface | class | observed behavior | disposition |
|---|---|---|---|---|
| G-054 | `signbit` inverted semantics | rust-side — **FIXED** (`bacc07b`) | the kernel wrote `is_positive()` (`Signed`-bound) — `signbit(-2.0)` was False; unsigned and half had no impl | `ExtReal::ext_signbit` (unsigned `false`, signed `< 0`, float/half `is_sign_negative` — the sign bit, so `-0.0`/`-NaN` are correct); `OpSignBitAPI` split out of the boolean-output table and re-bound `Signed` → `ExtReal`, both device modules |
| G-053 | `pow` dtype coverage | rust-side — **FIXED** (`bacc07b`) | `OpPowAPI` was `TA: num::Pow<TB>`, and `num` provides no `Int: Pow<Int>` (signed exponent) nor `Complex: Pow<Complex>`; the shim served float32/64 same-dtype only | `OpPowAPI` reworked into a promoted binary op (`TOut = TA::Res`, the `atan2`/`maximum` shape) with the kernel in `ExtNum::ext_pow` (`pow`/`powf`/`powc`), covering integer bases (with integer exponents) and complex bases; both device modules |

Corrections and caveats:

- **negative integer exponent** — the array API leaves `int ** int` with a
  negative exponent unspecified and NumPy raises `ValueError`; rstsr rejects it
  too (`ExtNum::ext_pow` returns `None` → an `InvalidValue` error, which the
  panic form `rt::pow` unwraps and the Python binding surfaces as `ValueError`)
  rather than returning a fabricated integer. This is **not** a NumPy
  divergence, so it carries no `numpy_differences.md` entry.
- switching `signbit` from a stub to a working kernel flips the suite's
  `_has_functional_signbit()` probe to true, which turns **on** the ±0 sign
  checks inside every strict float comparison — a latent-regression surface. The
  run showed no collateral flips (C1/C4 had already fixed the zero-sign cases).
- the shim gained `dispatch_t_real_bool!` and `dispatch_bin_pow!` — the latter
  reuses the real promotion arms of `dispatch_bin_promote!` and adds the
  complex64/128 arms; the `pow` output dtype is the promotion, matching the
  array API's `result_type` for every pair the suite draws (the rstsr and
  array-api promotion tables agree on all of them).
