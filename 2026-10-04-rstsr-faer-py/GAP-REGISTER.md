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
