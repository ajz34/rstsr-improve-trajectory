# Facts: np.where semantics and test surface

Distilled 2026-10-06 from sub-agent reads of the local checkouts
NumPy v2.5.2 (`~/Git-Others/numpy`) and array-api-tests
(`~/Git-Others/array-api-tests`). Source citations are file:line in those
checkouts.

## Semantics (NumPy 2.5.2)

- Entry: `numpy/_core/multiarray.py:406` `where(condition, x=None, y=None, /)`
  — positional-only, `__array_function__` dispatch; C core
  `PyArray_Where` at `_core/src/multiarray/multiarraymodule.c:3234`.
- **Broadcasting**: single 4-operand `NpyIter_MultiNew` (out, cond, x, y)
  (multiarraymodule.c:3325) → exact standard 3-way broadcast; mismatch =
  ValueError. Scalars / 0-d / array_like allowed for **every** argument
  (each arg `np.asarray`-ed).
- **cond dtype**: NOT required bool — iterator forces operand dtype to
  NPY_BOOL under unsafe casting (multiarraymodule.c:3323-3326), i.e. any
  dtype castable to bool accepted, truthiness applied (pinned by NumPy
  `test_dtype_mix`).
- **Output dtype** = `PyArray_ResultType(x, y)` only — **cond excluded**
  (multiarraymodule.c:3284). Scalar x/y follow NEP 50 weak-scalar rules
  (Python literal stays weak: `where([True], f32, 0.5)` → f32; `1e150`
  overflows with warning, stays f32). Python-int out of range for common
  dtype → OverflowError. bool+float → float; int+float → float64.
  No `casting=` kwarg; x/y never cast below their promotion.
- **Allocation**: output allocated by the iterator, `NPY_ITER_NO_SUBTYPE` —
  always base ndarray; subclasses NOT preserved (ma mask silently dropped;
  separate `np.ma.where`). Object/string/structured/datetime64/mixed
  byteorder all supported; zero-size OK.
- **1-arg form** = exactly `condition.nonzero()`: tuple of intp index
  arrays, C order (multiarraymodule.c:3246). Only one of x/y given →
  ValueError.

## array-api standard (2024.12 / 2025_12 stubs) and its tests

- Signature `where(condition: array, x1: array|int|float|complex|bool,
  x2: ..., /)` — positional; condition must be an **array** with bool dtype
  (should-level); **at least one of x1/x2 must be an array** (both-scalar
  forbidden). Scalar support + bool-cond clarification added 2024.12.
- Output dtype: promotion rules over x1/x2 only (cond excluded); Python
  scalars weak (do not promote the array).
- Suite: `test_searching_functions.py:202` `test_where` — Hypothesis,
  mutually-broadcastable 3 shapes, mutually-promotable dtype pairs for
  x1/x2, cond always bool; asserts shape + values, **never dtype**; no
  error-path tests; no 1-arg form (outside the spec).
  `test_operators_and_elementwise_functions.py:2269`
  `test_where_with_scalars` (min_version 2024.12): int32 array + Python
  scalar → dtype int32, shape unchanged.
- `dtype_helpers.py:585`: `where` not bool-returning.

## NumPy test surface (translation candidates for rstsr-core)

Class `TestWhere`, `_core/tests/test_multiarray.py:9931+`:

| test | pins | rstsr relevance |
| --- | --- | --- |
| `test_basic` | value selection many dtypes, scalar-bool cond, 0-d x/y, heavy strides | core parity |
| `test_ndim` | 2-D broadcast, cond on new axis, transposed ops | core parity |
| `test_error` | non-broadcastable → ValueError | core parity (Err) |
| `test_dtype_mix` | uint32+float64→float64, f32+i64→f64, **int cond truthiness** | parity only if cond-leniency/promotion chosen |
| `test_foreign` | byteorder variants | likely N/A in Rust |
| `test_exotic` | weak-Python-float NaN/inf/1e150 stays f32; object dtype; zero-size | weak-scalar half relevant |
| `test_string` | fixed-width string null-fill | N/A (no string dtype) |
| `test_scalar_overflow` | Python int 1000 + uint8 → OverflowError | relevant iff scalar overloads |
| `test_empty_result` / `test_largedim` | 1-arg form only | N/A if 3-arg-only |
| `test_kwargs` | kwargs rejected (positional-only) | N/A in Rust (no kwargs) |

Elsewhere: regression `test_endian_bool_indexing`/`test_endian_where`
(byteorder → mostly N/A), StringDType tests (N/A), `np.ma.where` family
(N/A), ufunc `where=` kwarg tests (**out of scope**: different feature).

## Tensions surfaced (feed Round 2)

1. **cond leniency**: NumPy pins non-bool cond (truthiness); array-api
   wants bool arrays. R1-Q4 recommended strict bool — that drops part of
   `test_dtype_mix` from translation scope.
2. **Promotion**: NumPy pins cross-category x/y promotion (int+float→f64);
   whether rstsr `where` promotes or is same-dtype-generic depends on the
   house binary-op convention (rstsr mechanics agent, pending).
3. **Scalars**: standard allows scalar x1/x2 (weak); NumPy allows scalar
   cond too. Rust-side overload decision pending mechanics facts.
