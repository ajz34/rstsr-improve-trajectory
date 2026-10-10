# tensordot for rstsr-core + rstsr-faer.api

Array-API `tensordot` end to end: an rstsr-core device op (CPU-serial + a
rayon twin) with a tensor-tier thin wrapper, exposed through the
`rstsr_faer.api` shim. Naive kernels are the baseline; the only acceleration
is a view-only GEMM fast path that never copies.

## Base

- rstsr (workspace `rstsr-local-workspace`): `d7056ba` — `#133`
  `feat: array-API linalg extension for rstsr_faer.api with n-dimensional slogdet`.
- Everything here is uncommitted in rstsr on purpose (user's instruction:
  work in the workspace, no rstsr commits). `proposed.patch` is the full diff.

## API design (settled by grilling)

- Signature: `rt::tensordot(a, b, (axes,))` where `axes: impl TryInto<AxesPairIndex>`
  (`AxesPairIndex<isize>`): `None` / `()` / `Val(n)` / `(a_axes, b_axes)`,
  matching array-API `tensordot(x1, x2, /, *, axes=2)`.
- `None`, `()` and the bare default `Val(2)` are one meaning (contract the
  last `n` of `a` with the first `n` of `b`). NumPy rejects `axes=None`; we
  accept it because array-API makes `axes` keyword-only with default 2.
- Bare collections are the rstsr extension: `[0, 1]` → `Pair(same, same)` =
  a{0,1} vs b{0,1}; the 2-tuple `(0, 1)` → `Pair(Val0, Val1)` = a-axis-0 vs
  b-axis-1. NumPy does NOT distinguish these (it collapses `[0,1]` to
  `(0,1)`); recorded as an intentional divergence in `numpy_differences.md`.
- `tensordot` is a top-level array-API function, not a `linalg` member, but
  `np.linalg.tensordot` also exists, so the shim exposes it in both spots.
- dtype promotion is resolved in the **device impl** (generic `TC`), never in
  the tensor tier or the device trait — tensor-tier functions stay
  `Mul`-free.

## Implementation

Device op:

- `rstsr-core/src/operators/linalg.rs` — `DeviceTensordotAPI<TA,TB,TC,DA,DB,DC>`.
- `rstsr-native-impl/src/cpu_serial/tensordot.rs` — naive kernel +
  `tensordot_gemm_layouts` predicate (a pure layout test — no data touched);
  `rstsr-native-impl/src/cpu_rayon/tensordot.rs` — rayon twin with a
  `PARALLEL_SWITCH` of 512 and a serial fallback.
- `rstsr-core/src/device_cpu_serial/linalg/tensordot.rs` — GEMM fast path if
  the layouts are view-only reshapeable, else the naive kernel.
- `rstsr-core/src/feature_rayon/auto_impl/tensordot.rs` — rayon twin, shared
  by symlink into `device_faer/rayon_auto_impl/` and the five
  `crates-device/*/src/rayon_auto_impl/`.

Tensor tier:

- `rstsr-core/src/tensor/linalg/tensordot.rs` — `tensordot` / `tensordot_f`
  (panicking / fallible, identical signatures), `tensordot_from` /
  `tensordot_from_f` (out-of-place-from), plus `TensorAny` methods;
  `resolve_tensordot_axes` normalizes every `AxesPairIndex` case.
- `rstsr-common/src/axes_pair_index.rs` — `impl From<()> for AxesPairIndex<isize>`.

Shim:

- `crates-interop/rstsr-faer-py/src/linalg.rs` — `linalg_tensordot` +
  `parse_tensordot_axes` (int / 2-seq / `None`); registered in `lib.rs` and
  wired into `python/rstsr_faer/api.py` as top-level `tensordot` and
  `_LinalgNamespace.tensordot`.

## GEMM fast path

`tensordot_gemm_layouts` builds the canonical `[free_a, ctr]`, `[ctr, free_b]`,
`[free_a, free_b]` layouts and asks `layout_reshapeable` whether each is a
no-copy reshape. If all three are, it hands the canonical 2-d layouts to
`matmul_uninit`; otherwise it returns `None` and the caller uses the naive
kernel. The predicate is a unit test (`crates/.../tensordot.rs`) that also
exercises a *decline* case (multi-axis contraction that cannot merge, e.g.
axes `[0, 2]` on `(2, 3, 4)`), so a regression in the trigger is caught.

## Tests

- `rstsr-core/tests/core_func/linalg/test_tensordot.rs`
  - `numpy_tensordot` — 1:1 transfer of NumPy `TestTensordot`
    (`numpy/_core/tests/test_numeric.py` L4229): `test_rejects_duplicate_axes`
    L4234, `test_zero_dimension` L4240, `test_zero_dimensional` L4248, each
    with the `// numpy: v2.5.2 | path::Class::method (L<n>)` header and the
    raw NumPy body inline. Error cases go through `rt::tensordot_f(..).is_err()`.
    (NumPy's second assert in `test_zero_dimension` is `np.einsum`; replaced by
    the equivalent zero-tensor comparison — rstsr has no einsum.)
  - `custom_tensordot` — 6 cases covering the array-API spec surface NumPy
    never tests: outer (`axes=0`) / matmul (`axes=1`), full contraction +
    `axes=2` default carried by `None` / `()`, the pair-axes 3-d docstring
    example with a negative-axis cross-check, `n` too large, negative `n`.
- `rstsr-core/tests/doc_draft/linalg/test_tensordot.rs` — doc examples.
- Tracking: 3 `TestTensordot` tuples added to `sync_numpy.py`; 3 rows inserted
  into `numpy_coverage.csv` (CRLF, binary-mode insert); the `[0,1]`-vs-`(0,1)`
  divergence noted in `numpy_differences.md`.

**Upstream coverage is genuinely thin**: `TestTensordot` has exactly three
methods, and the only other tensordot-specific test is
`numpy/linalg/tests/test_linalg.py::test_tensordot` (L2398, the alias).
Everything else (`test_einsum.py`, `tensorsolve`) uses tensordot as an oracle,
not as the subject.

## Results

- rstsr-core unit tests: 9/9 in `test_tensordot.rs` pass on both devices,
  row- and column-major; GEMM trigger fires and declines as intended.
- array-API conformance, tensordot nodes: **4 passed / 2 failed**.
  - green: `test_has_names[linear_algebra-tensordot]`,
    `test_has_names[linalg-tensordot]`, `test_func_signature[tensordot]`,
    `test_extension_func_signature[linalg.tensordot]`.
  - failing: `test_tensordot`, `test_linalg_tensordot` — both fail solely on
    the **shared mixed-dtype gap G-009** (`matmul/vecdot/tensordot: operands
    must share one dtype`), the same gap matmul and vecdot sit behind, not a
    tensordot defect.
- Full suite: **1294 passed / 55 failed / 0 error / 33 skipped** (1382 total),
  up from the 1287 / 62 / 33 shim-init baseline.

## Conclusion

The tensor/device API is implemented and array-API-shaped; the only remaining
red nodes are the two value tests blocked by the pre-existing G-009
mixed-dtype promotion gap. Implementing promotion is a separate scope (it
touches matmul/vecdot too) and was deferred in `PLAN.md`.

## Review follow-up

A `/code-review high` pass found no correctness bug (independent reference
across shapes × orders × devices, incl. transposed / negative-stride /
broadcast inputs and the rayon path). Its ten findings were addressed in
`44b6216`: the unresolved `vecdot` doc link, the missing anchor docstring
sections, the free `tensordot_from` return type, the array-API surface table,
the shim's `bin_numeric!` error names, value/output assertions in the doc test,
removal of the dead `order` parameter from the naive kernels, a shared
split-and-check helper, the duplicated in-src pair-axes test, and a new rayon
`PARALLEL_SWITCH` test.
