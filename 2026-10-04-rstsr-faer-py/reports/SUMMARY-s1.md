# S1 — skeleton + first red map (2026-10-04)

Package: `rstsr/crates-interop/rstsr-faer-py` on branch
`261004/rstsr-faer-py` (rstsr @ main `69c97bf`; **no rstsr-core edits** —
the crate binds through the facade `rstsr` with default features, i.e.
DeviceFaer). Installed into conda env `torch` as a local wheel (never
published). Zero rust-side fixes; every rstsr limitation found became a
register entry (G-016…G-020 new this step).

## What exists (vertical slice)

- Native (pyo3 0.29, abi3-py310, maturin): dtype-erased `AnyTensor` enum over
  the 13 canonical dtypes (`Tensor<T, DeviceFaer, IxD>`); asarray from
  flattened lists, zeros/ones/empty/full/arange, astype (own primitive-cast
  table — rstsr has no tensor-level cast), add/subtract/multiply/divide,
  negative/abs, 6 comparisons, all/any/isnan/isfinite/isinf, reshape,
  transpose; dtype/Device singletons; finfo/iinfo.
- Python (`rstsr_faer.api`): spec-exact signatures, `Array` protocol object
  (shape/dtype/ndim/size/T/mT/tolist/item/astype/to_device,
  `__array_namespace__`, `__dlpack_device__`; `__dlpack__` raises
  NotImplementedError by design until S2), operator dunders with
  within-kind weak-scalar forwarding, `__array_namespace_info__` +
  capabilities/default_dtypes/dtypes/devices, constants, `__all__` surface.

## First red map (suite @ `6c0b59f`, submodule `5f847a3`, suite-default 100
examples, derandomized; identical pins to the NumPy baseline)

Instrument note: the first whole-suite attempt was **OOM-killed** in one
pytest process — pytest + pytest-json-report hold every failure record
(tracebacks, array reprs) in RAM, and ~1000 failures × hypothesis shrinking
does not fit (~155 MB peak per file is fine; cumulative is not). `run.sh`
gained `CHUNKED=1` (one pytest process per suite file, per-chunk reports,
merged total; derandomization is per-test so numbers stay comparable to the
baseline's whole-suite run).

```
253 passed, 1026 failed, 87 skipped   (1366 collected over 18 chunks, ~5 min)
```

| chunk | passed | failed | skipped |
|---|---|---|---|
| test_array_object | 0 | 27 | 0 |
| test_constants | 4 | 1 | 0 |
| test_data_type_functions | 20 | 16 | 0 |
| test_dlpack | 0 | 3 | 0 |
| test_fft | 0 | 0 | 14 |
| test_has_names | 52 | 162 | 0 |
| test_indexing_functions | 0 | 2 | 0 |
| test_inspection_functions | 0 | 4 | 0 |
| test_linalg | 0 | 4 | 26 |
| test_manipulation_functions | 0 | 13 | 0 |
| test_operators_and_elementwise | 0 | 155 | 5 |
| test_searching_functions | 0 | 8 | 0 |
| test_set_functions | 0 | 6 | 0 |
| test_signatures | 43 | 127 | 42 |
| test_sorting_functions | 0 | 2 | 0 |
| test_special_cases | 134 | 483 | 0 |
| test_statistical_functions | 0 | 9 | 0 |
| test_utility_functions | 0 | 4 | 0 |

For calibration: NumPy 2.5.1 on identical pins is 1335/42/5 of 1382. The 16
items that vanish from our collection are suite-internal parametrizations
that collapse when names are missing. Constants 4/5 (only `nan` identity
convention differs — check in S3). 134 special-case passes show the numeric
core (arithmetic values, comparison semantics, default dtypes) is already
honest.

## Failure structure (what the 1026 are made of)

1. **Missing elementwise surface** — 241 failures are `test_special_cases`
   parametrizations whose `func` resolves to `None` (acos, asin, atan, ceil,
   floor, round, sign, sqrt, exp, pow, maximum, minimum, ...). rstsr's `rt::`
   has all of them; each is ~5 lines through the existing dispatch macros.
2. **`__getitem__` absence blocks data generation** — hypothesis's own
   `xps.arrays` strategy does `result[i]` while building the array, so every
   strategy-driven test (all 155 operator tests, most manipulation/
   statistical/searching tests) fails at draw time with
   `'Array' object is not subscriptable`. This is the single highest-leverage
   item: it converts "cannot even generate data" into real graded results.
3. **Missing namespaces** — `xp.linalg` (23), `xp.fft` (correctly absent;
   14 skips). `xp.linalg.matmul`/vecdot map to `rt::` matmul (fulfillment
   table's C-status `%`).
4. **Missing manipulation/creation functions** — concat, stack,
   broadcast_*, expand_dims, squeeze, moveaxis, flip, eye, linspace,
   tril/triu, meshgrid, unstack, where, nonzero, unique_*, searchsorted,
   sort/argsort (expected D-gaps), take*, cumsum*, stat functions
   (sum/max/min/mean/std/var/prod — reductions over axes, whole-array first).
5. **dtype-function surface** — result_type/can_cast/isdtype (needs the
   promotion story, G-008/G-009), 16 test_data_type_functions failures.
6. **Array object protocol** — test_array_object 0/27: `__getitem__`,
   `__len__` edge cases, `__floordiv__`/`__mod__`/`__pow__`/`__matmul__`,
   iadd-style in-place dunders, `__pos__` (D), bit operators.
7. **DLPack** — 3 failures, expected until S2.

## S3 priority order (derived from this map)

1. `__getitem__` basic indexing (unblocks the hypothesis machinery).
2. Full elementwise unary/binary surface via existing macros (~35 fns).
3. `xp.linalg` (matmul, matrix_transpose, vecdot, tensordot).
4. Manipulation/creation remainder (concat/stack/broadcast/reshape-family).
5. Statistical/searching/set functions (whole-array first, axes next).
6. result_type/can_cast/isdtype — needs a promotion decision (register).

## Environment facts

- pyo3 0.29.3 (`abi3-py310` + `py-clone`; Py<T>: Clone is feature-gated),
  maturin 1.15; wheel `rstsr_faer_py-0.9.0-cp310-abi3`.
- Build: `cargo build -p rstsr-faer-py` in the rstsr workspace (nightly, from
  rust-toolchain.toml), wheel via `maturin build --release -i <torch python>`;
  python-side edits need only `cp python/rstsr_faer → site-packages`, Rust
  edits need a wheel rebuild (~30 s incremental release).
- Chunk reports: `harness/reports/rstsr_faer_api-chunk-*.json`
  (this run's stamp: 20261004-183548).
