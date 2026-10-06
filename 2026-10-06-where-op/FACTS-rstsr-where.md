# Facts: rstsr mechanics for a where op (workspace @ 2ce2afd)

Distilled 2026-10-06 from a sub-agent sweep of `/home/a/rstsr_pack/rstsr`.
Paths repo-relative to rstsr.

## Existing state

- No where-like op anywhere; `where` has an **empty status row** in the
  Searching Functions table of `rstsr-core/src/docs/array_api_standard.md:238`
  (facade `rstsr/src/docs/array_api_standard.md` is a symlink — one edit).
- Bool-mask support today is only axis-wise: `bool_select` in
  `rstsr-core/src/tensor/adv_indexing.rs:295` (mask → index vec →
  `index_select`). No whole-tensor mask indexing, no nonzero. `Indexer` enum
  has no mask variant.
- Comparison ops output hardcoded `bool` (device duplicate table rows
  `[OpEqualAPI] [bool] [PartialEq]`); bool is a first-class dtype
  (blanket `impl<T> DeviceAPI<T>` per device). No numeric→bool cast trait.

## Elementwise op anatomy (traced via `maximum`; `nextafter` commit ae7a8e3)

Files to touch for a new **binary** elementwise op:

1. `rstsr-core/src/operators/ops/op_ternary_common.rs` — trait row in
   `#[duplicate_item]` table (house "ternary" = 2 in + 1 out).
2. `rstsr-core/src/device_cpu_serial/operators/op_ternary_common.rs` —
   serial impl row (closure → `op_mutc_refa_refb_func`).
3. `rstsr-core/src/feature_rayon/auto_impl/op_ternary_common.rs` — rayon
   impl row; **symlinked** by device_faer and every BLAS crate's
   `rayon_auto_impl/` → one edit covers faer + openblas/mkl/blis/aocl/kml.
4. `rstsr-core/src/tensor/operators/op_binary_common.rs` — 3 tables
   (TensorXxxAPI trait, broadcast impl, `func_binary!`).
5. `rstsr-core/src/prelude.rs` — 2 explicit lists (traits ~L30, funcs ~L139).
6. Docs: `rstsr-core/src/docs/api_specification.md` + status table.
7. Tests: `rstsr-core/tests/core_func/operators/` + `mod.rs`.

NOT needed: rstsr facade prelude (wholesale re-export), lib.rs,
rstsr-native-impl (generic closure kernels), BLAS crates (symlinks).

- Hand-written trait precedent with unusual shape: `OpIsCloseAPI`
  (op_ternary_common.rs:67) — bool output + extra `&IsCloseArgs` param.
- Promotion: common-op family promotes NumPy-style per element via
  `DTypePromoteAPI::promote_pair` (`rstsr-dtype-traits/src/promotion.rs`,
  table follows NumPy: f32+f64→f64, u64+i64→f64, i32+f32→f64; bool↔numeric
  covered); output `TA::Res`; missing impl = compile error. Arithmetic ops
  (add/sub/…) do NOT promote (std::ops bound). **Scalars are strong**:
  `rt::maximum(&f32arr, 0.5)` promotes to f64 — NOT NEP-50 weak.
- Scalar overloads: positional, via `op_mutc_refa_numb` /
  `op_mutc_numa_refb` impls (`TB: num::Num` bound at tensor layer). No
  tuple form for elementwise ops.
- Broadcasting: pairwise only. `broadcast_shape` / `broadcast_layout`
  (`rstsr-common/src/layout/broadcast.rs:166`) materialize stride-0
  layouts; no 3-way helper. Output dim `DA::Max` (`DimMaxAPI`).
- Kernel arity ceiling: dispatch helpers exist for 1/2/3 layouts
  (`rstsr-common/src/layout/iterator.rs:838/875/917`,
  `rstsr-common/src/par_iter.rs:89/125/166`). **A 3-input op is the first
  4-layout kernel** (out + cond + x + y). Generic closure kernels:
  `rstsr-native-impl/src/cpu_serial/op_with_func.rs:153`,
  `.../cpu_rayon/op_with_func.rs:227` (blocked-2d fast paths, PARALLEL_SWITCH
  fallback). Bridge traits in `rstsr-core/src/operators/ops/op_with_func.rs`.
- `clip` (also unimplemented) is the same 3-in/1-out shape → shared infra.

## Naming precedent

- **Zero raw identifiers** (`r#...`) in the workspace; zero trailing-
  underscore public fn names (besides `_f`). `where` has simply never been
  exposed. No house precedent either way.

## Test infrastructure

- Parity test home: `rstsr-core/tests/core_func/operators/test_where.rs` +
  `mod.rs` (`CATEGORY: "operators"`); `specify_test!` gating, `TESTCFG`
  device, `tensor_from_nested!`, `assert_equal` (float-only; bool compared
  via `.to_vec()`). Entry binaries symlink the tree per device
  (`entry_row_cpu.rs`, required-features row_major).
- Tracking: `rstsr-core/tests/tracking/numpy_coverage.csv`
  (`numpy_path,Class,test_fn,version,status,rstsr_location,source_hash,notes`),
  `numpy_differences.md`, pinned NumPy v2.5.2, `tracking/sync_numpy.py`.
  Doc-example tests: `tests/doc_draft/operators/`.
