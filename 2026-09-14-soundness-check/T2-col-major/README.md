# T2 — Column-major soundness check (rstsr @ acfa93e)

Subtask T2 of the 2026-09-14 soundness-check campaign: wire, run, and stress
the `col_major` configuration of rstsr (mission aspect S3: "the col_major
configuration is completely unwired").

- Worktree: `/home/a/rstsr_pack/tmp/snd-colmajor` (detached at `acfa93e`)
- Against: rstsr master `acfa93e` ("rstsr-native-impl: route 2-D
  transpose-pattern assigns through blocked orderchange kernels (#102)")
- Machine: Ryzen 9950X3D (16C), 64 GiB, rustc nightly `1.99.0-nightly
  (c79e8f89 2026-08-04)` per the repo's `rust-toolchain.toml`
- Deliverables: this README, `col-major-wiring-and-tests.patch`,
  `fixes.patch`. Nothing was committed to rstsr; all changes live as patches
  taken from the worktree.

## What was wired

1. `rstsr-core/tests/entry_col_cpu.rs` — new entry binary per ADR-0002:
   `type DeviceType = DeviceCpuSerial`, `TESTCFG` with
   `device.set_default_order(ColMajor)`, mod-including the SAME body as
   `entry_row_cpu.rs` (`core_func`, `doc_draft`, `test_issues`, `test_utils`)
   plus the new `col_func/` track (col-only, not included from the row entry).
2. `rstsr-core/Cargo.toml` — `[[test]] name = "entry_col_cpu"
   required-features = ["col_major"]`.
3. `rstsr-core/tests/col_func/` — the new col-major test track (66 tests, see
   below).

### The exact feature incantation that works

```
cargo test -p rstsr-core --no-default-features \
    --features "col_major,aligned_alloc,faer,faer_as_default,std" \
    --test entry_col_cpu
```

Notes on the incantation (the "expect friction" items):

- The task's suggested list worked **without** any `rstsr/col_major`
  dev-dependency feature: the workspace manifest declares `rstsr-core = {
  path = ..., default-features = false }`, so the umbrella `rstsr`
  dev-dependency (itself `default-features = false`) does **not** secretly
  re-enable `row_major` on the core. Feature unification therefore never sees
  both order features, and the `compile_error!` guard never fires.
- `std` **must** be added: `--no-default-features` drops `rstsr-common/std`,
  and the test bodies need std.
- The predicted compile friction in the `#[cfg(feature = "col_major")]`
  branches of `op_binary_assign.rs`, `op_binary_common.rs`, `adv_indexing.rs`
  (and the many others: `map_elementwise`, `reduction`, `iterator_*`,
  `op_tri`, `manipulation`, `op_binary_arithmetic`, ...) **did not
  materialize** — the col build compiled cleanly on the first attempt. Those
  branches are already consistent at this commit.
- `tests/tensor_sum.rs` (auto-discovered) is `#![cfg(not(feature =
  "col_major"))]` — correctly self-disables under col.

## Col-vs-row suite results

| Suite (rstsr-core) | row cfg (baseline) | col cfg before | col cfg after |
|--|--|--|--|
| `--test entry_row_cpu` / `--test entry_col_cpu` | 302 passed / 0 failed | 302 / 0 | **368 / 0** (302 body + 66 `col_func`) |
| `--lib` unit tests | 110 / 0 | 109 / 0 | **110 / 0** |
| `--doc` | not run (no col-sensitive change) | — | **183 / 0**, 2 ignored |

Why the existing body passed under col *before* any fix — triage of the
entire body found **zero failures to triage**, for a systemic reason
(finding 1 below): essentially every body test is order-explicit. Counts:

- 73 of the 71 body `test_*.rs` files call `set_default_order(RowMajor)` on a
  `TESTCFG.device.clone()` (several files do it in every test). Only
  `doc_draft/docs/test_order_semantics.rs` and a handful of `doc_draft/`
  creation/manipulation tests additionally construct `ColMajor` devices, and
  those already pin col behavior explicitly.
- Net effect: the col entry binary has, until now, been re-running the NumPy
  parity body under row-major semantics. Col-major *behavior* was covered
  only by the in-source `#[cfg(feature = "col_major")]` unit tests (which do
  exist and all pass) — nothing at the integration level.

## Findings

### F1 (test-architecture, category b): the body self-neutralizes col cfg

The `specify_test!`-tracked body deliberately pins `RowMajor` per test
(order-explicit tests are the right pattern), but the *default* pinned order
is hardcoded rather than derived from the entry's `TESTCFG` device order.
Consequence: `entry_col_cpu` gains nothing from the shared body beyond
compile-time cfg differences. Options for the future (not done here — would
touch every body file): derive the pinned order from `TESTCFG.device`
(`device.default_order()`), or keep as is and rely on `col_func/` + lib unit
tests for col semantics. ADR-0002's uniformity principle is unaffected.

### F2 (coverage defect, fixed): DeviceFaer matmul had ZERO col-major coverage

`rstsr-core/src/device_faer/matmul_impl.rs` — the only correctness test of
the faer GEMM path, `test_minimal_correctness`, was silenced under col with
`#[cfg(not(feature = "col_major"))]` (line 299 at acfa93e): under col,
`asarray(..).into_shape([3, 2])` packs the buffer column-major, the logical
matrices differ, and the expected element sum no longer matched. Rather than
porting the expectation, the test was cfg'd out — leaving the faer matmul
completely untested under `col_major` (and `faer_as_default` is the *default
build* of the crate, so the default device of a col build had no matmul test
at all).

**Fix** (in `fixes.patch`): added a `#[cfg(feature = "col_major")]` twin with
the correct expectation. Derivation: `sum(AB) = Σ_k colsum(A)_k ·
rowsum(B)_k`; with col packing, colsum(A) = (3+6i, 12+15i) and rowsum(B) =
(4+6i, 8+10i), giving sum = −78 + 282i (row twin: −78 + 270i). Verified
empirically before writing the assert. After: col lib suite 110/110, row lib
suite still 110/110.

### F3 (documented contract, now pinned): argmax/argmin flat index stays ROW-major under col

`tensor/reduction.rs` module docs state: "the all-element forms return a
row-major flat index even on a [`ColMajor`] device". Verified empirically
(including a scratch probe across layouts) and pinned by tests:

- f-contig [2,3]: max at logical [1,0] → 3 (row flat; buffer/col flat would
  be 1); max at logical [0,2] → 2 (col flat would be 4).
- 3-D [2,3,4]: max at logical (1,1,0) → 16 (= 1·12+1·4+0), not the col flat 3.
- Non-contiguous views: transposed view [4,3,2] of an f-contig [2,3,4]
  (strides [6,2,1]) returns the row-major flat of the *logical* position
  (3,2,1) → 23, not the buffer offset; partially-contiguous slices
  ([2,2,4] strides [1,2,6]; [3,2] strides [1,3]; c-strided [2,2] strides
  [4,1]) all return the correct row-major flat of the slice's own logical
  space.
- Ties: first in row-major visit order (all-equal → 0). Mechanism, for the
  record: `reduce_all_arg_cmp_cpu_serial`
  (rstsr-native-impl/src/cpu_serial/reduction.rs:832) folds an unraveled
  index and ravels it through a RowMajor pseudo-layout of the shape.

Same answers under both cfgs (checked) — this is order-independent by
construction, correctly documented, and needs no layout-dir change.

### F4 (documented contract, now pinned): Julia-style limited broadcasting

Under col, `broadcast_shapes(.., ColMajor)` and all elementwise ops align
shapes from the **first** axis (Fortran/Julia rule), per
`src/docs/order_semantics.md`:

- `[2,3] + [3]` → `InvalidLayout` error (NumPy would broadcast the trailing
  axis) — the "limited broadcasting" promised in Cargo.toml comments;
- `[3,2] + [3]` and `[3] + [3,2]` → broadcast along axis 0, values verified;
- `[2,3] + [2,1]` OK, `[2,3] + [3,1]` error, `[3,1] + [1,3]` → [3,3];
- explicit `RowMajor` still gives the NumPy rule (`broadcast_shapes` takes
  the order as an argument) — both pinned.

### F5 (documented contract, now pinned): creation/iteration/reshape semantics

- Shape-driven creation (`zeros/ones/full/empty`, `arange.into_shape`,
  `eye` with device default order) is F-contiguous under a col device
  (stride [1,2] for [2,3]); `asarray` with explicit `[2,3].c()` / `.f()`
  layouts works under col and yields the documented logical mapping.
- `tensor_from_nested!` is always row-major regardless of device order (the
  documented exception) — pinned.
- Iteration follows col-major logical reading: a C-contiguous tensor under a
  col device reads [0,3,1,4,2,5] and `reshape(-1)` must copy (`is_owned`);
  an f-contig tensor reads in buffer order and `reshape(-1)` is a view.
- Reshape copy/view dichotomy (the `([4,6], 9)` example from
  order_semantics.md): merging leading (f-contiguous) axes is free; merging
  across the contiguity boundary copies — pinned with `is_owned` checks.
- Fresh elementwise results and matmul outputs land F-contiguous; assignment
  into both F-order and C-layout outputs writes logically.

### F6 (no library bug found)

After F2, no genuine col-major *computation* bug surfaced in the exercised
surface (creation, iteration, indexing/slicing, manipulation, broadcasting,
elementwise ops incl. mixed F×C×broadcast-view operands, serial+faer matmul,
full reduction zoo, degenerate shapes). All 8 initial test failures during
track authoring were expectation errors in my own new tests (arithmetic slips
like transposing the wrong matrix, `count_nonzero` counting the zero of
`arange`, `to_vec` being 1-D only), each corrected against hand computation.
No `rstsr-common/src/layout/*` fix is required; nothing was edited there.

## The new `tests/col_func/` track

Structure: flat (chosen over per-category subdirs because the track is one
cohesive concern; mirrors `core_func/` file naming `test_<name>.rs`).
Documented deviations from the body conventions (in `col_func/mod.rs`):
plain `#[test]` fns — `specify_test!`/`TestCfg` is NumPy-parity machinery
(CONTEXT.md's "col_major body ... not NumPy-parity; a separate, deferred
track") — and each test builds its own explicit device via `col_dev()` /
`faer_dev()` helpers instead of `TESTCFG`, so no test can be silently
neutralized by a default-order mismatch (the F1 failure mode).

66 tests in 6 files:

| File | Covers |
|--|--|
| `test_creation_order.rs` (12) | FlagOrder default, F-contig creation (zeros/ones/full/empty/arange/eye), explicit C/F asarray, col packing, tensor_from_nested exception, tril/triu, 0-dim, size-1, empty |
| `test_iteration_indexing.rs` (8) | iter/reshape(-1) on F and C layouts, transposed-view iteration, non-contiguous + partially-contiguous slices, logical indexing, negative index, index_select, index_mut buffer placement |
| `test_manipulation_order.rs` (9) | reshape reading invariance + copy/view dichotomy, transpose & moveaxis round trips, flip, squeeze/expand_dims, to_contig/to_layout, broadcast_shapes both orders, broadcast_to, concatenate, diag |
| `test_broadcast_ops.rs` (8) | limited broadcasting (errors + values), size-1-axis cases, mixed-layout F×C elementwise, broadcast-view operands, assignment into F-order and C-layout outputs, in-place scalar ops, comparison ops |
| `test_matmul_col.rs` (9) | serial & faer: F/C/mixed inputs, transposed views, sliced operands, identity, tall (3×2) and wide (2×5), 1×1, k=0, m=0, n=0, vecdot; device agreement |
| `test_reduction_col.rs` (9) | sum/min/max/prod/mean/var/std on F and C layouts, axis reductions, argmax/argmin flat-index contract (incl. 3-D, non-contig, ties, axes forms), nanarg, count_nonzero/all/any, degenerate (0-dim, size-1, empty errors) |

## Efficiency notes

- Nothing performance-related stumbled on during this task. One expectation
  confirmed: the col build is a second full feature-unification variant, so
  alternating row/col invocations recompiles the dependency subtree each time
  (~50 s warm-ish on this machine for the release test profile; first build
  compiles all deps from scratch).
- The `col_func` track itself runs in ~10 ms (release); it adds no meaningful
  suite time.

## How to reproduce

```
# col entry + track
cargo test -p rstsr-core --no-default-features \
    --features "col_major,aligned_alloc,faer,faer_as_default,std" --test entry_col_cpu
# col lib (includes the fixed faer matmul col twin)
cargo test -p rstsr-core --no-default-features \
    --features "col_major,aligned_alloc,faer,faer_as_default,std" --lib
# col doctests
cargo test -p rstsr-core --no-default-features \
    --features "col_major,aligned_alloc,faer,faer_as_default,std" --doc
# row baselines
cargo test -p rstsr-core --release --test entry_row_cpu
cargo test -p rstsr-core --release --lib
```

All numbers above are `--release` in
`/home/a/rstsr_pack/tmp/snd-colmajor` at `acfa93e` + the two patches.

## Patch manifest

- `col-major-wiring-and-tests.patch` — `rstsr-core/Cargo.toml` [[test]]
  entry; `rstsr-core/tests/entry_col_cpu.rs`; `rstsr-core/tests/col_func/*`
  (7 files). Test-only + build-manifest.
- `fixes.patch` — `rstsr-core/src/device_faer/matmul_impl.rs`: col twin of
  `test_minimal_correctness` (+30 lines, including the derivation comment).
  No library code paths changed; nothing needed in `rstsr-common/src/layout/*`.
