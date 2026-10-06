# PROPOSAL — `rt::r#where(cond, x, y)` in rstsr-core (and device impls)

Status: grill closed 2026-10-06 (see `DECISIONS.md`); against rstsr `2ce2afd`.
Implementation awaits user go; branch `261006/rt-where`; no rstsr commits
without explicit instruction.

## API surface

```rust
// free functions (rstsr-core prelude rstsr_funcs; facade re-exports wholesale)
rt::r#where(&c, &x, &y)          // panicking
rt::where_f(&c, &x, &y)          // fallible
rt::r#where(&c, &x, 0.5)         // scalar y (house-strong promotion)   [Q8]
rt::r#where(&c, 0.5, &y)         // scalar x                            [Q8]
// no scalar cond, no both-scalar, no method form, no _with_output (follows
// the common-op family)
```

- `c: TensorAny<R, bool, B, D>` (strict bool, Q4); x/y promote like the
  `maximum` family: `TA: DTypePromoteAPI<TB>`, output `TA::Res`, no
  arithmetic bound (Q5). Broadcasting: 3-way NumPy, error on mismatch (Q6).

## Implementation map (7-file surface + new infra)

**Milestone 1 — 4-layout kernel family (Q7):**

> IMPLEMENTED 2026-10-06 (branch `261006/rt-where`, uncommitted). Naming note
> (user, mid-run): the mutable output operand is **`d`**, inputs are
> **a/b/c** (a = cond, b = x, c = y) — kernel
> `op_mutd_refa_refb_refc_func[_cpu_serial|_cpu_rayon]`, bridge trait
> `Op_MutD_RefA_RefB_RefC_API`, device methods
> `op_mutd_refa_refb_refc` / `op_mutd_refa_refb_numc` / `op_mutd_refa_numb_refc`.
> Scalar variants reuse the existing 3-layout `op_mutc_refa_refb_func` kernels
> with the scalar captured in the closure (no new native scalar kernels).

1. `rstsr-common/src/layout/iterator.rs` — serial
   `layout_col_major_dim_dispatch_4` (mirror of `_3`, line ~917).
2. `rstsr-common/src/par_iter.rs` — `_par_4` (mirror of `_par_3`, ~166).
3. `rstsr-native-impl/src/cpu_serial/op_with_func.rs` —
   `op_mutc_refa_refb_refc_func_cpu_serial` (CONTIG_SWITCH/blocked-2d
   pattern from `op_mutc_refa_refb_func_cpu_serial`, line 153).
4. `rstsr-native-impl/src/cpu_rayon/op_with_func.rs` — rayon twin
   (AtomicPtr write pattern, PARALLEL_SWITCH fallback, line 227 family).
   Export both in `rstsr-native-impl/src/prelude_dev.rs`.
5. `rstsr-core/src/operators/ops/op_with_func.rs` — bridge trait
   `Op_MutC_RefA_RefB_RefC_API` (+ scalar-arg variants per Q8 outcome);
   impls in `device_cpu_serial/operators/op_with_func.rs` and
   `feature_rayon/auto_impl/op_with_func.rs` (symlinks cover DeviceFaer +
   every BLAS crate).

**Milestone 2 — the op itself:**

6. `rstsr-core/src/operators/ops/op_quaternary_common.rs` (new; hand-written
   like `OpIsCloseAPI`) — `OpWhereAPI<TA, TB, D>` with
   `op_mutc_refa_refb_refc` (+ `numa`/`numc` scalar variants per Q8);
   impl rows for DeviceCpuSerial
   (`device_cpu_serial/operators/op_quaternary_common.rs`) and
   DeviceRayonAutoImpl (`feature_rayon/auto_impl/operators/…`).
   Closure: `|c, cond, a, b| { let (a, b) = TA::promote_pair(a, b); c.write(if cond { a } else { b }) }`.
7. `rstsr-core/src/tensor/operators/op_where.rs` (new) — `TensorWhereAPI`
   trait + impls + free functions. Broadcast: small private 3-way helper
   chaining `broadcast_layout` (x,y → S1; cond vs S1 → S; re-broadcast the
   S1 pair to S; NumPy rules associative). Output dim `DimMaxAPI` chaining;
   `uninit_impl` fresh output (Q6); same-device checks.

**Milestone 3 — scalars (Q8, with agreed fallback):**

Scalar x/y impls at the tensor layer (mirror of `op_mutc_refa_numb`/
`numa_refb` in `op_binary_common.rs:186-260`, but `DTypePromoteAPI`-bounded,
not `num::Num`). **Fallback criterion (user-set):** if the scalar trait
bounds/impl matrix become too complicated, drop to tensor-only v1
(scalars = 0-d tensors) and record the deferral here + in DECISIONS.md.

**Milestone 4 — wiring + docs:**

8. `rstsr-core/src/prelude.rs` — `TensorWhereAPI` (rstsr_traits ~L30);
   `r#where`, `where_f` (rstsr_funcs ~L139).
9. `rstsr-core/src/docs/array_api_standard.md:238` — fill the `where` status
   row (Searching Functions; facade symlink covers `rstsr/`);
   `api_specification.md` — add to the appropriate functions list.
   Docstring + Overloads Table per skill `api-doc-conventions` when written.

**Milestone 5 — tests (Q9):**

10. `rstsr-core/tests/core_func/operators/test_where.rs` + `mod.rs`
    registration. Modules: `numpy_where::test_basic`, `test_ndim`,
    `test_error`, `test_dtype_mix` (tensor-cond parts), `test_exotic`
    (0-d/zero-size), `custom_where` (bool×strided×broadcast, + scalar if
    Q8 holds). Bool outputs compared via `to_vec()` (assert_equal is
    float-only). Entry binaries pick it up via existing symlinks
    (entry_row_cpu + device crates).
11. `rstsr-core/tests/tracking/numpy_coverage.csv` rows (pinned v2.5.2,
    `source_hash` via `tracking/sync_numpy.py` conventions);
    `numpy_differences.md` rows: (i) cond must be bool (no truthiness),
    (ii) scalars strong-promote not NEP-50 weak, (iii) no scalar
    OverflowError analog, (iv) no 1-arg form.
12. `rstsr-core/tests/doc_draft/operators/` doc example.

## Verification

- Full: `RUST_BACKTRACE=1 cargo test -p rstsr-core --test entry_row_cpu
  --features "backtrace row_major" --no-default-features` (faer-free form
  per build memory; nightly toolchain).
- Doc: `RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --no-deps`
  doctest gate (`cargo test -p rstsr-core --doc`).
- Spot device run (e.g. blis entry) since tests symlink in.

## Risks / notes

- First 4-layout kernel: blocked-2d applicability guard must handle 4
  operands incl. stride-0 broadcast axes (skip fast path when any operand
  broadcast — same rule as binary today).
- `r#where` is the codebase's first raw identifier: check rustdoc/links,
  `specify_test!` FUNC strings stay plain `"where"`.
- Performance: v1 generic closure kernel (where is memory-bound); blend
  kernel + T4'-style tuning only if a later benchmark justifies.
- `clip` deliberately NOT in this proposal (same kernel family, separate
  op PR afterwards).
