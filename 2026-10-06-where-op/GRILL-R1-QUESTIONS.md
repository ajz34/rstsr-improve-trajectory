# Grilling Round 1 — rt::where proposal (consolidated full frontier)

Asked on screen 2026-10-06. All three fact-finds had already converged
(`FACTS-numpy-where.md`, `FACTS-rstsr-where.md`), so this single round
carries the whole frontier: the initially-asked questions enriched with the
facts, plus the branches the facts unblocked. No answers have been given
yet — answer `GRILL-R1-ANSWERS.md` (user-owned) or on screen.

---

## Q1 — Naming: `where` is a Rust keyword

The workspace has **zero raw identifiers and zero trailing-underscore
public names** — `where` was simply never exposed; either choice sets
precedent.

- (a) `r#where` — the NumPy/array-api/torch/jax name; rustdoc renders
  `where`; every call site reads `rt::r#where(&c, &x, &y)`; first `r#` in
  the codebase.
- (b) `rt::select` — clean call sites, ndarray precedent; permanent name
  divergence from every peer framework and from the status-table row
  (`array_api_standard.md` already links `where`).
- (c) dual: primary `r#where` + alias `select` (or vice versa).
- (d) `where_`-style mangling.

**Recommended: (a) `r#where`.** Parity is rstsr's identity; faer-py will
expose `api.where` regardless; rustdoc/doc-URLs show the plain name. The
`r#` is a per-callsite wart only.

## Q2 — Scope: 3-arg form only, or also the 1-arg form?

NumPy's 1-arg `where(cond)` is literally `nonzero()` (tuple of intp index
arrays, C order); the array-api standard has no 1-arg form and its suite
never calls it. G-038 (boolean-mask indexing) needs a nonzero-style
primitive, but with different machinery (index selection, data-dependent
shapes).

**Recommended: 3-arg only.** `nonzero` is its own future proposal; mixing
both under one function also muddies the Rust overloading story.

## Q3 — Scope: does "related implementations" include rstsr-faer-py?

faer-py G-037 records `where` as a rust-side gap; array-api-tests
`test_where` and `test_where_with_scalars` (2024.12+) would consume it.
Cover: (a) Rust only (rstsr-core + device impls + tests + docs), (b) Rust
+ pyo3 `api.where` binding, (c) Rust + binding + conformance run?

**Recommended: (a) Rust only now**; faer-py wiring as an explicitly named
follow-up in `2026-10-04-rstsr-faer-py` (W3 searching wave). One reviewable
PR.

## Q4 — Must `cond` be strictly bool-dtype?

NumPy accepts any cond dtype (truthiness via unsafe cast — pinned by
`test_dtype_mix`); array-api wants bool arrays; rstsr has no numeric→bool
cast trait (would need `!= 0` coercion invented); bool cond tensors are
first-class and exactly what comparison ops (`rt::greater` etc.) output.

**Recommended: strict `bool` cond** (`TensorAny<R, bool, B, D>`).
`test_dtype_mix`'s int-cond sub-case translates with explicit `a != 0` +
one `numpy_differences.md` row.

## Q5 — x/y dtype semantics: promote like the common family, generic over all dtypes

Follow the `maximum` common-op family: bound `TA: DTypePromoteAPI<TB,
Res: ...>`, output `TA::Res`. The promotion table already mirrors NumPy
(f32+f64→f64, u64+i64→f64, bool↔numeric), cond is excluded from promotion
— exactly NumPy's `result_type(x, y)`. No numeric-op bound is needed
(`where` copies elements, never computes) → dtype-generic: bool, ints,
floats, complex, halfs.

**Recommended: yes — common-family promotion + Clone-level genericity**
(no arithmetic bound; grouped as a select/elementwise op, status row under
Searching Functions per the existing table).

## Q6 — Output semantics: broadcasting and allocation

- Broadcasting: full three-way NumPy broadcast of cond/x/y. Mechanically:
  chain the existing pairwise `broadcast_layout` (NumPy rules are
  associative; a small 3-way helper or two chained calls); output dim via
  `DimMaxAPI` chaining. Mismatch → error.
- Allocation: always a fresh output (`uninit_impl` + write), like every
  common op; no storage-reuse path (output depends on all three inputs).
- Error variants: house pattern — fallible `where_f` + panicking variant,
  `rstsr_unwrap` inside the crate.

**Recommended: yes to all three.**

## Q7 — Kernel infrastructure: first 4-layout kernel vs composition

House elementwise machinery tops out at 3 layouts (out + 2 in; dispatch
helpers `_1/_2/_3` only). `where` needs out + cond + x + y. Options:

- (a) **First-class 4-layout family**: `layout_col_major_dim_dispatch_4` +
  `_par_4` in rstsr-common; `op_mutc_refa_refb_refc_func` kernels in
  rstsr-native-impl (serial + rayon, blocked-2d pattern); bridge trait
  `Op_MutC_RefA_RefB_RefC_API`; hand-written `OpWhereAPI` (the
  `OpIsCloseAPI` precedent). `clip` (also unimplemented, same 3-in/1-out
  shape) reuses the family later.
- (b) Composition over existing binaries: copy broadcast-y into out, then
  masked-assign x by cond — but masked-assign is itself a new 3-layout
  kernel, and it means two passes + double memory traffic.

Performance posture: v1 via the generic closure kernel (the same path
every common op was born on); a vectorized blend kernel is a later,
optional optimization (`where` is memory-bound; T4'-style tuning only if
it matters).

**Recommended: (a).** One new infra family, correctly generic; (b) saves
no work (still a new kernel) and bakes in 2× traffic.

## Q8 — Scalar arguments

House scalars are **strong** (promote): `rt::maximum(&f32arr, 0.5)` → f64 —
NumPy's NEP-50 *weak* scalars (`f32 + 0.5` stays f32) don't exist in rstsr
today. For `where`:

- (a) tensor-only v1 (Python literals in translated tests become 0-d
  tensors); scalar overloads later.
- (b) scalar x/y from day 1, house-strong promotion (+2 kernel variants
  `numa_refb_refc` / `refa_refb_numc`); cond tensor-only (array-api
  requires an array cond). NumPy weak-scalar tests (`test_exotic` f32-
  stays-f32, `test_scalar_overflow` OverflowError) become documented
  differences in `numpy_differences.md`.
- (c) scalars everywhere incl. cond (NumPy-max; un-idiomatic, array-api
  forbidden).

**Recommended: (b).** Ergonomic parity with `rt::maximum(&a, 0.5)`; the
marginal cost rides on Q7's kernel family. Weak scalars are a whole-library
question, not a `where` question — don't solve it here.

## Q9 — Test transfer scope (concrete list)

Proposed transfer from NumPy `TestWhere` (`_core/tests/test_multiarray.py`),
into `rstsr-core/tests/core_func/operators/test_where.rs`:

| transfer | adaptation |
| --- | --- |
| `test_basic` | as-is (dtypes, scalar-bool cond as 0-d bool tensor, strides) |
| `test_ndim` | as-is (2-D broadcast, cond new-axis, transposed operands) |
| `test_error` | as-is (non-broadcastable → Err/panic pair) |
| `test_dtype_mix` | tensor-cond parts only (int-cond → `!= 0` + difference row) |
| `test_exotic` | 0-d + zero-size parts; literal-weakness → 0-d tensors + difference row |
| + custom | bool-cond × strided × broadcast smoke; scalar x/y (if Q8b) |

Skip (documented in `numpy_differences.md`): `test_string` (no string
dtype), `test_foreign` (byteorder, N/A in Rust), `test_kwargs` (no kwargs),
`test_empty_result`/`test_largedim` (1-arg form), scalar-weakness/overflow
parts (Q8). All rows logged in `numpy_coverage.csv`; doc example in
`doc_draft/operators/`; bool results compared via `to_vec()`
(`assert_equal` is float-only).

**Recommended: this list.**

## Q10 — Process and records

This task directory is live (init prompt, facts, these questions; commits
per this repo's auto-commit rule). rstsr work would go on branch
`261006/rt-where`, **no rstsr commits without explicit user instruction**;
DECISIONS.md written here once answers land.

**Recommended: confirm as stated.**
