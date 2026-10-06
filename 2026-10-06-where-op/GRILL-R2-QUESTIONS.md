# Grilling Round 2 — rt::where proposal

Asked on screen 2026-10-06, after all three fact-finds converged
(`FACTS-numpy-where.md`, `FACTS-rstsr-where.md`). Round 2 = R1 items
re-prompted where the facts changed the picture, plus the branches the
facts unblocked. R1 answers may be folded into answers here — answer
`GRILL-R2-ANSWERS.md` (user-owned) or on screen.

---

## Q1 — Naming (re-prompt; new fact: zero precedent either way)

The workspace has **no raw identifiers and no trailing-underscore public
names** — `where` was simply never exposed, so either choice sets
precedent.

- (a) `r#where` — NumPy/array-api/torch/jax name; rustdoc renders `where`;
  every call site reads `rt::r#where(&c, &x, &y)`; first `r#` in the codebase.
- (b) `rt::select` — clean call sites, ndarray precedent; permanent name
  divergence from every peer framework and from the status-table row.
- (c) dual: primary `r#where` + alias `select` (or vice versa).
- (d) `where_`-style mangling.

**Recommended: (a) `r#where`.** Parity is rstsr's identity; the status table
already links `where`; faer-py will expose `api.where` regardless. The `r#`
is a per-callsite wart but rustdoc/doc-URLs show the plain name.

## Q2 — Scope, 1-arg form (re-prompt, quick)

NumPy's 1-arg `where(cond)` is literally `nonzero()`; array-api spec has no
1-arg form; NumPy's 1-arg tests (`test_empty_result`, `test_largedim`) are
nonzero tests.

**Recommended: 3-arg only.** `nonzero` stays its own future proposal
(also G-038's blocker).

## Q3 — Kernel infrastructure: first 4-layout kernel vs composition

House elementwise machinery tops out at 3 layouts (out + 2 in). `where`
needs out + cond + x + y. Options:

- (a) **First-class 4-layout family**: `layout_col_major_dim_dispatch_4` +
  `_par_4` in rstsr-common; `op_mutc_refa_refb_refc_func` kernels in
  rstsr-native-impl (serial + rayon, same blocked-2d pattern); bridge trait
  `Op_MutC_RefA_RefB_RefC_API`; hand-written `OpWhereAPI` (OpIsCloseAPI
  precedent). `clip` (also unimplemented, same 3-in/1-out shape) reuses it.
- (b) Composition over existing binaries: e.g. copy broadcast-y to out,
  then masked-assign x by cond — but masked-assign is itself a new 3-layout
  kernel, and it's two passes + double memory traffic.

Performance posture: v1 via the generic closure kernel (same path every
common op was born on); a vectorized blend kernel is a later optimization
(where is memory-bound; T4'-style tuning only if it matters).

**Recommended: (a).** One new infra family, correctly generic; composition
(b) saves no work (still needs a new kernel) and bakes in 2× traffic.

## Q4 — Scalar arguments

House scalars are **strong** (promote): `rt::maximum(&f32arr, 0.5)` → f64 —
NumPy's NEP-50 *weak* scalars (`f32 + 0.5` stays f32) do not exist in rstsr
today. For `where`:

- (a) tensor-only v1 (Python literals in translated tests become 0-d
  tensors); scalar overloads later.
- (b) scalar x/y from day 1, house-strong promotion (+ 2 kernel variants
  `numa_refb_refc` / `refa_refb_numc`); cond stays tensor-only (array-api
  requires an array cond). NumPy weak-scalar tests (`test_exotic` f32-
  stays-f32, `test_scalar_overflow` OverflowError) become documented
  differences in `numpy_differences.md`.
- (c) scalar everywhere incl. cond (NumPy-max; un-idiomatic, array-api
  forbidden).

**Recommended: (b).** Ergonomic parity with `rt::maximum(&a, 0.5)`; the
marginal cost rides on Q3's kernel family. Weak scalars are a whole-library
question, not a `where` question — don't solve it here.

## Q5 — cond dtype (re-prompt; mechanical backing arrived)

NumPy accepts any cond (truthiness); array-api wants bool arrays; rstsr has
no numeric→bool cast trait (would need `!= 0` coercion invented); `cond`
as `TensorAny<R, bool, B, D>` is first-class and matches comparison-op
outputs (`rt::greater` etc. produce exactly bool tensors).

**Recommended: strict bool cond.** `test_dtype_mix`'s int-cond sub-case is
translated with explicit `a != 0` + one `numpy_differences.md` row.

## Q6 — x/y dtype semantics: promote like the common family, generic over all dtypes

Follow `maximum`'s family: bound `TA: DTypePromoteAPI<TB, Res: ...>`, output
`TA::Res` — the NumPy promotion table already (f32+f64→f64, u64+i64→f64,
bool↔numeric, cond excluded from promotion, exactly `result_type(x, y)`).
No numeric-op bound needed (where copies, never computes) → works for
bool, ints, floats, complex, halfs.

**Recommended: yes — common-family promotion + Clone-level genericity.**

## Q7 — Test transfer scope (concrete list)

Proposed transfer from NumPy `TestWhere` (`_core/tests/test_multiarray.py`):

| transfer | adaptation |
| --- | --- |
| `test_basic` | as-is (dtypes, scalar-bool cond as 0-d bool tensor, strides) |
| `test_ndim` | as-is (2-D broadcast, cond new-axis, transposed) |
| `test_error` | as-is (non-broadcastable → Err/panic pair) |
| `test_dtype_mix` | tensor-cond parts only (int-cond → `!= 0` + difference row) |
| `test_exotic` | 0-d + zero-size parts; literal-weakness → 0-d tensors + difference row |
| + custom | bool-cond × strided × broadcast smoke; scalar x/y (if Q4b) |

Skip (documented): `test_string`, `test_foreign` (byteorder), `test_kwargs`
(no kwargs in Rust), `test_empty_result`/`test_largedim` (1-arg), scalar-
weakness/overflow parts (Q4). All rows logged in `numpy_coverage.csv`;
differences in `numpy_differences.md`; doc example in `doc_draft/operators/`.

**Recommended: this list.**

## Q8 — faer-py scope (re-prompt)

faer-py G-037 wants rust-side `where`; array-api `test_where` +
`test_where_with_scalars` (2024.12) consume it later.

**Recommended: Rust only now** (core + devices + tests + docs); faer-py
wiring is a follow-up in `2026-10-04-rstsr-faer-py` (W3 searching wave).

## Q9 — Process confirmation (tiny)

This task dir is live (R1/questions committed; facts files committed).
rstsr work would go on branch `261006/rt-where`, **no rstsr commits without
explicit instruction**; everything else (proposal/DECISIONS) recorded here.

**Recommended: confirm as stated.**
