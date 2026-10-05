# Status — rstsr_faer.api vs the Python array API (2026-10-05)

Current standing of the rstsr-faer-py conformance workstream: numbers, the
full failure table, and every open problem sorted by *where the fix has to
happen* — Python shim, rstsr fix, rstsr implementation, or a structural
limitation of the current scheme.

## 0. Artifacts

| artifact | where | tracked? |
|---|---|---|
| full per-test table (1382 rows, 16 cols), current | `harness/reports/COMPLIANCE-FULL-20261005-094602.csv` | no (gitignored; regenerate with the script) |
| generator | `harness/compliance_table.py` | yes |
| merged suite report (source of the CSV) | `harness/reports/rstsr_faer_api-MERGED-20261005-094602.json`, merged from the 19 `…-chunk-*-20261005-094602.json` | no (gitignored) |
| W2 narrative, per-file deltas, declines | `reports/SUMMARY-w2.md` | yes |
| W0+W1 state (this file's starting point) | stamp `20261004-233924`; kept in git history | — |
| gap register (per-divergence entries) | `../GAP-REGISTER.md` | yes |
| reduction plan (wave order) | `../FAILURE-REDUCTION-PLAN.md` | yes |

```bash
cd 2026-10-04-rstsr-faer-py/harness
NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh   # full red map, ~60 s, 19/19 chunks
# run.sh prints the totals only; merge the chunk reports into one json first
# (snippet in harness/README.md, "Merging chunk reports"), then:
python3 compliance_table.py reports/rstsr_faer_api-MERGED-<stamp>.json \
    -o reports/COMPLIANCE-FULL-<stamp>.csv --census
```

## 1. Headline numbers

Pins (identical to the NumPy baseline, so numbers compare directly):
array-api-tests `6c0b59f` + spec submodule `5f847a3`, standard 2025.12,
1382 collected items, derandomized, suite defaults (100 examples).
Subject wheel: rstsr branch `261004/rstsr-faer-py` @ `799c4b3`
(`1546e5e` = rust-side `positive`, `799c4b3` = shim elementwise + dunders).

| run | passed | failed | skipped | delta |
|---|---|---|---|---|
| NumPy 2.5.1 baseline (S0 gate) | 1335 | 42 | 5 | — |
| rstsr_faer.api S1 (first red map) | 255 | 1040 | 87 | `a52bf4f` skeleton |
| rstsr_faer.api S2 (DLPack) | 302 | 998 | 82 | +47 |
| rstsr_faer.api W0+W1 (indexing + quick wins) | 320 | 980 | 82 | +18, −18 |
| rstsr_faer.api W2 (elementwise + operator dunders) | **902** | **398** | **82** | **+582, −582** |

Per spec area (stamp `20261005-094602`; totals identical at `-020258`,
per-file counts wobble ±1 from hypothesis DB replay):

| suite file | pass | fail | skip |
|---|---|---|---|
| test_special_cases.py | 508 | 109 | 0 |
| test_has_names.py | 122 | 92 | 0 |
| test_signatures.py | 113 | 62 | 37 |
| test_operators_and_elementwise_functions.py | 100 | 55 | 5 |
| test_data_type_functions.py | 22 | 14 | 0 |
| test_manipulation_functions.py | 2 | 11 | 0 |
| test_creation_functions.py | 6 | 10 | 0 |
| test_statistical_functions.py | 0 | 9 | 0 |
| test_array_object.py | 19 | 8 | 0 |
| test_searching_functions.py | 1 | 7 | 0 |
| test_set_functions.py | 0 | 6 | 0 |
| test_linalg.py | 0 | 4 | 26 |
| test_utility_functions.py | 0 | 4 | 0 |
| test_inspection_functions.py | 1 | 3 | 0 |
| test_indexing_functions.py | 0 | 2 | 0 |
| test_sorting_functions.py | 0 | 2 | 0 |
| test_constants.py | 5 | 0 | 0 |
| test_dlpack.py | 3 | 0 | 0 |
| test_fft.py | 0 | 0 | 14 (self-skip) |

**62% of the remaining 398 are still surface-existence** (246 missing
names): W3+ territory — reductions/stats (`mean prod std var max min
cumulative_*`), manipulation (`concat stack expand_dims squeeze flip
moveaxis …`), creation (`*_like eye linspace tril triu meshgrid`),
`xp.linalg` (23), sorting (`sort argsort`), searching/set, dtype queries
(`result_type can_cast isdtype`), fft (14), plus `log1p` (20; declared
trait, device kernels are TODO — G-058). The rest is value adjudication
(79 wrong-value), rust-side declines that now raise register-referenced
errors (60 unexpected-exception), mask/fancy indexing (6), finfo-complex
(4), unclassified (2), a reduction `axis=` binding (1).

## 2. Failure census (398, W2) with fix layer

Root cause from the crash message, fix layer from the capability check
(Appendix). Cross-tab of the two (stamp `20261005-094602`):

| root cause | n | fix layer |
|---|---|---|
| missing-surface (name absent) | 246 | shim-bind 108, rust-impl 98, mixed 23, suite-scope 14, shim-py 3 |
| wrong-value (ran, wrong result) | 79 | mixed-values 79 |
| unexpected-exception | 60 | rust-fix-or-shim 60 |
| indexing-gap (mask/fancy) | 6 | rust-impl 6 |
| dtype-fn-bug (finfo complex) | 4 | shim-rs 4 |
| unclassified | 2 | unclassified 2 |
| axes-not-bound (reduction axis=) | 1 | shim-bind 1 |

Fix-layer totals: **shim-bind 109** (bind an existing `rt::` primitive —
W3/W4/W6 surface), **rust-impl 104** (no usable rstsr primitive, G-001…G-058
family), **mixed-values 79** (adjudication), **rust-fix-or-shim 60**
(register-referenced declines from the W2 pass), **mixed 23** (`xp.linalg`
namespace), **suite-scope 14** (fft), **shim-rs 4** (finfo), **shim-py 3**
(namespace-info), **unclassified 2**. (`log1p`'s 20 rows were re-classified
shim-bind → rust-impl during this status pass; see G-058.)

For reference, the W0+W1 census (980 failures at stamp `20261004-233924`):
missing-surface 762 (shim-bind 558, shim-alias 82, rust-impl 82, mixed 23,
suite-scope 14, shim-py 3), missing-dunder 159 (shim-bind 159),
promotion-decline 14 (rust-design 14), unexpected-exception 12,
wrong-value 12, marshal-reject 10 (shim-py 10), indexing-gap 6,
dtype-fn-bug 4, axes-not-bound 1. W2 collapsed the shim-bind + shim-alias
classes (800 failures) to 109 remaining shim-bind, and converted the
cross-dtype/marshal declines into register-referenced exceptions.

---

## 3. Category A — Python-specific (shim work; zero rstsr change)

Everything here rides an existing rstsr primitive; it is binding,
marshalling, validation, or signature work.

**W2 status:** A1 and A2 are **landed** (commit `799c4b3`); A3, A4 remain
open for W3–W6, plus the A5 shim bugs.

### A1. Elementwise surface (the single biggest unlock, ~500 tests)
**Landed in W2** — 56 names bound. Residual declines are registered:
integer inputs to `ceil/floor/trunc/round/conj` (G-052), `pow` int/complex
(G-053), `signbit` (inverted kernel, G-054), `expm1(complex)` and complex
special values (G-055), `log1p` (no device kernel, G-058). The pre-W2
inventory below stays as the record:

~50 names, all with an `rt::` counterpart:

- unary: `acos acosh asin asinh atan atanh ceil conj cos cosh exp expm1
  floor imag log log1p log2 log10 real reciprocal round sign signbit sin
  sinh sqrt square tan tanh trunc`
- binary: `atan2 copysign floor_divide hypot maximum minimum nextafter pow`
- unchanged semantics (same name in `rt::`): `abs`, `sign`, `square`, `sqrt`,
  `exp`, `log`, `round`, ...
- aliases (rt name differs): `remainder`→`rem`, `logaddexp`→`log_add_exp`,
  `bitwise_and/or/xor`→`bitand/bitor/bitxor`, `bitwise_left/right_shift`→
  `shl/shr`, `bitwise_invert`/`logical_not`→`not`,
  `logical_and/or/xor`→`bitand/bitor/bitxor` on bool (validated bool-only
  in the wrapper, as the spec requires).

Two gaps in the family are *not* in rt:: — see C.

### A2. Operator dunders (159 tests)
**Landed in W2** — all dunders bound, including `__pos__` via the new
rust-side `rt::positive` (`1546e5e`); `__matmul__` remains with W6
(`xp.linalg`).

`__pow__ __ipow__ __floordiv__ __ifloordiv__ __mod__ __imod__ __and__
__or__ __xor__` + in-place twins `__lshift__ __rshift__ __matmul__ __abs__
__neg__ __invert__`. All map to existing `rt::` fns (in-place dunders may
legally be implemented as `self = op(self, other)` — the spec does not
require true in-place behavior).
The shim's `_binary` helper already handles weak scalars within the same
kind; the operator definitions are a table plus the existing dispatch.

### A3. Reductions, manipulation, creation, linalg bindings
- reductions/stats: `max min mean prod std var argmax argmin count_nonzero`
  (+ `sum`/`all`/`any` already exist but whole-array only — `axis=`/`dtype=`
  unsupported, see A5).
- manipulation: `expand_dims squeeze flip moveaxis concat stack unstack
  broadcast_arrays broadcast_shapes` (`permute_dims`, `broadcast_to`,
  `reshape` done in W0/W1).
- creation: `zeros_like ones_like full_like empty_like eye linspace meshgrid
  tril triu` (all in `rt::`).
- linalg (already in `faer_impl`): `cholesky det eigh eigvalsh inv pinv
  solve svd svdvals` + core `matmul matrix_transpose vecdot`. Requires the
  shim to take up `rstsr-linalg-traits`; the `xp.linalg` namespace object
  itself is Python-only.
- `take` (per-axis) is in rt (`take_f`); `take_along_axis` is not.

### A4. Protocol / signature / introspection (shim-py, ~16 tests)
Found by this audit (all Python-layer):

1. `astype` signature lacks `device=` (spec 2025.12) — `Argument 'device'
   missing from signature`.
2. `reshape` declares `shape` positional-only; the suite requires pos-or-kw.
3. `__array_namespace_info__().devices()` returns a **list**; 2025.12
   requires a **tuple**.
4. `default_device()` is missing on the namespace-info object.
5. `dtypes()` takes no argument; the spec requires `dtypes(kind=...)` (and
   the kind vocabulary `bool / signed integer / unsigned integer / real
   floating / complex floating` plus composites `integral / numeric`).
6. `default_dtypes()` returns keys `real/integral/complex`; the required
   keys are `"real floating" / "complex floating" / "integral" / "indexing"`.
7. `capabilities()["boolean indexing"]` is **True but masking raises
   NotImplementedError** (G-038) — an overclaim the suite can act on;
   must be `False` until the rust primitive lands. `"data-dependent
   shapes": False` is correct today. `"max dimensions": 8` is a placeholder
   (rstsr has no ndim cap; decide `None` or a real bound).
8. `test_setitem` shape-(0,0) empty assign — see B3 (rust-side edge).

### A5. Behavior bugs found in this audit (shim, concrete)
Not in the register before; all reproduced on the current wheel:

| bug | evidence | fix |
|---|---|---|
| `isnan/isfinite/isinf` on **0-d non-float** returns shape `(1,)` (should be `()`) | `xp.isfinite(xp.asarray(0, dtype=xp.uint8)).shape == (1,)`; suite `out.shape=(1,), but should be ()` ×3 | shim: the int/bool constant-result fallback (G-017 workaround) builds a shape-(1,) array; must match the input shape |
| `all/any` ignore `keepdims=True` | `xp.all(xp.asarray([False]), keepdims=True).shape == ()`, should be `(1,)` | shim: honor keepdims when binding `*_axes` |
| `sum` lacks `axis=` and `dtype=` | `xp.sum(x, axis=0)` → NotImplementedError; `xp.sum(x, dtype=None)` → TypeError unexpected kwarg | shim: bind `sum_axes`; accept `dtype` (see B2 for the accumulation-dtype rule) |
| `sum(uint8 0-d)` returns `uint8`, spec expects `uint64` | suite `out.dtype=uint8, but should be uint64 [sum(uint8)]` | partly rust semantics — B2 |
| `finfo(complex64)` raises ValueError; spec requires real-part semantics | 4 tests | shim's `info.rs` validation (G-032) |

---

## 4. Category B — rstsr-side fixes (primitive exists, semantics incomplete)

| id | item | evidence | note |
|---|---|---|---|
| B1 | `negative`/unary minus refused for **unsigned** dtypes | suite grades `uint8` (spec/numpy wrap the value); `negative: not defined for bool/unsigned dtypes` | rt's `neg` deliberately refuses; spec expects defined behavior. Also gates `__neg__`. (bool exclusion is fine — spec excludes bool.) |
| B2 | integer reductions keep the input dtype; spec wants the default integer accumulation dtype | `sum(uint8)` → `uint8`, expected `uint64` (signed: `int64`) | affects `sum`/`prod` (+ `mean/std/var` promotion to float, to verify when bound) |
| B3 | empty-tensor assign rejected | `Array(shape=(0,0))[:, :] = ...` → `InvalidLayout: cannot assign to broadcasted tensor` | zero-size dims produce stride 0, which the write gate reads as broadcast; needs an `is_broadcasted`/assign gate fix for size-0 dims (rust-side; wrapper-only rule forbids a shim special case) |
| B4 | `all`/`any` implemented for bool only | G-017; spec requires truthiness on every dtype | widen existing impls (no new algorithm) |
| B5 | `isnan/isfinite/isinf` implemented for float/complex only | G-017; spec requires constant results for ints/bools | widen existing impls; also removes the shim's constant-array workaround (and the A5 shape bug) |
| B6 | `zeros/ones/full` tuple impls gated on `T: Num`; bool excluded | G-016 | shim currently assembles bool fills via `asarray(Vec<bool>)`; a bool-capable creation path removes the workaround |

## 5. Category C — rstsr-side implementations (no primitive today)

Every item below is a register entry in `GAP-REGISTER.md`; none can be
fixed shim-side under the wrapper-only rule.

**C1. Indexing/masking (the load-bearing rust item).** Mask gather/scatter
(`x[mask]`, `x[mask] = v` — G-038), integer-array fancy indexing with
broadcasting (G-039), array mixed with slices (G-035). rstsr has per-axis
`bool_select`/`take`/`index_select` only. The suite's own verification
helpers index with tuples/slices (fixed in W1), but the masking tests and
`where`-based verifications still block.

**C2. Elementwise ternary.** `where` (G-037) — owner ruled shim-side
implementation out; needs a rust primitive.

**C3. Sorting.** `sort`, `argsort`.

**C4. Manipulation.** `roll` (G-002), `repeat`, `tile` (G-003).

**C5. Statistics.** `cumulative_sum`, `cumulative_prod`, `clip`.

**C6. Searching/set (data-dependent shapes).** `nonzero`, `unique_all /
unique_counts / unique_inverse / unique_values`, `searchsorted`, `isin`,
`take_along_axis`.

**C7. Utility.** `diff`.

**C8. Linalg.** In `faer_impl` today: cholesky, det, eigh, eigvalsh, inv,
pinv, solve, svd, svdvals. Missing: `qr` (G-004), `slogdet` + faer
`solve_symmetric` (G-005), `matrix_rank`, `matrix_power`, `matrix_norm`,
`vector_norm`, `trace`, `outer`, `cross`, `tensordot`.

**C9. FFT.** No rstsr FFT. 14 names graded by `has_names`; the test file
itself self-skips. Open decision: implement vs. explicit scope-out with a
register entry (W7).

**C10. Runtime dtype queries.** `result_type`, `can_cast`, `isdtype`,
tensor-level `astype`/`into_dtype` (G-007/G-008/G-019). rstsr's promotion
is a compile-time associated type (`DTypePromoteAPI`), with no token-level
query. Options: rust-side runtime helper, shim-side static table mirroring
the trait (drifts), or scope-out (W5).

**C11. Cross-dtype arithmetic (G-009).** The shim declines mixed-dtype
operands for `add/subtract/multiply/divide/equal/…` (10 unexpected
exceptions + 14 promotion declines). rstsr tensor-level binary ops are
generic over the two input dtypes, but device kernels are bounded on Rust
traits that hold within a dtype family (`PartialOrd` etc.), so most mixed
pairs have no kernel; comparisons could be served with a pair-dispatch
(output is always bool), arithmetic needs a promotion map. Owner/rust-side
decision.

**C12. `positive` / `__pos__` — RESOLVED 2026-10-05 (`1546e5e`).** Rust-side
identity op landed as a first-class unary op (`OpPositiveAPI` device kernel +
`TensorPositiveAPI`; borrowed/view inputs copy through the kernel, an owned
tensor is returned as-is). The fulfillment table marks `positive`/`__pos__`
`Y`; the shim routes both through it.

## 6. Category D — difficult in the current scheme

Structural frictions, not individual missing functions. These are the
things that will not disappear when C is finally paid down.

**D1. Static dtype type-parameter vs runtime dtype token.** In rstsr the
dtype is a Rust type parameter; in Python it is a runtime value. Every
single binding needs a `match` over the handle's 13-dtype enum (13×13 for
binary ops), and the *output* dtype is a compile-time associated type, so
result-type computation cannot be expressed for mixed inputs without a
runtime promotion table. This is the root cause of C10/C11 and the reason
`astype` had to be built element-wise. Macro-generatable, but it multiplies
the work of every new op and leaks into the Python layer's dtype
validation.

**D2. Handle model has no view/alias representation.** The Python `Array`
wraps an owned tensor handle; index results, `astype`, `reshape`,
`permute_dims` are all copies. The spec permits copies for indexing, but
`asarray(copy=False)`/`astype(copy=False)` aliasing, and any future
`setitem`-through-view expectations, need a shared-storage handle
(`TensorArc`/`TensorDlpackShared` exist rust-side but are not in the handle
enum — G-014/G-036). Every op that "should" be a view becomes an allocation
and a semantic question.

**D3. Data-dependent shapes need gather/scatter kernels.** `nonzero`,
`unique_*`, boolean-mask indexing, `searchsorted`, `isin` cannot be
expressed by per-axis index_select; they need whole-tensor gather/scatter
primitives in rstsr (C1/C6). Under the wrapper-only rule no shim-side loop
is allowed (the earlier mask/fancy gathers were reverted on the owner's
ruling), so this is a hard rust-side dependency, not a schedule choice.

**D4. Promotion semantics are bigger than "cross-dtype arithmetic".** The
spec's promotion lattice (weak scalars, mixed arrays, reduction
accumulator dtypes, `dtype=None` defaults per op) is a runtime concept;
rstsr encodes it at the type level. Each reduction/op binding has to decide
its result dtype — see B2 and the `sum(dtype=...)` case.

**D5. Suite scope is asymmetric.** `test_has_names`/`test_signatures`
grade *extension* names (`linalg-*`, `fft-*`) unconditionally;
`--disable-extension` skips the extension test files but not the name
checks. So linalg must be implemented (planned) or explicitly skipped, and
fft is a binary implement/scope-out decision; there is no "declare it
optional" mechanism.

**D6. Signature and error-taxonomy exactness.** `test_signatures` compares
the Python callables' parameter lists (pos-only vs pos-or-kw, name,
presence) — the shim's Python defs have to mirror the spec signature by
signature. Separately, the spec mandates exception *types*
(`IndexError`/`AxisError`, `ValueError`, `TypeError`), which each new
binding must classify from rstsr's error variants (`err_py` does the
current mapping).

**D7. Scalar/0-d conventions.** Whole-array reductions must return 0-d
arrays (done for `sum/all/any` in W1), `item()`/`argmax` return Python
scalars, `bool()`/`int()` on size-1 arrays, etc. Small, but each new
binding re-encounters it.

**D8. Wrapper-only rule amplifies rust gaps.** Deliberately: no shim-side
algorithms means every missing primitive is a rust-side ticket, and the
failure count is honest but blocked. That is the intended cost model; it
does mean the red map cannot go green by shim work alone (≈88 tests plus
C11's decision sit on rust).

## 7. Trajectory

Wave plan and order of attack are in `FAILURE-REDUCTION-PLAN.md`; current
state against it:

- W0 (quick wins) — **landed** in `d7f4082`.
- W1 (indexing) — **basic indexing landed**; mask/fancy declined and
  registered (rust-side C1). The suite's helper crash fixed; manipulation
  failures are now value-level.
- W2 (elementwise + dunders, ~500+159 tests) — **landed 2026-10-05**
  (`1546e5e` rust-side `positive` + `799c4b3` shim): 320/980/82 →
  **902/398/82**. C12 resolved; declines registered as G-052…G-058;
  per-file deltas and census in `SUMMARY-w2.md`.
- W3 (reductions/stats/searching/set) — **next**; partially blocked by
  C3/C5/C6 (`where` G-037 remains the highest-value rust-side item).
- W4 (creation/manipulation) — mostly bindable today (roll/repeat/tile
  rust-side).
- W5 (dtype introspection) — needs a decision (C10).
- W6 (linalg) — bindable for the 9 faer kernels + core matmul/vecdot;
  QR/slogdet/norms are rust-side.
- W7 (fft) — decision: default scope-out with a register entry.

A realistic target after W3+W4 shim work: the red map drops to the
rust-blocked set (C1–C11) plus value adjudication; the honest end state is
*every remaining failure carries a register entry*, not 1382 green.

## Appendix — capability check used for the fix-layer column

Fix layer was assigned by checking each missing name against the rstsr
surface (`rstsr-core` prelude exports, `rstsr-linalg-traits/faer_impl`,
native device op macros), originally at `d7f4082`: `shim-bind` = `rt::` has
an equivalent primitive; `shim-alias` = same primitive under a different
name; `rust-impl` = absent. The W2 census re-verified the entries that
still matter against `799c4b3` (e.g. `log1p` corrected to `rust-impl` —
trait declared, no device kernel, G-058); the W3+ entries keep the W1
classification until their wave binds them. Pairs to re-check if a binding
surprises: `logical_*`→bool bit ops (`bool: BitAnd/BitOr/BitXor` —
present), `bitwise_invert`→`not` (Rust `!` is bitwise for integers —
present).
