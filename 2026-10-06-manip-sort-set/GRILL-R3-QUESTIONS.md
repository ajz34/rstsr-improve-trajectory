# Grill Round 3 — manip/sort/set wave (2026-10-06)

Agent-authored. Answers go in `GRILL-R3-ANSWERS.md` (user-owned) or on screen.

## Background — R2 answers as given on screen (agent-transcribed)

- Q1/2 (nonzero 3-tier; unique/isin kernel plan): Okay, conditional on
  array-api-tests passing.
- Q1/2 additional: col-major technique = reverse all axes → row-major
  computation → reverse output axes; user believed it lives in tier 2
  (between tensor layer and real implementation). → **Fact-checked this
  round**, see Q1.
- Q3 (RepeatArg), Q4 (AxisIndex), Q5 (searchsorted with_args), Q6
  (sort_custom per OpReduceCustomAPI): Okay.
- Q7 (comparator trait): traits in rstsr-dtype-traits start with `Ext*`;
  user floated `ExtSortCmpAPI`, unsure; must cover integers and real floats.
  → Fact-checked, see Q2.
- Q8 (workflow): single branch `261006/manip-sort-set`, 9 stage-commits,
  per-stage code review once (no implement-review loops), final review
  after everything. Models via claude-code-router: main = glm/glm-5.3
  ("Custom Opus", effort high); implementation subagents = glm/glm-5.3-flash
  ("Custom Sonnet", effort max); code review = DeepSeek/deepseek-flash
  ("Custom Haiku", effort high, exceptions max; final review max). Not all
  review comments must be applied — but unapplied ones must be noticed and
  addressed with justification. No push, no `gh pr`.
- Q9 (`diff`): **not** manipulation (involves real computation) →
  `tensor/diff.rs`; plain non-overloaded signature
  `diff(x, axis: isize, n: usize, prepend: Option<&TensorAny>, append: Option<&TensorAny>)`.

## New facts this round (from codebase + suite check)

- Axes-reversal: `Layout::reverse_axes` (rstsr-common layoutbase.rs:511);
  tier-2 `match default_order() { RowMajor => kernel, ColMajor =>
  reverse-axes(+swap/uplo-flip) => kernel }` at device_cpu_serial/linalg/
  matmul.rs:35-66, device_faer/matmul.rs:300-331, op_tri.rs (3 devices).
  Tier 3 uses reversal only to derive kernels (c2r from r2c). Tier 1 has no
  order normalization (to_contig is order-parameterized instead).
- The alternative, layout-generic idiom: argmax kernels traverse strided
  layouts with `IndexedIterLayout::new(la, RowMajor)` regardless of storage
  layout; device layer hard-passes `RowMajor`; documented contract
  "row-major flat index even on a ColMajor device" (tensor/reduction.rs:47-50).
- Output allocation for order-free outputs: `new_contig(None,
  device.default_order())` (concat, creation_from_tensor.rs:784).
- Gotcha: `reshape(-1)` is NOT a C-order flatten on a col-major-default
  device (docs/order_semantics.md:129-133) — flatten-contract ops must
  iterate RowMajor explicitly, never rely on reshape.
- rstsr-dtype-traits inventory: `ExtNum` (all numeric incl complex, no
  bool), `ExtReal` (ints+floats, no bool/complex — the exact union wanted),
  `ExtFloat` (floats), plus `DType*API`/`ValWriteAPI`/`IsCloseArgs` with
  API suffix. The `Ext*` family itself is suffix-less. No supertrait
  structure; no `Ordering`-returning method anywhere; no total_cmp usage;
  impls are explicit `#[duplicate_item]` dtype enumerations.
- **Suite requirement discovered**: `test_unique_*` draws all_dtypes
  *including complex* — unique kernels must order complex values
  (spec leaves the ORDER free, but SOME deterministic order is needed to
  sort). `test_sort`/`argsort` draw real_dtypes only (bool+ints+floats);
  `test_isin` ints only; `test_searchsorted` real only.
- array-api spec: sort/searchsorted/isin x-params say "should have a
  real-valued data type" — declining complex there is spec-aligned
  (confirmed the R1 intentional-deviation stance is actually
  array-api-conformant).

## Questions

❓ **Q1 — iteration-order plan for the new kernels (col-major handling).**
Proposal, per the two idioms found: (i) all new kernels are **layout-generic**
and iterate with an explicit order — C-order flat contracts (repeat/roll
axis=None flatten, nonzero scan, unique flatten, `indices`-first-occurrence)
use `IndexedIterLayout::new(la, RowMajor)` (argmax idiom), never
`default_order()` and never `reshape(-1)`; per-axis kernels (sort/argsort/
searchsorted/take_along_axis) split layout axis/rest and iterate the rest
greedily (order-free values contract); (ii) **no tier-2 reverse-axes
adaptation anywhere this wave** (it exists — tier 2 confirmed — but only
pays off when delegating to row-major-hardwired kernels like gemm; none of
ours are); (iii) output allocation for order-free outputs via
`new_contig(None, device.default_order())` (concat pattern); sort output
also `default_order()` per your R1 note.

➡️ Recommend: **as proposed** — layout-generic kernels + explicit RowMajor
where the contract is C-order; reversal technique documented as the tier-2
fallback if a future fused kernel is row-major-specialized.

❓ **Q2 — comparator trait: name, dtype coverage, semantics.**
Four sub-decisions:
(a) **Name**: in-crate precedent is mixed — `Ext*` family is suffix-less
(`ExtNum`/`ExtReal`/`ExtFloat`), the `DType*` family has `API`. Since this
is an `Ext*`-family trait, recommend **`ExtSortCmp`** (no suffix) with
method `fn ext_total_cmp(&self, other: &Self) -> Ordering`; `ExtSortCmpAPI`
also defensible if you prefer global API-suffix consistency.
(b) **Coverage**: impls for signed+unsigned ints (`Ord::cmp`), `bool`
(suite sorts bool — required), f32/f64/half (NaN-last), **and
`Complex<f32/f64>` lexicographic (re, then im), NaN-components-last** —
complex impls exist *for the unique kernels' internal sort* even though
sort/argsort/searchsorted/isin tensor APIs still decline complex at the
tensor layer (spec "real-valued" note). Unique's output order stays
unspecified-by-spec; lexicographic is just a deterministic choice.
(c) **Semantics**: `Equal` when `a == b` (covers ±0.0, matching NumPy's
treat-as-equal in sort), NaN (or NaN component) ordered last; NOT the raw
`total_cmp` bit trick (it orders −0 < +0 and NaN payloads, deviating from
NumPy visibly in rstsr-core parity tests).
(d) **Placement**: standalone trait in rstsr-dtype-traits (sibling of
`ExtReal`), not a method on `ExtReal` — `ext_min/ext_max` commit to IEEE
NaN-skipping semantics; a total order next to them would be a trap.

➡️ Recommend: **ExtSortCmp, coverage per (b), semantics per (c),
standalone per (d)**.

❓ **Q3 — confirmation ledger (small items; reply "okay" or correct
individually).**
1. `diff` axis parameter: you sketched `axis: isize`, but R2-Q4 settled
   `AxisIndex` for diff. Plan: parameter stays `impl TryInto<AxisIndex<isize>,
   Error: Into<Error>>` (accepts plain `isize` literals transparently, so
   your sketch still compiles at call sites); everything else in your sketch
   as written, with the return type concretized to `Tensor<T, B, D>` and the
   `_f` twin added.
2. Your Q8 answer's last sentence was cut off mid-thought ("After all code
   finishes, you are going to ask DeepSeek/deepseek-flash …"); interpreted
   per the bullet below it: final review by DeepSeek/deepseek-flash at
   effort max. Correct?
3. Router mapping used by the orchestrator: Agent-tool `model: "sonnet"` →
   glm/glm-5.3-flash (implementation), `model: "haiku"` → DeepSeek/deepseek-
   flash (review). Correct?
4. `take_along_axis`: strict same-shape-except-axis (no NumPy-style
   broadcast), matching the array-api description and the suite.
5. Complex `unique_*` is suite-required (hence Q2b); complex
   sort/argsort/searchsorted/isin stay declined at the tensor layer
   (spec-aligned), with `sort_custom` as the documented escape hatch.

➡️ Recommend: **all five as stated**.

## Frontier after this round (preview, not yet askable)

None — Q1–Q3 close every open branch. After your answers I write
`DECISIONS.md` and the wave plan (9 stages, models, review gates), then
implementation begins on branch `261006/manip-sort-set`.
