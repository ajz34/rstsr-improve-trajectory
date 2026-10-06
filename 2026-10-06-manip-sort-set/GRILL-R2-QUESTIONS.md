# Grill Round 2 — manip/sort/set wave (2026-10-06)

Agent-authored. Answers go in `GRILL-R2-ANSWERS.md` (user-owned) or on screen.

## Background — R1 answers as given on screen (agent-transcribed)

- Q1: `diff` **in scope**. clip / log1p / dtype functions / linalg out; fft out
  for this project at this time.
- Q2: delivery in **multiple commits, no PRs**. Agent granted to act
  autonomously (commit included — explicit session instruction), but
  **prohibited from pushing to remote and from `gh pr`**.
- Q3: `unique_*`×4 and `isin` are **3-tier device-dependent** (composition
  judged costlier / more memory-intensive). `nonzero` — user undecided, wants
  clarification (→ Q1 below). Kernel discipline: rstsr-native-impl kernels
  take `&mut output` + `&inputs`; auxiliary memory should stay **O(1)** beyond
  inputs/outputs unless efficiency demands an exception (matmul's
  contiguity-copy is the exemplar exception, and it is rare — check current
  status before adding more).
- Q4: rename `Repeats` → **`RepeatArg`**; open question whether it needs a
  tensor-input variant (→ Q3 below); introduce **`AxisIndex`** (exactly one
  axis) alongside `AxesIndex`; `SearchSide` gets Into/TryInto impls and is
  passed as a convertible parameter; `UniqueCounts`/`UniqueInverse`/
  `UniqueAll` are python named-tuples — implement `Into<(Tensor, Tensor)>`
  for the 2-field ones; numpy + array-api sources at `~/Git-Others`.
- Q5: no complex sort/argsort — intentional NumPy deviation, follow
  array-api. NaN/±0: array-api compliance first; if that is very inefficient,
  write a side function; where spec/tests are silent, agent's proposal stands
  (NaN last ascending, comparator-reversal descending, stable both
  directions). Sort output layout = device default order (not input layout).
  Allowed to edit rstsr-dtype-traits, add traits, make API-breaking changes
  when proper. Complex support traits only if array-api requests (tests do
  not).
- Q4/5 additional: provide **`rt::sort_custom`**-style custom-comparator sort
  (like `rt::reduce` custom); complex sort becomes a documented docstring
  example + test via custom comparator, but rstsr ships no complex sort
  itself. NumPy's `kind` parameter noted (not adopted).
- Q6: **(b)** — `sorter` is a numpy-parity rust-native parameter of
  `rt::searchsorted`; sub-question on entry style open (→ Q5 below).
- Q7: flip capabilities `data-dependent shapes` → True with the nonzero PR.
- Q8: correctness first, no bench deliverable, at agent's leisure.
- Q9: agreed; rstsr-core tests follow numpy-parity conventions
  (rstsr-agents skills `core-test` / `test-conventions`).
- Workflow note for implementation: task is huge; main session orchestrates,
  subagents do the real implementation and checking; **sequential, not
  parallel**; keep main-session token use low.
- Evidence note: `TensorViewAPI` introduces both `&TensorAny` / `TensorView`
  input forms — user unsure whether it applies here (fact-check running;
  folded into Q4 note below).

## Questions

❓ **Q1 — `nonzero`: semantics, and which tier?**
Semantics (array-api / NumPy): for x of shape (d0, d1, …), returns one 1-D
index array per dimension, together locating every element != 0 (bool: True;
complex: either component nonzero), in **strict row-major C order**; 0-d
input raises; indices int32/int64; suite tests all dtypes and checks C-order
position-by-position.
Options: (a) **3-tier**: kernel two-pass — pass 1 counts nnz, tensor level
allocates, pass 2 fills flat C-order indices; per-dim coordinate split is
host layout math; O(1) aux; (b) composition: elementwise `!= 0` → `to_vec`
→ host scan (materializes an O(n) host Vec, exactly the memory-heaviness you
moved unique/isin away from).

➡️ Recommend: **(a) 3-tier**, two-pass count+fill — consistent with the
unique/isin decision, honors the O(1)-aux discipline, and the count pass is
trivially rayon-parallel (a sum reduction over a bool map).

❓ **Q2 — `unique_*` / `isin` kernels inside the O(1)-aux discipline.**
Proposed kernel plan (all work happens in the output buffer, matmul-style
exceptions called out explicitly):
- `unique_values`: copy x flattened C-order **into the output storage**
  (capacity n), sort in place (stable), adjacent-dedupe compaction in place,
  truncate to u. Aux: O(1) beyond output (std/rayon sort's internal temp is
  implementation-internal).
- `unique_all` (one C-order re-scan of the input after values exist):
  binary-search each element into the sorted values for `inverse_indices`;
  first-hit sets `indices[k]` (needs a `seen` bitvec, **O(u) bits** — smaller
  than the `indices` output itself, so still within "no more than outputs");
  `counts` tallied in the same sweep. NaNs: each NaN element is its own
  entry, slots in C-order (consistent with stable NaN-block at sort end);
  signed zeros merge, first-seen sign kept.
- One device trait `OpUniqueAPI` with **two methods**: `unique_values`
  (sort+compact only) and `unique_all` (full sweep); `unique_counts` /
  `unique_inverse` wrap `unique_all` and discard the unneeded outputs.
- `isin`: sort x2's unique values in an **O(m) temp buffer** (binary search
  per x1 element, bool output). This temp buffer is a genuine
  beyond-inputs/outputs allocation — the numpy algorithm; register it as an
  efficiency-motivated exception (alternatives: O(n·m) pairwise (no aux,
  slow), hash set (O(u), also beyond-outputs)). `invert` = flip at write.

➡️ Recommend: **as proposed** — output-buffer sorting, two-method trait,
O(u)-bits justified as ≤ outputs, isin's O(m) temp registered as the one
explicit exception.

❓ **Q3 — `RepeatArg` final shape (tensor variant needed?).**
Python `repeats` is `Union[int, array]` where array is 1-D ints. A 1-D
tensor's logical element order is layout-independent, and the py wrapper
marshals via `tolist()` (the `take` precedent), so rust-side a tensor variant
buys nothing: propose `RepeatArg { All(usize), Elems(Vec<usize>) }` with
`From<usize>/From<Vec<usize>>/From<&[usize]>`. Semantics: `Elems` length must
be 1 (≡ All) or exactly the axis size (total size when axis=None); anything
else errors. axis=None flattens in **C-order** regardless of the device's
default order (array-api contract, same as argmax's flat contract — col-major
devices included).

➡️ Recommend: **no tensor variant**; py-side marshalling; C-order flatten
contract documented.

❓ **Q4 — `AxisIndex` introduction scope (+ where inputs' view-ness lives).**
New type in rstsr-common next to `AxesIndex`: `AxisIndex<isize>` — exactly
one axis (negative allowed, normalized at op level), `TryInto`-with-Error
mirroring `AxesIndex`. Used by the new single-axis knobs: `SortArgs.axis`,
`take_along_axis`, `diff`; `repeat`'s axis stays `AxesIndex` (its None means
flatten, which is genuinely none-or-one). **No retrofit** of existing
single-axis fns (`take`/`index_select`/`concat` keep `isize`) — a later
cleanup if wanted.
Related evidence note — **resolved by fact-check after this round was
written**: `TensorViewAPI` (`tensor/ownership_conversion.rs:678`) is a
single-method `.view()` conversion trait over ownership forms, and it IS the
established idiom for data-consuming compute free functions (reduce family,
zip drivers, matmul/vecdot/tensordot all take `impl TensorViewAPI<...>`);
only layout-only manipulations take plain `&TensorAny` (+ `into_*` by-value).
Therefore **every new op in this wave takes `impl TensorViewAPI`** (they all
read elements and produce owned outputs). The same fact-check confirmed:
`sort_custom` mirrors `OpReduceCustomAPI` exactly (by-value
`Fn + Send + Sync` comparator through the device trait; cpu-serial + rayon
impls); no comparison/ordering trait exists yet anywhere (kernels use bare
`PartialOrd`; NaN policy is the `ArgCmp` enum in rstsr-native-impl), so the
Q7 trait is new ground; `AxisIndex` (singular) does not exist —
`rstsr-common/src/axis_index.rs` hosts `AxesIndex` with the From/TryFrom
surface to mirror in reduced single-axis form (own impls; negative-axis
`rstsr_check_axis!` normalization is the load-bearing part).

➡️ Recommend: **as proposed** (new type, new fns only, no retrofit).

❓ **Q5 — `searchsorted` entry style: with_args or explicit entries?**
(a) with_args family like argmax: `searchsorted_f(x1, x2, args: impl
Into<SearchSortedArgs>)`, `SearchSortedArgs { side: SearchSide (default
Left), sorter: Option<Vec<usize>> }`, `From` impls for `()` (defaults),
`&str`/`SearchSide` (side only), tuples; panicking `searchsorted(x1, x2,
args)`. (b) explicit positional entries: `searchsorted_f(x1, x2, side,
sorter: Option<...>)` — callers write `None` explicitly.

➡️ Recommend: **(a) with_args** — two optional knobs is exactly the
`ReduceArgs` situation; extensible (e.g. a future `descending`); consistent
with the family users already know.

❓ **Q6 — `sort_custom` / `argsort_custom` surface.**
Mirror `rt::reduce` custom: device trait carrying the comparator closure
(`Fn(&T, &T) -> Ordering`, `+ Send + Sync` for the rayon impl), tensor-level
`argsort_custom_f(x, cmp, args) -> Tensor<usize>` and `sort_custom_f(x, cmp,
args) -> Tensor<T>`. Built-in sort/argsort are this machinery with the
default comparator (Q7) — no duplicated kernels. Docstring + doctest shows
complex lexicographic sort as the example; rstsr itself ships no complex
sort. NumPy's `kind=` recorded as intentional deviation (array-api has no
`kind`). Not exposed in the python wrapper (array-api has no such function).

➡️ Recommend: **as proposed**.

❓ **Q7 — comparison trait in rstsr-dtype-traits.**
NaN-last ordering needs `is_nan`-aware compare, but kernel bounds must cover
bool/ints too. Proposal: one small trait in rstsr-dtype-traits, e.g.
`SortCmpAPI { fn sort_cmp(&self, &Self) -> Ordering }` (floats: NaN-last
total-ish order; ints/bool: plain), implemented for all 13 dtypes, used by
the sort/argsort/searchsorted/unique kernels; additive only, no breaking
change needed. (`total_cmp` is not directly usable: it orders -NaN/+NaN and
payloads differently from NumPy's NaN-last + never-equal-NaN semantics.)

➡️ Recommend: **as proposed**; name and exact semantics fixed at
implementation review.

❓ **Q8 — branch/commit slicing without PRs.**
Single branch `261006/manip-sort-set` off main in rstsr-local-workspace;
commits sliced in stage order: (1) manip `repeat`/`roll`/`tile` core+tests,
(2) dtype-traits cmp trait + `AxisIndex`, (3) sorting `sort`/`argsort`/
`sort_custom` + kernels + tests, (4) `searchsorted` + tests, (5) set
`unique_*`/`isin` + kernels + tests, (6) `nonzero` + `take_along_axis`, (7)
`diff`, (8) faer-py bindings + capabilities flip + tracking-table updates,
(9) trajectory reports. No push, no PR. Each commit message follows
`git-commit-coauthor`; each stage compiles + tests green before the next.

➡️ Recommend: **one branch, stage-sliced commits** as listed (chained
branches per stage would complicate review without PRs).

❓ **Q9 — `diff` classification + placement.**
`diff(x, axis=-1, n=1, prepend, append)` = optional concat + n ×
(slice-pair subtraction) — expressible as tensor-level composition over
existing slicing/subtraction (no new kernels, no device trait; the n-loop is
host-side but each pass is a real device op). Placement
`tensor/manipulation/diff.rs` (array-api files it under utility, but it is
manipulation-shaped; NumPy keeps it in function_base — placement follows our
module map, docstring cites array-api).

➡️ Recommend: **device-independent composition**, `manipulation/diff.rs`.
