# Grill Round 2 — reduction APIs (rstsr-core)

Round 1 questions: `GRILL-R1-QUESTIONS.md`. Answers below arrived on screen
2026-10-05 and are transcribed here by the agent as background (the answers
file remains user-owned; on-screen answers were not written there).

## Transcription of R1 answers (agent-authored record)

> Q1/7: Okay.
>
> Q2/3: I'm considering that use a new function `sum_with_dtype` along with
> other existing functions, so that not going to make breaking changes.
> `sum_with_dtype` should be similar to `sum_with_args` that pass similar
> arguments, but with different type handling, and `sum_with_dtype` can have
> its own `TOut` parameter generic.
>
> Q4/5: Well good, I didn't thought of that before. Well you can create
> ReduceArgs/VarArgs/NormArgs. Also, `axes` should be also a field to these
> structs (not spell `axis` like numpy at rust side rstsr, this is intentional).
>
> Q5: Yes, defer SVD-dependent functions.
>
> Q6: Whatever that follows array-api.
>
> Q8: Well if you think 3 PRs are better, then you may need three branch names.
> But anyway, you should stop at the point on each task, and I will manually
> review and merge them.
>
> Q9: Maybe you need a fresh new sub-agent on this task. Also, you may note in
> some noticeable place that code style may change during development, and this
> is just as reference, and future tasks should not strictly follow this style:
> if existing rstsr-code-style is out-of-time or not documented correctly, you
> should ask the user what is divergent, and ask whether an update is needed.
>
> It may happen that I am not directly answering to your question. Re-ask if
> you still feel blur.

## Settled design tree (as of R2)

- **Q1 (contract)**: array-api 2025.12 normative; NumPy as test oracle.
- **Q2/Q3 (dtype)**: NO breaking changes. Existing `sum/sum_axes/...` keep
  `TOut = T`. New `*_with_dtype` function family with its own `TOut` generic
  (user-specified via turbofish; no runtime dtype value exists to infer from),
  taking the same style of args as `*_with_args`. Mechanism (carried from
  R1 Q3, uncontested): cast-then-reduce composition on top of a new public
  tensor-level cast API — folds happen in the wide type (spec anti-overflow
  semantics), costs one intermediate allocation (accepted; efficiency not the
  concern). Plain `sum(int<64)` keeps narrow accumulation; docs should steer
  rust users to `sum_with_dtype` for narrow ints.
- **Q4 (args structs)**: `ReduceArgs { axes, keepdims }`, `VarArgs { axes,
  keepdims, correction }`, `NormArgs` (shape → R2 Q1), all ReshapeArgs-style
  (all-`Option`/sensible `Default`, `From`-overloads, `impl Into<Args>`).
  `axes` is a struct FIELD (plural spelling intentional, unlike numpy `axis`).
  `<fam>_with_args[_f](x, args)` generated for all reduction families incl.
  arg*/all/any/count_nonzero (keepdims everywhere, fixes G-041); existing
  `<fam>`, `<fam>_all`, `<fam>_axes` remain untouched thin wrappers.
  Whole-reduce + keepdims → shape `(1,) * ndim`.
- **Q5 (norm)**: NormOrd enum (1, 2, Inf, -Inf, 0, -1, -2, P(f64), Fro, Nuc)
  with `From<f64>` overloads; vector ords + Fro + matrix 1/-1/inf/-inf via
  composition now; SVD-dependent matrix 2/-2/nuc DEFERRED (register as gap);
  `l2_norm` stays as fast path; int input cast to float; `TOut = T::Real`.
- **Q6 (cumulative)**: follow array-api — names `cumulative_sum/prod`; single
  axis; `axis=None` valid for 1-D only (spec-strict: error for ndim>1, since
  "providing an axis must be required"); `CumSumArgs`-style struct with
  `include_initial`; no `reverse`; spec unsigned-widening when dtype is given;
  serial kernels first, rayon twin later. NumPy cumsum transcription tests
  will document the flatten-on-None divergence (np flattens, spec errors).
- **Q7 (custom reduce)**: `reduce_all[_f]` / `reduce_axes[_f]` with
  init/fold/combine/finalize closures, accumulator and output generic,
  `Send + Sync` bounds, device trait over closures; axes/keepdims via the same
  args struct (axes moved into struct per Q4 answer); Lp-norm doc example.
- **Q8 (process)**: three PRs on three branches (proposed:
  `261005/reduction-withargs-dtype`, `261005/norm-custom-reduce`,
  `261005/cumulative`); after each PR is opened, STOP for manual review/merge
  before starting the next. Fork `ajz34`, PRs to RESTGroup/rstsr.
- **Q9 (skill)**: fresh subagent drafts `rstsr-agents/skills/rstsr-code-style/`
  at wrap-up from this task's actual procedure; SKILL.md carries a noticeable
  notice: reference-not-normative, style may change during development, future
  tasks must not follow it strictly; on divergence/out-of-date content, ask
  the user what diverged and whether to update.

faer-py consequence (derived, uncontested): Python-visible widening defaults
(e.g. uint8 sum → uint64 when dtype=None) are produced by the shim dispatching
to `sum_with_dtype::<u64>` — the W2 generated-dispatch pattern, driven by
core-side type-level traits, not a shim-side algorithm.

## Round 2 questions

❓ **Q1** - **norm signature: where does `ord` live?**
Two spellings, given that `axes`/`keepdims` now live in `NormArgs`:

(a) `ord` stays the first positional argument everywhere:
`norm(x, ord)` (whole-flatten), `norm_all(x, ord)` (scalar?),
`norm_with_args(x, ord, args: NormArgs { axes, keepdims })`.

(b) `ord` is a field of `NormArgs`:
`norm(x, args: impl Into<NormArgs>)` with `NormArgs { ord, axes, keepdims }`
and `From`-overloads so `norm(x, 2.0)` / `norm(x, NormOrd::P(1.5))` still read
naturally — closer to how `ReshapeArgs`/`DiagonalArgs` fold all options into
one struct.

➡️ Recommend (b): consistent with the just-settled "everything in the args
struct" decision, one fewer positional, `From` overloads keep the short
spellings; `norm(x, ord)` shorthand comes free via `From<NormOrd>`.
Scalar variant `norm_all(x, ord)` kept as a thin wrapper (ord positional
there is fine since it is the only argument).

❓ **Q2** - **`*_with_dtype` family scope and spelling**:
Proposed: `with_dtype` variants only for accumulator/float-typed families —
`sum`, `prod`, `mean`, `var`, `std`, `cumulative_sum`, `cumulative_prod` —
NOT for min/max (no accumulation, spec has no dtype), arg*/count (usize is
the index contract), all/any (bool contract). Spelling:
`sum_with_dtype::<TOut>(x, args)` — args is the same `ReduceArgs` (axes +
keepdims inside), `TOut` turbofished; `_f` twin; TensorAny method. Bounds:
`T: DTypeCastAPI<TOut>` plus the family's reduce bounds on `TOut`
(impl detail). mean/var/std with dtype: `TOut: Float`-ish constraint;
this is also the shim's route for integer inputs (mean(int) → f64) since
plain `mean` stays `ComplexFloat`-only (non-breaking).

(a) confirm the family scope above;
(b) confirm `sum_with_dtype` and `sum_with_args` BOTH exist (with_args = keepdims
entry at native dtype; with_dtype = same + explicit TOut), vs collapsing
into only `*_with_dtype` (with_args would be redundant — `with_dtype::<T>`
covers it, at the cost of a pointless turbofish for the common case).

➡️ Recommend: (a) as listed; (b) keep BOTH — `sum_with_args` is the
ergonomic keepdims entry, `sum_with_dtype` the dtype-controlling one; the
macro can define with_dtype in terms of with_args internally.

❓ **Q3** - **name of the new tensor-level cast**:
It does not exist yet (only scalar `DTypeCastAPI::into_cast` and the assign
kernel). A dtype change always copies, which collides with rstsr's
`to_` = zero-copy-view / `into_` = consumes-ownership conventions. Options:
`astype` (numpy/array-api name; the pyo3 shim wants exactly this name
eventually), `into_cast` (matches the scalar trait but takes `&self`, not
consuming), `to_cast` (violates the to_-means-view convention).

➡️ Recommend `astype[_f](x) -> Tensor<TOut, B, D>` (borrowing, always
copies, name matches numpy/array-api; document in its docstring why it
deliberately steps outside the to_/into_ naming convention).
