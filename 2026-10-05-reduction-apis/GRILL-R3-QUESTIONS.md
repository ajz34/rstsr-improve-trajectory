# Grill Round 3 — reduction APIs (rstsr-core)

R1: `GRILL-R1-QUESTIONS.md`. R2: `GRILL-R2-QUESTIONS.md`.

## Transcription of R2 answers (agent-authored record; answers file user-owned)

> Q1/2: Agreed.
>
> Q3: Well I'm considering that, whether `astype` give TensorCow, and make
> this function's borrowing rule similar to `reshape`? I'm not sure if there's
> technical difficulties. And I'm actually not asking you to implement astype
> function, can you tell me what's the reason you proposed that?

Settled in R2: Q1 — `NormArgs { ord, axes, keepdims }` with `From` overloads
(`norm(x, 2.0)` shorthand), `norm_all(x, ord)` positional. Q2 — both
`*_with_args` and `*_with_dtype` exist; `with_dtype` scope =
sum/prod/mean/var/std/cumulative_sum/cumulative_prod; `TOut` turbofished;
defined internally via `with_args`.

## Q3 re-ask: astype — rationale, then the TensorCow question

### Why astype was proposed at all (answer to the user's question)

The agreed `*_with_dtype` mechanism is cast-then-reduce composition, and a
**tensor-level dtype cast does not exist publicly today** (only the scalar
`DTypeCastAPI::into_cast`, the internal `OpAssignAPI` assign kernel, and the
generic `map_f`). So this task needs the building block regardless. I proposed
making it *public* (rather than `pub(crate)`) because:

1. ~7 call sites in this task need it (with_dtype families, norm's int→float
   cast, cumulative), and more later;
2. the faer-py shim will need a core-side cast to bind the array-api
   `astype` method in a later wave — under the wrapper-only directive the shim
   cannot implement conversion itself;
3. it fills an existing user-facing gap (rust users currently emulate it with
   `map_f(|x| x.into_cast())` or assignment tricks).

It was a scope proposal beyond the stated aims — with_dtype works identically
if the cast stays crate-internal this round.

### TensorCow / reshape-like borrowing analysis (user's idea)

Feasible, no fundamental difficulty, and arguably the more rstsr-idiomatic
design:

- Rule mirrors reshape exactly: reshape copies only when the layout cannot be
  a view; `astype` copies only when the dtype differs (`T == TOut` → view,
  else convert). Semantics ≈ numpy `astype(copy=False)`.
- Same-type branch is expressible **without unsafe**: enumerate the dtype
  matrix with the existing `duplicate!` macro pattern (the same one that
  generates the scalar `into_cast` impls) — the identity impl
  (`Tensor<f64> → TensorCow<f64>`) expands to `to_view()` (safe), cross-type
  impls expand to convert-and-own (safe, via the assign kernel). No
  specialization needed since impls are enumerated, not blanket.
- Real payoff inside this task: `mean_with_dtype::<f64>` on an f64 tensor, or
  any same-type `with_dtype`, becomes a free view instead of a pointless
  full-array copy.
- Ergonomics caveat: `TOut` is a *trait* parameter, so method-call turbofish
  does not work (`x.astype::<f64>()` is not expressible); primary spelling
  must be the **free function** `rt::astype::<f64>(&x)`, matching rstsr's
  free-function-first conventions. Type-annotation inference also works for
  a method form.
- Owned twin `into_astype[_f]` (identity case moves storage, zero copy)
  mirrors `reshape`/`into_shape`.
- Minor semantic decision: converting branch preserves layout (no implicit
  contig normalization), documented.

### Question

❓ **Q3'** - **astype surface**:
(a) public `astype[_f](&x) -> TensorCow<'a, TOut, B, D>` + `into_astype[_f]`
    owned twin, reshape-like conditional copy (recommended);
(b) public but always-owned `astype -> Tensor<TOut>` (original proposal);
(c) keep the cast crate-internal (`pub(crate)`) this round, decide the public
    surface in the W5 dtype wave.

➡️ Recommend (a) — your Cow suggestion, implemented via the macro-enumerated
safe impl matrix; free-function-first spelling; `into_astype` for ownership
reuse; document the copy=False-like semantics under "Notes of API accordance".
