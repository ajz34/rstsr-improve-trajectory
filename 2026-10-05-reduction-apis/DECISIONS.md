# DECISIONS — reduction APIs (rstsr-core), grilled 2026-10-05

Grill record: `GRILL-INIT-PROMPT.md`, R1–R3 question files (answers arrived on
screen; transcribed into each next round's file). Status: frontier empty;
pending final owner confirmation.

## Objective

Enhance rstsr-core reductions: keepdims + `*_with_args`, explicit output dtype
(`*_with_dtype`), `norm` family, `cumulative_sum/prod`, custom user reductions,
public tensor cast (`astype`), and the rust-side enablers for rstsr-faer-py's
statistical-function failures (W3). Efficiency is not the priority — but memory
discipline holds (D5: no input-sized intermediates).

## D1 — Contract
array-api 2025.12 is normative; NumPy is the test oracle (numpy-transcription
tests). NumPy-style supersets allowed only where they cannot fail the suite,
documented under "Notes of API accordance".

## D2 — No breaking changes
Existing `sum/sum_axes/sum_all/...` keep `TOut = T` and current bounds
(`mean`/`var`/`std` stay `ComplexFloat`-only). All new capability is additive.

## D3 — `*_with_args` (keepdims)
- `ReduceArgs { axes, keepdims }` — ReshapeArgs style: all-`Option` fields,
  `Default` (axes None = whole-reduce, keepdims false), `From`-overloads,
  `impl Into<ReduceArgs>`. **`axes` is a struct field; plural spelling
  intentional** (not numpy's `axis`).
- `VarArgs { axes, keepdims, correction }` for var/std (array-api correction;
  `M - correction <= 0` → NaN).
- `<fam>_with_args[_f](x, args)` + TensorAny methods generated for ALL
  reduction families incl. arg*/all/any/count_nonzero (keepdims everywhere;
  fixes faer-py G-041). Existing `<fam>`/`<fam>_axes`/`<fam>_all` stay thin
  wrappers. Whole-reduce + keepdims → shape `(1,) * ndim`.

## D4 — `*_with_dtype` (explicit output dtype)
- Both `*_with_args` and `*_with_dtype` exist. `with_dtype` scope:
  sum/prod/mean/var/std/cumulative_sum/cumulative_prod only (not min/max/
  arg*/count/all/any — no accumulation or fixed output contract).
- Spelling: `sum_with_dtype::<TOut>(x, args)` (+ `_f`, methods); same args
  structs; `TOut` turbofished (dtypes have no runtime value).
- Bounds: `T: DTypeCastAPI<TOut>` + family reduce bounds on `TOut`.

## D5 — dtype mechanism: element-cast fold, NO intermediate tensor
(owner directive, R3) The existing kernel closure signature is the mechanism:

```rust
I:    Fn() -> TS + Send + Sync,   // init
F:    Fn(TS, TI) -> TS + Send + Sync, // element fold: TI input, TS accumulator
FSum: Fn(TS, TS) -> TS + Send + Sync, // combine partials (rayon; associative)
FOut: Fn(TS) -> TO + Send + Sync,     // finalize; usually TO == TS (identity)
```

- `TI` = input dtype, `TS` = casted/accumulator dtype, `TO` = output (usually
  `== TS`). `*_with_dtype` routes `TI = T`, `TS = TO = TOut` into the existing
  TI/TS/TO-generic kernels (`reduce_{all,axes}_cpu_serial[_rayon]` already are),
  with the cast **inside `F`** (per element). Whole-tensor pre-cast
  (`astype` then reduce) is FORBIDDEN for reductions: O(input) intermediates
  are unacceptable; reduction memory stays O(1) per output element
  (O(output) scratch for strided/multi-axis is fine).
- Same principle for `norm` and friends: elementwise pre-ops (abs, pow) are
  absorbed into the fold closure, not materialized as tensors.
- Cumulative kernels get the same TI/TS separation (cast per element while
  scanning).
- Device-trait shape (new `*DtypeAPI` trait vs added methods) is an
  implementation detail for the implementing agent, following existing
  duplicate-item patterns.

## D6 — `astype` (public tensor cast, this stage)
- Public now (owner: "implement in this stage"): `rt::astype::<TOut>(&x)` →
  `TensorCow<'a, TOut, B, D>` — reshape-like conditional copy: view iff
  `T == TOut`, else convert-copy (≈ numpy `astype(copy=False)`); owned twin
  `into_astype[_f]` (identity case moves storage). Free-function-first
  spelling (method turbofish not expressible for trait params).
- Implementation: macro-enumerated dtype-pair impls (same `duplicate!` pattern
  as scalar `into_cast` impls) — identity pairs → `to_view()` (safe),
  cross-type → convert via assign kernel (safe). Converting branch preserves
  layout. Not the mechanism behind `*_with_dtype` (see D5).
- Purpose: general gap-fill + future faer-py `astype` binding (wrapper-only).
- Docstring documents the deliberate step outside to_/into_ conventions.

## D7 — `norm` family
- `NormArgs { ord, axes, keepdims }` with `From` overloads (`norm(x, 2.0)`
  shorthand); `norm_all(x, ord)` keeps ord positional (sole argument).
- `NormOrd` enum: 1, 2, Inf, -Inf, 0, -1, -2, `P(f64)`, Fro, Nuc; `From<f64>`
  maps 1.0→One, 2.0→Two, ±inf→Inf, else P. scipy/NumPy semantics: axis=None
  flattens; int input cast to float; `TOut = T::Real` (complex → real).
- Scope this round: vector ords (incl. non-norms 0/-inf/p<1) + Fro + matrix
  1/-1/inf/-inf. SVD-dependent matrix 2/-2/nuc DEFERRED to linalg wave
  (register as gap). `l2_norm` stays as fast path.
- Implementation per D5: fold-embedded abs/pow, no intermediates.

## D8 — `cumulative_sum` / `cumulative_prod`
- Follow array-api: names as written; single axis; `axis=None` valid for 1-D
  only (error for ndim>1); args struct with `include_initial`; NO `reverse`;
  dtype via `*_with_dtype` variants; spec unsigned-widening when dtype given;
  NumPy-transcription tests document the flatten-on-None divergence.
- New scan kernels, serial first, rayon twin later.

## D9 — Custom user reduction
- rt-level free functions `reduce_all[_f]` / `reduce_axes[_f]`
  `(x, args, init, fold, combine, finalize)` surfacing the D5 closure
  machinery; accumulator `TS` / output `TO` generic; closures `Send + Sync`;
  new device trait generic over closures; axes/keepdims via ReduceArgs;
  Lp-norm as the doc example. No TensorAny methods (expert-level).
  Document associativity requirement on `combine`.

## D10 — faer-py consequence
Python-visible widening defaults (uint8 sum → uint64 when dtype=None) and
int→float mean are produced by the shim dispatching to `*_with_dtype`
instantiations (W2 generated-dispatch pattern driven by core type traits);
no shim algorithms (wrapper-only). Shim binding work itself is the later W3
wave, not this task.

## D11 — Process
- Three PRs, three branches (fork `ajz34` → RESTGroup/rstsr):
  PR1 `261005/reduction-withargs-dtype` — ReduceArgs/VarArgs + with_args +
  with_dtype (element-cast fold) + astype/into_astype.
  PR2 `261005/norm-custom-reduce` — NormOrd/norm family + public reduce_axes.
  PR3 `261005/cumulative` — cumulative_sum/prod + scan kernels.
- HARD STOP after each PR for owner review/merge before starting the next.
- Each PR: numpy-transcribed + custom tests (core_func/reduction + doc twins),
  `array_api_standard.md` parity-table update, docstring conventions per
  `api-doc-conventions` skill.
- Q9 (skill): a FRESH subagent drafts `rstsr-agents/skills/rstsr-code-style/`
  at wrap-up from this task's actual procedure. The skill carries a prominent
  notice: reference-not-normative; code style may change during development;
  future tasks must not strictly follow it; on divergence or stale content,
  ask the user what diverged and whether to update. (`rules/code-concepts.md`
  remains the concept reference; the skill is the procedure reference.)

## D12 — bool sum unification REJECTED; way (b) is final
Owner asked whether bool sum could be unified into `OpSumAPI` (free fns for
bool) — the "way (a)" previously attempted and abandoned for way (b)
(`TensorSumBoolAPI`, method-only). Root cause re-confirmed empirically:
E0119 — coherence does no negative reasoning, so `impl OpSumAPI<bool, D>`
cannot coexist with the blanket `impl<T: Zero + Add> OpSumAPI<T, D>` (rustc
must assume `core`/`num` may add `Add`/`Zero` for `bool`). Only stable-Rust
route: close the blanket into a per-dtype enumeration. Owner rejected
enumeration ("compile-time dispatch is also dispatch"; the open blanket keeps
unencountered future/foreign `Zero + Add` types automatically summable).
Nightly `specialization`/`negative_impls` also rejected (incomplete features
in a public crate). A probe impl was compiled to capture the exact E0119
diagnostics, then reverted; working tree restored to committed PR1 state.
Consequence: bool stays method-only on `TensorSumBoolAPI` (complete surface
from PR1: sum/sum_axes/sum_with_args[_f] + with_dtype via cast-in-fold);
`rt::sum` family stays numeric-only. Bool min/max (ExtReal) deferred; bool
count_nonzero has the same shape and is likewise out of scope. Do NOT retry
unification unless the owner reopens it (e.g. at a 1.0 coherence refactor).
