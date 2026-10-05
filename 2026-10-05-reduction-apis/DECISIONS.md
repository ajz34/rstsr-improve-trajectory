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

## D13 — norm redesign mid-implementation: enforced `ord` + ReduceArgs reuse + string ords
Owner interjected while PR2 was being implemented (2026-10-05), superseding
parts of D7:
- `ord` is an ENFORCED positional parameter on every norm function:
  `norm(x, ord)`, `norm_axes(x, ord, axes)`,
  `norm_with_args(x, ord, args)`. The no-ord default spelling (`norm(x)`)
  was dropped; NumPy's `ord=None` default corresponds to passing `2`, and
  the `ord=None` raveled 2-norm of an n-D array maps to `l2_norm`.
- The bespoke `NormArgs` struct was DELETED; the args form reuses
  `ReduceArgs` (axes + keepdims) directly.
- `NormOrd: TryFrom<&str>`/`TryFrom<String>` for `"l1"`/`"L1"`/`"l2"`/`"L2"`
  /`"fro"`/`"nuc"`; ord parameter type is
  `impl TryInto<NormOrd, Error: Into<Error>>` so numerics (via `From`,
  Infallible -> Error) and strings both work at one call site.
- fro/nuc stay matrix-only (two reduced axes); single-axis or ndim != 2
  whole-input use is `InvalidValue` (NumPy-compatible).
Implementation notes (for the record):
- Matrix ±1/±Inf norms are two-stage; stage 2 lives in a dedicated device
  method `OpNormAPI::norm_matrix_cmp_axes` (axis adjustment after stage 1 is
  device-side; empty max stage yields 0 = NumPy `max(initial=0)`).
- rustc E0391: `B: OpMaxAPI<B::TOut, IxD>`-style bounds (trait WITH an
  associated type, arg = projection of the same self type) cycle bound
  elaboration; routing stage 2 back through `OpNormAPI` with `T = B::TOut`
  hit the same wall - hence the dedicated device method.
- `feature_rayon/auto_impl/reduction.rs` is SYMLINKED into
  `device_faer/rayon_auto_impl/` - one source, two cfg contexts; kernels take
  `Option<&ThreadPool>`.

### D14 — flatten RETRACTED; NumPy vector/matrix convention stands; nuc excluded
After D13's "no matrix/vector distinction" direction was implemented and the
L1-divergence table was shown (`norm([[1,2],[3,4]],1)` = 10 vs NumPy 6), the
owner retracted: "I made wrong assertion to you. You are going to follow
NumPy's convention." Final PR2 norm semantics = the pre-flatten design:
- NumPy vector/matrix classification (1 axis = vector; 2 axes = matrix,
  axis-tuple order significant for ±1/±Inf; >2 axes = InvalidValue).
- Matrix 2/-2 raise `UnImplemented` (SVD deferred to linalg wave).
- `nuc` EXCLUDED from `NormOrd` entirely (owner directive kept from the same
  review): `"nuc"` does not parse; documented as an intentional gap.
- `fro` string/`Fro` variant kept (matrix Frobenius; == 2-norm fold for
  vectors, single-axis Fro is InvalidValue per NumPy).
- enforced `ord` + `ReduceArgs` reuse + string ords from D13 unchanged.
Note: the owner's staged snapshot turned out to predate the norm
implementation (partial staging) - the four src files were reset to HEAD and
the implementation reconstructed; any diff review must use the working tree,
not the staging area.

## D15 — norm REVERTED from rstsr-core; moved to future rstsr-sci-traits
Owner (2026-10-05, after seeing `norm(a)` on 2-D resolve to Frobenius under
the enforced-ord mapping): "I wish to revert this PR [the norm part], and
leave this to be something rstsr-sci-traits to be implemented. However,
custom reduce is still needed at this time."

Consequences:
- PR2 scope is now custom reduce ONLY (`reduce_all`/`reduce_axes`/
  `reduce_with_args` + `OpReduceCustomAPI` + device impls). All norm code,
  tests, exports, docs, and tracking rows removed from rstsr-core.
- The complete norm prototype (implementation + passing tests + settled
  design + gotchas) is preserved at `postponed-norm/` in this directory
  (README + full diff + the two test files) as the reference for the future
  `rstsr-sci-traits` wave. Key settled points: enforced positional `ord`,
  `ReduceArgs` reuse (no `NormArgs`), string ords l1/L1/l2/L2/fro, nuc
  excluded, NumPy vector/matrix convention with axis-tuple order, matrix
  2/-2 SVD-deferred, E0391 projection-bound gotcha -> dedicated device
  two-stage method.
- Owner also flagged (review round): array-api's linalg table does not
  track norm the way I had added it -> `array_api_standard.md` reverted and
  stays untouched by this task.
