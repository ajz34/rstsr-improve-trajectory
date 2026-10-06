# Grilling Round 1 — rt::where proposal

Asked on screen 2026-10-06. Round 1 covers only decisions independent of the
fact-finding then in flight (rstsr op mechanics, NumPy semantics, test
surface). Downstream branches (dtype promotion, scalar overloads, trait
shape, test placement) wait for Round 2.

Answer here (`GRILL-R1-ANSWERS.md`, user-owned) or on screen.

---

## Q1 — Naming: `where` is a Rust keyword

`where` is reserved in Rust, so the free function cannot be plain `where`.
Options:

- (a) raw identifier: `pub fn r#where(...)` — call sites read
  `rt::r#where(&c, &x, &y)`; rustdoc shows `where`; greppability slightly hurt;
- (b) different name with parity elsewhere: `rt::select(c, x, y)`
  (ndarray precedent, but diverges from NumPy/array-api/torch/jax naming);
- (c) mangled variant like `where_` — consistent with the `_f`/`type_`
  underscore conventions but reads awkwardly at call sites.

**Recommended: (a) `r#where`** — rstsr's identity is NumPy parity, and every
peer framework says `where`; the raw identifier is a one-time wart, a rename
is a permanent divergence.

## Q2 — Scope: 3-arg form only, or also the 1-arg form?

NumPy's `np.where` has two personalities: `where(cond, x, y)` (elementwise
select) and `where(cond)` (tuple of index arrays — a reduction/indexing op
with completely different machinery).

**Recommended: 3-arg only.** The 1-arg `nonzero`/`argwhere` form is a
different op family (index selection) and its own future proposal; mixing
both under one function also muddies the Rust overloading story.

## Q3 — Scope: does "related implementations" include rstsr-faer-py?

`where` is in the array-api standard, so `rstsr_faer.api.where` is eventually
needed for conformance. Cover: (a) Rust only (rstsr-core + devices + tests),
(b) Rust + pyo3 `api.where` binding, (c) Rust + binding + conformance run?

**Recommended: (a) Rust only now**, faer-py as an explicitly named follow-up
phase — keeps one PR reviewable.

## Q4 — Must `cond` be strictly bool-dtype?

NumPy applies truthiness to any dtype; the array-api standard requires an
actual boolean array.

**Recommended: strict `bool`** at the type level
(`cond: TensorAny<R, bool, B, D>`), matching the array-api standard and Rust
idiom — no implicit `!= 0` coercion.

## Q5 — Dtype generality: arithmetic-free op?

`where` needs no math on `x`/`y` — only element copy. It can be generic over
every dtype (bool, integers, floats, complex, ...) with Clone-level bounds,
unlike `add`/`sin`. This also decides its home: logical/select family rather
than arithmetic.

**Recommended: fully dtype-generic** (house dtype trait only, no numeric
bound), grouped with the logical functions.

## Q6 — Output semantics: broadcasting and allocation

- Broadcasting: full three-way NumPy broadcast of cond/x/y (reusing the
  binary-op broadcast machinery) — yes/no?
- Allocation: always a fresh output (no storage-reuse path like
  `rt::add(a, &b)`; output depends on all three inputs, reuse buys nothing).
- Error variants: house pattern — `where_f` returning `Result` + panicking
  variant.

**Recommended: yes to all three.**

## Q7 — Process and records

(a) proposal/discussion recorded in this task directory
(`2026-10-06-where-op/`), auto-commit allowed here per this repo's rules;
(b) implementation in rstsr on branch `261006/rt-where`, **no commits** until
the user explicitly instructs. Alternative: ADR in `rstsr-book/dev` if this
sets precedent for a select-family.

**Recommended: (a)** — single op, not an architecture decision; lift the
naming question to an ADR only if it turns contentious.
