# Testing the Python Array API Standard — What to Test, and How

**Context.** We are building a tensor library in a **non-Python** language. This document
answers two questions, deliberately in this order:

1. **What** must be tested to claim Array API conformance — the invariants, semantics,
   error conditions and edge cases.
2. **How** to test it — what the official suite does mechanically, and what our options are
   given that our implementation is not a Python module.

Sources are cited as `path:line` against `~/Git-Others/array-api` (the specification repo,
`data-apis/array-api`) and the conformance suite `data-apis/array-api-tests`. Every
non-obvious claim below was read out of the source, not recalled. Claims I could not verify
are marked as such.

**Target version: `2025.12`** — the current released standard (`spec/2025.12/conf.py:9`,
`CHANGELOG.md:7`). `spec/draft/` holds the next revision.

---

## 0. The short answer

The Array API standard has **three layers**, and only one of them is actually about Python.
Splitting them is the single most useful decision for our project:

| Layer | What it is | Language-neutral? | Who needs it |
|---|---|---|---|
| **1. Numerical semantics** | dtype promotion lattice, broadcasting, indexing rules, special-value (NaN/inf) tables, reduction/accumulation rules, integer division & `%` sign, error conditions | **Yes** — pure math/behaviour | **Everyone.** This is the real prize. |
| **2. API surface** | ~135 required functions + 2 optional extensions, names, arguments, argument conventions | Mostly — names and *meaning* port; Python syntax (`/`, `*`) does not | Our public API design |
| **3. Python object protocol** | `__array_namespace__`, 33 dunders, `__bool__`/`__int__`/…, DLPack | **No** — Python-only | Only if we want Python interop, or want the *official* suite to grade us |

**Recommendation in one line:** adopt Layer 1 wholesale and test it hard in our own language;
follow Layer 2's names and semantics but not its syntax; treat Layer 3 as a shim we build
when we want the official conformance suite to run against us.

**And a notable fact:** as of this writing there is **no prior art** — no Rust, Julia, C++ or
other non-Python library that claims Array API conformance, and no non-Python conformance
tooling. Adoption is entirely within Python (NumPy, CuPy, PyTorch, Dask, JAX). We would be
first, which means the strategy below has to be derived rather than copied.

---

## 1. What we are actually targeting

### 1.1 Where the normative text lives

This trips people up. In 2025.12, `spec/2025.12/API_specification/*.rst` module pages are
mostly **autosummary indexes** — they list function names, they don't describe them. The
per-function normative prose is the **docstring in the stub modules** under
`src/array_api_stubs/_2025_12/*.py`, which Sphinx renders into the spec pages. So:

- **Function semantics** → `src/array_api_stubs/_2025_12/*.py` docstrings
- **Cross-cutting rules** → the prose pages in `spec/2025.12/API_specification/`:
  `array_object.rst`, `broadcasting.rst`, `data_types.rst`, `indexing.rst`,
  `type_promotion.rst`, `sorting_functions.rst`, `function_and_method_signatures.rst`
- **Rationale and design intent** → `spec/2025.12/design_topics/*.rst`
- **What's mandatory vs optional** → `spec/2025.12/purpose_and_scope.md`,
  `spec/2025.12/assumptions.md`, `spec/2025.12/extensions/index.rst`

**This is good news for us.** The stubs are a real Python package, one module per API
category, machine-readable, importable. We can parse them to generate our own API inventory
and to diff our surface against the standard mechanically. (Minor wart: the top-level
`src/array_api_stubs/__init__.py` imports `_2021_12`…`_2024_12` and `_draft` but omits
`_2025_12`, despite the package existing. Import it by full path.)

### 1.2 The normative vocabulary

The spec uses RFC 2119 keywords (`spec/2025.12/terms_and_definitions.md:7-22`):

- **must** — absolute requirement
- **must not** — absolute prohibition
- **should** — valid reasons may exist to ignore, but full implications must be understood
- **may** — genuinely optional

This matters enormously for test design: **a `should` violation is not a bug.** Split your
conformance report into "must-violations" (fail the build) and "should-deviations" (warn).
The word "shall" appears exactly **once** in the whole 2025.12 tree, in a sentence about
version strings (`spec/2025.12/future_API_evolution.md:35`) — so `must`/`should` really are
the whole normative vocabulary.

### 1.3 What "conforming" means

> "A conforming implementation … must provide and support all the functions, arguments, data
> types, syntax, and semantics described in this specification. A conforming implementation
> … **may** provide additional features … beyond those described."
> — `spec/2025.12/purpose_and_scope.md:406-414`

So conformance is a floor, not a ceiling. Extra functions, extra dtypes, extra keyword
arguments are all explicitly allowed. Partial implementations are anticipated too — the spec
encourages libraries to "provide details on the level of (non-)conformance"
(`purpose_and_scope.md:417-419`), and the mechanism for declaring the gaps is the
`capabilities()` inspection API (§2.9).

### 1.4 Version declaration

A conforming namespace must expose `__array_api_version__` as a string in `'YYYY.MM'` form
(`spec/2025.12/future_API_evolution.md:38-41`). Comparison is by string comparison
(`:43-45`). There is no alpha/beta/rc in the versioning scheme (`:35-37`), and the stated
intent is 100% backwards compatibility between revisions — exceptions need "strong
rationales" (`:16-25`).

Caveat: **errata get backported to older revisions** (e.g. `clip` NaN behaviour backported
to v2023.12, `CHANGELOG.md:102`; fft changes backported to v2022.12, `CHANGELOG.md:335-343`).
A version string alone does not fully pin semantics — pin a revision of the spec repo.

---

## 2. WHAT to test — the conformance checklist

This is the core of the document. It's organised as ten layers, roughly cheapest-to-test
first. For each: what's normative, what to assert, and the trap.

Throughout, "the spec" means `spec/2025.12/`.

---

### Layer 1 — API surface

**What's normative.** The presence of every required function and method, with the right
name in the right namespace.

**The required surface (2025.12), verified by counting `def`s in the stubs:**

| Module | Count | Notes |
|---|---|---|
| `elementwise_functions` | **67** | the bulk of the API |
| `creation_functions` | 16 | |
| `manipulation_functions` | 15 | |
| `statistical_functions` | 9 | |
| `data_type_functions` | 6 | |
| `info` | 6 | `__array_namespace_info__` + its 5 methods |
| `searching_functions` | 6 | |
| `set_functions` | 5 | |
| `constants` | 5 | `e inf nan newaxis pi` |
| `linear_algebra_functions` | 4 | matmul, matrix_transpose, tensordot, vecdot — **required, not the extension** |
| `utility_functions` | 3 | |
| `indexing_functions` | 2 | `take`, `take_along_axis` |
| `sorting_functions` | 2 | `sort`, `argsort` |
| `array_object` | 7 properties + 33 dunders + `to_device` | §2.8 |
| `data_types` | 13 dtypes | bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128 |
| `version` | 1 | `__array_api_version__` |

**Total: 135 required top-level functions** across the module set, plus the array object
protocol.

Note the trap: **`matmul`, `matrix_transpose`, `tensordot`, `vecdot` are required top-level
functions** (`spec/2025.12/API_specification/linear_algebra_functions.rst:9`). They are *not*
part of the optional `linalg` extension, even though `linalg` re-exports them as aliases.
Don't put them behind an "optional linalg" feature flag.

**How to test.** Parse the stub modules for `def` names; diff against our exported symbol
table. This is a pure codegen task, doable in CI, no numerics involved. It is the highest
value-per-unit-effort test we can write.

**Growth trajectory** (elementwise functions, AST-verified): 56 (2021.12) → 59 (2022.12) →
65 (2023.12) → 67 (2024.12) → 67 (2025.12). The API is converging, not churning.

---

### Layer 2 — Signatures and argument conventions

**What's normative** (`spec/2025.12/API_specification/function_and_method_signatures.rst`,
`API_specification/index.rst:8-9`):

- Positional parameters **should** be positional-only (`/`); optional parameters **should**
  be keyword-only (`*`).
- A single array parameter is named `x`; multiple are `x1, x2, …`
  (`function_and_method_signatures.rst:29-31`).
- "There are a few exceptions to rules (1) and (2)" (`:61-63`).

**Verified argument matrix across 2025.12** (AST scan of the stubs):

| Parameter | Count | Where |
|---|---|---|
| `axis` | 37 | reductions, sorting, searching, manipulation, fft |
| `dtype` | 21 | allocation, `astype`, `isdtype`, `fftfreq`, `linalg.trace`, `sum`/`prod`/`cumulative_*` |
| `device` | 18 | creation, `astype`, `fftfreq`, `info.dtypes` — always keyword-only, defaults `None` |
| `keepdims` | 14 | reductions |
| `copy` | 4 | `asarray`, `from_dlpack`, `astype`, `reshape` (+ `__dlpack__`) |
| `correction` | 2 | `std`, `var` |
| `descending`, `stable` | 2 each | `sort`, `argsort` |
| `include_initial` | 2 | `cumulative_sum`, `cumulative_prod` |
| `stream` | 2 | `__dlpack__`, `to_device` |
| **`out`** | **0** | **does not exist anywhere** |
| **`where=`** | **0** | **does not exist as a keyword** |

Two of these are worth stating loudly because they're the opposite of NumPy intuition and
I verified them by grepping the entire stub tree (zero hits each):

- **There is no `out=` parameter in the standard, anywhere.** This was a deliberate design
  decision, not an omission: the standard "chooses to (a) leave out `out=`, and (b) not
  specify another method of reusing arrays"
  (`spec/2025.12/design_topics/copies_views_and_mutation.rst:61-75`). If we expose `out=`,
  that's an extension beyond the standard — allowed, but the suite will never require it and
  we must not let it shape our core design.
- **There is no `where=` keyword.** `where` exists only as the top-level three-argument
  function `where(condition, x1, x2, /)` (`src/array_api_stubs/_2025_12/searching_functions.py:180`).
  Conditional elementwise selection is done by calling `where`, not by passing it to `add`.

**The signature-rule exceptions worth knowing** (these are the ones a naive code generator
will get wrong):

- `clip(x, /, min=None, max=None)` — `min`/`max` are positional-or-keyword; there is no `*`
  (`elementwise_functions.py:834`)
- `isdtype(dtype, kind)` — **no `/` at all** (`data_type_functions.py:178`)
- `expand_dims(x, /, axis)`, `squeeze(x, /, axis)`, `permute_dims(x, /, axes)`,
  `moveaxis(x, source, destination, /)`, `broadcast_to(x, /, shape)`, `tile(x, repetitions, /)`,
  `take(x, indices, /, *, axis=None)` — the shape/axis argument is **positional-only**
- `arange(start, /, stop=None, step=1, *, dtype=None, device=None)` — `stop`/`step` are
  positional-or-keyword
- `linspace(start, stop, /, num, *, …)` — `num` is positional-only and required
- `meshgrid(*arrays, indexing='xy')`, `broadcast_arrays(*arrays)`, `broadcast_shapes(*shapes)`,
  `result_type(*arrays_and_dtypes)` — variadic
- `concat(arrays, /, *, axis=0)`, `stack(arrays, /, *, axis=0)` — sequence as first argument
- `where(condition, x1, x2, /)` — three positional args, first named `condition`, and
  "At least one of `x1` and `x2` must be an array"
- `isin(x1, x2, /, *, invert=False)` — same "at least one must be an array" rule

**How to test.** Parse the stubs' signatures, compare against ours structurally. Port the
*arity and meaning* of each argument; Python's `/` and `*` conventions have no direct
equivalent in most languages, so record the intent (which args are optional, which are
keyword-like) in our own API and test that.

---

### Layer 3 — Dtype objects, default dtypes, and promotion

This is where most real bugs will live.

**Dtype objects.** The namespace must expose 13 dtype objects: `bool`, `int8/16/32/64`,
`uint8/16/32/64`, `float32/64`, `complex64/128`
(`spec/2025.12/API_specification/data_types.rst:12-40`). They must support `__eq__`; **no
attributes are required** — so don't build logic that depends on `dtype.name` or `dtype.itemsize`
(`data_types.rst:42-53`). Extra dtypes (float16, bfloat16, int128) are allowed and dtype
*kinds* may be extended, but must stay consistent and be documented
(`data_types.rst:56-59`, `data_type_functions.py:204-207`).

**Default dtypes must be declared, not assumed.** A conforming library defines a default real
floating, default complex floating, default integer and default index dtype
(`data_types.rst:92-105`). The default integer/index dtype **may differ by platform and by
device** — `data_types.rst:103-110` explicitly allows 32-bit vs 64-bit variation. **Never
hardcode int64 as the default index dtype.** Read it from `__array_namespace_info__().default_dtypes()`.

**Promotion is a join on a lattice** (`type_promotion.rst:8-13`). The tables are laid out
explicitly: signed integers (`:33-50`), unsigned (`:55-72`), mixed signed/unsigned (`:77-87`),
float/complex (`:92-111`).

Rules to encode and test:

- **Only the dtypes participate, never the values.** A 0-D array promotes exactly like an
  N-D one (`:116`).
- **Promotion is associative for >2 operands** (`:28`).
- **Mixed integer/float promotion is deliberately unspecified** — "behavior varies between
  implementations" (`:119-120`). **Do not test it. Do not rely on it.**
- **Non-numeric → numeric promotion is unspecified** (e.g. bool → int) (`:117`).
- Libraries may add *extra* promotion rules (`:20`) — so extra rules are conformant, missing
  ones are not.

**Python scalar interop** (`:128-163`) — this is the subtlest rule in the standard and it is
**not** NumPy's "weak scalar" behaviour. When you do `array <op> scalar`:

> the semantics are "convert the scalar to a **0-D array with the same dtype as the array
> operand**", then perform the operation.

So `float32_array + 1.0` stays `float32` — the scalar adopts the array's dtype rather than
promoting it. The exceptional case: a Python `complex` scalar combined with a real array
converts to the **same-precision** complex type — float32→complex64, float64→complex128
(`:145-157`).

Defined combinations: bool scalar → bool array; in-bounds int → integer array; int/float →
real floating array; int/float/complex → complex array. This applies to built-in operators
**including in-place, excluding `@`**.

Out of bounds / undefined: out-of-range integers, Python `float` + integer array, Python
`complex` + integer array — all explicitly unspecified (`:159-163`).

**Casting** (`data_type_functions.py:19-37`): `astype` casts "irrespective of type-promotion";
complex→real "should not be permitted"; bool→real is exactly 1/0; bool→complex exactly
1+0j/0+0j; real→bool is nonzero→True; complex→bool is `0+0j`→False else True.

**Integer overflow is not specified.** I grepped for any wraparound or saturation rule: there
is none. The only integer-edge mandates are (a) `abs`/`neg` of the minimum representable
signed integer is implementation-dependent (`array_object.py:129`,
`elementwise_functions.py:82,2242-2243`), (b) integer division by zero is unspecified
(`elementwise_functions.py:1396,2527`), (c) integer `pow` with negative exponent is
unspecified (`:2401`). **Pick a behaviour, document it, and don't test it as conformance.**

**How to test.** This layer is highly testable with pure table-driven tests:
`result_type(dtype_a, dtype_b) == expected` for every cell of the four promotion tables,
plus the 0-D/same-as-N-D property, plus associativity for triples. Cheap, exhaustive, and
catches real bugs.

---

### Layer 4 — Shapes and broadcasting

**Broadcasting** is specified as an explicit step-by-step algorithm: align trailing
dimensions, prepend singleton dimensions, and raise on mismatch
(`spec/2025.12/API_specification/broadcasting.rst:22-64`). Note the asymmetry made explicit
in the examples: "singleton dimensions can only be prepended, not appended" (`:109-110`).

**The in-place/broadcasting rule is a hard requirement and easy to get wrong:**

> in-place element-wise operations (including `__setitem__`) **must not change the shape**
> of the in-place array as a result of broadcasting. Such operations "should only be supported
> in the case where the right-hand operand can broadcast to the shape of the left-hand
> operand, after any indexing operations are performed."
> — `broadcasting.rst:112-128`

Concretely, for `x` of shape `(2,3,4)` and `a` of shape `(1,3,4)`: `x[...] = a` is allowed,
but `x[1, ...] = a` is **not** (`:124-128`).

**Testable shape invariants:**

- Every elementwise binary op broadcasts per the algorithm.
- `broadcast_to` raises on incompatible shapes; result keeps the input dtype
  (`manipulation_functions.py:72-90`).
- `broadcast_shapes(*shapes)` — new in 2025.12 — returns `()` for no args, should raise
  `ValueError` on incompatibility, and must propagate `None` (unknown) dimensions for
  graph-building libraries (`manipulation_functions.py:45-68`).
- Reductions with `keepdims=True` must restore the reduced axis as size 1.
- `shape` must be tuple-like, immutable, indexable; its elements may be `None` **iff** a
  dimension is unknown; `size` must equal the product of dims, and is `None` iff any dim is
  unknown (`array_object.py:69-103`). We should support the `None` case only if we build a
  lazy/graph front end — otherwise always concrete.

---

### Layer 5 — Values: the special-case tables

This is the largest body of testable content, and the part where "just compare against NumPy"
fails — the standard differs from NumPy in places.

**Accuracy requirements — know which functions are held to what**
(`spec/2025.12/design_topics/accuracy.rst`):

- `+ - * / %`, `reciprocal`, `sqrt` on real floats **must return correctly rounded** results
  per IEEE 754-2019 (`:11-38`). Default rounding `roundTiesToEven` (`:26`).
- The ~22 transcendental functions (trig, exp, log, …) are **approximations**: only the
  listed boundary cases are required (`:40-72`).
- **Statistical and linalg routines have no accuracy requirement** — only a best-effort
  expectation (`:77-93`).
- Subnormals / DAZ / FTZ are left unspecified and implementation-defined
  (`data_types.rst:61-64`).

This means our test suite needs **two tolerance regimes**: exact-or-ULP for the correctly-rounded
set, and a loose relative tolerance for everything else.

**Integer division and `%` — the C-vs-Python question.** This is the classic portability trap:

- `floor_divide` rounds toward **−∞** — "the greatest (closest to +infinity) integer-value
  number that is not greater than the division result" (`elementwise_functions.py:1378`).
  This is **not** C's truncation-toward-zero. For `inf // finite` and `finite // inf` the
  spec documents a preferred result but explicitly permits libraries to return `NaN` / `-1.0`
  to match Python (`:1400-1407, 1422-1429`).
- `remainder` — "Each element-wise result **must** have the same sign as the respective
  element `x2_i`" (`:2521`), and "in the remaining cases, the result must match that of the
  Python `%` operator" (`:2555`). The spec warns it does not follow IEEE 754 for floats
  (`:2531-2532`). C's `%` follows the dividend's sign — **the opposite**. This is a real bug
  we will write if we bind to a C math library naively.
- `divide` on integer inputs: implementation-dependent — either raise, or return a real
  floating-point array (`:1103-1105`). Note that integer `divide` **promotes to float**
  (unlike `floor_divide`).

**NaN / inf — the tables worth encoding as data, not code:**

- `maximum`/`minimum`: **NaN propagates** — if either operand is NaN the result is NaN
  (`elementwise_functions.py:2118,2154`). This **differs from C's `fmax`/`fmin`**, which
  return the non-NaN operand, and from `(a > b) ? a : b`. Easy bug.
- Signed-zero choice for max/min is implementation-defined (`:2111,2147`) — don't test it.
- Reductions `max`/`min`: NaN propagates; empty reduction is implementation-defined (may
  raise, return NaN, or return the dtype's min/max) (`statistical_functions.py:155-168,249-262`).
- `all`/`any`: **±inf and NaN must evaluate True**; empty → `True` for `all`, `False` for
  `any` (`utility_functions.py:34-36,70-72`).
- `clip`: NaN in `x`, `min`, or `max` → NaN (`elementwise_functions.py:869-871`).
- `equal`: NaN → **False**; `±0` are equal; complex compares component-wise but **any NaN
  component makes the whole comparison False** (`:1196-1209`).
- `isnan`/`isinf`/`isfinite` on complex: `isnan` True if either component NaN; `isinf` True
  if either component is ±inf **even when the other is NaN**; `isfinite` False if either is
  non-finite (`:1575-1681`).
- `signbit` must be True for `-0` and for negatively-signed NaN (`:2652-2687`).
- `pow`: `x**0 == 1` **even when the base is NaN** (`:2411-2449`).
- `nextafter`: `-0 → +0` and `+0 → -0` are special-cased (`:2266-2298`).
- `round`: **ties-to-even**, preserving ±0, ±inf, NaN (`:2562-2604`).
- **Sign of NaN is implementation-defined** (`:2196-2197`) — never assert it.

**Reduction and accumulation dtype rules** — the "integer accumulation" contract:

- `sum`/`prod` with `dtype=None` return **the same dtype as `x`**, *except* when `x` is an
  integer dtype narrower than the default integer dtype: then it widens to the default
  signed integer (signed input) or a same-width unsigned type (unsigned input)
  (`statistical_functions.py:378-384, 284-290`). So `int16.sum()` → `int64`, never `int16`.
- If `dtype` differs from `x.dtype`, the input **should be cast before computing** — this is
  the anti-overflow rule (`:384`).
- `mean`: integer input is allowed but the result **must** be the default real float
  (`:194-199`).
- `std`/`var`: `correction` is the divisor adjustment; if `M - correction <= 0` the result
  **must be NaN** (`:344-357, 438-451`).
- Empty `sum` **must be 0**, empty `prod` **must be 1** (`:401, 307`).
- Index-returning functions (`argmax`, `argmin`, `count_nonzero`, `nonzero`, `unique_*`
  counts/indices) use the **default array index dtype** (`searching_functions.py:25,52,83,112`).
- **Historical trap:** before 2023.12, `sum`/`prod`/`linalg.trace` on float32 could
  accumulate to float64. Since 2023.12 they **must preserve the float dtype**
  (`CHANGELOG.md:327-328,345`). Don't copy old NumPy behaviour.

**Sorting / searching / sets:**

- `sort`/`argsort` default `stable=True`; when stable, equal elements **must** keep relative
  order (`sorting_functions.py:21-22`).
- **NaN and signed-zero sort order is unspecified** — "implementations may choose to sort
  signed zeros (`-0 < +0`) or rely solely on value equality", and should document their NaN
  convention (`spec/2025.12/API_specification/sorting_functions.rst:9-17`). **Do not test.**
- `searchsorted`: `'left'` → `x1[i-1] < v <= x1[i]`; `'right'` → `x1[i-1] <= v < x1[i]`;
  out-of-range → `0` or `M` (`searching_functions.py:147-153`).
- `unique_*`: NaN values **should be distinct** (each with count 1) — this differs from
  `==`-based dedup; `-0` and `+0` **should not** be distinct, and which one survives is
  implementation-defined; element **order is unspecified**
  (`set_functions.py:70-78,115-121`).
- `nonzero` returns row-major (C-order) index tuples (`searching_functions.py:88-121`).
- `argmax`/`argmin` return the **first** occurrence; `axis=None` gives the flat index
  (`:11,18,38,45`).

**Complex-specific rules** (`spec/2025.12/design_topics/complex_numbers.rst`):

- The standard requires **no specific complex order relation** (`:49`); if a library defines
  one it must document it (`:51-53`). Complex ordering in `sort`/`max`/`searchsorted` is
  therefore untestable.
- Complex arithmetic follows C99 / one-infinity conventions, and **branch cuts are
  explicitly provisional** — "consumers should not assume that branch cuts are consistent
  between implementations" (`:27-32`). Branch cut for `log`/`pow`/`sqrt` is the negative real
  axis.
- **Value-based promotion is forbidden**: `sqrt` of a real array **must** return real, even
  if the input contains negatives (which give NaN) — the output dtype depends only on input
  dtypes (`:59-61`, `elementwise_functions.py:2865`).
- Complex special-case behaviour is "unlikely to be consistent across implementations"
  (`:1156-1159,2222-2228`) — only all-NaN inputs are pinned.

**`vecdot` conjugates its first argument** — `sum(conj(a_i) * b_i)` — while `matmul` and
`tensordot` **must not** conjugate or transpose (`linear_algebra_functions.py:58,120,134-139`).
Getting this backwards is a classic.

---

### Layer 6 — Indexing, views and mutation

**Indexing** (`spec/2025.12/API_specification/indexing.rst`) is the densest cluster of
`must` statements in the whole spec (60 occurrences of "must" in that one file).

Core rules that differ from NumPy and are therefore prime bug territory:

- **Any selection of a single value returns a 0-D array, not a scalar.** "should be an array
  of rank 0, not a NumPy scalar" (`:41-42,149`). So `x[0]` on a 1-D array yields a 0-D array.
  Our internal scalar type must not leak out here.
- Basic slicing supports `i:j:k` with the full default/negative-index machinery, and the
  selection-count formula is spelled out with quotient/remainder (`:68-104`).
- `i == j` gives an empty axis (`:108`). **Out-of-range slice bounds need not be clipped** —
  `0:100` on a size-10 axis is *not* required to equal `0:10` (`:112-113,124-125`). Nonexistent
  in NumPy, where clipping is standard. Don't test clipping.
- Integer indexing out of bounds: **bounds checking is not required** (`:16-37`).
- Exactly **one** ellipsis allowed; >1 → `IndexError` **must** be raised (`:153`).
- `None`/`newaxis` inserts a size-1 dimension at that position (`:160-166`).
- Count of index expressions (excluding `None`): fewer than N → `IndexError` **should** be
  raised; more than N → **must** be raised (`:168-175`).
- Flat indexing is not supported; use `reshape(x, (-1,))[i]` (`:170-173`).
- Integer-array indexing is a **reduced subset** of NumPy's: slices combined with integer
  arrays are **unspecified**; list/tuple indices are **unspecified** (`:186-194`); all index
  arrays broadcast together and the result takes the broadcast shape (`:213-219`); duplicate
  indices **must be preserved** (`:208`).
- Boolean indexing: the boolean array must be the **sole** index — it cannot combine with
  anything else, including `None` (`:229-232`). `A[B]` replaces the first `M` dims with one
  dim of size `count(True)` in C order, equivalent to `A[nonzero(B)]` (`:234`). `N < M` →
  `IndexError` **must**; per-dimension size mismatch (non-zero) → `IndexError` **must**
  (`:239-241`). A 0-D boolean mask gives shape `(1, ...)` for True and `(0, ...)` for False
  (`:245`).
- Result dtype **must** equal the indexed array's dtype (`:250`).

**Views vs copies — the weakest guarantee in the standard.** This is important enough to
quote:

> "It is not always clear, however, when a library will return a view and when it will return
> a copy. **This standard does not attempt to specify this—libraries may do either.**"
> — `spec/2025.12/design_topics/copies_views_and_mutation.rst:31-34`

**So: never assert aliasing for ordinary operations.** No `shares_memory`-style tests for
slicing, `reshape`, transpose, `conj`, `real`, `imag` — those are all explicitly unspecified
(`elementwise_functions.py:915,2470`).

What *is* specified is the `copy=` three-way contract, per function:

| Function | `copy=True` | `copy=False` | `copy=None` |
|---|---|---|---|
| `asarray` | must copy | must never copy **for buffer-protocol input**, and **must raise `ValueError`** if a copy would be needed (`creation_functions.py:106-107`) | reuse if possible |
| `astype` | newly allocated array **must** be returned | if dtype matches, **must return the input array**; else new allocation (`data_type_functions.py:15-46`) | — |
| `reshape` | **must** always copy (`manipulation_functions.py:281-305`) | **must never** copy; `ValueError` **should** be raised if impossible | must avoid copying if possible, may copy otherwise |
| `from_dlpack` / `__dlpack__` | always copy | never copy; `BufferError` if impossible | reuse if possible |

Note `astype`'s default is `copy=True` but its `copy=False` + same-dtype case must return
**the identical object**, not merely an equal one. That's an identity test, not a value test.

There is **no general "functions must not mutate their inputs" rule** anywhere in 2025.12.
The mutation rules are narrower:

- In-place ops **must not change dtype or shape** as a result of promotion or broadcasting
  (`array_object.rst:180`).
- If promotion yields the LHS's existing dtype, `x1 += x2` **must equal** `x1[...] = x1 + x2`,
  including special cases (`:182`).
- If promotion yields a *different* dtype, the result is **implementation-defined** — casting
  and intermediate precision are the implementer's choice (`:184`).
- A library with no mutation support may simply **omit the in-place operators** — Python falls
  back to the binary op (`:177-178`). So in-place support is effectively optional, and we can
  ship without it.
- `__setitem__` must not change `self.dtype`; the value **must be promoted to `self.dtype`**
  (new normative text in 2025.12) (`array_object.py:1108-1141`).

The `copy=True` consumer contract is the useful flip side: if we honour `copy=True` with a
real copy, consumers are entitled to mutate the result in place
(`copies_views_and_mutation.rst:83-102`).

---

### Layer 7 — Errors

**The headline: the standard mostly does not mandate exception *types*.** From
`spec/2025.12/design_topics/exceptions.rst:6-28`: it "does not attempt to specify exception or
warning types to the extent needed in order to do exception handling in a portable manner";
guidance in a *Raises* section is phrased as *should*; and often the spec "will only specify
that an exception should or must be raised, but not mention what type".

Consequences for testing:

- Build the test plan around the small set of **`must`-with-a-type** rules (table below).
- For untyped "must raise" rules, assert only *that* something was raised.
- For "should" rules, report deviations but do not fail the build.

**The complete set of mandated exception types** (verified against the stubs):

| Situation | Exception | Force | Source |
|---|---|---|---|
| `__float__` on complex array | `TypeError` | must | `array_object.py:539` |
| `__int__` on complex array | `TypeError` | must | `:717` |
| `__index__` on float array | `TypeError` | must | `:692` |
| `__int__` of ±inf | `OverflowError` | must | `:743` |
| `__int__` of NaN | `ValueError` | must | `:744` |
| lazy impl hitting `__bool__`/`__int__`/`__float__`/`__complex__`/`__index__` | `ValueError` | should | `:246,288,558,704,751` |
| >1 ellipsis in an index | `IndexError` | must | `indexing.rst:153` |
| more index expressions than dims | `IndexError` | must | `indexing.rst:175` |
| fewer index expressions than dims | `IndexError` | should | `indexing.rst:168` |
| boolean index with `N < M` | `IndexError` | must | `indexing.rst:239` |
| boolean index dim-size mismatch (non-zero) | `IndexError` | must | `indexing.rst:241` |
| integer index arrays not broadcastable | *untyped* exception | must | `indexing.rst:213` |
| `broadcast_shapes` incompatible | `ValueError` | should | `manipulation_functions.py:59-62` |
| `expand_dims` invalid axis | `IndexError` | should | `:139-142` |
| `reshape(copy=False)` would copy | `ValueError` | should | `:301-304` |
| `squeeze` on non-singleton | `ValueError` | should | `:354-357` |
| `asarray(copy=False)` would copy | `ValueError` | **must** | `creation_functions.py:107` |
| `from_dlpack` export failure | `BufferError` (propagate) | must | `:250-258` |
| `from_dlpack` missing `__dlpack__` | `AttributeError` | may | `:259-262` |
| `from_dlpack(copy=False)` would copy | `ValueError` | must | `:263-264` |
| `__dlpack__(copy=False)` / bad `dl_device` | `BufferError` | must | `array_object.py:360-377,392-396` |
| invalid `axis` in most reductions/manipulation | *untyped* — "must raise an exception" | must (raise only) | e.g. `statistical_functions.py:33` |
| `nonzero` on 0-D | *untyped* | must | `searching_functions.py:107` |
| `matmul` shape error / 0-D operand | spec literally writes `Raises: Exception` | should | `linear_algebra_functions.py:50-62` |

**Note what is absent: there is no `AxisError` in the standard** (verified: zero occurrences).
Also note that `matmul` on 0-D operands **must raise** — a 0-D array is not a scalar
(`array_object.py:891-897`).

---

### Layer 8 — Devices and interchange

**Devices** (`spec/2025.12/design_topics/device_support.rst`):

- The model is "local control": execution happens on the device where the argument arrays
  live (`:16`).
- The surface is: a `.device` property returning an opaque `Device` supporting `==`/`!=`
  **within one library only** (`:45-52` — cross-library device comparison is out of scope);
  a `device=` keyword on array-creation functions; and `.to_device(device, *, stream=None)`.
- **There is no universal `Device` object in the standard** — the standard "does not provide
  a means of instantiating a `Device` object" (`:54-58`). Libraries supply their own.
- The semantics are **recommendations, not hard requirements** (`:73-88`): respect explicit
  `device=`; preserve device assignment; "raise an exception if an operation involves arrays
  on different devices (i.e. avoid implicit data transfer)"; scalars inherit their array's
  device; precedence is `device=` kwarg > input array's device > global default.
- Device *inference* rules are normative per function: `*_like` and `astype` inherit from `x`
  unless `device=` is given; `asarray` must infer from the input array
  (`creation_functions.py:104-105,163-164,366-367`, `data_type_functions.py:47-48`).
- `to_device` on the same device may copy or return `self`; sync vs async is
  implementation-dependent (`array_object.py:1220-1249`).
- Out of scope: cross-library device identity, global default device, stream/queue control,
  distributed allocation, memory pinning (`:91-102`).

**DLPack** is the mandated interchange mechanism (`design_topics/data_interchange.rst:42-49`),
chosen for zero-copy where possible, support for all spec dtypes, and a stable C ABI.

- A library that can never export may **omit** `__dlpack__`/`__dlpack_device__` entirely, in
  which case `from_dlpack` raises `AttributeError` (`:100-105`).
- `__dlpack_device__` returns `(device_type, device_id)`; the enum values are CPU=1, CUDA=2,
  CPU_PINNED=3, OPENCL=4, VULKAN=7, METAL=8, VPI=9, ROCM=10, CUDA_MANAGED=13, ONE_API=14
  (`array_object.py:484-497`).
- `copy=False` semantics are strict: `BufferError` if a copy would be needed, and `ValueError`
  from `from_dlpack(copy=False)`.
- **DLPack 1.0's read-only flag should be ignored by consumers** (`array_object.py:453-457`).
- `stream` semantics are CUDA/ROCm-specific and support beyond `None` is optional.
- The **draft** adds `__dlpack_c_exchange_api__` (a `PyCapsule` holding the DLPack C-API
  exchange struct, DLPack 1.3+) for C-level exchange without going through Python
  (`src/array_api_stubs/_draft/array_object.py`, draft `array_object.rst:276`). **For us this
  is the most interesting line in the draft** — it is the standard's first move toward
  non-Python interop.

The Python buffer protocol is supported as an **input** to `asarray` (CPU only) but is never
an output protocol (`creation_functions.py:86-92`, `data_interchange.rst:48-49`).

`__array__`, `__array_priority__`, `__array_ufunc__`, `__array_function__` are **not part of
the standard** (verified: zero occurrences). `__array__` is mentioned only as a NumPy-specific
mechanism the standard declined to adopt (`data_interchange.rst:65-72`).

---

### Layer 9 — The Python object protocol (only if we want Python interop)

This is the layer that is genuinely Python-specific. It's what makes an object "array API
compliant" *as far as a consumer can detect*:

> "The recommended approach to check for compliance is by checking whether an array object
> has an `__array_namespace__` attribute, as this is the one distinguishing feature of an
> array-compliant object." — `spec/2025.12/purpose_and_scope.md:338-352`

**The array object surface** (verified by enumerating `array_object.py` in 2025.12):

- **7 properties**: `dtype`, `device`, `mT`, `ndim`, `shape`, `size`, `T`
  (`:24,35,46,59,70,88,106`). Note `T` **must** be two-dimensional and `mT` requires ≥2-D —
  they are matrix transpose, not NumPy's reverse-all-axes. Use `permute_dims` for the general
  case. Shape/size rules are in Layer 4.
- **33 dunders + `__init__` + `to_device`**:
  - arithmetic: `__abs__ __add__ __floordiv__ __matmul__ __mod__ __mul__ __neg__ __pos__
    __pow__ __sub__ __truediv__`
  - bitwise: `__and__ __invert__ __lshift__ __or__ __rshift__ __xor__`
  - comparison: `__eq__ __ge__ __gt__ __le__ __lt__ __ne__` — all return **bool arrays**, not
    Python bools. `__lt__/__le__/__gt__/__ge__` are only defined for real-valued dtypes;
    `__eq__`/`__ne__` for any dtype (`array_object.rst:166-167`).
  - interop: `__array_namespace__ __bool__ __complex__ __dlpack__ __dlpack_device__
    __float__ __index__ __int__ __getitem__ __setitem__`
  - `to_device` — the only non-dunder method.
- **In-place and reflected operators are not in the stub file at all** — they appear only in
  `array_object.rst:195-253`. All 12 in-place (`+= -= *= /= //= **= %= @= &= |= ^= <<= >>=`)
  and 14 reflected (`__radd__ __rsub__ __rmul__ __rtruediv__ __rfloordiv__ __rpow__ __rmod__
  __rmatmul__ __rand__ __ror__ __rxor__ __rlshift__ __rrshift__`). "The results of applying
  reflected operators must match their non-reflected equivalents" (`:220-253`).
- **Absent by design**: `__array__`, `__buffer__` (output), `__iter__`, `.item()`, `.tolist()`,
  `out=`. I verified all of `out=`, `.item()`, `.tolist()` have **zero occurrences** in the
  2025.12 spec and stubs.

**The operator/function duality is a testable invariant**: for every op, `__op__` must equal
the corresponding function's result (`array_object.py:145,170,195,…`; `array_object.rst:74-76`).
Test one against the other — it's free coverage.

**The scalar-conversion dunders are contract-bound and highly testable**
(`array_object.py:217-758`):

- They apply to **zero-dimensional arrays**; behaviour on other ranks is unspecified.
- `__bool__`: NaN → True, ±inf → True, ±0 → False; complex is the logical OR of real/imag.
  A lazy implementation **should** raise `ValueError`.
- `__int__`: complex → **must** `TypeError`; ±inf → **must** `OverflowError`; NaN → **must**
  `ValueError`; `-0` → `0`.
- `__float__`: complex → **must** `TypeError`.
- `__index__`: float → **must** `TypeError`.
- A lazy/graph implementation has exactly two legal choices: compute the value, or raise
  (`design_topics/lazy_eager.rst:16-30`).

**Inspection namespace** — `__array_namespace_info__()` must provide 5 methods:
`capabilities()`, `default_device()`, `default_dtypes()`, `devices()`, `dtypes()`
(`src/array_api_stubs/_2025_12/info.py`, `API_specification/inspection.rst:29-42`).

- `capabilities()` must contain `"boolean indexing"` (bool), `"data-dependent shapes"` (bool)
  and `"max dimensions"` (`None` = arbitrary) (`info.py:53-73`).
- **Warning — spec inconsistency:** the normative prose says the key is `"max dimensions"`
  (`info.py:64`) but the `Capabilities` TypedDict in `_types.py:147` spells it `"max rank"`.
  We should emit **both**, and our tests should read `"max dimensions"`.
- `capabilities()` is also **the mechanism by which we declare partial conformance** — see
  Layer 9's optional-omission list below.

**`__array_namespace__(*, api_version=None)`**: must accept that one keyword; `None` means
latest; an invalid or unimplemented version **should** raise; and the returned namespace
"should have every top-level function defined in the specification as an attribute"
(`array_object.py:198-215`, `purpose_and_scope.md:318-331`).

**What may be legally omitted — and must then be declared** (`capabilities()`):

| Omittable | Condition | Declared via |
|---|---|---|
| Boolean array indexing | graph-building libraries may omit | `capabilities()["boolean indexing"]` |
| `nonzero`, `unique_all/counts/inverse/values`, array-valued `repeat(repeats=...)` | graph-building libraries may omit | `capabilities()["data-dependent shapes"]` |
| `__dlpack__` / `__dlpack_device__` | only if never able to export | absence → `from_dlpack` raises `AttributeError` |
| In-place operators | any library may omit | Python falls back to binary op |
| `linalg`, `fft` extensions | entirely optional | `hasattr(xp, 'linalg')` |
| Subnormals; non-`None` streams; extra dtypes | allowed | — |

Note `repeat` with a **literal `int`** must be supported regardless — only the array-valued
form is omittable (`manipulation_functions.py:240-243`).

**Extensions are all-or-nothing.** If a library exposes `xp.linalg`, consumers may assume
**all 25** of its functions are present; the spec says implementors "must aim to provide all
functions and other public objects in an extension" so that `hasattr(xp, 'extension_name')`
is sufficient (`extensions/index.rst:15-22`). Same for `fft` and its 14 functions.
This is a **directly testable invariant**: extension present ⇒ complete.

---

### Layer 10 — Optional extensions

**`linalg`** — 25 functions, namespace name must be exactly `linalg`
(`extensions/linear_algebra_functions.rst:11-17,94-118`). New in 2025.12: `eig`, `eigvals`.

Output contracts worth testing (all are namedtuples):

- `svd` → `(U, S, Vh)`, `S` in **descending** order (`linalg.py:786-793`); `svdvals` same order.
- `qr` → `(Q, R)`; shapes per `mode`; **the sign of `R`'s diagonal is not unique** — different
  devices may differ (`:616-617`). Don't assert signs.
- `eigh` → `(eigenvalues, eigenvectors)`, eigenvalues at real precision, eigenvectors dtype
  = `x` (complex for complex input) (`:266-272`).
- `eig` → **complex eigenvalues even for real input** (`:218-223`).
- `slogdet` → `(sign, logabsdet)`, `sign` dtype = `x`, `logabsdet` real; zero determinant →
  `sign=0`, `logabsdet=-inf` (`:671-697`).
- `matrix_rank` → default integer dtype, shape `x.shape[:-2]` (`:506-509`).
- `pinv` → shape `(..., N, M)` (`:574-577`).
- `vector_norm`/`matrix_norm` have explicit `ord` tables (`:914-945, 419-454`); complex input
  → matching real precision, and a real dtype input → real float output (`:950`).
- **Non-uniqueness is pervasive**: eigenvectors, singular vectors and QR signs may differ
  across devices and implementations (`:203-209,253-256,616-620,774-777`). Eigenvalue **sort
  order is unspecified**. Test invariants (e.g. `A @ V ≈ V @ diag(w)`, orthonormality,
  reconstruction `U @ diag(S) @ Vh ≈ A`), never raw output values.
- Whether the routines validate Hermitian/symmetric/full-rank/invertible input is
  unspecified (`:56-57,258-259,301-302,613-614,720-721`).

**`fft`** — 14 functions, namespace name must be exactly `fft`
(`extensions/fourier_transform_functions.rst:9-21`).

- **Round-trip invariants are the tests**: `ifft(fft(x)) == x` and `irfft(rfft(x)) == x`
  within numerical accuracy (`fft.py:32-33,84-85,252-253`).
- `norm` ∈ `'backward' | 'ortho' | 'forward'`, default `'backward'` = no normalization.
- `rfft` output axis is `n//2 + 1`; `irfft` defaults `n = 2*(M-1)`.
- **2023.12 dtype contract**: `fft/ifft/fftn/ifftn` take complex input and return the same
  dtype; `rfft/rfftn` real → matching-precision complex; `irfft/irfftn/hfft` → matching real;
  `ihfft` → matching complex (`CHANGELOG.md:335-343`).

---

## 3. What NOT to test — the unspecified list

Equally important. Asserting any of these will produce false failures, and *relying* on them
will produce an implementation that isn't conformant.

**Explicitly unspecified / implementation-defined:**

- **View vs copy for any operation** — `copies_views_and_mutation.rst:31-34`. No aliasing tests.
- **Aliasing of `conj` / `real` / `imag`** — `elementwise_functions.py:915,2470`.
- **`from_dlpack` view-or-copy** — `creation_functions.py:248`.
- **Mixed integer↔float promotion** — `type_promotion.rst:119-120`.
- **Non-numeric → numeric promotion** (bool → int/float) — `:117`.
- **Python `float`/`complex` with an integer array; out-of-range integer scalars** — `:159-163`.
- **Integer overflow/underflow in arithmetic** — no rule exists (no wraparound, no saturation).
- **`abs`/`neg` of the minimum signed integer**; **integer division by zero**; **integer `pow`
  with negative exponent** — `array_object.py:129`, `elementwise_functions.py:1396,2401`.
- **Sign of NaN** — `elementwise_functions.py:2196-2197`.
- **Signed-zero ordering in `max`/`min`; empty-array `max`/`min` result** —
  `statistical_functions.py:155-168,249-262`.
- **Out-of-bounds integer indexing; out-of-range slice clipping; flat indexing; slices mixed
  with integer arrays; list/tuple indices** — `indexing.rst:25,35,113-125,178-194`.
- **`take` with out-of-bounds indices or 0-D input** — `indexing_functions.py:29-31`.
- **NaN / signed-zero sort order; `unique_*` element order** —
  `API_specification/sorting_functions.rst:9-17`, `set_functions.py:70-78,115-121`.
- **Complex ordering and complex special cases; branch-cut consistency; complex accuracy** —
  `complex_numbers.rst:27-32,49`.
- **Statistical and linalg accuracy** — `accuracy.rst:77-93`.
- **Eigenvector/singular-vector/QR-sign non-uniqueness; eigenvalue sort order** —
  `linalg.py:203-209,253-256,616-620,774-777`.
- **Whether device-mismatch operations raise** (recommendation, not requirement) —
  `device_support.rst:79`.
- **`to_device` sync vs async** — `array_object.py:1245`.
- **`__setitem__` with integer-array keys; behaviour when `value` cannot promote** —
  `array_object.py:1133-1137`.
- **Iteration of 0-D or ≥2-D arrays** — `array_object.py:645-653`.
- **`asarray` with nested sequences containing arrays; precision overflow on input** —
  `creation_functions.py:71-123`.
- **`reshape`/`asarray` `copy=False` raising is `should` for `reshape`, `must` for `asarray`** —
  treat accordingly.

**Also out of scope entirely per `purpose_and_scope.md:139-207`:** execution model
(lazy vs eager), parallelism, the C API, bfloat16/datetime/string/object dtypes, I/O,
polynomials, error handling portability, testing routines, packaging, subclassing, masked
arrays, ufunc/gufunc APIs, and mixing multiple libraries in one program.

**Never hardcode the default dtypes** — they may vary by platform *and* by device
(`data_types.rst:103-110`). Always read them from `default_dtypes()`.

---

## 4. HOW — what the official suite actually measures

Source: `data-apis/array-api-tests` (HEAD `6c0b59f`, spec submodule pinned at `5f847a38`).

### 4.1 Architecture

Plain pytest, no runner binary. The repo-root `conftest.py` is the whole plugin: it declares
CLI options, registers the Hypothesis profile, and does all skip/xfail/extension filtering at
collection time.

**The spec repo is a git submodule, and it is used as *executable test data*, not as
documentation.** This is the key architectural fact:

- `array_api_tests/stubs.py` imports the real spec stub modules and builds
  `name_to_func`, `category_to_funcs`, `array_methods`, `array_attributes`, `EXTENSIONS`,
  `extension_to_funcs` from them.
- `test_has_names` parametrizes over exactly those names.
- `test_signatures` compares library signatures against the stub signatures.
- **`test_special_cases` regex-parses the `**Special cases**` blocks out of the stub
  docstrings** — the same docstrings we identified in §1.1 as the normative text — and builds
  one pytest case per sentence, then checks the result with exact semantics.
- `dtype_helpers` regex-parses "Should have an X data type" out of the stubs to learn each
  function's legal input dtypes.

So the suite's oracle for edge cases **is the spec prose itself, executed.** That is the
single most important thing to copy for a non-Python implementation, and we can do it from
the same source files (see §5).

**How a partial library still gets useful results.** `array_api_tests/_array_module.py`
binds every spec name — if the library lacks it, it binds an `_UndefinedStub` whose use
raises `AssertionError("<name> is not defined in <xp>")`. Tests therefore *collect and run*
against an incomplete library instead of dying at import, and failures are attributed to the
missing name. Missing **core** names are not skipped, though — they fail.

**Capability discovery** (why a partial implementation still yields meaningful signal):
dtypes are probed by `getattr` and dropped from every strategy tuple if absent; devices come
from `__array_namespace_info__().devices()` and are probed for DLPack and dtype compatibility;
extensions are skipped wholesale when `hasattr(xp, 'linalg')` is false; `min_version` markers
auto-skip on older `api_version`; `ARRAY_API_TESTS_SKIP_DTYPES` removes more. Note that `hh`
calls `__array_namespace_info__()` at import time — **in practice any library under test must
provide the inspection namespace.**

**Version selection:** `ARRAY_API_TESTS_VERSION` env → else `xp.__array_api_version__` → else
`"2025.12"`. This drives which spec-stub directory is loaded, and therefore which tests and
which special-case tables run.

### 4.2 The four oracles — how expected results are defined

Confirmed: **NumPy is not an oracle.** No array library is imported by the suite (only
`ndindex` and Hypothesis). Four distinct mechanisms:

1. **Python-scalar reference implementations.** Value correctness is defined by iterating
   elements, casting each to a Python scalar, and applying a `math`/`cmath`/`operator`/builtin
   function. E.g. `test_sum` compares against `sum(python_scalars)`; `test_prod` against
   `math.prod`; `test_sort` against Python `sorted`; `test_unique_*` against `set`/`Counter`.
2. **Spec-derived tables hardcoded in the harness** — the promotion table, accumulation result
   dtypes, dtype ranges, `isdtype` kind membership, operator↔function maps. Broadcasting and
   index mappings for manipulation functions are **simulated in pure Python**.
3. **Special-case tables parsed verbatim from the spec stubs' docstrings** (see 4.1). Each
   parsed sentence becomes a case checking exact semantics — NaN distinguished from NaN,
   ±0 distinguished, complex results checked component-wise with sign wildcards.
4. **Cross-call consistency invariants** — no external truth at all. Examples:
   `linspace(endpoint=False) == first num of linspace(num+1)`;
   `diff(prepend/append) == diff(concat(...))`; `matrix_power(x, 0) == eye`;
   `asarray(copy=False)` aliasing verified by mutating the input and observing the output.

**What this means:** the suite measures *declared type/shape bookkeeping, spec-prose special
cases, and Python-semantics agreement on ordinary finite values*. It does **not** measure
numerical accuracy, and for linalg and fft it barely measures correctness at all.

### 4.3 What is *not* covered — verified gaps

This is the most actionable part for us, because these are the areas where passing the
official suite does **not** mean we're correct:

- **Floating-point accuracy: no ULP bound at all.** Default elementwise tolerance is
  `rel_tol=0.25, abs_tol=1` — extremely loose. Special-case approximate results use
  `abs_tol=0.01`. Exact `==` is used for integer/bool and for a short list of float functions
  (`round`, `ceil`, `floor`, `trunc`, `sign`, `signbit`, `reciprocal`, `maximum`, `minimum`).
  So a badly-inaccurate transcendental would still pass.
- **Values are simply not tested** (explicit TODOs in the source) for: `mean`, `std`, `var`,
  `cumulative_prod`, `pow`, `astype`, `broadcast_arrays`, `broadcast_to`, `searchsorted`,
  `tile`, `isin` — and for linalg: `det`, `eigh`, `eigvalsh`, `eig`, `eigvals`, `inv`,
  `matrix_norm`, `pinv`, `matrix_rank`, SVD orthogonality, QR orthonormality, `solve`
  residual. **`test_matmul` has no mathematical oracle at all** — it checks error behaviour,
  shape rules, and stack-consistency. **The entire fft module checks dtype and shape only; no
  values whatsoever.** `pinv` and `matrix_rank` are assertion-free calls.
- `test_remainder` is **unconditionally skipped as flaky** (`:1953`) — the `%`-sign rule from
  §Layer 5, one of the highest-risk areas for a C-family implementation, is not checked by the
  official suite.
- **Devices:** non-default-device behaviour is untested; DLPack tests have TODOs.
- **Performance, memory, concurrency:** nothing.
- **Error taxonomy:** only a handful of mandated exceptions are checked.
- **Signature conformance is deliberately partial.**
- Subnormals are excluded globally (the suite monkey-patches `st.floats` with
  `allow_subnormal=False`); overflow raised by the library is treated as an untestable
  example, not a bug; the most-negative-integer `abs`/`neg` case is out of scope.

The README itself admits the gap list is unwritten: "some aspects of the spec are impractical
or impossible to actually test, so they are not covered… TODO: note what these are."

### 4.4 Strictness knobs

| Knob | Effect |
|---|---|
| `--max-examples N` | Hypothesis examples per test (code default **100**; README says 20 — stale) |
| `--disable-deadline` | default per-example deadline is 800 ms; CI normally disables it |
| `--hypothesis-derandomize` | reproducible runs |
| `--disable-extension EXTS` | skip `linalg` / `fft` |
| `--disable-data-dependent-shapes` | skip `nonzero`, `unique_*`, boolean-mask indexing |
| `--skips-file` / `--xfails-file` | repeatable, merged; substring/prefix match on test IDs |
| `ARRAY_API_TESTS_XFAIL_MARK=skip` | turn the xfails file into skips (faster) |
| `ARRAY_API_TESTS_SKIP_DTYPES=a,b` | drop dtypes (bool and the default float cannot be skipped) |
| `ARRAY_API_TESTS_MAX_ARRAY_SIZE` | default 1024 |
| `ARRAY_API_TESTS_VERSION` | pin the spec revision under test |
| `-o xfail_strict=True` | recommended only with caveats — Hypothesis xfails are flagged flaky |
| `--json-report` | via `reporting.py`; attaches per-test metadata incl. the array-API function name — this is the conformance-dashboard mechanism |

There is **no `--strict` flag** and no assumption-vs-requirement mode: the "assumption" layer
is Hypothesis `assume`/`reject` plus reference implementations returning `None`; the
"requirement" layer is plain asserts.

**Authority:** "The spec is the suite's source of truth. If the suite appears to assume
behaviour different from the spec, or test something that is not documented, this is a bug —
please report such issues" (`README.md:151-154`). The suite is calver-versioned and **new
tests can break previously-passing libraries** — pin a tag.

**Doc drift to be aware of** (README vs code, all verified): default `--max-examples` (20 vs
100), default xfail filename (`fails.txt` vs `xfails.txt`), a `generate_stubs.py` that isn't
tracked, and a `docs/` directory that doesn't exist.

---

## 5. HOW — strategy for a non-Python library

### 5.1 Three things the suite's design tells us

**1. The spec stubs are executable test data — so we can consume them too.** The suite wires
the spec repo in as a git submodule and uses it as *input*: it enumerates the surface from
the stubs, compares signatures against the stubs, and — the crown jewel — **regex-parses the
`**Special cases**` blocks out of the stub docstrings to generate its edge-case tests.** We
can run the same parse in a build step and emit our own special-case test tables in our own
language. That gives us the spec's NaN/±0/inf/branch-cut tables as data, from the same
authoritative source, with no Python in the loop and no re-derivation by hand.

**2. The oracle is not NumPy — which means it's portable.** Value checks are defined by
Python scalars (`math`/`cmath`/`operator`), spec-derived tables hardcoded in the harness,
spec-prose special cases, and cross-call identities (§4.2). All four mechanisms re-express in
any language. The suite's value to us is therefore **not the runner** — it's the *encoded
knowledge* of which identities and edge cases matter, plus the parsing machinery that extracts
them from the spec.

**3. Passing the suite does not mean being correct.** §4.3 is a long list of what it doesn't
check: no accuracy bound, no values for `mean`/`std`/`var`/`pow`/`astype`/`isin`/`searchsorted`,
**no fft values at all**, barely any linalg values, untested device behaviour, and
`test_remainder` unconditionally skipped. If we treat a green run as "done", we will ship
those gaps. Our own suite has to cover what theirs doesn't — which is also where the
differentiation is, since a tensor library that gets linalg *values* right is worth more than
one that merely returns the right shapes.

### 5.2 Options

**Option A — thin Python binding + the official suite (the acceptance gate).**
Expose our library as an importable Python object and run the suite against it. Two details
from the source make this more attractive than it looks:

- `ARRAY_API_TESTS_MODULE` accepts **an `exec(...)` snippet**, not just a module name: the
  value `exec('import mylib; xp = mylib.namespace(...)')` is exec'd and must bind `xp`. So a
  factory-produced namespace works — we don't need a real importable module hierarchy.
- **nanobind-wrapped libraries are an explicitly supported case.** `test_signatures.py` has a
  branch for nanobind functions that report a generic `(*args, **kwargs)` signature, falling
  back to `_test_uninspectable_func`, which calls the function with auto-derived arguments
  (trying float64 → bool → int64 → complex128) or skips with a message. If we bind via
  PyO3/nanobind/pybind11, signature conformance degrades gracefully rather than failing.

Requirements to plan for: the namespace must expose the full spec surface, provide
`__array_namespace_info__` (the strategy layer calls it at import time), and ideally
`__array_api_version__` (otherwise the suite assumes 2025.12). Pin a **release tag** — the
suite is calver-versioned and new tests can break previously-passing libraries. Vendor it if
we want reproducibility: it's explicitly standalone, depending only on `pytest` and
`hypothesis`. Use `--json-report` to get per-test metadata including the array-API function
name, which is the basis for a conformance dashboard.

Phased adoption, so we get signal before we're complete: low `--max-examples` first;
`--disable-extension linalg` / `fft`; `--disable-data-dependent-shapes`;
`ARRAY_API_TESTS_SKIP_DTYPES=...`; `ARRAY_API_TESTS_XFAIL_MARK=skip` for speed. Caveat:
there is **no core-function skip** — a missing required name fails `test_has_names` and
cascades. Extensions are the only thing that skips cleanly.

**Option B — port the checks into our language (the bulk of our suite).**
The four oracles in §4.2 re-express cleanly. We lose Hypothesis's automatic case generation
but can substitute our own property-based/shape-fuzzing harness. This is the only option that
gives fast, in-process, CI-friendly conformance testing with no Python toolchain in the loop —
and the only way to cover the §4.3 gaps.

**Option C — generate surface and signature tests from the stubs.**
`src/array_api_stubs/_2025_12/*.py` enumerates every required name and signature. Parse it to
generate symbol-presence and arity tests. Cheap, exhaustive for Layer 1/2, and it re-runs
automatically when we bump spec versions. (Remember the spec's conventions are `should`-level
for `/` and `*`, so treat signature mismatches as deviations, not failures.)

**Option D — differential testing against a reference implementation.**
`array-api-strict` is a deliberately minimal, strict reference implementation;
`array-api-compat` gives NumPy/PyTorch/etc. a conforming surface. Both are good oracles
**only where the spec pins behaviour** — §3 lists where it doesn't. Most useful for the
promotion tables and the special-value tables, which are fully deterministic.

Note `array-api-strict` is itself the **only** library exercised in the suite's own CI, and
its workflow is a worked example worth copying (loops over `2023.12` / `2024.12` / `2025.12`,
runs with `--hypothesis-disable-deadline --max-examples 500 -n 4`). Even it still xfails
~30 spec special cases, because NumPy itself deviates from the standard. That's a useful
calibration: **the reference implementation is not bug-free, so a differential mismatch
against it is a question, not an answer.**

**Recommended combination:** C for the surface (generated, cheap), B for semantics (the bulk),
**plus a port of the docstring-parse special-case generator from A's source**, and A as the
acceptance gate once the binding lands. D as a cross-check on promotion and special values.

### 5.3 What this means for our API design

- **Don't** put `out=` in our core API, or `where=` as a keyword on elementwise ops — neither
  exists in the standard (§Layer 2), and designing around them will make conformance harder.
- **Do** make in-place ops optional from day one — the standard permits omitting them
  entirely.
- **Do** design a scalar-return path that respects "indexing yields 0-D arrays, never scalars"
  (Layer 6). This is a deep design constraint, not a surface detail.
- **Do** build `capabilities()` from the start — it's how we honestly declare partial
  conformance rather than silently failing.
- **Do** keep the option of a lazy/graph front end in mind: `shape` elements may be `None`,
  `capabilities()["max dimensions"]` may be `None`, and several APIs may be omitted by
  graph-building libraries. Even if we ship eager, matching these signatures now is cheap.

---

## 6. A concrete plan

### Phase 1 — spec-derived, no Python required, highest value per hour

1. **Surface + signature tests, generated from the stubs.** Parse
   `src/array_api_stubs/_2025_12/*.py`, extract names/arities/argument names, emit a
   presence-and-arity test in our language. Re-run the generator when we bump spec versions.
   Covers Layers 1–2 exhaustively and cheaply.
2. **Promotion tables as data.** Encode the four tables from `type_promotion.rst:33-111` and
   assert `result_type(a, b) == expected` for every cell, plus the 0-D-acts-like-N-D property
   and associativity for triples. Pure table-driven, catches real bugs, no Python.
3. **Port the special-case generator.** Re-implement the docstring `**Special cases**` parser
   from `array_api_tests/test_special_cases.py` against the same stub sources, emitting our
   own edge-case tables (NaN, ±0, ±inf, branch cuts). This is the highest-value item in the
   whole plan — it converts the spec's prose into our test data automatically and permanently.
4. **The `must`-typed error table** from Layer 7. Small, fully specified, easy to get wrong.

### Phase 2 — semantics, in our own language

5. **Property/identity tests** for everything with algebraic structure: broadcasting shapes,
   reduction identities, `sort`↔`argsort`↔`searchsorted` consistency, manipulation index
   remapping (assert the output element at each output index equals the input element at the
   computed source index — this is how the official suite tests `expand_dims`, `flip`,
   `permute_dims`, `roll`, `concat`, …), `linspace(endpoint=False)` vs `linspace(num+1)`.
6. **Cover the §4.3 gaps the official suite leaves open** — the places where a green official
   run would still leave us wrong:
   - `mean`, `std`, `var`, `cumulative_prod`, `pow`, `astype`, `isin`, `searchsorted` values
   - all fft values, via round-trips (`ifft(fft(x)) ≈ x`, `irfft(rfft(x)) ≈ x`) and Parseval
   - linalg values via invariants rather than raw output — reconstruction (`U@diag(S)@Vh ≈ A`),
     orthonormality, residual norms (`‖Ax−b‖`), `matrix_power(x,0)==eye`, `solve(A, A@x) ≈ x`.
     Never assert eigenvector signs, QR signs, or eigenvalue order.
   - `remainder`/`floor_divide` sign semantics, which the official suite skips entirely
   - accuracy: enforce the correctly-rounded set (`+ - * / %`, `reciprocal`, `sqrt` on real
     floats) to ULP, and use loose relative tolerance only for the transcendental set.
7. **NaN/inf/±0 tables.** Both the generated ones (Phase 1.3) and hand-written checks for the
   rules that differ from the C library we're probably sitting on — `maximum`/`minimum` NaN
   propagation, `remainder` sign, `x**0 == 1` for NaN base, `signbit(-0)`.

### Phase 3 — the official gate

8. **Build the Python binding** and run `array-api-tests` against it. Start with
   `--disable-extension linalg --disable-extension fft`, low `--max-examples`, and a skips
   file; then ratchet. Turn on `--json-report` and keep the report as a conformance artifact.
9. **Differential cross-check** against `array-api-strict` on the deterministic subset
   (promotion tables, special values) — remembering that a mismatch is a question, not an
   answer, since the reference implementation itself xfails ~30 spec cases.

### Definition of done

Passing the official suite is **necessary but not sufficient**. A defensible conformance claim
needs all three: the official suite green on our binding, our in-language suite covering the
§4.3 gaps, and a published `capabilities()` + `__array_api_version__` declaring exactly what
we do and don't support. The spec explicitly encourages publishing the level of
(non-)conformance rather than staying silent (`purpose_and_scope.md:417-419`).

---

## Appendix A — API inventory (2025.12)

*Full function lists per module.* See §Layer 1 for counts. The authoritative list is
`src/array_api_stubs/_2025_12/`. Key lists:

- **creation (16)**: `arange asarray empty empty_like eye from_dlpack full full_like linspace
  meshgrid ones ones_like tril triu zeros zeros_like`
- **elementwise (67)**: `abs acos acosh add asin asinh atan atan2 atanh bitwise_and
  bitwise_left_shift bitwise_invert bitwise_or bitwise_right_shift bitwise_xor ceil clip conj
  copysign cos cosh divide equal exp expm1 floor floor_divide greater greater_equal hypot imag
  isfinite isinf isnan less less_equal log log1p log2 log10 logaddexp logical_and logical_not
  logical_or logical_xor maximum minimum multiply negative nextafter not_equal positive pow real
  reciprocal remainder round sign signbit sin sinh square sqrt subtract tan tanh trunc`
- **manipulation (15)**: `broadcast_arrays broadcast_shapes concat expand_dims flip moveaxis
  permute_dims repeat reshape roll squeeze stack tile unstack`
- **statistical (9)**: `cumulative_prod cumulative_sum max mean min prod std sum var`
- **data_type (6)**: `astype can_cast finfo iinfo isdtype result_type`
- **searching (6)**: `argmax argmin count_nonzero nonzero searchsorted where`
- **set (5)**: `isin unique_all unique_counts unique_inverse unique_values`
- **required linalg (4)**: `matmul matrix_transpose tensordot vecdot`
- **utility (3)**: `all any diff`
- **indexing (2)**: `take take_along_axis`
- **sorting (2)**: `argsort sort`
- **constants (5)**: `e inf nan newaxis pi`
- **linalg extension (25)**: `cholesky cross det diagonal eig eigh eigvals eigvalsh inv matmul
  matrix_norm matrix_power matrix_rank matrix_transpose outer pinv qr slogdet solve svd svdvals
  tensordot trace vecdot vector_norm`
- **fft extension (14)**: `fft ifft fftn ifftn rfft irfft rfftn irfftn hfft ihfft fftfreq
  rfftfreq fftshift ifftshift`

## Appendix B — Version deltas that matter to a new implementation

- **2022.12** — complex numbers added wholesale (complex64/128, complex promotion, Python
  complex scalars, `conj`/`imag`/`real`); `__array_api_version__` added; **fft extension
  added** (`CHANGELOG.md:503-548`).
- **2023.12** — **breaking**: `sum`/`prod`/`linalg.trace` must preserve float dtype
  (`:327-328,345`); fft dtype contract tightened (`:335-343`); `vecdot` axis restricted.
  Added the whole `__array_namespace_info__` inspection API, plus `clip`, `copysign`,
  `cumulative_sum`, `hypot`, `maximum`, `minimum`, `moveaxis`, `repeat`, `searchsorted`,
  `signbit`, `tile`, `unstack` (`:297-319`); lazy implementations allowed to raise from
  `__bool__`/`__int__`/etc. (`:261-267`).
- **2024.12** — scalar argument support for ~30 binary elementwise functions (`:119-152`);
  added `count_nonzero`, `cumulative_prod`, `diff`, `nextafter`, `reciprocal`,
  `take_along_axis`; **integer array indexing became normative** (`:169`). **Breaking**:
  Python complex + real array promotion (`:196`), device-context-aware `can_cast`/`result_type`
  (`:200-201`).
- **2025.12** (current) — added `broadcast_shapes`, `isin`; `linalg.eig`/`eigvals`;
  `expand_dims` tuple axis; negative axes in `permute_dims`; `__setitem__` value promotion.
  **Breaking**: `broadcast_arrays`, `meshgrid` and `info.devices` now return **tuples** where
  they previously returned lists (`CHANGELOG.md:7-76`).
- **draft (next)** — `top_k` (searching), `__dlpack_c_exchange_api__` (C-level DLPack 1.3+
  exchange, no Python in the loop), relaxed dtype requirement ("should provide" instead of
  "must provide", with a floor of bool + one integer + one real float), device-dtype
  validation on creation. Verified: these are the *only* API-surface deltas in the draft stubs.
