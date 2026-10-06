# Grill Round 4 — complex `unique_*` algorithm/bound (2026-10-06)

Agent-authored. Answers go in `GRILL-R4-ANSWERS.md` (user-owned) or on screen.

## Background — R3 answers as given on screen (agent-transcribed)

- Q1 (iteration-order plan): Okay. Added principle: device implementations
  stay layout-generic; the **output tensor's layout is at the disposal of
  the high tier** — C-contiguous preferred on a row-major-default device,
  F-contiguous preferred on col-major-default (i.e. allocate with
  `new_contig(None, device.default_order())`; kernels never decide order).
- Q2 (comparator trait): Okay — `ExtSortCmp` (suffix-less name), ints +
  bool + floats NaN-last + complex lexicographic impls, `Equal` on `==`
  (covers ±0.0), standalone sibling of `ExtReal`.
- Q3 ledger items 1–4 (diff axis via `TryInto<AxisIndex>`; final review =
  DeepSeek/deepseek-flash effort max; router mapping sonnet→glm-5.3-flash
  implementation / haiku→DeepSeek review; take_along_axis strict
  same-shape-except-axis): no objection raised → stand as recommended.
- Q3 ledger item 5 (complex unique via the complex `ExtSortCmp` impls):
  **re-opened for discussion** — see Q1 below.

## The re-opened question

❓ **Q1 — `unique_*` algorithm and trait bound: sorted-for-orderable,
naive-PartialEq-for-the-rest, or one path for all?**

Your framing: `unique_*` could relax the bound to `PartialEq` (covers
complex trivially); with extra memory and orderable types (ints/floats/
bools), the sorted algorithm is much faster; dispatch naive for common +
complex, sorted for orderable; correctness first — if the efficient path
is too complicated, naive `PartialEq` for all types is acceptable.

The complication is a Rust mechanics fact: **there is no specialization on
stable Rust**. A single tensor-level `unique_values_f<T>` cannot
conditionally engage a faster path for `T: ExtSortCmp` while requiring only
`T: PartialEq` — the two device impls (`impl<T: ExtSortCmp> OpUniqueAPI<T>`
and `impl<T: PartialEq> OpUniqueAPI<T>`) overlap and are rejected. The
concrete options:

- **(A) One sorted path, bound `T: ExtSortCmp` (R2/R3 status quo).**
  All 13 dtypes incl. complex have `ExtSortCmp` impls (complex =
  lexicographic re-then-im, NaN-components last), so the single sorted
  algorithm covers everything: O(n log n), output sorted, deterministic.
  Bonus: NumPy's `np.unique` returns **sorted** values and sorts complex
  lexicographically (re, then im) — so this is NumPy parity for complex
  too, not just a spec-free order choice (to be verified against
  `~/Git-Others/numpy` when authoring parity tests, esp. NaN edge details).
  The "naive" path is not needed for any current dtype. If a future dtype
  is only `PartialEq`, a naive `OpUniqueAPI` variant can be added then as a
  separate trait (additive).
- **(B) Naive for all, bound `T: Clone + PartialEq`.**
  O(n·u) first-occurrence scan; output order = first occurrence in C-order
  (deterministic, spec-legal since the spec leaves `values` order free).
  Cost: *not* NumPy-parity order for any dtype (NumPy returns sorted), so
  core parity tests need order-insensitive comparison and a
  numpy_differences entry; slower on large inputs; simplest bounds.
- **(C) Dual path (your description, literally).**
  Requires one of: two tensor-level functions (API bloat: e.g.
  `unique_values` vs `unique_values_sorted` — but then the array-api name
  maps to which?); or a `dyn`-comparator indirection threaded through the
  trait (vtable overhead, bounds still awkward); or autoref-specialization
  arcana (not house style). This is exactly the "efficient implementation
  too complicated" case you said to avoid.

Note `isin` is unaffected either way (suite draws ints only; spec
real-valued; goes through `searchsorted` → `ExtSortCmp`, complex declined
at the tensor layer like searchsorted).

➡️ Recommend: **(A)** — the sorted path is not merely the efficient
optimization, it is the *simplest* single-path design, it already covers
complex via the approved `ExtSortCmp` complex impls, and it buys NumPy-parity
ordering as a side effect. (B) is the fallback only if you value the
semantic minimalism of `PartialEq`-bounds over parity and speed; (C) is not
worth the machinery on stable Rust.

## Frontier after this round

Empty — after this answer I write `DECISIONS.md` + the stage plan and the
grill closes.
