# Grill Round 2 — array indexing (fancy indexing) (2026-10-08)

Agent-authored. Answers go in `GRILL-R2-ANSWERS.md` (user-owned) or on screen.

## Round-1 answers (background, agent-transcribed from screen — not authoritative)

- **Q1 (semantics):** agreed — NumPy vectorized indexing **with the placement
  rule** on the row-major path.
- **Q2 (return type):** agreed — `TensorCow<'_, T, B, IxD>`.
- **Q4 (API surface):** agreed — dedicated `array_index`/`array_index_f` +
  `TensorAny` methods; leave `i` and `[]` untouched.
- **Q5 (device contract):** agreed — hoist basic indexing first; tensor tier
  owns shape + placement; the device receives resolved index buffers and does
  not receive the `Basic` arm.
- **Q3 (boolean):** integer-array first; **boolean may be skipped if too
  difficult** (not required this pass).
- **Q6 (col-major):** divergence from NumPy/Julia is **expected and acceptable**
  (like `reshape`); add tests that check the divergence is *reasonable*, not
  buggy (deterministic, documented), rather than NumPy parity.
- **Q7 (input type / placement):** `ArrayIndexer<B>` lives in **rstsr-core**, in
  a new file `array_indexer.rs`, alongside a `array_indexing.rs` (tensor-tier
  fns). Negative indices allowed and bounds-checked.

## Round-2 frontier

❓ **Q1 — placement implementation: construct the final output layout directly
(no physical transpose), and the view-return rule.**

Given we allocate a fresh owned buffer, the NumPy "transpose the result" step is
purely a *layout-construction* choice, not a data movement: we can build the
output `Layout<IxD>` with the broadcast dims already at `consec` and have the
kernel gather straight into it (each output element is an independent gather,
so iteration order is free). That avoids both a transpose copy and a transpose
view. The alternative is the literal NumPy approach: materialize the gather in
`[bulk][subspace]` order, then return `transpose(...)`.

Sub-decision: the **view-return rule** for `TensorCow`. A pure-basic tuple (no
`ArrayIndex` after lowering; note 0-d int arrays lower to `Select`) is
expressible as `i(...)` → borrowed view; anything else → owned.

➡️ Recommend: **build the final layout directly** (no transpose copy, no
transpose view). And **return the view iff no non-trivial array indexer remains
after lowering** (empty advanced set → delegate to `i`); otherwise owned.

❓ **Q2 — do integer scalars and 0-d integer arrays count as "advanced" for the
placement rule?**

NumPy's placement rule counts a plain **integer** (`x[arr, :, 1]`-style `1`) as
advanced for grouping, so separated integer/array indexers push the bulk dims to
the front. Our `ArrayIndexer::Basic(Select(_))` is a *basic* indexer by
construction. To match NumPy we would have to reclassify `Select` (and 0-d
integer arrays, which the spec says behave as integers) as advanced **only for
the `consec` computation**, while still lowering them as axis-dropping selects.

➡️ Recommend: **follow NumPy — reclassify `Select` and 0-d integer arrays as
"advanced" for the contiguity/`consec` computation only** (they contribute 0
bulk dims and still drop their axis). This is what makes `x[arr, :, 1]` place
the bulk dims at the front.

❓ **Q3 — the exact `ArrayIndexer<B>` `From`/`TryFrom` surface, and whether to
introduce a separate `*Args` type.**

INIT lists: `Vec` / `&Vec` / `&[T]` / `[T; N]` for `T ∈ {isize, usize, i64,
u64, i32}`, plus `Tensor<isize, B, IxD>` / `TensorView` / `&TensorAny<isize, B,
IxD>`. Open: is the public input `impl TryInto<AxesIndex<ArrayIndexer<B>>>`
directly (no `*Args` struct), and does the single-array convenience form
`array_index(x, arr)` (no tuple) also resolve (tuple `(arr,)`)?

➡️ Recommend: **input type is `impl TryInto<AxesIndex<ArrayIndexer<B>>, Error:
Into<Error>>`** (no separate Args struct — the tuple already is the group);
provide `From`/`TryFrom` on `ArrayIndexer<B>` for the exact INIT list (all
integer dtypes via a macro like `impl_try_from_axes_index!`, both `Vec`/slice/
array and owned/borrowed, and the three tensor forms); make the **single-array
form** resolve so `array_index(x, arr)` == `array_index(x, (arr,))` (via
`From<T> for AxesIndex<T>`). Bespoke tuple `TryFrom` (not the exported macro) if
the generic type blocks it.

❓ **Q4 — operator trait name and the device-boundary contents.**

INIT calls it `OpArrayIndexAPI`; the index-op family in the codebase uses
`Device*API` (`DeviceIndexSelectAPI`, `DeviceTakeAlongAxisAPI`,
`DeviceMaskIndexAPI`). Boundary: with the tensor tier owning placement and index
resolution, the device needs the output (raw + `Layout`), the source (raw +
`Layout`), and per index array its `(src_axis, index-data, index-Layout)`.

➡️ Recommend **`DeviceArrayIndexAPI`** (matches the index-op family; `Op*API` is
for elementwise/reduction). Signature takes `(&mut out_raw, &Layout<IxD>,
&in_raw, &in_layout, indexers: &[(axis, &Raw/Vec<isize>, &Layout<IxD>)])` — or a
tiny device-side struct `ArrayAuxIndexer<'a, V>` if the tuple is unwieldy — with
all indices resolved to `usize` and validated at the tensor tier. (Exact kernel
internals = Round 3.)

❓ **Q5 — error taxonomy.**

Cases: out-of-bounds integer index; index arrays not mutually broadcastable;
tuple arity exceeding rank; non-array basic indexer errors (existing); too many
advanced indexers (NumPy's `NPY_MAXDIMS`); empty index arrays (valid, empty
output); 0-d tensor input. NumPy raises `IndexError` for the first four with
specific messages.

➡️ Recommend: **out-of-bounds / broadcast-mismatch / too-many-indexers →
`IndexError`** (NumPy-parity, message-only per ADR-0005); **axis-related errors
keep the structured `AxisError`**; no artificial advanced-indexer cap unless a
real device limit exists (document if added). Match NumPy message shapes where
cheap ("shape mismatch: indexing arrays could not be broadcast together with
shapes …").

❓ **Q6 — test-transfer plan and the col-major divergence tests.**

Forward target: array-api nodes (`test_getitem_arrays_and_ints_1/2`) via the
rstsr-faer-py suite; a curated NumPy `test_indexing.py` subset (broadcast
gather, duplicates, negative indices, empty index, mixed slice+array including
the in-place-vs-front placement rule, out-of-bounds, 0-d/dtype preservation);
`doc_draft` doctests per `api-doc-conventions`. Plus the user-requested
**col-major divergence tests** documenting "reasonable, not buggy".

➡️ Recommend: new rstsr-core test dir `tests/core_func/indexing/` (or
`adv_indexing/`) with a curated NumPy-cited subset + `custom_*` col-major
divergence modules (assert deterministic documented order, not NumPy parity);
run the `C6b` conformance measurement through rstsr-faer-py.

❓ **Q7 — rstsr-faer-py wiring, setitem scope, and PR delivery.**

`parse_key` currently rejects `NativeArray` keys and `api.py::__getitem__`
declines; extend to build an `ArrayIndexer<B>` from a python `Array` key and
route through `array_index`. Open: is **`__setitem__` with an integer-array
key** in scope this pass (needs a scatter, which does not exist — only scalar
`mask_fill`), and is this one PR or staged?

➡️ Recommend: **getitem only this pass** (setitem with advanced keys deferred —
it is a genuinely separate scatter feature); `parse_key` builds
`ArrayIndexer::Array` from `Array` keys (0-d int → `Select`), routed through
`rt::array_index`. **One PR** (core + faer-py getitem + tests), size ≈ the
manip/sort/set stages; `capabilities()` unchanged unless a test reads it.
