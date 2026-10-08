# Grill Round 1 — array indexing (fancy indexing) (2026-10-08)

Agent-authored. Answers go in `GRILL-R1-ANSWERS.md` (user-owned) or on screen.

Evidence base: six comprehension reports — (1) rstsr-common indexer types,
(2) rstsr-core tensor/device indexing, (3) the 3-tier device-op pattern,
(4) existing take/gather/select ops, (5) NumPy + array-api indexing semantics
and the NumPy C algorithm, (6) `rstsr-faer-py::__getitem__` + book/ADR prior
art. Reference checkouts: array-api-tests `6c0b59f9`, array-api spec
`ff497ed8`, NumPy `v2.5.2` (`48fecee5`), CuPy `c52b7e02`. Prior art in-repo:
`GRILL-INIT-PROMPT.md` (this folder); gap **G-039** / checklist item **C6b** in
`2026-10-07-arrayapi-convergence/CHECKLIST.md`.

## Settled by evidence (not asked, flag if wrong)

- **`[]` today is scalar element access only.** `impl Index<I> for Tensor*`
  requires `I: AsRef<[usize]>` and returns `&T`; `a[i, j]` does not compile
  (a tuple has no `AsRef<[usize]>`). There is no `TensorIndex` trait.
  (`rstsr-core/src/tensor/indexing.rs:540-654`.)
- **Basic slicing is a copy-free view**: `i`/`slice`/`i_mut`/`slice_mut` take
  `I: TryInto<AxesIndex<Indexer>, Error: Into<Error>>` and end in
  `Layout::dim_slice(&[Indexer])` → `TensorView` (no device op, no copy).
  `Indexer = { Slice(SliceI), Select(isize), Insert, Ellipsis }`
  (`Insert` = `NewAxis`). (`rstsr-common/src/layout/indexer.rs`.)
- **`AxesIndex<T> = { None, Val(T), Vec(Vec<T>) }`**; `AxisIndex<T>` (single,
  `None` = last axis); `AxesPairIndex<T>`. Conversions are a surface of
  blanket `From`/`TryFrom` impls — there is **no** `AxesIndexToIndex` trait.
  Tuples → `AxesIndex` via the exported `impl_from_tuple_to_axes_index!` macro
  (arity 1..=10; note it **panics** on a failed per-element conversion).
- **`Layout<IxD>` exists** (`IxD = Vec<usize>`; stride = `Vec<isize>`); no
  `Default`. Construct via `Layout::<IxD>::new(shape, stride, offset)` or the
  `From<Vec<usize>>` / `new_c_contig` / `new_contig(order)` helpers.
- **`adv_indexing.rs` already implements named advanced ops** (never via `[]`):
  `index_select`/`take` (per-axis host integer list), `bool_select` (per-axis
  host bool mask), `take_along_axis` (index *tensor*, same-rank, one axis),
  `mask_select`/`mask_fill` (whole-tensor bool mask gather / scalar scatter).
  Device traits: `DeviceIndexSelectAPI`, `DeviceTakeAlongAxisAPI`,
  `DeviceMaskIndexAPI` (`rstsr-core/src/operators/adv_indexing.rs`).
- **3-tier house pattern**: op trait in `operators/`, tensor wrapper in
  `tensor/`, impls in `device_cpu_serial/` + `feature_rayon/auto_impl/`
  (symlinked into `device_faer/rayon_auto_impl/`), kernels in
  `rstsr-native-impl`. `Raw = Vec<T>` for both CpuSerial and Faer (Faer *is*
  the rayon device wrapper). Tensor tier resolves indices host-side to
  `Vec<usize>` + a contiguous layout; the device only sees resolved indices.
  Allocation via `DeviceCreationAnyAPI::uninit_impl` / `assume_init_impl`.
- **Index ops use the `Device*API` naming**, while elementwise/reduction ops
  use `Op*API` — the codebase is not uniform here.
- **Conventions** (manip/sort/set wave): `*Args` argument-group structs with
  `From`/`TryFrom` overloads; `func_f` (fallible) + `func` (panicking) +
  `TensorAny` method twins; read-only inputs take
  `impl TensorViewAPI<Type = T, Backend = B, Dim = D>`.
- **Semantics (authoritative)**: array-api defines only the **reduced subset** —
  all-int / all-int-array tuples, mutually broadcast (trailing-aligned), result
  shape = the broadcast shape, zipped gather; **slices mixed with arrays are
  explicitly left unspecified**. NumPy implements full **vectorized indexing**,
  including the **placement rule** (advanced bulk dims injected in-place when
  the advanced indexers are contiguous in the tuple, moved to the front when
  separated; integer scalars count as advanced for this grouping) and
  boolean-array expansion via `nonzero`. NumPy C ref:
  `mapiter_fill_info` / `_get_transpose` / `PyArray_MapIterSwapAxes`
  (`numpy/_core/src/multiarray/mapping.c`).
- **`rstsr-faer-py`**: basic keys + whole-tensor bool mask (G-038, done) work;
  integer-array keys are hard-declined (**G-039**). A **wrapper-only rule**
  forbids re-implementing gathers Python-side — it must be rust-side.

## Questions

❓ **Q1 — semantic contract: NumPy vectorized indexing (with the placement
rule) or the array-api reduced subset?**

Option **(a) reduced subset only**: every indexer is an int / int-array, they
broadcast trailing-aligned, result = broadcast shape, zipped gather. Exactly
what array-api-tests gate; mixed slice+array would be rejected or given a fixed
convention. Option **(b) NumPy vectorized**: additionally supports mixed
basic+advanced tuples with the placement rule (advanced dims injected in-place
when all advanced indexers are contiguous, front otherwise), matching
`test_indexing.py`. Note your own example `tensor[1:3, [0,1,2], [0,2,1]]` needs
(b) to match NumPy (`(2,3)`, not `(3,2)`); (b) costs a post-gather transpose in
general.

➡️ Recommend **(b)** — mixed slice+array is explicitly in scope (INIT-PROMPT
§"We do not implement … grouped" / but do implement the unparenthesized form),
and it is the only way the stated example matches NumPy. The reduced subset
then falls out as the all-array special case.

❓ **Q2 — return type: `TensorCow<'_, T, B, IxD>` or owned `Tensor<T, B, IxD>`?**

`TensorCow` lets the degenerate case (no non-trivial array indexer — all
scalars / 0-d ints, i.e. reducible to basic slicing) return a borrowed view and
everything else return an owned tensor. NumPy advanced indexing copies in
essentially all non-degenerate cases, so the borrowed arm only fires in that
degenerate case. Owned-only is simpler (no lifetime in the signature).

➡️ Recommend **`TensorCow`** per the INIT-PROMPT sketch: borrowed view exactly
when the parsed index set reduces to basic indexing, owned otherwise. Flag: the
common path is owned, so the view arm is a small optimization, not a headline.

❓ **Q3 — are boolean-array indexers in scope for this feature?**

`ArrayBool` is in the INIT-PROMPT enum. But a bool array is *not* per-axis: it
consumes as many axes as its ndim and expands to `nonzero` integer arrays
(NumPy). Whole-tensor `x[mask]` already exists as `mask_select`; the new thing
would be bool *mixed* into a tuple (`x[1:3, mask]`). That expansion is a
sizeable extra mechanism and interacts with the placement rule.

➡️ Recommend **integer-array indexing first**: keep `ArrayBool` in the type,
route a lone bool array to the existing `mask_select`, and raise
`UnImplemented` for bool-mixed-in-tuple as a documented follow-up. Keeps PR one
focused; the failing conformance nodes for `C6b` are integer-array
(`test_getitem_arrays_and_ints_1/2`).

❓ **Q4 — API surface: a dedicated `array_index`/`array_index_f`, or fold into
`i`/`slice`/`[]`?**

`i`/`slice` have a clean contract — *always a view, never copy*; folding array
indexers in would silently make `i` sometimes copy. Extending `[]` would
collide with its scalar `AsRef<[usize]>` role and cannot return `Result`.

➡️ Recommend **a dedicated `array_index` / `array_index_f` free fn +
`TensorAny` method pair**, taking
`impl TryInto<AxesIndex<ArrayIndexer<B>>, Error: Into<Error>>`. Leave `i` and
`[]` untouched. (`array_index` = the method name for the "array indexing"
coinage.)

❓ **Q5 — device-layer contract, and how placement survives basic-indexing
hoisting.**

The INIT-PROMPT prefers "hoist all possible basic indexing first
(`t = tensor[1:3]`), pass the device only the array indexers with their axes".
That is clean for the *gather*, but the NumPy placement rule needs `consec` =
the output-dim count before the first advanced indexer, which depends on the
slices/newaxis *surrounding* the arrays — information the hoist drops. Two
shapes: **(a)** hoist at the tensor tier, compute the final output layout
*including the placement transpose* there, and pass the device
`(src_axis, out_axis, V, Layout)` per array indexer (device just gathers);
**(b)** pass the whole `&[ArrayAuxIndexer<V>]` (Basic + Array) to the device
and let it decode structure.

➡️ Recommend **(a)**: keep the device dumb. Tensor tier owns shape,
placement/transpose, and index resolution (negatives, bounds, broadcast); the
device receives resolved index buffers + explicit `(src_axis, out_axis)` pairs
and a target layout — mirroring how `take_along_axis` already resolves indices
host-side. Consequence: `ArrayAuxIndexer<V>`'s `Basic` arm is **not** sent to
the device.

❓ **Q6 — column-major stance.**

Row-major is the stated parity target. Column-major advanced indexing genuinely
differs (visit/flatten order), and the project's col-major story is a stub that
is *not* NumPy-parity.

➡️ Recommend **row-major is the gate; column-major gets a device-default-order
fallback that is self-consistent but not parity-tested**, with the divergence
documented (same treatment `mask_select` already documents for its non-C-order
caveat).

❓ **Q7 — input-type design: where `ArrayIndexer<B>` lives, and the host-list
variants.**

`ArrayIndexer<B>` holds `Tensor<isize, B, IxD>`, so it cannot live in
`rstsr-common` (no `Tensor` there) — it must live in `rstsr-core`. But the
tuple→`AxesIndex` conversion macro is exported from `rstsr-common` and
instantiated for a concrete `$t:ty`; instantiating
`impl_from_tuple_to_axes_index!` for a *generic* `ArrayIndexer<B>` likely will
not compile. The `OneDimIndex(Vec<isize>)`/`OneDimBool` variants exist only so
`From<Vec<isize>>` etc. resolve — and a host `Vec<isize>` unifies with the
`Array` arm directly, since `ArrayAuxIndexer<V>`'s `V` is just `Vec<isize>`
(no device tensor needs materializing).

➡️ Recommend define **`ArrayIndexer<B>` in `rstsr-core`**, keep the `OneDim*`
variants as the `From` entry points, lower all of them (host `Vec` / `&[T]` /
`[T; N]` / `Tensor` / `TensorView` / `&TensorAny<isize, B, IxD>`) to
`Array(Vec<isize>, Layout<IxD>)` at the tensor tier, and wire the tuple
conversion with a **bespoke `TryFrom<(F1, ..)> for AxesIndex<ArrayIndexer<B>>`**
if the generic macro cannot take the type. **Allow negative indices and
bounds-check them → `IndexError`** (NumPy allows negatives; spec does not test
them but does not forbid them).

## Deferred to later rounds (frontier downstream of Q1–Q7)

- the exact `consec` computation + transpose placement algorithm;
- the precise `From`/`TryFrom` surface for `ArrayIndexer<B>` (integer dtypes,
  `&Vec`/`&[T]`/`[T; N]`, tensor forms) and name/placement of any `*Args` type;
- error taxonomy (mask shape mismatch, broadcast failure, out-of-bounds,
  to-many-advanced-index limits) and the `AxisError` vs `IndexError` split;
- whether/when a single-array index can reuse `index_select` / how the kernel
  generalizes `take_along_axis`;
- test-transfer plan (array-api `test_getitem_arrays_and_ints_*`, NumPy
  `test_indexing.py` subset) and the `C6b` conformance-flip measurement;
- rstsr-faer-py `parse_key`/`api.py` wiring for the integer-array key.
