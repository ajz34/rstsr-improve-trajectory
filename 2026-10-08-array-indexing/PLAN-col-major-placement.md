# Column-major placement of the array-indexing broadcast block (plan)

Draft, 2026-10-09. Proposal by the maintainer ("contiguity is the first
priority; larger row/column divergence is acceptable"), worked out here. **Not
implemented.** Supersedes the col-major part of the "behaves identically under
RowMajor and ColMajor" doctrine that the tests written earlier today pin.

## 1. The problem

`array_index` inserts the broadcast ("fancy") block of the index arrays into the
output. Row-major takes NumPy's placement rule as-is; a column-major device then
re-uses the same *axis position* under the other reading direction, so the
block's **contiguity role flips**:

`a = np.arange(24).reshape(2, 3, 4)`, `a[[0, 1], :, [2, 0]]`:

| | shape | strides | broadcast axis |
|--|--|--|--|
| row-major (NumPy) | `(2, 3)` | `[3, 1]` C-contig | axis 0, stride 3 — the **most**-strided (outermost) |
| column-major (today) | `(2, 3)` | `[1, 2]` F-contig | axis 0, stride 1 — the **least**-strided (innermost) |

Same shape, same logical values, but the broadcast axis plays the opposite role
in the memory order.

## 2. The rule

Keep NumPy's placement rule — it is a row-major statement ("`consec` base axes
precede the block; a displaced run goes to the front") — and **measure it in the
device's reading direction**:

> On a column-major device the broadcast block is inserted at
> `consec_col = base_ndim - consec`.

* the base axes keep their subscript order (so the sliced/inserted axes line up
  with the input axes exactly as they do under basic indexing),
* the bulk block keeps its internal broadcast order (trailing-aligned with the
  index arrays),
* the output layout stays `new_contig(None, order)`, so the result is
  F-contiguous on a column-major device (built directly — no transpose copy).

Consequences of the rule:

* **in-place runs stay in place**, mirrored: whenever `consec == base_ndim -
  consec` (e.g. a single base axis with the block on either side) the shape is
  unchanged and only the arrangement differs — today's behaviour, kept;
* **a displaced run goes to the back** on a column-major device (to the front on
  a row-major one);
* a **1-D result is unchanged** (no base axis), e.g. a key consuming every axis;
* in the two-axis case (one base axis, one broadcast axis) the column-major
  result is exactly the row-major one read transposed — same buffer,
  `reverse_axes`-equivalent, zero cost.

For the maintainer's example this gives shape `(3, 2)`, F-strides `[1, 3]`,
buffer `[2, 6, 10, 12, 16, 20]` (identical to the row-major buffer):

```
[[ 2 12]
 [ 6 16]
 [10 20]]
```

Rejected alternatives:

* **full axis reversal** (`col = reverse_axes(row)`, identical buffer): also
  reverses the *base* axes, so `a[:, :, [0, 1]]` would return the input's first
  axis last — inconsistent with basic indexing (`a[:, :]` keeps the axes) ✗.
* **"always put the block at the slowest end"**: not expressible while keeping
  NumPy's rule for row-major — the rule legitimately places the block last in
  the in-place cases, and forcing it to the end would either change row-major
  (breaking parity) or make the in-place positions unrepresentable ✗.

## 3. Case table (`a = arange(24).reshape(2,3,4)`, `b4 = 4-D build`)

| key | base / bulk / consec | row-major (NumPy) | column-major (proposed) |
|--|--|--|--|
| `a[[0,1], :, [2,0]]` | `(3,)` / 1 / 0 | `(2,3)`, `[[2,6,10],[12,16,20]]`, C-strides `[3,1]` | `(3,2)`, `[[2,12],[6,16],[10,20]]`, F-strides `[1,3]`, **same buffer** |
| `a[:, :, [0,1]]` | `(2,3)` / 1 / 2 | `(2,3,2)`, buffer `[0,1,4,5,8,9,12,13,16,17,20,21]` | `(2,2,3)`, `C[p,i,j] = R[i,j,p]`, F-strides `[1,2,4]`, buffer `[0,1,12,13,4,5,16,17,8,9,20,21]` |
| `b4[0:2, [0,1], [1,0], :]` (block in the middle) | `(2,4)` / 1 / 1 | `(2,2,4)` | **unchanged** `(2,2,4)`, F-strides `[1,2,4]` |
| `b4[[0,1], :, [1,0], :]` | `(2,4)` / 1 / 0 | `(2,2,4)` | `(2,4,2)`, `C[j,l,p] = R[p,j,l]`, F-strides `[1,2,8]` |
| `a[1, [0,1,2], [2,0,1]]` | 1-D result | `(3,)` | **unchanged** `(3,)` |
| all-basic keys | — | a view | **unchanged** (a view) |

## 4. Consequences

* **Row-major: byte-for-byte unchanged** — NumPy parity stays the anchor, and
  the array-API surface (the `rstsr_faer.api` shim is row-major) is unaffected.
* **Column-major: the shape-level divergence grows.** For displaced placements
  the result is no longer "the same logical tensor, arranged differently": it is
  the row-major result with the broadcast block re-inserted at the mirrored
  point (a documented axis permutation, and in the two-axis case a pure
  transpose). The pair of in-src order tests must be rewritten accordingly —
  `test_array_index_order_invariance` becomes "identical for in-place placements
  and 1-D results, otherwise the documented permutation" and
  `test_array_index_order_arrangement` gets the permuted expectations.
* **Docs**: `array_index` / `array_index_f` replace the "behaves identically
  under `RowMajor` and `ColMajor`" one-liner with a **Row/Column Major Notice**
  (the house warning div) stating the rule; `order_semantics.md`'s gathering
  paragraph and the `col-major-transfer` entry in `numpy_differences.md` are
  re-scoped (that entry currently promises "only the arrangement differs" for
  `array_index`).
* **Device tier: no change.** `lc` + `consec` already describe the result
  completely; the tensor tier just hands over the mirrored `consec`, so
  `array_index_cpu_serial` and the adapters are untouched and every output
  element is still written exactly once.
* **Out of scope (needs a separate decision)**: `mask_select`'s count axis has
  the analogous role question, but its `(count, *trailing)` shape is fixed by
  the array API and its sequence divergence is already registered; the other
  gathers (`index_select`, `take_along_axis`, `nonzero`) insert no axis and are
  unaffected.

## 5. Implementation sketch

* `rstsr-core/src/tensor/array_indexing.rs`, after the placement walk and before
  the output shape is assembled:
  `let consec = if order == ColMajor { base_layout.ndim() - consec } else { consec };`
  (the device call, the layouts, and the kernel stay as they are).
* Rewrite the two order tests' column-major expectations; add the permutation
  assertion (`col == permute(row, perm)` with the block moved from `consec` to
  `consec_col`) for the displaced cases.
* Update docs + tracking as in §4; record the new contract in `DECISIONS.md`.
* Verify: both CI order jobs (`--lib` row-major and col-major), the row-major
  batteries byte-identical, doctests, fmt/clippy, the array-api run unchanged.

## 6. Decisions to confirm

1. the rule in §2 (recommended) vs full axis reversal (rejected in §2);
2. for n-dimensional index arrays the block moves **as a unit**, keeping its
   internal broadcast order (recommended);
3. scope: `array_index` now, `mask_select` separately (recommended);
4. land it on `261009/array-indexing` with the doc/tracking updates above
   (recommended).
