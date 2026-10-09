# Column-major placement of the array-indexing broadcast block (plan, rev. 2)

Draft started 2026-10-09; **rule decided** the same day (maintainer:
"access-contiguity-first axes layout"). This supersedes the col-major part of
the "behaves identically under RowMajor and ColMajor" doctrine that the tests
written earlier that day pinned.

## 1. The problem

`array_index` inserts the broadcast ("fancy") block of the index arrays into the
output. Row-major takes NumPy's placement rule as-is; a column-major device then
re-uses the same *axis position* under the other access order, so the block's
**contiguity role flips**:

`a = np.arange(24).reshape(2, 3, 4)`, `a[[0, 1], :, [2, 0]]`:

| | shape | strides | broadcast axis |
|--|--|--|--|
| row-major (NumPy) | `(2, 3)` | `[3, 1]` C-contig | axis 0, stride 3 — the **most**-strided (outermost) |
| column-major (before this change) | `(2, 3)` | `[1, 2]` F-contig | axis 0, stride 1 — the **least**-strided (innermost) |

## 2. The rule (decided: flip-displaced-only)

NumPy's placement rule has two regimes: a run of advanced indexers that is
**separated** by other indexers is *displaced* to the front of the result; a run
that stays **together** is emitted *in place*, where it appears in the subscript
(leading, middle or trailing).

Column-major devices access the axes in the opposite index order, and the
priority is that the broadcast block keep its contiguity role — the displaced
block is the outermost (most-strided) axis under row-major, so under
column-major it must be the most-strided axis too, which is the **last** one:

> On a column-major device a **displaced** run is placed at the **back**
> (`consec = base_ndim`); a run that stays **together** keeps its subscript
> position in both orders.

Everything else is unchanged: base axes keep their subscript order, the block
moves as a unit with its internal broadcast order, and the output is allocated
with `new_contig(None, order)` (F-contiguous on a column-major device, built
directly — no transpose copy).

Observations:

* the *shape* difference between the two orders occurs exactly when the advanced
  indexers are **apart**; together runs differ only in their arrangement
  (C- vs F-contiguous), as before;
* for a displaced run on a column-major device the result is the row-major
  result with the block moved from the front to the back — in the two-axis case
  (one base axis, one broadcast axis) that is exactly the transpose, i.e. the
  same buffer, so it costs nothing;
* 1-D results (no base axis) and all-basic keys are unchanged in both orders;
* "apart" is a distinct state of the placement walk, **not** the same as
  `consec == 0`: NumPy's algorithm yields `consec = 0` both for a displaced run
  and for a run that merely *leads* the subscript; only the former flips. In the
  port that state is the `consec_status == 2` branch.

Rejected alternatives:

* **mirror the insertion count** (`consec_col = base_ndim - consec`): it moves
  together runs too — `a[:, [0,1], [2,0]]` would go from `(3,2)` to `(2,3)`,
  contradicting the maintainer's K2 expectation;
* **full axis reversal** (`col = reverse_axes(row)`, identical buffer): also
  reverses the *base* axes, inconsistent with basic indexing (`a[:, :]` keeps
  the axes).

## 3. Case table

`A = arange(24).reshape(2,3,4)` (`A[i,j,k] = 12i+4j+k`),
`C = arange(60).reshape(3,4,5)` (`C[i,j,k] = 20i+5j+k`),
`B` = the 4-D broadcast build of the tests (`B[i,j,k,l] = 16i+8j+4k+l`).

| key | run | row-major (NumPy) | column-major (decided) |
|--|--|--|--|
| `C[[0,1], :, [2,0]]` | apart | `(2,4)` `[[2,7,12,17],[20,25,30,35]]` | `(4,2)` `[[2,20],[7,25],[12,30],[17,35]]`, F-strides `[1,4]` (transpose; same buffer) |
| `C[:, [0,1], [2,0]]` (K2) | together, after one base axis | `(3,2)` `[[2,5],[22,25],[42,45]]` | **unchanged** `(3,2)`, F-strides `[1,3]` |
| `C[[0,1], [2,0], :]` (K3) | together, leading | `(2,5)` `[[10..14],[20..24]]` | **unchanged** `(2,5)`, F-strides `[1,2]` |
| `A[[0,1], :, [2,0]]` | apart | `(2,3)` `[[2,6,10],[12,16,20]]` | `(3,2)` `[[2,12],[6,16],[10,20]]` (transpose; same buffer) |
| `A[:, :, [0,1]]` | together, trailing | `(2,3,2)` | **unchanged** `(2,3,2)`, F-strides `[1,2,6]` |
| `B[[0,1], :, [1,0], :]` | apart | `(2,2,4)` | `(2,4,2)`, F-strides `[1,2,8]`, `col[j,l,p] = row[p,j,l]` |
| `B[0:2, [0,1], [1,0], :]` | together, middle | `(2,2,4)` | **unchanged** `(2,2,4)`, F-strides `[1,2,4]` |
| `A[1, [0,1,2], [2,0,1]]` | 1-D result | `(3,)` | **unchanged** `(3,)` |
| all-basic keys | — | a view | **unchanged** (a view) |

## 4. Consequences

* **Row-major: byte-for-byte unchanged** — NumPy parity is the anchor, and the
  array-API surface (the `rstsr_faer.api` shim is row-major) is unaffected.
* **Column-major**: shape divergence for apart runs; the arrangement divergence
  (C- vs F-contiguous) remains for every rank-≥2 result; 1-D results stay
  identical. The order tests are rewritten: the invariance claim narrows to 1-D
  results and together runs, and the arrangement test gains the displaced
  expectations plus the K2/K3 configurations.
* **Docs/tracking**: `array_index`'s docstring replaces the "behaves identically
  under `RowMajor` and `ColMajor`" one-liner with a **Row/Column Major Notice**;
  `order_semantics.md`'s gathering paragraph and the `col-major-transfer` entry
  in `numpy_differences.md` are re-scoped (they currently promise that only the
  arrangement differs).
* **Device tier: no change.** `lc` + `consec` describe the result completely;
  the tensor tier hands over the effective `consec`, so the kernel and the
  adapters are untouched and every output element is still written once.
* **`mask_select`: resolves to no change.** Its count axis replaces the leading
  axes of the tensor and `check_mask_axes` forbids anything else, so it is
  structurally the *leading together* configuration (K3) and keeps
  `(count, *trailing)` — the shape the array API fixes. Its *selection sequence*
  divergence (the mask visit order follows the device) is a separate,
  already-registered item, not a placement question. `take_along_axis`,
  `index_select` and `bool_select` insert no axis, and `nonzero` returns a tuple
  of coordinate lists — all unaffected.

## 5. Implementation

`rstsr-core/src/tensor/array_indexing.rs`: before the output shape is assembled,
the effective insertion point becomes `base_ndim` on a column-major device when
the run was displaced (`consec_status == 2`); the device call, the layouts and
the kernel are untouched.
