# Zero-size tensors: stride policy and BLAS leading dimensions

- **Date**: 2026-10-19
- **rstsr**: `c1a6b79d9c7514381e373d1aaff0ed20921f754d` (v0.10.2, 2026-10-15)
- **Issue**: [RESTGroup/rstsr#161 — solve_triangular with a zero-size RHS passes an illegal LDB to DTRSM](https://github.com/RESTGroup/rstsr/issues/161)
- **Reference**: NumPy `v2.5.2` checkout (`~/Git-Others/numpy`), runtime `2.5.1`; OpenBLAS `0.3.34` checkout (`~/Git-Others/OpenBLAS`), LAPACK (`~/Git-Others/lapack`)
- **Status**: investigation only; no rstsr source changed, nothing committed.

## TL;DR

The issue reproduces exactly (`LDB = 1`, `M = 18`, `N = 0` → `DTRSM` parameter 11). The
failure is not the zero-size shape itself; it is that rstsr has **a semantic rule but no
policy for the stride values of empty layouts**, and the BLAS layer consumes those stride
values directly as leading dimensions.

- rstsr already implements the NumPy semantic rule ("size 0 ⇒ both C- and F-contiguous,
  strides irrelevant") in its predicates (`f_prefer`, `c_prefer`, `c_contig`, `f_contig`,
  `check_strides`, `bounds_index`, `is_broadcasted`, `Layout::eq`).
- But the *values* produced by `stride_f_contig` / `stride_c_contig` differ between static
  `Ix<N>` and dynamic `IxD` shapes, and no consumer clamps them. `ld_row` / `ld_col` only
  special-case a dimension of length 1, not length 0.
- The semantic rule makes `f_prefer()` / `c_prefer()` true for every empty layout. This
  short-circuits the contiguous-layout normalization in
  `overwritable_convert_with_order`, so a bogus leading dimension survives into BLAS for
  owned/mut tensors, while the same shape as a *view* is normalized and works. The outcome
  depends on the tensor representation.
- BLAS-3 errors have no return channel (`xerbla` only prints); LAPACK errors are mapped to
  `InvalidValue`, so the same defect surfaces as an "internal bug" error on the LAPACK
  side (verified with `solve_general` / `DGESV`, argument 7 = `LDB`).

**Recommendation.** Treat empty strides as *unspecified and never consumed* (adopt as
normative), make the contiguous constructors deterministic (all-zero strides for zero-size
shapes, both `Ix` and `IxD`), clamp `ld_row` / `ld_col` to the BLAS minimum, and add
per-routine early returns for empty operands. The early return is mandatory regardless of
the stride-value rule, because empty *views* can carry arbitrary strides.

## 1. Issue anatomy

Reproducer (issue #161):

```rust
let b = rt::asarray((&mut buf[..], [0, naux].f(), &device)).into_reverse_axes();
// shape [18, 0], strides [1, 1]
rt::linalg::solve_triangular((rt::eye((naux, &device)).t(), b, Lower));
```

Chain, with source locations:

1. `[0, 18].f()` is a `Layout<Ix2>`; `stride_f_contig` multiplies by
   `shape[i-1].max(1)`, so the zero axis is treated as length 1:
   stride `[1, 1]` (`rstsr-common/src/layout/shape.rs:89-95`).
2. `into_reverse_axes()` reverses shape and stride → shape `[18, 0]`, stride `[1, 1]`.
3. `SolveTriangularAPI` converts the RHS to `Ix2` and calls
   `ref_impl_solve_triangular_f`
   (`rstsr-linalg-traits/src/blas_impl/solve_triangular.rs:24-38`).
4. `TRSM::internal_run` checks `a.f_prefer()`; for `size == 0` this is unconditionally
   true (`rstsr-common/src/layout/layoutbase.rs:87-95`) and is used by
   `overwritable_convert_with_order` to skip normalization
   (`rstsr-blas-traits/src/util/util_tensor.rs:34-52`). The empty owned/mut RHS is kept
   with strides `[1, 1]`.
5. `let ldb = b.view_mut().ld_col().unwrap();`
   (`rstsr-blas-traits/src/blas3/trsm.rs:127`). `ld_col` special-cases only `shape[1] == 1`;
   for `shape[1] == 0` it falls through to `stride()[1] = 1`
   (`rstsr-core/src/tensor/tensor2_impl.rs:39-52`).
6. OpenBLAS validates `ldb < MAX(1, m)` (`interface/trsm.c:201`) **before** its
   `m == 0 || n == 0` early return (`interface/trsm.c:362`), calls `xerbla`, then returns.
   The CBLAS driver returns `()`, so rstsr reports success
   (`rstsr-blas-traits/driver_impl/cblas/blas3/trsm.rs`).

The contrast in the issue (direct `[18, 0]` f-order works) is explained by the same static
stride rule: `[18, 0].f()` has stride `[1, 18]`, so `ld_col() == 18`.

## 2. What NumPy does

### 2.1 Documented semantic policy

`doc/source/reference/arrays.ndarray.rst:146-155` states it explicitly:

> If an array has no elements (`self.size == 0`) there is no legal index and the strides
> are never used. Any array with no elements may be considered C-style and Fortran-style
> contiguous.

The flag computation implements this: the first `dim == 0` sets **both** `C_CONTIGUOUS`
and `F_CONTIGUOUS` and returns (`numpy/_core/src/multiarray/flagsobject.c:127-131`).

### 2.2 Value convention at allocation

`PyArray_NewFromDescr_int` allocates 1 byte for a zero-size array and then sets **all
strides to 0** (`numpy/_core/src/multiarray/ctors.c:820-830`). This comes from PR #21477
(commit `dd5ab7b115`, 2022), whose motivation is memory safety: no pointer arithmetic
should be possible on the single-byte allocation.

This is an *allocation* convention, not a global invariant. NumPy restores ordinary
computed strides on reshape/ravel even for zero-size arrays:

```
a = np.empty((0, 3))          strides (0, 0)
a.reshape(3, 0)               strides (8, 8)
a.ravel()                     strides (8,)
np.empty((18, 5))[:, 0:0]     strides (40, 8)     # empty view of a base
```

All of these still report `C=True F=True`. So even in NumPy "strides are never used" is the
real contract; the zeroing is only a safe default.

### 2.3 The BLAS/LAPACK boundary

NumPy never passes an empty array's raw stride to BLAS/LAPACK:

- `numpy.linalg` copies inputs into a contiguous scratch buffer (`linearize_matrix`) and
  derives the leading dimension from the **shape**: `ld = max(N, 1)`
  (`numpy/linalg/umath_linalg.cpp:1724`, likewise 2184, 2408, 2762, 2894).
- Explicit empty handling exists where LAPACK cannot cope:
  `_is_empty_2d` (`numpy/linalg/_linalg.py:265`), `cond` raises (line 1996), `pinv`
  returns an empty `(..., n, m)` (line 2251), and `lstsq` allocates one extra RHS column
  when `n_rhs == 0` and slices it off afterwards (lines 2532-2545).

Observed behavior (`numpy_output.txt`): `solve(eye(18), zeros((18,0)))`,
`solve(eye(18), base[:, 0:0])`, `inv(zeros((0,0)))`, `det(zeros((0,0)))`,
`matmul(eye(18), zeros((18,0)))` all succeed without a BLAS error.

## 3. What rstsr does today

### 3.1 Semantic predicates: already NumPy-compatible

| Rule | Location |
|---|---|
| `size() == 0 ⇒ f_prefer() == true` | `rstsr-common/src/layout/layoutbase.rs:87-95` |
| `size() == 0 ⇒ c_prefer() == true` | `rstsr-common/src/layout/layoutbase.rs:117-125` |
| `size() == 0 ⇒ f_contig()/c_contig()` | `layoutbase.rs:150-188`, `191-202` |
| `check_strides` unconditionally OK | `layoutbase.rs:283-292` |
| `bounds_index` returns `(offset, offset)` | `layoutbase.rs:235-249` |
| `is_broadcasted` false (empty is writable) | `rstsr-common/src/layout/broadcast.rs:280-282` |
| `Layout::eq` ignores strides on dim 0/1 | `layoutbase.rs:600-616` |
| `Debug` prints `Empty` | `rstsr-common/src/format_layout.rs:16` |

### 3.2 Stride values: no single rule

`Ix<N>` multiplies by `.max(1)`; `IxD` multiplies literally by the dimension (so a zero
axis zeroes every stride outside it):

| shape | F static | F dynamic | C static | C dynamic |
|---|---|---|---|---|
| `[0, 18]` | `[1, 1]` | `[1, 0]` | `[18, 1]` | `[18, 1]` |
| `[18, 0]` | `[1, 18]` | `[1, 18]` | `[1, 1]` | `[0, 1]` |

Sources: static `rstsr-common/src/layout/shape.rs:89-106`; dynamic `:199-216`. The same
shape therefore gets different strides depending on whether it went through `Ix2` or
`IxD`. Created tensors keep whichever layout they were built from (probe-layout output):

```
rt::empty(([0,18].f(), &dev))    stride [1, 1]     (static source)
rt::empty((vec![18,0], &dev))    stride [0, 1]     (dynamic, default C order)
rt::zeros([0usize,18].f())       stride [1, 1]
```

The `max(1)` rule is not coherent under permutation either: `[0,18].f()` is `[1,1]`, and its
transpose `[18,0]` is still `[1,1]`, whose `ld_col` (1) is too small for `M = 18`.

### 3.3 Leading dimensions

`ld_row` / `ld_col` return the raw stride unless the *other* axis has length 1
(`rstsr-core/src/tensor/tensor2_impl.rs:22-52`):

```rust
// ld_col
if !self.f_prefer() { None }
else if self.shape()[1] == 1 { Some(self.shape()[0]) }   // can return 0
else { Some(self.stride()[1] as usize) }                 // can return 0 / 1
```

`shape()[1] == 0` is not handled. Even the `== 1` branch returns `0` when `shape()[0] == 0`,
although BLAS requires `ld >= max(1, m) = 1`. (Minor: the comments in `ld_row`/`ld_col`
call `[1, n]` a "col-vector" and `[m, 1]` a "row-vector"; they are swapped.)

BLAS/LAPACK call sites consume these values unchanged:
`blas3/gemm.rs:111-112,139`, `blas3/syhemm.rs:107-108,133`, `blas3/trsm.rs:114,127`,
`lapack_solve/gesv.rs:47-48`, `lapack_solve/potrf.rs:42`, and the eigh/svd wrappers.

### 3.4 Representation-dependent normalization

`change_contig_f` normalizes the layout of a contiguous tensor, including empty ones
(`rstsr-core/src/tensor/manipulation/to_contig.rs:8-45`):

```
reverse then to_contig_f(F):      shape [18, 0], stride [1, 18]
```

But `overwritable_convert_with_order` only takes the normalizing path when the argument is
a *reference* (`a.is_ref()`); an owned/mut/COW tensor that reports `f_prefer()` is passed
through untouched (`rstsr-blas-traits/src/util/util_tensor.rs:34-52`). For non-empty
layouts this is sound (`f_prefer` then implies usable strides); for empty layouts it is
not, because `f_prefer` carries no information about strides.

Measured through `solve_triangular` (probe-blas output, `** On entry to DTRSM` = `xerbla`):

| # | input | shape | strides | `ld_col` | result |
|---|---|---|---|---|---|
| 1 | issue: static `[0,18].f()` reversed (owned/mut) | `[18,0]` | `[1,1]` | 1 | **xerbla, silent success** |
| 2 | static `[18,0].f()` direct | `[18,0]` | `[1,18]` | 18 | OK |
| 3 | dynamic `[18,0]`, explicit RowMajor, owned | `[18,0]` | `[0,1]` | 1 | **xerbla, silent success** |
| 4 | f-order `[5,18]`, rows `0..0`, transposed (view) | `[18,0]` | `[5,1]` | 1 | OK (view normalized internally) |
| 5 | row-major `[18,5]`, cols `0..0` (view) | `[18,0]` | `[5,1]` | 1 | OK (view normalized internally) |
| 6/7 | owned copy of 4/5 | `[18,0]` | `[1,18]` | 18 | OK (`to_owned` recomputes) |
| 8 | `solve_general(eye(18), issue b)` (`DGESV`) | `[18,0]` | `[1,1]` | 1 | **`InvalidValue` "illegal argument 7"** |

Case 8 is the LAPACK side of the same defect: LAPACK validates `LDB` before returning
(`lapack/SRC/dgetrs.f:168`; same pattern in `dpotrf.f:149-153`), returns `INFO = -7`, and
rstsr maps negative `INFO` to `InvalidValue` in `map_lapack_info`
(`rstsr-blas-traits/src/util/util_error.rs:12-30`).

Note also `ld_row = 0` in case 3: row-major LAPACK drivers would trip the analogous check.

## 4. Gap summary

- **G1 — no documented policy.** Empty-layout strides are neither specified nor declared
  off-limits; `Debug` and DLPack export them, BLAS reads them.
- **G2 — static/dynamic divergence.** `Ix` and `IxD` disagree on zero-size strides (and
  on which axes are zeroed). The same tensor constructed through different shape types
  carries different layouts.
- **G3 — `ld_*` is not empty-safe.** It consumes a meaningless stride, has no `max(1, ·)`
  lower bound, and can return 0 even in its singleton-axis branch.
- **G4 — `f_prefer`/`c_prefer` collapse the distinction.** Because they are true for every
  empty tensor, the "already preferred ⇒ no normalization" shortcut in
  `overwritable_convert_with_order` keeps the bogus layout. The same shape works as a view
  and fails as an owned/mut tensor.
- **G5 — BLAS-3 has no error channel.** `xerbla` prints to stderr and the call reports
  success. LAPACK at least returns an error.
- **G6 — no tests.** No BLAS/LAPACK tests exercise zero-size operands. LAPACK's `potrf`
  on `0×0` and SVD/eigh on empty shapes are also uncovered.

## 5. Proposed policy

### P1 — Semantics (normative)

A layout with `size() == 0` has no addressable element. Therefore:

1. its strides are **unspecified** and must not be consumed as data-addressing metadata;
2. it is both C- and F-contiguous and both c-/f-preferred (current behavior, keep it);
3. any consumer that needs a scalar derived from strides (leading dimension, pointer
   bound, DLPack, ...) must compute it from the **shape**, clamped to its contract
   minimum — or short-circuit on the empty case.

This is NumPy's documented rule and validates the predicates rstsr already has.

### P2 — Contiguous-constructor values (implementation)

Change `stride_f_contig` / `stride_c_contig` to return **all zeros when
`shape_size() == 0`**, for both `Ix<N>` and `IxD`. Rationale:

- one deterministic value rule across static and dynamic shapes;
- matches NumPy's allocation convention and what a fresh NumPy array exports;
- zero is unambiguous ("no motion"), whereas `max(1)` fabricates a plausible but wrong
  leading dimension.

This is *cosmetic/deterministic*, not a safety guarantee: views and slices of non-empty
bases may still carry arbitrary strides (as in NumPy), which is why P3 remains mandatory.

Rejected alternative: "as if zero dimensions were 1" (the current static rule). It is not
stable under permutation (`[0,18] → [18,0]` keeps `[1,1]`), so it cannot serve as a
policy even though it accidentally produces a valid `ld` for directly created `[18,0]`.

### P3 — Consumers (mandatory)

**Leading dimension.** Return a value that satisfies the BLAS/LAPACK contract for the
logical shape:

```text
ld_col: shape[1] <= 1  -> max(1, shape[0])
        otherwise      -> max(stride[1], max(1, shape[0]))
ld_row: shape[0] <= 1  -> max(1, shape[1])
        otherwise      -> max(stride[0], max(1, shape[1]))
```

For every non-empty prefer layout these are exactly today's values (a valid f-prefer
layout has `stride[1] >= shape[0]`, and the transpose for row-major is symmetric).

**Early return.** A BLAS/LAPACK call with no elements to process must return the
mathematically correct empty result without calling the library, after the ordinary shape
and device validation. Minimum matrix:

| routine | short-circuit condition | result |
|---|---|---|
| `gemm` | `m == 0 \|\| n == 0` | `C` unchanged (empty) |
| `syhemm` | `m == 0 \|\| n == 0` | `C` unchanged (empty) |
| `trsm` | `m == 0 \|\| n == 0` | `B` unchanged (empty) |
| `gesv` | `n == 0` → empty; `nrhs == 0` → factor `A`, skip the solve (or pass `ldb = max(1,n)`) | |
| `getrf` | `m == 0 \|\| n == 0` | empty pivots |
| `getri` | `n == 0` | return |
| `potrf` / `sysv` | `n == 0` | return |
| `syev` / `syevd` / `sygv` / `sygvd` | `n == 0` | empty eigenvalues/eigenvectors |
| `gesvd` / `gesdd` | `m == 0 \|\| n == 0` | correctly shaped `U`, empty `S`, `Vt` |

`gemm` with `k == 0` is *not* a no-op (`C := beta*C`); handle it explicitly or rely on the
clamped `lda/ldb/ldc` and let BLAS run. Note that the LAPACK work-query paths also need a
zero-size check, since some drivers query workspace with `n == 0`.

### P4 — Documentation and tests

- Add the policy (P1+P2) to the layout documentation and to the `ld_row`/`ld_col`
  docstrings; fix the swapped "row-vector"/"col-vector" comments.
- Unit tests in `rstsr-common`: stride values for zero-size shapes, `Ix` vs `IxD`;
  `ld_*` lower bounds for `[0,0]`, `[0,n]`, `[m,0]`, `[0,1]`, `[1,0]`.
- BLAS/LAPACK tests: `solve_triangular` with `[18,0]` / `[0,18]` in both orders and in
  owned / mut / view reprs; `solve_general` with `nrhs = 0`; `potrf` `0×0`; `svd` on
  `0×n` / `m×0`; assert no `xerbla` output.
- Consider installing an OpenBLAS `xerbla` handler through
  `openblas_set_xerbla` (the binding exists in `rstsr-openblas-ffi`) so a test fails loudly
  on an illegal parameter instead of relying on stderr matching.

## 6. Why the earliest fix should be the early return

Cases 4-7 show that the same shape can arrive with any stride, including from an empty view
of a non-empty buffer, and even the `ld` clamp cannot recover a "correct" stride because
there is none. A leading dimension is an *input contract*, so the only local answer for an
empty operand is to not call the routine. The stride-value rule (P2) and the `ld` clamp
(P3, first item) then become defense in depth: they keep the API self-consistent and make any missed
call site fail with a legal parameter instead of a silent wrong result.

## 7. Reproduction

```bash
# layout-only probe (no BLAS)
cd probe-layout && cargo run            # output.txt

# BLAS probe (uses the system OpenBLAS through dynamic loading)
cd probe-blas && for i in 1 2 3 4 5 6 7 8; do cargo run -- $i; done   # output.txt

# NumPy reference
python3 numpy_probe.py                  # numpy_output.txt
```

Both probes link rstsr by relative path (`../../../rstsr/rstsr`) and were run with
`CARGO_TARGET_DIR=<rstsr>/target`; the BLAS probe enables `col_major`, `openblas`,
`linalg` and `dynamic_loading`.
