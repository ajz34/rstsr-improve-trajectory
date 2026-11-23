# Grill R3 — BLAS-2 GEMV dispatch for matmul (`openblas` / `faer`)

Session 2026-11-23 (round 3; the grill opened 2026-11-18). rstsr `d5c9b86`
(`261023/astype`, clean tree); this repo clean. **No rstsr file has been
edited.** Recommended answers marked ➡️.

---

## R2 answers — agent-authored transcript (chat; no answers file exists)

- **Q1 = (a)**. The `256 × 452676` case is two *physically contiguous* matrices
  `A(256, 452676)` and `Aᵀ(452676, 256)`; no `.t()`-view cell.
- **Q2 = (a)** `GEMV_` is order-free.
- **Q3 = (b)**, with the note: *"Please re-check the details on `flip_trans`. It
  returns `TensorCow` (not necessary owns data), and if input c-prefer, it will
  automatically handle N/T cases. For C (complex conjugate), it may perform a
  conjugate operation and return an owned tensor wrapped by `TensorCow`. So at
  least for f32/f64, and c32/c64 if no conjugate, any input 2-dim f-prefer or
  c-prefer tensor should make no copy and call correct GEMV operation with
  proper trans code. This is also the most important and very careful code in
  GEMM or TRSM, so you may need to re-read those part of code, and transfer your
  learning on GEMV."*
- **Q4** approximately correct, plus: *"For usual cases, please refer Q3's
  answer and carefully check how `flip_trans` works on matrices. `flip_trans` do
  not need to be operated on vectors, and as your recommendation, pass the
  stride size."*
- **Q5**: *"I think probably we already have some functions? Like `dim_chop`
  which is probably just for this usage."*
- **Q6 = okay**, with: *"note this is at matmul implementation function, not in
  `GEMV_` of BLAS wrapper."* (the threading rule lives in the device matmul
  code).
- **Q7 = (a)** keep the drift, patch all five copies.
- **Q8 = okay**, with: *"`CHANGELOG.md` is only updated when version updates,
  not now."*
- **Q9 = (a)**; **Q10 = two patches**.

### Settled decisions

- D1–D15 from R1/R2 stand, amended by:
- **D16** The `GEMV_` container owns the matrix normalisation, via
  `flip_trans(ColMajor, transa, a, false)` — the same code path GEMM uses. No
  copy for c-prefer/f-prefer 2-D inputs with a legal `ld`; a copy only appears
  when the layout is neither-prefer (or when `C` needs a conjugated buffer). The
  device matmul path keeps its own matrix conversion (the `gemm_blas_no_conj_*`
  pattern), i.e. Q3(b), not the shared-helper option of D9.
- **D17** `flip_trans` is only ever applied to the 2-D matrix; vectors are never
  flipped, their strides are passed through.
- **D18** Threading changes live only in the device matmul implementation
  (`matmul_row_major_blas`'s broadcast loop), not in the BLAS-2 wrapper.
- **D19** No `CHANGELOG.md` edit now; changelog entries happen at version bump.

---

## New facts gathered for R3

**F6 — the squeeze helpers already exist.** `rstsr-common/src/layout/indexer.rs`:
`dim_eliminate(axis)` (decl `:220`, impl `:264`) raises unless `shape[axis] == 1` and
returns the layout with that axis removed; `dim_chop(axis)` (decl `:224`, impl `:284`)
removes the axis with no check (used for zero-size axes in reductions/advanced
indexing). Both are on `IndexerSmallerOneAPI`, whose `DOut` for `Ix<2>` is
`Ix<1>` (`rstsr-common/src/layout/dim.rs:141`), so
`la.dim_eliminate(0)?` turns the `(1,K)` matrix layout directly into a
`Layout<Ix1>` of length `K` — no `into_dim` step. Both traits are in scope in
the device crates through the `layout::exports::*` glob in `prelude_dev`.

**F7 — negative vector increments.** Verified against the system OpenBLAS
(0.3.32): `cblas_dgemv` accepts negative `incX`/`incY`, but with the F77/CBLAS
convention that the pointer is the *array start* and the first logical element
sits at the far end (`base + (len-1)*|inc|`). rstsr layouts store the pointer at
the logical first element, so forwarding a negative stride requires a pointer
bump of `(len-1)*stride`; a raw pass-through reads out of bounds. This is the
concrete reason the "copy when stride ≤ 0" rule stands (BLIS/AOCL/KML behaviour
for negative increments is untested here).

**F8 — `TensorMutable::ToBeCloned` write-back exists** (`clone_to_mut`,
`rstsr-core/src/tensor/tensor_mutable.rs:87`) and `assign_f`/`assign` are
available on mutable tensors (`rstsr-core/src/tensor/assignment.rs:154/161`), so
the container can handle a non-positive-stride `y` without new machinery.

---

## Q1 — Where the device matmul path's gemv wrapper lives, and through which FFI

- (a) a new `gemv_blas_no_conj_{f32,f64,c32,c64}` in each device's
  `matmul_impl.rs` (five copies, the same place as `gemm_blas_no_conj_*` and
  `syrk_blas_no_conj_*`), calling **`B::driver_gemv`** so the cblas signature and
  the complex-pointer protocol live once in
  `rstsr-blas-traits/driver_impl/cblas/blas2/gemv.rs`;
- (b) same placement but calling `lapack_ffi::cblas::cblas_*gemv` directly, to
  mirror how the gemm/syrk wrappers call cblas;
- (c) put the wrapper itself in `rstsr-blas-traits` as a thin f-prefer-only
  function (what D9 originally said, superseded by D16).

➡️ (a). It keeps the new trait load-bearing (the container and the `%` path go
through the same `driver_gemv`), avoids re-deriving the `*const c_void` complex
protocol per crate, and leaves the wrapper in the same file and style as its
gemm/syrk siblings.

---

## Q2 — Vector strides in the device path: guard-and-fall-back or copy?

When the fast path is entered from `(1,2,1)` / `(2,1,1)` / the `(2,2,2)`
reduction, `x` or `y` may have a stride ≤ 0 (reversed or broadcast view).

- (a) **guard**: if either vector stride is ≤ 0, do not take the gemv fast path
  — fall through to today's broadcast-config + gemm path (correct, no copy);
- (b) **copy**: copy the affected vector to a unit-stride temporary (and write
  `y` back afterwards) inside the gemv wrapper (five copies of the copy/write
  back logic).

➡️ (a). The old path is already correct for those layouts, the guard is two
comparisons, and F7 shows the copy would be the only reason to add write-back
code to five crates. (The `GEMV_` container cannot fall back, so it does copy —
Q3.)

---

## Q3 — How `GEMV_` handles a non-positive-stride `y`

- (a) reuse GEMM's `TensorMutable::ToBeCloned(y, buffer)` + `clone_to_mut()`
  (F8): copy `y` into a contiguous buffer, run `driver_gemv` into it, write back;
- (b) teach the driver `incx`/`incy` as `isize` and forward negative increments
  with the pointer bump from F7;
- (c) reject such `y` (error).

➡️ (a). No new driver surface, no vendor-dependent negative-increment reliance,
and the copy is O(m) against the O(m·k) gemv. For `x` the container copies into
an owned contiguous buffer with `to_contig`, same rule.

---

## Q4 — `GEMV_` `run` / `internal_run` contract (confirm)

Proposed, mirroring `GEMM_`:

- `run()`: reject `FlagTrans::CN` (`rstsr_invalid!`, like `gemm.rs:56-65`); apply
  `flip_trans(ColMajor, transa, a, false)` unconditionally (order-free, D16/D17);
  rebuild the builder with `a_cow.view()` and `order: Some(ColMajor)`;
  call `internal_run()`.
- `internal_run()`: assert `order == Some(ColMajor)`; device-equality checks for
  `a`/`x`/`y`; derive `m`, `n`, `k` from `transa`; assert `x.len() == k` and
  `y.len() == m` when supplied; resolve `y = None` into `zeros_f([m])`;
  early-return on `m == 0`; `k == 0` → `y := beta*y` (fill path, `A`/`x`
  untouched); assert `a.f_prefer()`; `lda = a.testify_ld(ColMajor, "A")`;
  compute `incx`/`incy` under the Q2/Q3 rule; call
  `B::driver_gemv(ColMajor, transa, m, n, alpha, ptr_a, lda, ptr_x, incx, beta,
  ptr_y, incy)`; return `TensorMutable::Mut(y)` (or `Owned` when allocated).

➡️ Yes. Note one asymmetry to accept: `driver_gemv` keeps an `order` parameter for
symmetry with `driver_gemm`, even though the container always passes `ColMajor`.

---

## Q5 — Size-1 axis removal in the `(2,2,2)` reduction: `dim_eliminate` or `dim_chop`

- (a) `dim_eliminate` — validates `shape[axis] == 1` and returns `Layout<Ix1>`
  directly;
- (b) `dim_chop` — unchecked, returns the same type.

➡️ (a). The size-1 fact is an invariant of the reduction (it is exactly what the
arm tests), and letting the helper assert it costs nothing; `dim_chop`'s niche is
zero-size axes, which the reduction must not accept.

---

## Q6 — P0 cell table (final, for confirmation)

Per device (`DeviceOpenBLAS`, `DeviceFaer`), f64 unless noted; one `%` /
`rt::matmul` cell per row (fresh output, `beta = 0`), plus the extras below.

| matrix | cells | note |
|---|---|---|
| `(1024, 1024)` | `A @ x`, `xᵀ @ A` | L2/L3-resident |
| `(4096, 4096)` | both | L3-resident |
| `(8192, 8192)` | both | DRAM-bound |
| `(65536, 64)` | both | skinny, large `K` |
| `(256, 452676)` | both | 0.93 GB, DRAM-bound, output 256 / 452676 |
| `(452676, 256)` | both | the physical transpose, 0.93 GB |
| `(1024, 1024)` f32 | both | dtype spot |
| `(1024, 1024)` | `matmul_from(beta = 1, out=)` | write-back path |
| `(64, 1024, 1024)` batched | `A @ x` (rule 4) | batched/threading |

Anchors: raw `cblas_dgemv` for the same 10 f64 cells and numpy 2.5 for the
`(1024,1024)` and `(256,452676)` cells. Protocol: criterion, 3 repetitions,
`sample_size(10)`/`measurement_time(5s)` for the 0.93 GB cells, thread count
pinned through the OpenMP mechanism (F5/R1) and asserted at bench start.

➡️ Confirm, or trim/adjust cells. Runtime estimate: dominated by the 0.93 GB
cells (≈20 ms per call, 10 samples × 3 reps × 4 cells × 2 devices ≈ 1 min) plus
build; total under 10 min per device.

---

## Q7 — Which OpenBLAS for P0

The system `libopenblas.so.0` is 0.3.32, `DYNAMIC_ARCH ... USE_OPENMP`, and
reports `openblas_get_corename() = Cooperlake` on this Zen 5 box — an Intel
kernel chosen by the distro build, which can distort the gemm-vs-gemv gap.

- (a) P0 on the system library as-is (what rstsr links here), record the config,
  and re-run only the headline cells on a locally built Zen-5 OpenBLAS if the
  numbers look anomalous;
- (b) build a Zen-5 OpenBLAS first and make it the P0 primary, system lib as a
  cross-check.

➡️ (a). The system library is the one the developer's rstsr link line resolves
to, so its numbers are the honest baseline; building a second OpenBLAS is a
one-day detour that does not change the dispatch decision.
