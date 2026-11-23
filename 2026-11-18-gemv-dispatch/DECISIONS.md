# Decisions — BLAS-2 GEMV dispatch (grill closed 2026-11-23)

Grill rounds: R1 (campaign shape), R2 (API/normalisation), R3 (mechanics), R4
(vector policy + go/no-go), R5 (post-P0 scope). Answers are given in chat and are
transcribed at the top of the following round's questions file. Grill facts were
gathered at rstsr `d5c9b86` (`261023/astype`); **the P0 harness was built and run
against `062c970` (`main`, v0.10.4)** whose matmul dispatch is unchanged. No rstsr
file has been edited yet.

## Measurement

- **D1** P0 measures first; P1 (blas-traits BLAS-2) and P2 (device dispatch)
  start only after the P0 numbers are reviewed.
- **D2** Every benchmark artefact lives in this directory (or a local scratch
  place). rstsr receives code only — no `benches/`, no criterion wiring, no
  timing tests.
- **D3** P0 surface: `%` / `rt::matmul` for both orientations (1-D operands and
  degenerate 2-D operands), one `matmul_from` cell (`beta = 1`, caller `out=`),
  one batched cell.
- **D4** P0 devices: `DeviceOpenBLAS` and `DeviceFaer`.
- **D5** Anchors: raw `cblas_?gemv` on the same linked library (ceiling) and
  numpy 2.5 (cross-language sanity). Thread-control mechanism recorded and
  pinned.
- **D6** Cells: `1024²`, `4096²`, `8192²`, `65536×64`, `256×452676`,
  `452676×256` (all contiguous; no `.t()`-view cell), f32 spot at `1024²`.
- **D7** Success: match raw `cblas_?gemv` within ~10 % on the target shapes; no
  regression on the plain square-GEMM cells. Gates P2.
- **D25** P0 runs on the system OpenBLAS 0.3.32 as the baseline (record
  `openblas_get_config()`/`get_corename()`); a Zen-5 build is only a cross-check
  if numbers look anomalous.

## `rstsr-blas-traits` (P1)

- **D8** Add `blas2/gemv.rs` with `GEMVDriverAPI<T>`, the `GEMV_` builder
  container and `SGEMV`/`DGEMV`/`CGEMV`/`ZGEMV` aliases; register the module in
  `lib.rs` and the public prelude; compose `GEMVDriverAPI<T>` into
  `BlasDriverAPI<T>` (a breaking change for external implementors; the five
  in-tree devices get it from the shared template).
- **D16** `GEMV_` owns the matrix normalisation via
  `flip_trans(ColMajor, transa, a, false)`: no copy for c-/f-prefer 2-D input
  with a legal `ld`; a conjugate copy only for `C` on a c-prefer complex matrix.
- **D17** `flip_trans` is only applied to the 2-D matrix; vectors are never
  flipped.
- **D22** `GEMV_::run` rejects `CN`, flips to `ColMajor`, then `internal_run`
  relies on `testify_ld(ColMajor, "A")` (which also enforces f-prefer), checks
  `x.len() == k`/`y.len() == m`, allocates `zeros_f` when `y` is `None`, early
  returns on `m == 0`, fills `y := beta*y` on `k == 0`, calls
  `B::driver_gemv(ColMajor, …)` and returns `TensorMutable::Mut(y)`.
- **D27 (R4/Q2 = b)** `driver_gemv` has **no** `incx`/`incy`: unit stride only,
  minimal surface. Re-add them only if a consumer needs increments.
- **D28 (R4/Q1)** Vectors: contiguous (unit-stride) → no copy; non-contiguous →
  copy (gather in, scatter out for `y`), in both the container and the device
  path.
- **F8 note** the container reuses `TensorMutable::ToBeCloned` + `clone_to_mut()`
  for a non-unit-stride `y`.
- **D9 (R1/Q9 = a)** One shared implementation of the layout/ld/vector
  normalisation, not five.
  *Superseded in part by **D20** (R3/Q1 = a): only `driver_gemv` is shared;
  the slice-level `ld`/vector normalisation is a per-device
  `gemv_blas_no_conj_*` in each `matmul_impl.rs`, because it needs the device's
  own `assign_cpu_rayon` staging and `Layout` → `FlagTrans` decisions. Putting
  the wrapper in `rstsr-blas-traits` was R3/Q1 option (c), rejected.*

## Device dispatch (P2)

- **D20 (R3/Q1 = a)** New `gemv_blas_no_conj_{f32,f64,c32,c64}` in each device
  `matmul_impl.rs` (five copies, next to the gemm/syrk wrappers), routed through
  `B::driver_gemv`; no new direct cblas call site.
- **D21 (R3/Q2 = b, R4/Q1)** The device path copies non-unit-stride vectors
  instead of guarding/falling back; no increment forwarding anywhere.
- **D23** The `(2,2,2)` size-1 reduction uses `Layout::dim_eliminate`
  (validated, returns `Ix1`), not `dim_chop`.
- **D10** Fast path accepted for positive strides in the original plan —
  superseded by D28: unit stride or copy.
- **D11 (R2/Q6, R3/Q6)** Batched gemv: see D29, which supersedes the "rayon over
  batches with one BLAS thread per task" rule.
- **D18** That threading change lives in the device matmul implementation, not in
  `GEMV_`.
- **D7 (R3/Q4 = a)** Keep the drift; patch all five `matmul.rs` copies.
- **D12** `DeviceGEMVAPI` stays dead (historical; may be removed later).
- **Known divergence (file, do not fix in patch-2)** `gemm_blas_no_conj_*` and
  `syrk_blas_no_conj_*` zero-fill the output at `k == 0`
  (`matmul_impl.rs:79` / `:332`), ignoring `beta`, while `matmul_from`
  documents `c = beta*c + alpha*(a@b)`. The gemv wrapper (and `GEMV_`) does
  `beta*y`; the gemm side is a separate issue with its own test.
- **D8 (R2/Q8)** No `CHANGELOG.md` edit now — changelog entries happen at
  version bump.

## Verification and handoff

- **D14** Tests in the openblas copy: in-crate tests in `matmul.rs` for the new
  dispatch and a new `tests/blas2_gemv.rs` module for the `GEMV_` container
  (wired into `entry_row_openblas.rs`); gates
  `cargo test -p rstsr-openblas --test entry_row_openblas`,
  `cargo test -p rstsr-core --test entry_row_cpu --no-default-features
  --features "backtrace row_major"`, `cargo check` for the other four device
  crates, clippy on touched crates. *Amended 2026-11-23 after review:* the
  container body moved to
  `rstsr-blas-traits/src/blas_tests_impl/driver_impl/blas2_gemv.rs`, symlinked
  into each device crate and run by `tests_driver_impl` (ADR-0012 §4); the
  OpenBLAS-local `tests/blas2_gemv.rs` and the `entry_row_openblas.rs` wiring
  were dropped, because a module pulled in from the entry would compile the
  same `#[test]`s twice.
- **D10 (R2/Q10)** Two patches: patch-1 = blas-traits BLAS-2 API + cblas
  template + container test; patch-2 = the five `matmul.rs` dispatch sites +
  dispatch tests.
- **D2** Handoff = `proposed.patch` files + README numbers; no push/PR without
  explicit permission.

## Scipy cross-check (2026-11-23, `~/Git-Others/scipy` @ `791ccf2df0`)

`scipy.linalg.blas.dgemv(alpha, a, x, beta=0, y=None, offx=0, incx=1, offy=0,
incy=1, trans=0, overwrite_y=0)` (`scipy/linalg/src/blas/blas_l2.cpp:21-53`;
docstring `blas_docs.cpp:409`). It forces `NPY_ARRAY_F_CONTIGUOUS` on every
array (`as_in`, `blas_helpers.hpp:236-242`), derives `lda` from the shape, and
exposes `off`/`inc` only to step inside an already-contiguous buffer (bounds
checked with `abs(inc)`, so negative increments are legal, with the buffer-start
pointer convention). `y=None` allocates a zero-filled vector; `overwrite_y` is
f2py's `intent(in,out)` opt-in.

Consequence for us: scipy pays for strided views at the numpy conversion
boundary (copy), exactly as D28 does at the rstsr boundary; its `inc` arguments
are not a general strided-tensor mechanism. Keeping `incx`/`incy` out of
`driver_gemv` (D27) is consistent with that, at the cost of needing a trait
change if a future caller wants zero-copy strided vectors.

## Execution order

1. P0 harness + numbers (this directory) — **in progress**.
2. Review gate.
3. patch-1 (`rstsr-blas-traits`), then patch-2 (devices), each with tests.
4. P3 re-measure with the same harness, README conclusion.

## R5 answers (post-P0) and D29

- **Q1 = (a)**: this campaign is BLAS-only. Faer is used for benchmarking only;
  do not improve efficiency by means other than BLAS in normal cases. Faer's
  1.4–2.4× gaps are recorded as a follow-up, not patched here.
- **Q2**: **D29** — for batched cases (`n_task > 1`) in the BLAS device matmul
  implementation, always iterate batches **in the main thread** and let BLAS be
  multi-threaded (`with_num_threads(nthreads, …)`). Do **not** spawn rayon tasks
  that each run single-threaded BLAS (`with_num_threads(1, …)` inside rayon).
  This applies to **every** `n_task > 1`, however large — the
  `n_task >= 4 * nthreads` rayon-outer branch is removed. This is a change in the
  generic batch loop, so batched GEMM shapes are affected too and must be
  re-measured in P3 (a batched rule-7 cell is added to the P0 harness as the
  before/after datum).

  *Scope narrowed by the maintainer 2026-11-23.* Batch threading keys off the
  **inner** matmul shape. Vector-shaped inner product (`m == 1` or `n == 1`,
  i.e. it becomes `gemv`): always iterate batches in the main thread with
  `with_num_threads(nthreads, …)`, for every `n_task` — gemv is memory-bound
  and rayon-over-batch would fragment the cache. Real matrix product: keep the
  existing `n_task >= 4 * nthreads` rule (rayon outer,
  `with_num_threads(1, …)`) unchanged. Batched GEMM therefore keeps its current
  threading, and the rule-7 20.41 ms datum is no longer a risk for patch-2.
- **Q3 = okay** → patch-1 implemented on branch `261123/blas2-gemv` (see below).

## Patch-1 state (2026-11-23, branch `261123/blas2-gemv`, commits `3c3999d`
and `718864a`)

Added `rstsr-blas-traits`:

- `src/blas2/{mod,gemv}.rs`: `GEMVDriverAPI<T>` (no `incx`/`incy`, per D27) and
  the order-free `GEMV_` container with `SGEMV/DGEMV/CGEMV/ZGEMV` aliases.
  `run()` rejects `CN`, applies `flip_trans(ColMajor, …, false)`;
  `internal_run()` checks devices/shapes, converts a non-unit-stride `y` through
  `TensorMutable::ToBeCloned`, allocates `zeros` when `y` is omitted, returns
  early on an empty output, scales `y := beta*y` when the contraction is empty
  for every `beta != 1` (BLAS returns before scaling, so the container does it),
  copies a non-unit-stride `x`, and calls the driver with the `testify_ld`
  leading dimension.
- `lib.rs` `pub mod blas2;`, `prelude.rs` export, `GEMVDriverAPI<T>` composed into
  `BlasDriverAPI<T>`.
- `driver_impl/cblas/blas2/gemv.rs` (+`mod.rs`, `cblas/mod.rs`): the four-type
  `duplicate_item` impls calling `cblas_{s,d,c,z}gemv` with unit increments.
- The container needs `B: OpAssignArbitaryAPI<T, Ix1, Ix1>` (for the vector
  copy paths); that is the only extra bound over `BlasDriverBaseAPI`.
- `rstsr-blas-traits/src/blas_tests_impl/driver_impl/blas2_gemv.rs` (10 tests:
  matvec, trans, beta with caller output, strided y/x, column-major input, zero
  contraction with `beta = 2` and `beta = 0`, complex conj-trans, f32, CN
  rejection), symlinked into all five device crates and run by
  `tests_driver_impl` (see the D14 amendment). References are elementwise, so
  the assertions survive patch-2 routing `%` through `GEMV_`.
- Gates: `cargo fmt --check`, `cargo clippy -p rstsr-blas-traits --all-targets
  --all-features -- -D warnings`, `cargo clippy -p rstsr-openblas --all-targets
  -- -D warnings`, `cargo test -p rstsr-openblas --test tests_driver_impl blas2`
  (10/10), `cargo check --tests -p rstsr-kml` (second device compiles the shared
  body).

Review pass (same day; `3c3999d` = reviewed state, `718864a` = fixes):

- Fixed: empty contraction with `beta == 0` returned the caller `y` unscaled,
  while `GEMM_` and the dispatched `matmul` give zeros. Regression test added
  (it fails against the old guard).
- Fixed: five container tests used `&a % &x` as oracle, which patch-2 reroutes
  through `GEMV_` itself; references are now elementwise.
- Fixed: test placement per ADR-0012 §4 (D14 amendment).
- Fixed: names (`len_x`/`len_y`, `transa`) and trimmed comments that overstated
  what the code guarantees (no-`order` rationale, driver "unit-stride" contract,
  `ld` derivation).
- Left as is, judged sound: aliasing preconditions between `x`/`y`/`a` (same as
  `GEMM_`), no `order` field (D16), and extra tests for a `beta != 0` strided
  `y` or a neither-prefer matrix (both traced correct by hand; deferred, since
  the user asked not to over-test at this stage).

## Patch-2 state (2026-11-23, branch `261123/blas2-gemv`, commit `c93fdde`)

- Each device `matmul_impl.rs` gained `gemv_blas_no_conj_{f32,f64,c32,c64}`
  next to the gemm/syrk wrappers (D20), routed through `B::driver_gemv` (no new
  cblas call site): `lda` from shape/stride, `x`/`y` staged through contiguous
  buffers (`beta == 0` never reads `y`), c-prefer matrix viewed transposed with
  the flag flipped, otherwise copied, and `y := beta*y` on an empty contraction.
- Each device `matmul.rs` gained the `(2,1,1)` and `(1,2,1)` arms and routes the
  `(2,2,2)` arm through a shape-aware wrapper: a length-1 output axis is
  eliminated with `Layout::dim_eliminate` (D23) and dispatched to gemv. `k == 1`
  stays gemm — that product is an outer product, not a matvec.
- Batch loop: `vec_inner` (length-1 output axis) always iterates in the main
  thread with `with_num_threads(nthreads, …)`; real matrix batches keep
  `n_task >= 4 * nthreads` with rayon-outer and `with_num_threads(1, …)`. This
  is the 2026-11-23 scope narrowing of D29.
- Mixed dtypes keep the naive `gemv_naive_cpu_serial` / `gevm_naive_cpu_serial`
  fallback.
- `TB: Mul<TA, Output = TC>` is now required by the gemv dispatch and
  propagated to `matmul_row_major_blas` / `matmul_uninit_row_major_blas`
  (acceptable to the maintainer). The `(1,2,1)` and `m == 1` forms swap matrix
  and vector roles, so the transposed orientation needs it, and a
  `gemv`-against-`la.reverse_axes()` fallback does not remove it from the
  swapped call sites. `DeviceMatMulAPI` already carried the bound, so the trait
  surface is unchanged.
- Tests: 10 dispatcher tests per device copy (both arms, the three length-1
  output collapses, `k == 1` outer product, f-prefer/strided/broadcast
  operands, `beta` with a strided output, empty contraction, complex, batched
  shapes including `n_task = 256 > 4 nthreads`).
- Gates: `fmt --check` on five crates, clippy `--all-targets -D warnings`,
  `--lib` 17 passed, `entry_row_openblas` 501/501, `tests_driver_impl blas2`
  10/10, `cargo check --tests` for the other four devices, five copies still
  two hashes (four identical + OpenBLAS's extra rule-7 test).
- Not done yet: P3 re-measurement (needs a quiet machine), and the gemm/syrk
  `k == 0` zero-fill divergence still to be filed.
