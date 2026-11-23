# Grill R2 — BLAS-2 GEMV dispatch for matmul (`openblas` / `faer`)

Session 2026-11-18 (round 2). rstsr `d5c9b86` (`261023/astype`, clean tree); this
repo clean. **No rstsr file has been edited.** Recommended answers marked ➡️.

---

## R1 answers — agent-authored transcript (the answers arrived in chat, not in a file)

Recorded here verbatim in substance; the user may restate any of them.

- **Q1 = (a)** P0 measurement first, then review, then the patch. Plus: *"The
  sentence of 'we probably never dispatch to GEMV, and OpenBLAS gemm is not very
  efficient on some m×k, k×1 matmul' should be updated after GEMV abstraction
  landed."*
- **Q2–Q4, Q7–Q8 = recommendations** (dir/artefacts; call surface; OpenBLAS +
  Faer; success criterion; driver + `GEMV_` container + `BlasDriverAPI`
  supertrait).
- **Q5 = okay**, with: *"benchmark details should not leave into rstsr itself,
  but only in ruaa-traj or some local place. Let rstsr only handle code work and
  it is published as opensource."*
- **Q6 = yes**, plus *"Also a case on 256 x 452676 and its transpose. This should
  substantially exceed L3 cache."*
- **Q9 = (a)**, with: *"be very careful on how to handle dimension, tensor
  view/mut/own relations, to avoid explicit copy unless very necessary, let both
  contiguous/c-or-f-preferred/strided tensors also able to give correct results.
  The GEMM/TRSM should be good example, but that requires very careful design."*
- **Q10 = okay**, and *"try (b) since this is achievable in principle"* (positive
  strides passed directly as `incx`/`incy`; copy only when the stride is ≤ 0).
- **Q11 = (b)** gemv-specific threading: *"probably in any situation use
  parallel, and since it's memory-bounded, parallel/serial should theoretically
  give similar performance, so accept parallel."*
- **Q12 = "show me the drift, and let me decide"** → §Q7 below.
- **Q13 = (a)** leave the dead `DeviceGEMVAPI` (*"historical reason, may be
  removed in future"*).
- **Q14 = okay.**

### Settled decisions (carried forward, not re-asked)

- D1 Phasing: P0 measurement-only with a review gate → P1 `rstsr-blas-traits`
  BLAS-2 → P2 device dispatch → P3 re-measure.
- D2 All benchmark artefacts live in this repo (`2026-11-18-gemv-dispatch/`) or
  a local scratch place. **rstsr itself gets code only**: no `benches/`, no
  criterion wiring, no `#[ignore]` timing tests added by this campaign.
- D3 P0 surface: `%`/`rt::matmul` for both orientations (1-D and degenerate 2-D
  operands) + one `matmul_from` (`beta ≠ 0`, caller `out=`) cell + one batched
  cell.
- D4 P0 devices: `DeviceOpenBLAS` + `DeviceFaer`.
- D5 Harness: T0-style standalone criterion crate; anchors = raw `cblas_?gemv`
  (ceiling) + numpy; thread-control mechanism recorded and pinned.
- D6 Cells: 1024²/4096²/8192², `K = 65536` with output length 64 (both ways),
  one asymmetric cell — **plus** `256 × 452676` and its transpose (see Q1).
- D7 Success: match raw `cblas_?gemv` within ~10 % on the target shapes; no
  regression on the plain square-GEMM cells.
- D8 `rstsr-blas-traits`: `GEMVDriverAPI` + `GEMV_` container + composition into
  `BlasDriverAPI`; shared cblas template `driver_impl/cblas/blas2/gemv.rs`.
- D9 One shared layout-normalisation implementation in `rstsr-blas-traits`
  (not five copies), correct for c-prefer / f-prefer / strided operands, no
  explicit copy unless unavoidable, mirroring GEMM/TRSM discipline.
- D10 Fast path accepts positive vector strides directly (`incx`/`incy`).
- D11 gemv-specific parallelism: always parallel (details in Q6).
- D13 `DeviceGEMVAPI` left dead.
- D14 Tests in the openblas copy, gates as listed, handoff = `proposed.patch` +
  README numbers.

---

## New facts gathered for R2

**F1 — the five copies' drift.** `crates-device/{mkl,blis,aocl,kml}/src/matmul.rs`
are byte-identical to each other (blob `b77e2019`); the openblas copy has exactly
one extra test, `test_matmul_rule7_broadcast` (+27 lines, not `#[ignore]`d).
`matmul_impl.rs`: the four are byte-identical (`460335de`), the openblas copy
differs by one comment (`// the threading of openblas should be handled outside
this function.` vs `// the threading should be handled outside this function.`).
Nothing else has drifted.

**F2 — memory.** 61 GB RAM, 51 GB available; a `256 × 452676` f64 buffer is
0.93 GB, so both orientations plus outputs fit comfortably.

**F3 — the shared helper does not need a device instance to copy.**
`rstsr-core::prelude_dev` re-exports `rstsr_native_impl::prelude_dev::*`
(`rstsr-core/src/prelude_dev.rs:28`), so `rstsr-blas-traits` can use the
slice-level `assign_cpu_rayon` / `fill_cpu_rayon` / `uninitialized_vec` already
used by `matmul_impl.rs`. A helper generic over `B: GEMVDriverAPI<T>` is enough;
no `&DeviceBLAS` parameter has to be threaded through
`matmul_row_major_blas` / `matmul_uninit_row_major_blas` (five crates).

**F4 — size-1 axis removal needs no new API.** `Layout::dim_split_at(-1)`
(`rstsr-common/src/layout/indexer.rs:439`) splits a 2-D layout into
(first axis, last axis). For the m=1 reduction, `A(1,K)` → suffix `(K,)` is the
x vector and `C(1,N)` → suffix `(N,)` is y; for the n=1 reduction,
`b(K,1)` → prefix `(K,)` and `c(M,1)` → prefix `(M,)`. Both give `Layout<IxD>`
that `.into_dim::<Ix1>()` narrows.

**F5 — trans conjugation.** `FlagTrans::flip(hermi=false)` maps `N↔T`, `C→N`
(`rstsr-common/src/flags.rs:576`). A reversed c-prefer matrix therefore needs the
caller to conjugate the buffer to keep `C` semantics (exactly what
`flip_trans` does for GEMM).

---

## Q1 — What "256 × 452676 and its transpose" means

- (a) two physically contiguous matrices: `A(256, 452676)` and a separate
  `Aᵀ(452676, 256)` buffer;
- (b) (a) **plus** a third cell that uses a `.t()` view of the `(256, 452676)`
  buffer (strided matrix access, no physical transpose);
- (c) only the `.t()` view cell (one buffer).

➡️ (b). The two contiguous shapes give the DRAM-scale row-vector and
column-vector baselines; the `.t()` view is the only cell that exercises the
copy / strided-normalisation path that D9 demands we prove, and it is the worst
case we should report honestly rather than hide. 0.93 GB per buffer, 51 GB free
(F2); at ~50–60 GB/s a single call is ~16–20 ms, so criterion stays fast.

---

## Q2 — Is `GEMV_` order-free?

The output `y` is 1-D, so the device order has no observable meaning for it; the
matrix's preference alone decides whether the helper uses `trans` or `NoTrans`.

- (a) no `order` field in `GEMV_`; document the API as order-free;
- (b) keep `order: Option<FlagOrder>` for symmetry with `GEMM_`, resolved through
  `get_output_order` — knowing it can only affect a tie-break.

➡️ (a). GEMM needs `order` because it also fixes the 2-D output layout; a field
in `GEMV_` that cannot change behaviour is a trap for callers.

---

## Q3 — Who owns the matrix normalisation: the shared helper or the container?

- (a) **helper owns it**: it takes `a` as-is plus `transa ∈ {N, T, C}` and
  internally picks direct / reversed / conjugated-copy; the container stays thin
  (device + shape checks, `y` allocation and `ToBeCloned` write-back). The
  matmul path and the container share one code path.
- (b) **container owns it** via `flip_trans` (as GEMM does), the helper only
  accepts f-prefer, and the matmul path keeps its own matrix conversion — i.e.
  five copies of the logic.

➡️ (a), matching D9. Consequence to accept: the helper must handle `C` itself,
which for a c-prefer complex matrix means a conjugated copy (rare; only reachable
through the container, never from `matmul`).

---

## Q4 — Copy and stride policy, concretely (confirm)

Proposed rule table for the helper:

| operand situation | action |
|---|---|
| matrix f-prefer, legal `ld` | direct, `NoTrans` |
| matrix c-prefer, legal `ld` | reverse axes + flip `N↔T`, no copy |
| matrix neither prefer, or `ld` illegal (issue #161 style) | copy to f-contig, `NoTrans` |
| `C` requested on a c-prefer complex matrix | conjugated copy |
| vector stride > 0 | pass `inc = stride` |
| vector stride ≤ 0 (incl. broadcast stride 0) | copy to unit stride |
| vector `size ≤ 1` | `inc = 1` |
| `k == 0` | `y := beta * y` (fill path), A/x untouched |

➡️ Yes. This is the "no copy unless very necessary" reading of D9: only
non-contiguous matrices and non-positive-stride vectors pay.

---

## Q5 — Size-1 axis removal in the `(2,2,2)` reduction

- (a) reuse `Layout::dim_split_at(-1)` (F4) + an explicit `shape[axis] == 1`
  assert;
- (b) add a validated `Layout::dim_remove(axis)` in `rstsr-common` as the dual
  of `dim_insert`.

➡️ (a). Zero new public API, and the assert documents the precondition. (b) is a
reasonable future helper but not needed by this campaign.

---

## Q6 — Batched threading rule (concretising D11)

For **gemv shapes only**, and only in the BLAS device's broadcast path:

- `n_task == 1` → one call with BLAS threads = `nthreads` (BLAS does the
  parallelism; we do **not** hand-split a single gemv);
- `n_task > 1` → rayon over batches, BLAS threads = 1 per task, regardless of the
  existing `n_task >= 4 * nthreads` threshold;
- gemm shapes keep the current `4 * nthreads` rule untouched.

➡️ Yes. Note this deletes the `with_num_threads(nthreads, ...)` batch loop for
gemv only; for `n_task == 1` the call is identical to today's sequential branch,
so the surface is small.

---

## Q7 — Drift decision (your Q12)

Drift is exactly: openblas `matmul.rs` carries one extra test
(`test_matmul_rule7_broadcast`, 27 lines) and `matmul_impl.rs:42` has one
reworded comment; the other four crates are byte-identical to each other
(F1). Options:

- (a) leave the drift, patch all five copies in P2;
- (b) sync the extra test into all five (or remove it) first, then patch;
- (c) converge `matmul.rs` to one symlinked shared file now (the
  `driver_impl/` pattern), then patch once.

➡️ (a). The drift is cosmetic; (b) touches files for no functional gain; (c) is a
real refactor with its own verification burden across five crates and would
obscure the performance diff. Revisit (c) as a separate task if the copies keep
drifting.

---

## Q8 — Where "the sentence" gets updated

Your Q1 note: the claim *"we probably never dispatch to GEMV"* must be corrected
once the abstraction lands. Proposed targets:

- (i) this directory's `README.md` — restate the hypothesis as measured fact at
  P0 and as implemented behaviour at P3;
- (ii) a short comment on the new arms in the BLAS `matmul.rs`;
- (iii) `CHANGELOG.md` entry for the BLAS-2 addition;
- (iv) leave `rstsr-core/src/device_faer/matmul.rs:3` ("does not specialize
  gemv") **unchanged**, because it stays true for Faer in this campaign;
- (v) no change to `matmul` / `matmul_from` rustdoc (it documents semantics, not
  dispatch).

➡️ (i)–(v). Say if you want (iv) revised too (that would imply a Faer patch in
this campaign, which D4 deferred).

---

## Q9 — Where the `GEMV_` container gets tested

The container can only be instantiated for a device implementing
`GEMVDriverAPI`, and only the five device crates do. So
`cargo test -p rstsr-blas-traits` cannot exercise it. Options:

- (a) a new module `crates-device/rstsr-openblas/tests/blas2_gemv.rs`, wired into
  `tests/entry_row_openblas.rs`;
- (b) append to the existing openblas-only `tests/zero_size/blas3.rs` file;
- (c) a mock device inside `rstsr-blas-traits` tests.

➡️ (a). It is the only place that can cover strided / `C` / `beta ≠ 0` / zero-size
cases that the `matmul` path cannot express, and a dedicated file keeps the
existing zero-size suite focused.

---

## Q10 — Patch packaging for P1/P2

- (a) two patches: **patch-1** = `rstsr-blas-traits` BLAS-2 API + cblas template
  (+ container test), **patch-2** = the five `matmul.rs` dispatch sites + the
  dispatch tests;
- (b) one combined patch.

➡️ (a). Patch-1 is new public surface and compiles/tests without changing any
device behaviour, so it can be reviewed and integrated independently of the
performance change; patch-2 is then a small, reviewable delta per crate.
