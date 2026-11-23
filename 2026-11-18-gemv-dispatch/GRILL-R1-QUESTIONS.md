# Grill R1 — BLAS-2 GEMV dispatch for matmul (`openblas` / `faer`)

Session 2026-11-18. rstsr `d5c9b86` (`261023/astype`, clean tree); this repo
`d97449e`, clean. **No rstsr file has been edited for this campaign.**

Recommended answers are marked ➡️; "yes" accepts the recommendation. Facts in
§1–§6 were verified in-repo today (file:line given); decisions are yours.

---

## Factual baseline (agent-authored, 2026-11-18)

### 1. Where matmul dispatches today

Call chain for `a % b` / `rt::matmul`: `op_refa_refb_matmul`
(`rstsr-core/src/tensor/linalg/matmul.rs:279`) allocates uninit storage and calls
`DeviceMatMulAPI::matmul_uninit` (:304) → device `matmul_uninit_row_major_*`.
`matmul_from` / `matmul_with_output` / `matmul` (beta path) share the same
dispatch with the initialized output.

**BLAS device** (`crates-device/rstsr-openblas/src/matmul.rs`; the mkl / blis /
aocl / kml `matmul.rs` copies are byte-identical, the openblas copy has one extra (non-ignored) test, and its
`matmul_impl.rs` differs only by one comment):

- `matmul_row_major_blas` matches `(la.ndim(), lb.ndim(), lc.ndim())` at
  `:118` with exactly two fast arms: `(1,1,0)` → naive rayon inner dot
  (`:119–126`) and `(2,2,2)` → `gemm_blas_ix2_no_conj_dispatch` (`:127–135`).
- Everything else — including `(1,2,1)` (`x(K) @ B(K,N) → c(N)`) and `(2,1,1)`
  (`A(M,K) @ x(K) → c(M)`) — reaches the broadcast config at `:140`.
  `layout_matmul_dyn_row_major_with_lc` resolves those rules into 2-D layouts by
  inserting a size-1 axis (`dim_insert`; `rstsr-common/src/layout/matmul.rs:341`
  and `:343` for rule 3, `:360` and `:361` for rule 4) and marks them
  `MatMulType::GEVM` / `MatMulType::GEMV` — a tag nothing downstream reads.
- So today both vector orientations run as **cblas gemm with one 1-sized
  dimension**: the row-vector orientation as `m = 1` (row-major c-prefer
  operands, `la` single-row so both orders prefer, `lda` becomes the inner
  stride) and the column-vector orientation as `n = 1`
  (`matmul_impl.rs:93–120` picks `NoTrans`/`Trans` and the leading dimensions;
  the degenerate cases do *not* trigger the `lc` flip at `:45–64`, because a
  size-1 output axis reports both orders as preferred).
- `cblas_?gemv` is called nowhere in the workspace (only the FFI bindings
  declare it).
- `matmul_uninit_row_major_blas` (`:218–281`) special-cases only `(1,1,0)` with
  the write-only dot kernel (`:246–251`) and otherwise delegates into
  `matmul_row_major_blas` with `beta = 0`, so one implementation of the new
  arms serves both entry points.

**Faer device** (`rstsr-core/src/device_faer/matmul.rs`) has the same two arms
(`:121–138`), the same broadcast fallback (`:141–146`), and its module doc says
outright: *"This implementation does not specialize gemv. We always use gemm for
matmul."* (`:3`). `gemm_faer` hands the raw strides to
`MatRef::from_raw_parts` and calls `faer::linalg::matmul::matmul`
(`device_faer/matmul_impl.rs:45–84`). faer 0.24.4 internally dispatches
`dst.ncols()==1` / `dst.nrows()==1` to SIMD `matvec_*` kernels and `k==1` to
`rank_update` (`faer-0.24.4/src/linalg/matmul/mod.rs:1214, 1247, 1283`), but only
when the layout preconditions hold (`try_as_col_major` ⇔ row stride 1,
`try_as_row_major` ⇔ col stride 1). **Whether rstsr's layouts reach those
kernels is exactly what P0 must measure** — the answer is not obvious.

**Native fallbacks** (`rstsr-native-impl/src/cpu_rayon/matmul_naive.rs:131`,
serial `:277`): same two arms; matrix–vector is a 1-sized-dim gemm. The rayon
path has no gemv kernel at all; the serial `gemv_naive_cpu_serial` /
`gevm_naive_cpu_serial` (`:128`, `:169`; uninit twins `:367`, `:404`) have no
callers.

### 2. `rstsr-blas-traits` today, and the cost of a BLAS-2 addition

- BLAS-3 only: `src/blas3/{gemm,syhemm,trsm}.rs`; the template is
  `XDriverAPI<T>` (gemm.rs `:4–21`) + `#[derive(Builder)] X_` (`:23–45`) +
  `run`/`internal_run` (`:52`/`:104`) + aliases (`:179–183`), registered in
  `lib.rs:19` and `prelude.rs:5`, composed into `BlasDriverAPI<T>`
  (`trait_def.rs:43–48`).
- Driver impls are a **template tree**, not a crate module:
  `rstsr-blas-traits/driver_impl/cblas/{mod.rs, blas3/…}`. Every device crate
  symlinks the whole directory (`crates-device/<dev>/src/driver_impl/cblas →
  ../../../../rstsr-blas-traits/driver_impl/cblas/`), so adding
  `driver_impl/cblas/blas2/gemv.rs` + `pub mod blas2;` in
  `driver_impl/cblas/mod.rs` reaches all five devices with **no symlink change**.
  Each device also has an empty `impl BlasDriverAPI<T> for DeviceBLAS {}`
  (`crates-device/<dev>/src/driver_impl/mod.rs:15–19`).
- All five FFI crates already declare `cblas_{s,d,c,z}gemv` and expose it as
  `lapack_ffi::cblas::*` (openblas `ffi_extern.rs:287/301/315/329`; mkl `:380`;
  aocl `blis/ffi_extern.rs:56998`; blis `blis_x86_64/ffi_extern.rs:23963`; kml
  `:756`), so a single template file covers all five.
- Gaps a gemv hits: `ld_row` / `ld_col` / `testify_ld` are
  `impl TensorBase<S, Ix2>` only (`tensor2_impl.rs:29–95`); `flip_trans` is
  hard-wired to `Ix2` (`util_tensor.rs:61–66`); nothing in the crate knows
  `incx`/`incy`.
- Container consumers today: `GEMM` only at
  `rstsr-linalg-traits/src/ref_impl_blas.rs:134` (pinv), `TRSM` at `:217`,
  `SYHEMM` at no production site. A `GEMV_` container would therefore have **no
  in-tree consumer other than tests** — worth weighing in Q8.

### 3. Dead scaffolding that already exists

- `DeviceGEMVAPI<TA,TB,TC>` with `gemv` + `gevm`
  (`rstsr-core/src/operators/matmul.rs:116–146`): zero implementors, zero call
  sites.
- `MatMulType::{GEVM,GEMV}` (`rstsr-common/src/layout/matmul.rs:33–42`):
  assigned, never read outside that file.
- Serial naive `gemv`/`gevm` kernels: no callers (§1).

### 4. Indicative probe — **not** the campaign measurement

A one-off C probe against the system `libopenblas.so.0` (`OpenBLAS 0.3.32
DYNAMIC_ARCH NO_AFFINITY USE_OPENMP`, `openblas_get_corename()` = `Cooperlake`
on this Zen 5 box — itself a caveat), f64, col-major `A`, `2·P·Q` flops,
best-of-repeats. Two idioms, both mathematically the shapes at stake:

- **row-vector idiom** — `y(Q) = A(P,Q)^T x(P)`: `gemv(NoTrans)` vs
  `gemm(m=1, n=Q, k=P)`;
- **column-vector idiom** — `y(Q) = A(P,Q) x(P)` … `gemv(Trans)` vs
  `gemm(m=Q, n=1, k=P, transa=Trans)`.

| P×Q | idiom | gemv 1T ms | gemm 1T ms | ratio | gemv 16T ms | gemm 16T ms | ratio |
|---|---|---|---|---|---|---|---|
| 1024²   | row (m=1) | 0.068 | 0.155 | 2.28× | 0.026 | 0.046 | 1.77× |
| 1024²   | col (n=1) | 0.065 | 0.254 | 3.90× | 0.007 | 0.075 | 10.50× |
| 2048²   | row (m=1) | 0.394 | 1.115 | 2.83× | 0.123 | 0.075 | 0.61× |
| 2048²   | col (n=1) | 0.413 | 1.793 | 4.34× | 0.028 | 0.084 | 2.97× |
| 4096²   | row (m=1) | 2.456 | 5.162 | 2.10× | 1.084 | 1.202 | 1.11× |
| 4096²   | col (n=1) | 2.636 | 9.096 | 3.45× | 0.743 | 0.968 | 1.30× |
| 8192²   | row (m=1) | 11.489 | 20.744 | 1.81× | 8.292 | 9.584 | 1.16× |
| 8192²   | col (n=1) | 11.301 | 36.812 | 3.26× | 8.080 | 8.333 | 1.03× |
| 65536×64 | row (m=1) | 0.436 | 1.085 | 2.49× | 0.059 | 0.061 | 1.04× |
| 65536×64 | col (n=1) | 0.434 | 2.007 | 4.62× | 0.039 | 0.385 | 10.00× |

Reading: single-threaded, `gemv` wins every cell by 1.8–4.6×. Multithreaded, the
**`m=1` idiom is the pathological one** (10× at 1024² and at 65536×64), while
square DRAM-bound cells close to 1.0–1.3×; one cell (2048² row, 0.61×) is
anomalous and must be re-measured. Thread caveat: the distro OpenBLAS is
OpenMP-built, so `OPENBLAS_NUM_THREADS` is ignored — threads must be set through
`OMP_NUM_THREADS`/`omp_set_num_threads` (which `with_num_threads`
in `crates-device/rstsr-openblas/src/threading.rs` already does). This probe
does **not** include rstsr's layout normalisation, copies or broadcast path, so
rstsr's real gap can be larger.

### 5. Benchmark infrastructure

In-repo: none. `criterion 0.8` is declared in the workspace and as a
`rstsr-core` dev-dependency but unused; no `benches/`, no `[[bench]]`; timing
exists only as `#[ignore]` tests (e.g. `syrk_efficiency`,
`crates-device/rstsr-openblas/src/matmul.rs:501`). The established harness is
this repo's `2026-09-09-bench-harness-baseline/`: standalone non-workspace crate,
`src/lib.rs` devices + thread assertions, `benches/*.rs` under criterion,
`examples/correctness.rs` gate, `numpy_ref/`, `results/`, `reproduce.sh`.
Only `rstsr-openblas` is demonstrably buildable here (no `target/` artefacts for
the other four; MKL/AOCL need `RSTSR_DEV=1` plus library paths; KML is ARM-only).

### 6. What the answer changes downstream

- Which of the five `matmul.rs` copies get the code (Q12) decides the patch size.
- Where the normalisation lives (Q9) decides whether the device crates depend on
  `rstsr-blas-traits` for the hot path.
- The layout condition (Q10) decides whether `matmul_from` with a strided `out=`
  benefits.
- Q8 decides whether this is one commit or three, and whether it is semver-minor
  or breaking.

---

## Q1 — Campaign shape and the go/no-go gate

The trigger is a hypothesis, not a measurement: "we probably never dispatch to
GEMV, and OpenBLAS gemm is not efficient on some `m×k, k×1` matmul". Options:

- (a) **P0 measurement only** (no rstsr edits) → you review the table → P1
  (`rstsr-blas-traits` BLAS-2) → P2 (device dispatch) → P3 (re-measure + report).
- (b) measure and patch in one wave, numbers as evidence in the same directory.
- (c) measurement only; no patch in this campaign.

➡️ (a). It is the literal request ("check on the behavior"), it matches this
repo's charter (numbers before patch, human integrates), and it keeps P1/P2
design adjustable by the P0 numbers.

---

## Q2 — Task directory and artefacts

Proposed `rstsr-improve-trajectory/2026-11-18-gemv-dispatch/` containing:
`GRILL-INIT-PROMPT.md`, `GRILL-R<N>-{QUESTIONS,ANSWERS}.md`, then P0's
standalone crate (`Cargo.toml`, `benches/`, `examples/correctness.rs`,
`numpy_ref/`, `results/`, `reproduce.sh`), `README.md` (commit pin, environment,
baseline table, post-change table, conclusion), `TASKS.md`, and for P1/P2 a
`proposed.patch` + porting notes. Alternative slugs: `2026-11-18-blas2-gemv`,
`2026-11-18-matmul-gemv`.

➡️ Yes — `2026-11-18-gemv-dispatch`, with `README.md` carrying the conclusion and
`proposed.patch` (not a branch in rstsr) as the handoff artefact.

---

## Q3 — What P0 measures: which call surface

- (a) only `%` / `rt::matmul` (i.e. `matmul_uninit`, `beta = 0`, fresh output)
  for both orientations, 1-D operands and degenerate 2-D operands;
- (b) (a) + one `matmul_from` cell with `beta ≠ 0` and a caller-provided `out=`
  + one batched cell (e.g. `[64, K] @ [64, K, N]`-style rule 6/7, and a
  `[B, M, K] @ [K]` rule 4);
- (c) only the degenerate 2-D forms, since those are what the trigger names.

➡️ (b). The new code path is shared by `matmul` and `matmul_uninit`, so the beta
path is one cheap cell; the batched cell is the only evidence we would have for
the threading question (Q11) without guessing.

---

## Q4 — Devices measured in P0

Note on wording: there is no standalone rayon matmul device — `DeviceMatMulAPI`
is implemented only by `DeviceFaer`, `DeviceCpuSerial`, and the five BLAS
devices (`grep impl.*DeviceMatMulAPI`). "Rayon parallel" in the prompt therefore
means (i) `DeviceFaer` (the rayon device; the `DeviceCpu` alias is `DeviceFaer`
when the `faer` feature is on) and (ii) the BLAS device's rayon-driven batched
outer loop.

- (a) `DeviceOpenBLAS` + `DeviceFaer` only;
- (b) (a) + `DeviceCpuSerial` spot cells as the "no BLAS" reference;
- (c) (a) + MKL/AOCL/BLIS spot cells.

➡️ (a). OpenBLAS is the trigger; Faer is the control that tells us whether faer's
internal matvec dispatch already covers us (a real possibility given §1). Skip
`DeviceCpuSerial` (it is orders of magnitude slower and not the owner's use
case) and MKL/AOCL/BLIS (they share the same `cblas` template after P1 and
cannot be built locally without extra environment work).

---

## Q5 — Harness and anchors

Recommendation: a standalone criterion crate modelled on the T0 harness —
release profile, `RAYON_NUM_THREADS=16`, thread assertions, one benchmark file
for matmul shapes, `examples/correctness.rs` as a pre-perf gate, `results/` raw
output. Anchors: raw `cblas_dgemv` (the achievable ceiling for the same linked
library) and numpy 2.5 on the same cells (cross-language sanity), plus a
`openblas_get_config()` / `get_corename()` record and explicit thread control.

➡️ Yes. Add one thing to the T0 pattern: record and pin the *thread-control
mechanism* per call (`omp_set_num_threads` vs `openblas_set_num_threads`), since
the distro library is OpenMP-built and the naive env-var is ignored (§4).

---

## Q6 — Shape cells

Proposed grid, both orientations, f64, one f32 spot:

- `1024²`, `4096²`, `8192²` square (cache-resident → DRAM-bound);
- `K = 65536`, output length 64 (`64×65536` and `65536×64`);
- `M/K/N = 64/1024/1024` asymmetric pair to overlap the cache and DRAM regimes.

➡️ Yes, with the `65536×64` class kept mandatory: it is the cell where the
multithreaded `m=1` gemm gap is largest (10×) and where a regime switch at
~`4·nthreads` could bite.

---

## Q7 — Success criterion (gates P2, informs P3)

- (a) no regression on the plain square-GEMM cells, and a measurable win
  wherever the probe shows ≥1.5× headroom;
- (b) match raw `cblas_?gemv` within ~10 % on the four target shapes, and no
  regression elsewhere;
- (c) ship the dispatch, report whatever it gives.

➡️ (b) as the target and (a) as the hard gate. Without a number to hit, "we now
call gemv" is an implementation claim, not an efficiency result.

---

## Q8 — `rstsr-blas-traits` BLAS-2 API surface

- (a) driver trait only (`GEMVDriverAPI::driver_gemv` + the shared cblas
  template);
- (b) (a) + a public `GEMV_` builder container mirroring `GEMM_`
  (`alpha`/`beta`/`transa`/`order`, `SGEMV`/`DGEMV`/`CGEMV`/`ZGEMV` aliases,
  prelude export);
- (c) (b) + add `GEMVDriverAPI<T>` to `BlasDriverAPI<T>`
  (`src/trait_def.rs:44`).

Tradeoffs: (b)'s container would have no in-tree consumer beyond tests (§2);
(c) is a breaking change for any external implementor of `BlasDriverAPI`
(rstsr-blas-traits is published at 0.10.3, so a minor bump covers it) but makes
BLAS-2 structurally identical to BLAS-3, and all five device crates get the impl
for free from the symlinked template.

➡️ (c) — all three, in one commit series: driver, container, composition. This is
what "similar work on rstsr-blas-traits" means in BLAS-3 terms, and the only
`BlasDriverAPI` implementors in existence are the five in-tree devices.

---

## Q9 — Where the layout normalisation for the fast path lives

- (a) one shared generic function in `rstsr-blas-traits/src/blas2/gemv.rs`
  (slices + layouts + `B: GEMVDriverAPI<T>`) that the five device `matmul.rs`
  files call;
- (b) a local `gemv_blas_no_conj_*` in each device's `matmul_impl.rs`, mirroring
  the existing `gemm_blas_no_conj_*` / `syrk_blas_no_conj_*` (five copies,
  direct cblas);
- (c) raw `B::driver_gemv` from `matmul.rs` with the normalisation duplicated
  there (five copies).

➡️ (a). It is the only option that keeps the "matrix needs a temp copy",
"`x`/`y` increment", and "leading dimension" rules in one reviewed place; (b)
would add a second cblas call site per type and copy the trickiest logic five
times, which is exactly the drift the symlinked `driver_impl/` tree exists to
avoid.

---

## Q10 — Which layouts take the fast path

- (a) fast path only when `x` and `y` are unit-stride (and the matrix has a legal
  leading dimension); every other layout keeps today's gemm path, which is
  correct for arbitrary layouts;
- (b) pass `incx`/`incy` for positive strides, copy when the stride is ≤ 0;
- (c) always copy non-unit-stride vectors into contiguous temporaries.

➡️ (a) for the first cut, with (b) as an explicit follow-up knob. The measured
target shapes are all contiguous, and (a) keeps the change verifiable: the fast
path either fires or it does not, with no new buffer behaviour inside the
matmul hot loop. Note (a) means a strided `out=` in `matmul_from` does not
benefit yet — acceptable, and measurable later.

---

## Q11 — Batched shapes and thread policy

- (a) keep the existing BLAS-device rule (`n_task >= 4 * nthreads` → rayon over
  batches with one BLAS thread each, else sequential batches with
  `nthreads` BLAS threads), and simply call gemv per batch;
- (b) add a gemv-specific bias (e.g. prefer rayon-over-batch much earlier,
  because OpenBLAS gemv threading is weak for small matrices).

➡️ (a) in the first cut, decide from the P0 batched cell. A heuristic change and
a new kernel in the same patch would make a regression impossible to attribute.

---

## Q12 — Propagation to the five device copies

- (a) patch all five `crates-device/*/src/matmul.rs` (status quo; the files
  already drift, §1);
- (b) also collapse `matmul.rs` into one symlinked shared file as part of this
  work.

➡️ (a). (b) is a hygiene refactor with its own verification cost (five crates
compile the same file against five FFI aliases) and would obscure the performance
diff; note it as a separate task.

---

## Q13 — The dead `DeviceGEMVAPI` in `rstsr-core`

- (a) leave it dead and out of scope;
- (b) make it the seam: implement it per device and have `DeviceMatMulAPI`
  dispatch the degenerate cases through it;
- (c) delete it as part of this work.

➡️ (a). `DeviceMatMulAPI` already covers every shape; routing through a second
trait would force extra bounds on the matmul impl for no measured gain. Deleting
it is defensible but unrelated — keep the campaign diff about the matmul path.

---

## Q14 — Verification gates and the handoff artefact

Proposed:

- Correctness: parity already exists (device crates symlink
  `rstsr-core/tests/core_func`, where `%` covers rules 3–7 and the col-major
  contract). Add targeted tests in the **openblas** copy's in-crate test module
  only: degenerate 2-D forms (`1×K @ K×N`, `M×K @ K×1`, and the both-1 corner),
  a transposed/strided operand case that must stay correct, a `beta ≠ 0`
  `matmul_from`, a batched rule-3/rule-4 case, and one `ColMajor` default-order
  case (the reverse-and-swap path turns GEMV into GEVM).
- Gates: `cargo test -p rstsr-openblas --test entry_row_openblas`,
  `cargo test -p rstsr-core --test entry_row_cpu --no-default-features
  --features "backtrace row_major"`, `cargo test -p rstsr-blas-traits`
  (new container's doctest/unit test), `cargo check` for the other four device
  crates (same template, not linked locally), plus clippy on the touched crates.
- Handoff: `proposed.patch` + README numbers; no branch push, no PR without
  explicit permission (charter).

➡️ Yes. If you prefer the patch to land on a rstsr branch instead of an
uncommitted `proposed.patch`, say so in the answers — the default here is
`proposed.patch`.
