---
name: rstsr-gemv-dispatch-grill
description: "BLAS-2 GEMV dispatch for matmul (grill 2026-11-18..23): current dispatch map, probe numbers, and the decisions that bind the P0/P1/P2 campaign"
metadata:
  type: project
---

Grill on how rstsr's matmul handles vector–matrix products on `DeviceOpenBLAS`
and `DeviceFaer`, and on adding BLAS-2 `gemv` to `rstsr-blas-traits`
(rstsr `d5c9b86`, branch `261023/astype`). Task dir:
`2026-11-18-gemv-dispatch/` (`GRILL-INIT-PROMPT.md`, `GRILL-R1..R3-QUESTIONS.md`).
Answers are quoted in the R2/R3 files because the user answered in chat (no
`GRILL-R*-ANSWERS.md` exists yet).

**Current dispatch (verified).** BLAS device and Faer device both fast-path only
`(1,1,0)` (naive inner dot) and `(2,2,2)` (gemm) in
`matmul_row_major_blas` / `matmul_row_major_faer`; `(1,2,1)` and `(2,1,1)` reach
the broadcast config, whose `dim_insert` turns them into 2-D with a size-1 axis
(`rstsr-common/src/layout/matmul.rs:341/343/360/361`), i.e. **cblas gemm with
m=1 or n=1**. `cblas_?gemv` is called nowhere. `a % b` goes through
`matmul_uninit` (beta = 0), not `matmul`. Faer internally dispatches
`nrows==1`/`ncols==1` to SIMD matvec kernels
(`faer-0.24.4/src/linalg/matmul/mod.rs:1214/1247`) — whether rstsr's layouts
reach them is what P0 measures. Dead scaffolding: `DeviceGEMVAPI`
(`rstsr-core/src/operators/matmul.rs:116`), `MatMulType::GEMV/GEVM`, serial
`gemv_naive_*`. There is **no standalone rayon matmul device** (`DeviceFaer` is
the rayon path; `DeviceCpu` aliases it when `faer` is on).

**Probe (C, system OpenBLAS 0.3.32 `DYNAMIC_ARCH USE_OPENMP`,
`corename=Cooperlake` on Zen 5, f64).** Single thread: gemv beats gemm-with-a-1
by 1.8–4.6× in every cell (1024²…8192², 65536×64). At 16 threads the **m=1
gemm** is pathological (10× at 1024² and at 65536×64), DRAM-bound 8192² closes to
1.0–1.3×. Gotchas: the OpenMP-built lib ignores `OPENBLAS_NUM_THREADS` (use
`OMP_NUM_THREADS`/`omp_set_num_threads`); `cblas_dgemv` negative increments use
the F77 convention (pointer = array start, first element at the far end), so
rstsr layouts cannot forward a negative stride without a pointer bump.

**Decisions (binding).** P0 measurement first (benchmarks live only in this
repo — rstsr gets code only, no benches); `GEMV_` + `GEMVDriverAPI` +
`BlasDriverAPI` supertrait + shared `driver_impl/cblas/blas2/gemv.rs`;
`GEMV_` is order-free and normalises the matrix with
`flip_trans(ColMajor, …, hermi=false)` (no copy for c/f-prefer, conjugate copy
only for `C` on c-prefer complex); vectors are never flipped, strides are passed;
the device path keeps its own matrix conversion and **copies** non-unit-stride
vectors (D21/D28: copy, never guard); the `(2,2,2)` size-1 reduction uses
`Layout::dim_eliminate` (validated, returns `Ix1`); the gemv wrapper is a local
`gemv_blas_no_conj_*` in each device `matmul_impl.rs` (five copies, D20 /
R3-Q1 = a) routed through `B::driver_gemv`, not a shared helper in
`rstsr-blas-traits` (that was the rejected option c); batch threading keys off
the inner shape — vector-shaped inner products always iterate in the main
thread with multi-threaded BLAS (D29, scoped 2026-11-23), real matrix products
keep `n_task >= 4 * nthreads` rayon-outer with `with_num_threads(1)` — and this
lives in the device matmul code, not in `GEMV_` (D18); all five device
`matmul.rs` copies are patched (drift today = one extra openblas test + one
comment); `DeviceGEMVAPI` stays dead; two patches (blas-traits API first, then
the dispatch); no CHANGELOG edit until a version bump.

**R4/R5.** Vector policy final: copy any non-contiguous vector, no copy when
contiguous; `driver_gemv` has no `incx`/`incy` (minimal surface; scipy's
`dgemv(alpha, a, x, beta, y, offx, incx, offy, incy, trans, overwrite_y)`
wrapper pays for strided views at the numpy conversion boundary, so rstsr copying
at its own boundary is the same trade). R5 asks: Faer scope after P0 (rec:
BLAS-only now, Faer follow-up), the batched-cell anomaly (rec: fold into P2's
acceptance), and the patch-1 go-ahead.

**Patch-1 (2026-11-23, branch `261123/blas2-gemv`, commits `3c3999d` +
`718864a`).** `rstsr-blas-traits` gained `src/blas2/gemv.rs` (`GEMVDriverAPI`
with **no** `incx`/`incy`, order-free `GEMV_` using `flip_trans(ColMajor, …,
false)`, ToBeCloned write-back for a non-unit-stride `y`, `beta*y` scaling on an
empty contraction for every `beta != 1`), `driver_impl/cblas/blas2/gemv.rs`
(4-type `duplicate_item`), `BlasDriverAPI` composition, and 10 container tests in
`blas_tests_impl/driver_impl/blas2_gemv.rs` — the **shared** device body, run by
`tests_driver_impl` in all five device crates (not an OpenBLAS-local
`tests/blas2_gemv.rs`, which Cargo would also have compiled twice if the entry
imported it). The only extra bound over `BlasDriverBaseAPI` is
`OpAssignArbitaryAPI<T, Ix1, Ix1>`. Review pass before commit fixed two real
defects: `beta == 0` on an empty contraction left the caller `y` unscaled (while
`GEMM_`/dispatched `matmul` give zeros), and five tests used `%` as oracle —
which patch-2 reroutes through `GEMV_` itself, making them tautological; the
references are now elementwise. Diff captured as `proposed-patch-1.diff`.

**P0 (done, 2026-11-23, rstsr `062c970` v0.10.4).** Harness + numbers in
`2026-11-18-gemv-dispatch/` (criterion, 16 threads, system OpenBLAS 0.3.32
`core=Cooperlake`). OpenBLAS device slower than raw `cblas_dgemv` on every f64
cell: 1.04–11.8× (11.8× at 1024² matvec; **55.0 → 15.9 ms, 39 ms wasted**, at
the 0.93 GB `256×452676 vecmat`; f32 1024² matvec 14.6×). Faer is mixed: DRAM
cells 0.73–1.00× of the cblas anchor (its internal matvec is reached) and
`65536×64 vecmat` is 6.3× faster than cblas gemv, but the transposed orientation
still loses 1.4–2.4× at 1024²/4096²/65536×64 and 1.69× (11 ms) at
`256×452676 vecmat`. Clean pass `run4`: openblas 1024² matvec 8.96×, `65536×64 vecmat` 2.42×,
`256×452676 vecmat` 55.5 vs 16.0 ms (39.5 ms/call). The batched rule-6 cell
`(64,1024,1024) @ x` is 11.0 / 9.1 ms (20–40× off ideal, the branch D29
rewrites); the batched rule-7 GEMM datum is 20.4 ms — D29's universal policy
must not exceed it. Small cells move ±10–20 % between passes; the anchors moved
up to 133 % in a pass that overlapped a compile, so P3 needs a quiet machine.

**Record lesson (2026-11-23).** Two decisions were reconstructed wrongly from
the round files: D9's "one shared implementation" was later narrowed by D20
(the *driver* is shared, the normalisation is five copies), and D11's
"rayon-over-batch" wording is superseded by D29 + its 2026-11-23 scope note.
Read the **later** round and the D-item, not the earliest option list.

**Build gotchas recorded:** the sandbox makes `~/.cargo` read-only, so builds
need a writable `CARGO_HOME` (this session used
`/home/a/rstsr_pack/tmp/cargo`, a copy of the registry + `config.toml`), and
`RSTSR_DEV=1` is required for `rstsr-openblas` to link the system libopenblas
and `gomp`. `rstsr_core::prelude::*` does **not** re-export the free functions;
use `rstsr_core::prelude::rt::*`. Cargo does **not** notice edits to a shared
`blas_tests_impl` body through the per-crate symlink (the symlink's mtime is
unchanged), so incremental `clippy`/`test` can silently skip the rebuild — run
`cargo clean -p <device>` (or touch the symlink itself) after editing one.
