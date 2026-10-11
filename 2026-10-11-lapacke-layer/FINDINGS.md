# Findings — comprehension base for the LAPACKE-translation grill (2026-10-11)

Agent-authored consolidation of four read-only comprehension reports
(rstsr inventory, LAPACKE reference, scipy layer, SVD cross-comparison).
Facts below are as-verified on this date; they feed the grill rounds
(GRILL-R1-QUESTIONS.md → R2+). Not yet decisions.

## 1. What rstsr has today (rstsr-blas-traits)

Layout of the layer (correcting the init prompt's assumption about paths):

- `src/{blas3,lapack_solve,lapack_eigh,lapack_svd}/` — dtype-generic builder
  structs (`GESVD_<'a,B,T>` + `derive_builder`, aliases `S/D/C/ZGESVD`) and
  per-routine `XxxDriverAPI<T>` traits (`unsafe fn driver_xxx(..) -> blas_int`).
  No `_f` suffix at this tier (everything already returns `Result`);
  `internal_run` + thin `run`.
- `driver_impl/{lapacke,lapack,cblas}/` — hand-written FFI impls per vendor,
  selected by **crate feature**: `lapack` (default, all 5 vendors) vs `lapacke`
  (openblas+mkl only). `duplicate_item(T; [f32];[f64];[Complex<f32>];[Complex<f64>])`
  in two blocks per file (real/complex split).
- `blas_int` = `i32`, or `i64` under feature `ilp64`; vendor ffi crates gate the
  integer type accordingly.
- Consumers: `rstsr-linalg-traits/src/ref_impl_blas.rs` free functions →
  vendor `blas_impl/*` symlinks → builder `.run()`, wrapped in
  `device.with_blas_num_threads(..)`.
- Tests: `blas_tests_impl/driver_impl/f64/` — **f64 only, 10 routines**;
  complex drivers exercised only indirectly via `linalg_func/func_c64`.

**The taxonomy is aspirational.** `f_prefer()`/`c_prefer()` exist but are only
used to pick an *order*; the generic layer then normalizes every input via
`overwritable_convert[_with_order]` → `into_contig_f(order)`/`to_contig_f(order)`
to a **contiguous** buffer, and `ld` is derived from that contiguous buffer
(`a.view().ld(order)`). No routine passes a strided buffer with a derived
`ld > rows/cols` today. "f-prefer/c-prefer with ld≠dim" pass-through is a
design decision to make, not current behavior.

Three coexisting order-selection idioms:
(a) `if b.f_prefer() && !b.c_prefer() {ColMajor} else {RowMajor}` (gesv, sysv,
syev(d), getrf, getri, potrf); (b) 4-arm match with `device().default_order()`
fallback (gesvd, gesdd); (c) `get_output_order(..)` (blas3: gemm/syhemm/trsm).
(a) and (b) disagree on both-prefer tensors (e.g. 1-column).

## 2. LAPACKE reference protocol (the authority to mirror)

- **Two-layer API**: `lapacke_xxx` (layout check → nancheck → workspace
  query/alloc/free → forward) over `lapacke_xxx_work` (the layout 3-way:
  COL direct / ROW transposed temps / bad-layout error). `matrix_layout` is
  arg 1 everywhere.
- **Row-major bridging**: row-major caller's `ld` = column count (checked
  `ld ≥ n`); the col-major temp uses `ld_t = MAX(1, rows)`. Only 2-D matrices
  transpose (`dge/dsy/dpo_trans`; sygv uses full `dge_trans`); `ipiv/w/s/tau`
  pass through. Output transpose conditional on job (e.g. syev `jobz=='V'`).
- **Workspace policies are per-routine, not uniform**:
  - pure query (`lwork=-1`): getri, dgeqrf, dsysv, dsyev, dsygv, dgesvd;
  - double query in one `_work` call (lwork+liwork): dsyevd, dsygvd;
  - gesdd: iwork by documented formula `MAX(1,8*MIN(m,n))` (allocated *before*
    the query — LAPACK exposes no iwork query), work queried;
  - gesvd: query (NOT a formula — verified over file history); complex adds
    `rwork = MAX(1,5*MIN(m,n))` by documented formula; `superb[i]=work[i+1]`
    (real) / `rwork[i]` (complex), i < MIN(m,n)−1;
  - gesvdq: triple query (work+iwork+rwork).
  - Query calls pass the *template* lds (`lda_t=MAX(1,m)`, …), i.e. the
    transposed ones.
- **Errors**: `info<0` from LAPACK → `info-1` (matrix_layout inserted as arg 1);
  `info>0` untouched; malloc failures −1010 (work) / −1011 (transpose) via
  `xerbla`; ld violations → negative param index. NaN check: full-pass over
  each input matrix, run-time toggle (env, default ON), compile-time removable.

## 3. scipy's layer (comparison point; deliberately different)

- **Never passes custom lds** — lds are always natural dims (`lda=m|1`,
  `ldu=u_shape0`, …; sole exception `ldb=max(1,m,n)` for gels*). Every input
  is copied into true F-order scratch (`copy_slice_F`, strides consumed only
  for reading). The one C-buffer-as-F reinterpretation is the **Cholesky uplo
  flip** (`uplo_f = lower ? 'U' : 'L'` for C-ordered data). No QR-via-LQ,
  no SVD-of-Aᵀ, no LU-of-transpose anywhere.
- **One-malloc arena**: query once → single `malloc` chopped into
  `work | data | U | Vh`, reused across the whole batch loop. No arena class.
- **`_calc_lwork` hardening**: `nextafter` bump for f32/c64 queries, optional
  fudge factor (1.01 on getri probes — OpenBLAS segfault workaround),
  overflow → sentinel. `lwork==0 → 1` clamp. Complex `?geev` needs rwork
  allocated *before* the query (segfaults with NULL).
- **Errors**: C layer never raises; per-slice status dicts → Python
  `LinAlgError`/`ValueError`/`LinAlgWarning`; `check_finite` is a pure-Python
  full-pass scan, default on.
- **Output assembly**: C-contiguous outputs; transposes fused into the output
  copy (`copy_slice_F_to_C`) or in-place blocked `swap_cf`; triangle-only
  copies for R/Cholesky.

## 4. SVD cross-comparison — where rstsr deviates (ranked, most severe first)

1. **The two rstsr builds disagree on error codes.** The manual `lapack/` path
   bypasses LAPACKE: no `info<0 → info-1` remap, no ld validation
   (`lda<n`→−7 etc.), no NaN check, no xerbla; transpose failures `.unwrap()`
   (panic) instead of error codes; alloc sentinels −1010/−1011 (LAPACKE's are
   −1010/−1011 too but the *info channel semantics* differ — same bug yields
   different codes under `--features lapacke` vs default).
2. **Zero-size short-circuit** (`minmn==0` in the generic layer) returns
   zero U/VT; LAPACK returns **identity sub-blocks**. Also means drivers are
   never exercised at m==0/n==0.
3. **Row-major query uses untransposed lds** (`m.max(n)`) instead of LAPACKE's
   template lds — benign on reference LAPACK (LWORK provably ld-independent
   in `SRC/dgesvd.f`) but latent for other backends.
4. **`u`/`vt` passed as user pointers where LAPACKE passes NULL** for
   jobu/jobvt ∉ {A,S}.
5. **No `lwork==0→1` clamp** or overflow/negative guard (scipy has all).
6. More mallocs than scipy (work, a_t, u_t, vt_t, iwork, rwork separately) —
   matches LAPACKE granularity, so not a regression vs LAPACKE.
7. gesdd 'N' complex `lrwork = 7mn` over-allocates vs documented `5mn` — safe.
8. Faithful already: gesvd superb indices (`work[i+1]`/`rwork[i]`), complex
   gesvd `rwork=5·minmn`, gesdd `iwork=8·minmn`, complex gesdd lrwork formula
   (all documented fixed minima — LAPACK has no query for them).
9. gesvd vs gesdd selection: none anywhere (LAPACKE: caller names it; scipy:
   `lapack_driver='gesdd'` default). rstsr: two builders, call site picks.

## 5. Other rstsr inconsistencies (non-SVD, from the inventory)

- **ipiv base convention**: getrf/getri/sysv decrement or not inconsistently
  (getrf −1, getri +1 to invert, sysv nothing) — silent 1-off pivot risk.
- **sygv/sygvd** map errors via `rstsr_assert_eq!(info,0,InvalidLayout)` while
  everyone else uses `rstsr_errcode!`.
- **`info as i32`** cast while the API is `blas_int` — truncation under ilp64.
- **gesvd returns a 4-tuple** (extra `superb`) vs gesdd's 3-tuple; higher
  layer discards superb.
- `.unwrap()` on `orderchange_out_*` results in raw drivers (panic channel).
- Mixed `overwritable_convert` vs `_with_order` within one routine (gesv/sysv):
  a 1-column `b` can produce a derived order ≠ b's chosen order → latent ld
  mismatch.
- No batching anywhere in the BLAS driver layer (2-D only; faer covers n-dim
  on its own device).

## 6. Decision points these findings open (R2 candidates, gated on R1)

1. Zero-copy generality: make f-prefer/c-prefer **actual** pass-through with
   derived ld (user's taxonomy), or keep normalize-to-contiguous? (gated on
   R1-Q4 fidelity principle)
2. c-prefer bridging mechanism per family: LAPACKE transposed-copy vs
   call-on-transpose tricks (SVD-of-Aᵀ + U/V swap, QR-via-LQ, LU-of-transpose)
   vs scipy's copy-to-F + uplo-flip-only. (gated on R1-Q4)
3. Workspace policy table for the guideline: per-routine query/formula table
   copied from LAPACKE §2 above + `lwork==0→1` clamp + template-lds queries +
   optional `_calc_lwork`-style hardening (f32 nextafter? fudge?) — which
   hardening is mandated vs forbidden?
4. Unify `driver_impl/lapack/` vs `driver_impl/lapacke/`: one canonical
   strategy (retire the other / keep both / make lapacke the reference shape)?
5. Error mapping: adopt LAPACKE's info-remap + −1010/−1011 semantics at the
   driver boundary, or a rstsr-native `Error` mapping (and where does the
   remap live)? Fix `as i32`, `.unwrap()`, assert_eq sites as part of it?
6. ipiv base: unify on 0-based (rust-facing) with a single conversion point?
7. Zero-size: identity sub-blocks (LAPACK-faithful) vs current zeros?
8. NaN-check: adopt at some layer (default-on toggleable, like LAPACKE /
   scipy's check_finite), or reject as out-of-scope for the driver tier?
9. superb: keep gesvd's 4-tuple or drop to 3-tuple (LAPACKE keeps superb as
   an out-param; higher layer discards it)?
10. Order-selection idiom: collapse (a)/(b)/(c) into one mandated rule?
11. Shared helpers the guideline should ship as reference shapes: layout
    classifier (f-prefer/c-prefer/non-contig + ld derivation), trans-in/out,
    workspace query helpers, error mapping — new module in `util/`?
12. Test additions: 3-case layout matrix per tensor input (f-prefer ld>dim,
    c-prefer ld>dim, fully non-contig), complex driver coverage (c64), and
    zero-size cases. (extends R1-Q6)

## 7. R3 fact-agent reports (2026-10-11)

### 7.1 Output order — numpy vs scipy (empirical + source)

- **numpy (2.5.1/2.5.2): input-order-INDEPENDENT, always C.** inv/pinv/
  cholesky/qr/svd/eigh/eig/lstsq → C outputs for C, F, and strided inputs
  alike; solve's b 2-D F-order → C. No `order=` logic exists in
  `numpy/linalg/_linalg.py`; gufunc allocation is KEEPORDER → C uniformly.
- **scipy (1.18.0): C by default; F ONLY via buffer reuse.** The batched C
  module allocates every output C-ordered (`PyArray_SimpleNew`), copies
  inputs F-ward (`copy_slice_F`) and results back (`copy_slice_F_to_C`).
  The only F outputs: `overwrite_*` on genuinely F-contiguous writable
  inputs, where the returned array IS the reused input buffer (solve b,
  inv, cholesky, qr R, lu U, lstsq x) — gated Python-side on F_CONTIGUOUS;
  and legacy f2py routines (solve_triangular, lu_factor, eigh/syevd) that
  are F-ordered unconditionally.
- Implication: the numpy-compatible rule for fresh outputs is "always
  device-default order"; F-preservation is an explicit in-place/overwrite
  contract, never implicit input-order inheritance.

### 7.2 LAPACKE NaN-check rationale (verdict)

- NOT caller-side C memory safety (no source claims it; the nancheck helpers
  themselves had UB bugs — commit 23b489104).
- Purpose = early diagnostics (clean error code instead of garbage; netnet
  design doc) + callee-side robustness: NaN pivot in `?sptrf/?hptrf` reads
  uninitialized offsets and writes past `ipiv` while `INFO=0` (LAPACK commit
  ac6db4873); scipy `check_finite` docstring: "crashes, non-termination" if
  disabled; convergence loops are iteration-bounded (dbdsqr MAXITR*N²,
  dstein MAXITS=5, dstebz bisection) → garbage results, not hangs;
  dgesdd/dgeev self-check NaN (`INFO=-4` via DISNAN).
- For a rust caller of the same LAPACK, the callee hazards (silent OOB,
  crash, garbage) remain; only a pre-call finiteness pass guards them.
  arXiv 2207.09281 §2.4 records the numerical rationale (Inf scaling →
  all-zero matrix, defeating the algorithm).
