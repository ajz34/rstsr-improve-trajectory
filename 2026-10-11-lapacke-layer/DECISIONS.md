# DECISIONS — LAPACKE translation layer grill (closed 2026-10-11)

Grill rounds R1–R5 (questions in GRILL-R<N>-QUESTIONS.md of this directory;
user answers given on screen and transcribed by the agent into each next
round's questions file). Fact base: FINDINGS.md (four comprehension agents +
two fact agents). Status: **closed — shared understanding reached**.
Execution awaits the user's explicit go.

1. **Deliverables**: skill `lapacke-translation` (rstsr-agents, new branch)
   + ADR-0009 (rstsr-book, new branch) + code retrofit (rstsr, new branch
   `261011/…`). Skill = recipe-with-rationale + mandatory gap-notice rule
   ("recipe lacks coverage → stop and report; do not improvise silently");
   ADR = the why. New branch in EACH repo touched.

2. **Input policy**: f-prefer view inputs pass zero-copy into LAPACK with
   derived `ld`; all non-f-prefer inputs (c-contig / c-prefer / fully
   non-contiguous) are cloned; routines that overwrite the input clone or
   consume regardless of contiguity. Fully-non-contiguous support is our
   extension beyond LAPACKE's two-layout world (LAPACKE accepts only
   c-prefer/f-prefer).

3. **Output policy (hybrid)**: fresh outputs in the device default order
   (numpy parity — numpy is input-order-independent, always C; scipy C by
   default); in-place/overwritable outputs reuse the input buffer and its
   order (scipy `overwrite_*` semantics — the f-prefer-mutable-`b` solve
   case keeps F order, zero bridge copies); f-prefer views exploited
   maximally everywhere.

4. **Bridge copies**: per-device orderchange infra with ambient threading
   replaces the `_cpu_serial` kernels currently hardcoded in the driver
   files (a NEW deviation vs existing translation code, matching elementwise
   behavior). Full-matrix copies; no `dsy_trans`/`dtr_nancheck`-style
   triangle-only variants this campaign (footprint O(N²) ≪ O(N³) compute;
   uniform kernel inventory for subagent determinism) — recorded as a
   sanctioned future optimization. Open flag (user): multi-threaded
   transpose policy is the most severe open problem — ignored for this
   campaign; agent may re-ask.

5. **Vendor file shape**: trait `driver_xxx` signatures UNCHANGED this
   campaign. Per-file free `pub fn xxx_work` (public for symlink
   reachability, internal-purpose, not prelude-exported) = pure FFI thunk
   with `lwork = -1` query semantics; query call and compute call textually
   separate; workspace ownership + clamps live at the builder layer.

6. **Workspace policy**: per-routine table, LAPACKE-faithful — query
   (`lwork=-1`) where LAPACKE queries (gesvd, getri, dgeqrf, dsysv, syev,
   sygv; double query for syevd/sygvd), documented fixed formulas where
   LAPACKE has no query (gesdd `iwork = MAX(1, 8·min(m,n))` allocated before
   the query; complex gesvd `rwork = MAX(1, 5·min(m,n))`; complex gesdd
   lrwork formula). Hardening (all O(1) budget, accepted): query with the
   real col-major lds; `MAX(1,·)` on formulas; `lwork==0 → 1` clamp; checked
   int conversion of query values; `nextafter` guard for f32/c64 query
   values; NO fudge factors (scipy's 1.01 rejected).

7. **Error mapping**: thunk returns raw LAPACK info; builder maps `info<0` →
   illegal-argument internal-bug error (with arg index), `info>0` →
   per-routine semantic message; allocation failures as normal rstsr errors
   (drop the −1010/−1011 sentinels); NO LAPACKE `info−1` remap (we have no
   `matrix_layout` arg); fix `info as i32` truncation under ilp64; replace
   `.unwrap()` panics and `rstsr_assert_eq!` (sygv/sygvd) with the mapping.

8. **ipiv**: 0-based at the builder boundary; ±1 conversion only inside the
   thunk. getrf/getri/gesv already consistent; sysv leak fixed in Part 2.

9. **gesvd superb**: kept (4-tuple at driver tier; higher layer discards).
   Arbitration rule on vendor disagreements: reference LAPACK / OpenBLAS
   behavior first.

10. **Zero-size matrices** (`min(m,n)==0`): synthesize LAPACK's identity
    sub-blocks in the builder without calling the driver ('A' ⇒ I_m / I_n;
    'S' ⇒ zero-width); never exercise the driver at m==0/n==0. Current SVD
    space settings de-conservatized in Part 1 (user: too conservative and
    memory-costly).

11. **NaN check**: none this campaign — deliberate divergence (numpy
    precedent: no finite check anywhere), with the callee-hazard evidence
    (silent OOB past `ipiv` with INFO=0 on NaN pivots; scipy "crashes,
    non-termination") recorded in the ADR. Future path (NOT this campaign;
    separate structural task): the user's `_adv_` trait shape — defaulted
    legacy `driver_xxx` forwarding to required `driver_adv_xxx(nthreads,
    nan_check, ..)` — evaluated sound; six amendments + cleaned sketch in
    GRILL-R5-QUESTIONS.md. Multi-threading-transpose handling is the
    blocking open problem there.

12. **lapack vs lapacke**: both kept indefinitely, mutually exclusive
    features (`cfg(not(feature="lapacke"))` guards `mod lapack` — verified).
    `lapack/` primary (most vendors ship no LAPACKE); `lapacke/` stays for
    openblas+mkl. No both-features parity module possible; instead the
    existing driver suite runs under BOTH configs separately; numpy/scipy
    as correctness oracle. (Precedent: paths once disagreed — cause was a
    LAPACKE bug.)

13. **Validation**: OpenBLAS ONLY, locally and in CI; blis/aocl/kml
    untested/low priority. KML's local `tests/driver_impl/f64/gesvd.rs`
    override kept (per-vendor local test overrides allowed). Tests small
    (≤ 20 per side). Required new test dimensions: 3-case layout inputs per
    tensor (f-prefer with ld > dim, c-prefer with ld > dim, fully
    non-contiguous), zero-size cases, c64 driver coverage for gesvd/gesdd
    (Part 1).

14. **Process**: part-by-part with review stops after each part; NO
    auto-commit in the rstsr repos; PRs on explicit user go only (fork
    `ajz34`, per CLAUDE.local.md). Part 1 = gesvd+gesdd retrofit + tests.
    Part 2+ = solve family (ipiv, error mapping), syev/syevd/sygv/sygvd,
    getrf/getri/potrf; blas3 untouched. The trajectory repo records the
    campaign (auto-commit allowed here).
