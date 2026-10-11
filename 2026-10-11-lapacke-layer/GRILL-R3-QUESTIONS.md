# Grill R3 — LAPACKE translation layer (session 2026-10-11)

## R2 answers (agent-transcribed from the on-screen session)

- **Q1 two impls**: (c) keep both indefinitely. `driver_impl/lapack/` is the
  primary (most vendors ship no LAPACKE in their shared library); `lapacke/`
  stays an optional feature. Note: the two paths once disagreed in results —
  root cause was a LAPACKE bug. Cross-validation between paths has value.
  Also: `orderchange_out_r2c_ix2_cpu_serial` is our `dge_trans` analog; we
  have no `dsy_trans` analog today; it is single-threaded — threading policy
  open (→ R3-Q1).
- **Q2 workspace tier**: mirror LAPACKE's two-tier split (thin wrapper doing
  query-then-compute; `_work` form handling query+computation) but keep both
  tiers in one physical file. (→ R3-Q2 skeleton confirm)
- **Q3 error mapping**: approved as recommended.
- **Q4 output order**: RE-ASK after checking what scipy and numpy do (fact
  agents dispatched; lands in R4). User principle stated: mutable inputs
  (e.g. solve's b) that are f-prefer must maximally utilize that f-prefer —
  forcing device-default row-major order would add a transposed copy +
  memory; following the input (c) keeps fortran order with zero extra budget.
- **Q5 ipiv**: user not familiar — agent to verify end-to-end and make it
  sound (→ R3-Q3).
- **Q6 superb**: keep. Dispute rule: reference LAPACK / OpenBLAS behavior is
  first priority when distributions disagree (goes into ADR + skill).
- **Q7 zero-size**: (a) identity sub-blocks, provided no testing issues.
- **Q8 NaN check**: pending facts — is the LAPACKE check for caller-side
  memory safety, arithmetic robustness (hangs), or diagnostics? Re-ask in R4.
- **Q9 hardening**: accepted, conditional on redundant-allocation budget
  being small O(1) — it is (all items are clamps/1-ULP bumps/checked casts;
  no extra allocation). scipy's 1.01 fudge rejected.
- **Q10 validation**: OpenBLAS ONLY locally (it is also the only vendor in
  CI); blis/aocl/kml untested, low priority.
- **Q11 execution**: confirmed with amendments — NO auto git commit in the
  rstsr repo; split work into small parts; STOP for user review after each
  part. PR only on explicit go.

---

## R3 frontier

❓ **Q1 - Bridge-copy threading policy**: `orderchange_out_r2c_ix2_cpu_serial`
(our `dge_trans` analog) is single-threaded; LAPACKE's `dge_trans` is serial
too. On large matrices the bridge copy is O(mn) — same order as one LAPACK
pass. (i) mandate serial always (LAPACKE-faithful, simplest, predictable);
(ii) reuse the per-device orderchange infra as-is (on rayon devices it is
parallel; thread count = ambient rayon pool; BLAS-thread injection stays at
the linalg-traits tier `with_blas_num_threads`, unchanged); (iii) dedicated
parallel transpose with its own knob.

➡️ Recommend (ii): it is the codebase's existing behavior for every other
copy, adds no new knob, and the guideline stays silent about thread counts
(ambient device policy governs rust-side copies — same rule as any
elementwise op).

❓ **Q2 - One-file two-tier vendor shape**: confirm the vendor-file skeleton
mirroring LAPACKE's split, both tiers in one physical file (shared by vendor
symlinks):

```rust
// driver_impl/lapack/svd/gesvd.rs
// "_work"-equivalent: pure FFI thunk. lwork = -1 ⇒ LAPACK query semantics.
unsafe fn dgesvd_work(.., a, lda, s, u, ldu, vt, ldvt,
                      work: &mut [T], lwork: blas_int, rwork: &mut [R])
    -> blas_int;
// thin wrapper: query call (lwork=-1) → alloc (formulas + clamps) →
// compute call via dgesvd_work. Error mapping stays in the builder layer.
unsafe fn driver_dgesvd(..) -> blas_int;
```

Query call and compute call become textually separate; both go through
`dgesvd_work`. Confirm this shape (it is essentially today's code with the
two calls split into named functions, plus the R2-Q3/R2-Q9 hardening)?

➡️ Recommend confirm.

❓ **Q3 - ipiv comprehension & fix (R2-Q5 delegated)**: Verified end-to-end
today: `getrf` returns ipiv **0-based** (decrements), `getri` expects 0-based
(re-increments before the FFI call) — the pair is self-consistent; `gesv`
also returns 0-based. **`sysv` leaks the raw 1-based ipiv** (no decrement) —
the one inconsistency; higher layers never consume sysv's ipiv, so it is a
latent trap, not a live bug. Fix: mandate "index vectors (ipiv, and jpvt for
future QR) are 0-based at the builder boundary; ±1 conversion lives only in
the thunk"; change sysv to decrement; add ipiv-value assertions to the
driver tests (vs the same factorization computed on the f-contig path).
Confirm?

➡️ Recommend yes.

❓ **Q4 - lapack↔lapacke parity test**: given your anecdote (paths once
disagreed; cause was a LAPACKE bug), add a parity test module compiled only
under `#[cfg(all(feature = "lapack", feature = "lapacke"))]` that runs the
same inputs through both paths and asserts identical outputs (bitwise for
deterministic routines; U/VT up to sign for SVD)? Runs in a CI matrix cell
with both features on (openblas), or manual-only?

➡️ Recommend yes, CI matrix cell (openblas, both features).

❓ **Q5 - Zero-size semantics confirm (R2-Q7)**: synthesize LAPACK's result
without calling the driver: for `min(m,n) == 0` — `s` empty; jobz='A' ⇒
`U = I_m`, `VT = I_n` ('S' ⇒ `U = [I 0]`-shaped m×k / k×n with k=0 ⇒
zero-width); same for gesdd. Tests: m=0, n=0, both, each job flag, f64+c64,
small. Any objection (e.g. prefer calling the driver on OpenBLAS only)?

➡️ Recommend confirm (a) with those tests.

❓ **Q6 - Triangle-only bridge copies (`dsy_trans` analog)**: today our
bridge copies the full matrix; LAPACKE's symmetric routines copy only the
referenced triangle in (`dsy_trans`/`dpo_trans`) and transpose out with
`dge_trans` only when jobz='V'. Under our input policy the bridge fires only
for non-f-prefer symmetric inputs. (i) adopt triangle-only copies now (halves
the copy; needs new kernels; must still allocate a full square buffer since
syev jobz='V' overwrites all of A); (ii) keep full-matrix copies (uniform
kernel inventory, LAPACKE-faithful in outcome), record triangle-only as a
sanctioned future optimization in the skill.

➡️ Recommend (ii) for this campaign — one bridge shape everywhere is worth
more to subagent determinism than the half-copy saving; the skill notes the
exception explicitly.

---

## R3 addendum — re-asks unblocked by the fact agents (independent of Q1–Q6)

Fact base (see FINDINGS.md §7): numpy is **input-independent, always C** —
no order logic exists anywhere in `numpy/linalg` (gufunc KEEPORDER → C). scipy
is **C by default**; F outputs occur ONLY in the `overwrite_*` fast path where
the returned array IS the reused input buffer (solve b F+overwrite_b, inv,
cholesky, R, lu U), plus legacy f2py routines that are unconditionally F.

❓ **Q7 - Output order policy (R2-Q4 re-ask)**: Your f-prefer-maximization
scenario (mutable `b`, f-prefer, row-major device, solve) is *exactly* scipy's
overwrite fast path — which our in-place machinery gives for free. Proposals:
(a) **hybrid**: fresh outputs → device `default_order()` (numpy parity; one
output bridge copy on row-major devices, the price scipy also pays); in-place /
overwritable outputs → the input's own buffer and order (returned tensor IS
`b`, F-ordered even on a row-major device; zero bridge copies — scipy
`overwrite` semantics). View-only f-prefer inputs still pass zero-copy into
LAPACK (R1-Q4), regardless. (b) always device-default even for in-place
(forces the transpose you want to avoid); (c) follow input order always
(matches neither numpy nor scipy; leaks F outputs for view-only inputs).

➡️ Recommend (a).

❓ **Q8 - NaN check adoption (R2-Q8 re-ask)**: Verdict from the evidence:
LAPACKE's nancheck is NOT caller-side memory safety — it is early diagnostics
plus callee-side robustness. The hazard survives a rust caller because the
callee is the same C/Fortran: NaN pivots in `?sptrf/?hptrf` read from
uninitialized offsets and write past `ipiv` **while reporting INFO=0**
(LAPACK commit ac6db4873); scipy documents "crashes, non-termination" when
`check_finite=False`; `dbdsqr`/`dstein`/`dstebz` are iteration-bounded
(garbage, not hangs); `dgesdd`/`dgeev` self-check (`INFO=-4`). Options:
(a) no check at any tier now — deliberate, documented divergence (numpy
precedent: no finite check at all), hazard recorded in the ADR, with (b) as
the sanctioned future upgrade; (b) `debug_assert!`-only finiteness scan
(guards dev/test builds, zero release cost); (c) opt-in runtime flag at the
linalg-traits user tier; (d) always-on (LAPACKE parity).

➡️ Recommend (a) per your "deviate by intention at this time", with the
callee-hazard paragraph going into the ADR and (b) recorded as the named
future upgrade path in the skill.
