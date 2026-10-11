# Grill R5 (final round) — LAPACKE translation layer (session 2026-10-11)

## R4 answers (agent-transcribed from the on-screen session)

- **Q1 NaN check**: no knob this campaign; user proposes the future trait
  shape (defaulted `driver_syev` → required `driver_adv_syev(nthreads:
  Option<usize>, nan_check: bool, ..)`) and asks for an evaluation, not
  implementation. → Evaluated below; confirm amendments.
- **Q2**: confirmed; new branch in EACH repo (rstsr, rstsr-book, rstsr-agents).
- **Q3 Part 1 scope**: confirmed.
- **Q4 KML override**: keep; local per-vendor test overrides are allowed.

## Evaluation of the user's `_adv_` trait proposal

Verdict: sound. Defaulted legacy method forwarding to a required advanced
variant preserves all call sites, allows per-crate incremental migration
(explicit impls override the default), and places knobs at the FFI boundary
where LAPACKE itself puts them. Amendments:

1. **Cleaned sketch** (the R4 pseudo-code referenced `Some(nc)` with nothing
   in scope; explicit control enters only via the `_adv_` call):

   ```rust
   pub trait SYEVDriverAPI<T>
   where
       T: BlasFloat,
   {
       /// Legacy entry: knobs resolved from compile-time feature / ambient policy.
       unsafe fn driver_syev(
           order: FlagOrder, jobz: char, uplo: FlagUpLo,
           n: usize, a: *mut T, lda: usize, w: *mut T::Real,
       ) -> blas_int {
           #[cfg(feature = "nan-check")]
           let nan_check = true;
           #[cfg(not(feature = "nan-check"))]
           let nan_check = false;
           unsafe { Self::driver_adv_syev(None, nan_check, order, jobz, uplo, n, a, lda, w) }
       }

       /// Advanced entry: explicit knobs; vendor crates implement only this.
       unsafe fn driver_adv_syev(
           nthreads: Option<usize>, nan_check: bool,
           order: FlagOrder, jobz: char, uplo: FlagUpLo,
           n: usize, a: *mut T, lda: usize, w: *mut T::Real,
       ) -> blas_int;
   }
   ```

2. **Feature plumbing**: the `nan-check` feature must be declared in
   rstsr-blas-traits (where the default body compiles) and forwarded by each
   vendor crate (`nan-check = ["rstsr-blas-traits/nan-check"]`); the cfg
   string must match the feature name exactly.
3. **Edition-2024**: unsafe-fn bodies are no longer implicit unsafe blocks —
   the forwarding call needs the explicit `unsafe { }` shown above.
4. **Knob-count threshold rule** (for the skill): positional knobs are fine
   up to ~2 (`nthreads`, `nan_check`); the moment a third knob appears, the
   `_adv_` signature must be refactored to a single `DriverAdvArgs`-style
   struct with `Default` before appending again (appended booleans/options
   break all 5 vendor impls each time).
5. **`nthreads` semantics assignment**: it governs vendor-library threading
   at the FFI boundary (omp-style control just before the LAPACK call);
   rust-side bridge copies keep following the ambient device policy (R3-Q1)
   — the driver tier never spawns its own threads for the bridge.
6. **Nancheck scan semantics** (when implemented): scan only the logical
   matrix through `lda` (and uplo-referenced triangle for sy/po matrices),
   mirroring LAPACKE's `dge/dsy/dtr_nancheck` — never the whole allocation.

❓ **Q1 - Confirm the amendments** above as the recorded future path in
ADR-0009 (proposal credited as user design; implementation deferred)?

➡️ Recommend confirm.

❓ **Q2 - Shared understanding / close**: the full decision set as grilled
(R1–R5) is summarized in DECISIONS.md-draft form below; on your confirmation
the grill closes, DECISIONS.md is written to this directory, and execution
awaits your explicit go (part-by-part with review stops, no auto-commit in
rstsr; trajectory repo may auto-commit its own files).

➡️ Recommend confirm.

---

## DECISIONS.md draft (for confirmation)

1. **Deliverables**: skill `lapacke-translation` (rstsr-agents, new branch)
   + ADR-0009 (rstsr-book, new branch) + code retrofit (rstsr, new branch
   `261011/…`). Recipe-with-rationale + mandatory gap-notice rule ("recipe
   lacks coverage → stop and report, do not improvise").
2. **Input policy**: f-prefer view inputs pass zero-copy with derived ld;
   all non-f-prefer inputs (c-contig/c-prefer/fully-non-contig) clone;
   overwrite-routines clone/consume regardless. Fully-non-contiguous is our
   extension beyond LAPACKE's two-layout world.
3. **Output policy (hybrid)**: fresh outputs in device default order;
   in-place/overwritable outputs reuse the input buffer and its F order;
   f-prefer views exploited maximally.
4. **Bridge**: per-device orderchange infra (ambient threading) replaces the
   `_cpu_serial` kernels in driver files; full-matrix copies (no dsy/dtr
   triangle-only variant this campaign — recorded future optimization).
5. **Vendor file shape**: trait `driver_xxx` signatures unchanged; per-file
   free `pub fn xxx_work` (internal-purpose) = pure FFI thunk; query and
   compute calls textually separate; workspace+clamps at the builder layer.
6. **Workspace**: per-routine policy table (LAPACKE-faithful: query vs
   documented formulas: gesdd iwork 8·minmn, complex gesvd rwork 5·minmn,
   complex gesdd lrwork formula); query with real col-major lds; MAX(1,·);
   lwork==0→1 clamp; checked int conversion; nextafter guard for f32/c64;
   no fudge factors.
7. **Errors**: raw LAPACK info at the thunk; builder maps info<0 → illegal-
   argument internal-bug error, info>0 → per-routine semantic message;
   allocation failures as normal rstsr errors (no −1010/−1011 sentinels);
   no `info−1` remap (no matrix_layout arg); fix `as i32` truncation; replace
   `.unwrap()`/`assert_eq!` sites (sygv/sygvd) in Part 2.
8. **ipiv**: 0-based at the builder boundary, ±1 only in the thunk; sysv
   fixed in Part 2.
9. **gesvd superb**: kept (4-tuple); reference LAPACK/OpenBLAS behavior is
   the arbitration priority on distribution disagreements.
10. **Zero-size**: identity sub-blocks synthesized, no driver call; current
    SVD space settings to be de-conservatized in Part 1.
11. **NaN check**: none this campaign (numpy precedent; deliberate
    divergence; callee-hazard evidence in ADR). Future path: the `_adv_`
    trait shape with the six amendments above.
12. **lapack vs lapacke**: both kept indefinitely (mutually exclusive
    features; lapack primary — most vendors ship no LAPACKE); existing test
    suite run under BOTH configs; numpy/scipy as oracle.
13. **Validation**: OpenBLAS only locally and in CI; KML local test override
    kept; tests small (≤20 per side); 3-case layout inputs (f-prefer ld>dim,
    c-prefer ld>dim, fully-non-contiguous) + zero-size; c64 for gesvd/gesdd
    in Part 1.
14. **Process**: part-by-part with review stops; no auto-commit in rstsr;
    PR on explicit go only; trajectory repo records the campaign.
