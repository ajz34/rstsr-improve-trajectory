# Session 2026-10-12 — Part-1 execution decisions (supplements DECISIONS.md)

Working branch `261011/lapacke-svd-retrofit` (commits `6c70dea`, `2f6c035`,
plus one pending F10 commit). Companion ADR branch
`261011/adr-0009-lapacke-layer`, skill branch
`261011/lapacke-translation-skill`. Review source:
`tmp-review-gesdd.md` (recovered `/code-review high`, all 10 findings
verified against code before acting).

## Resolved this session

1. **F1 (safety-doc lint)** — resolved as **module-level allow**, not
   per-thunk `# Safety` docs and not crate-level allows:
   `#![allow(clippy::missing_safety_doc)]` in
   `driver_impl/lapack/mod.rs` (shared by all 5 vendor crates via the
   directory symlink; also covers Part-2 families). User: "we do not need
   safety docstrings here", "better in module level".
2. **F2 (test lints)** — the diag-scale loops replaced by the broadcast
   form `&u * s.i((None, ..))` (mixed-dtype Complex×real multiply
   type-checks through the `Op` bound — compiler-verified); the no-op
   `size_of` branch became a real f32→f64 widening (`norm as f64`).
3. **F3 (query read-back)** — new `util_error::lwork_from_query`:
   immediate f64 conversion (rstsr-prelude vs `num::Float` method
   ambiguity makes generic `T::Real: Float` code an E0034 trap),
   non-finite/negative rejection (`None` → driver returns −1010 →
   MemoryError), f32 one-ulp round-up (`1 + f32::EPSILON` growth only when
   `size_of::<T::Real>() == 4`), `blas_int::MAX` overflow guard,
   `max(1)` clamp. Applied at all 4 sites. Note: vendor driver files see
   `rstsr_blas_traits::prelude` only — the helper needs an explicit
   `use rstsr_blas_traits::util::util_error::lwork_from_query;` per
   driver file.
4. **F4 (non-contig test)** — `slice!(None, None, 2)` over the (m+4, n+4)
   buffer then `.i((0..m, 0..n))`: step-2 view genuinely non-contiguous
   in both dims, tested matrix exactly (m, n) (old construction was
   stop-clamped by the indexer).
5. **F6 (complex gesdd lrwork)** — outer `MAX(1,·)` added, byte-matching
   `lapacke_cgesdd.c:65`.
6. **F8 (`convert_f_lapack`)** — removed (zero callers; SVD builders
   hand-roll order selection + `overwritable_convert_with_order`). If a
   Part-2 solve-family builder wants it, re-add with its first consumer.
7. **F9 + the output-order policy (user arbitration)** — **call-order
   policy codified**: the builder picks one order per call from the input
   preference (f-prefer → ColMajor zero-copy, c-prefer → RowMajor,
   ambiguous → device default); fresh outputs (incl. zero-size
   synthesized identities) allocate in the call order; output strides
   follow input layout for prefer inputs; values stay numpy-parity.
   This revises ADR §2's "fresh outputs in device default order" and the
   R6 tier-table's `order = device.default_order()` row. F9 closed as
   consistent-by-policy (no code change).
8. **F10 (gesvd superb backup)** — backup moved before the `info != 0`
   early return at both gesvd sites (LAPACKE does it unconditionally).
9. **F5 (sentinels vs ADR §5)** — documentation resolution: driver tier
   keeps reusing LAPACKE's −1010/−1011 values on the info channel;
   builder decodes via `map_lapack_info` → MemoryError. ADR §5 + skill
   §3b amended accordingly.
10. **F7 (partial `map_lapack_info` adoption)** — deferred to Part 2 by
    design (the 7 remaining `info as i32` sites are the Part-2 error
    mapping scope).

## ADR-0009 amendments applied (branch `261011/adr-0009-lapacke-layer`)

- §1: clone target is "the call's order" (was "fresh F-order buffer").
- §2: retitled "Call order: one order per call, derived from the input";
  documents the preference-derivation rule, call-order fresh outputs,
  zero-size synthesis in call order, and *why* device-default was
  abandoned (one-order driver contract; post-call bridge or wasted clone
  would defeat zero-copy f-prefer exploitation).
- §3: thunk described as carrying the layout 3-way branch (per-branch
  query short-circuit, template lds, sentinels); trait impl = the
  `lapacke_xxx` high-level tier (formula workspaces before query,
  hardened read-back); policy-vs-mechanics split restated.
- §5: sentinel wording inverted — they DO ride the info channel by
  design, decoded by `map_lapack_info`.

## Skill amendments applied (branch `261011/lapacke-translation-skill`)

- §2 input table: clone target "the call order".
- §2a: call-order fresh outputs + zero-size synthesis in call order;
  device-default rule removed.
- §3 skeleton: `order: FlagOrder` in the thunk signature (rstsr's
  replacement of LAPACKE's `matrix_layout`), layout-3-way description,
  hardening ownership moved to the trait-impl tier, generic builder =
  policy only.
- §3b: −1010/−1011 semantics (decode, not ban), `map_lapack_info`
  referenced.

## Still open

- GESVD superb comment alignment + `cargo +nightly fmt` + F10 commit
  (in progress when this file was written).
- R6 answers were never transcribed (R6 questions file exists, no
  answers file); this session's items 7/9 effectively settle R6-Q1/Q2
  in code+ADR — record here rather than inventing an answers file.
- Skill §4 test matrix may deserve a note that the non-contig case must
  avoid indexer stop-clamping (the F4 failure mode).
- Part-2 scope unchanged (solve family ipiv + error mapping, syev*,
  getrf/getri/potrf; blas3 untouched; `map_lapack_info` rollout = F7).

## Addendum 2026-10-12 (later) — ADR scope split + revision

Acting on the user's scope review of ADR-0009.

### Scope split

- **ADR-0009 narrowed** to the **LAPACKE translation layer**
  (`driver_impl/lapack/**` + `lapacke/**`): two-tier file shape, layout
  branch, translation conventions (`duplicate_item`, `T::Real` signature
  sharing), the `lapack/`≡`lapacke/` equivalence invariant, workspace
  protocol + hardening, error-channel sentinels. All input/output/order
  policy moved out.
- **New ADR-0010 — BLAS/LAPACK interface layer** (`src/lapack_*/` builders):
  call-order-from-input, input taxonomy (f-prefer zero-copy / clone /
  clone-or-consume), output layout (call-order fresh outputs; scipy
  `overwrite_*`-style reuse), zero-size synthesis, `map_lapack_info` decode,
  2-D-only.
- Both rewritten to the charter's Context / Decision / Considered /
  Consequences; defensive inline prose dropped.

### Negative-`info` numbering → Fortran (reference-LAPACK)

- **Finding**: LAPACKE's `_work` remaps `info<0 → info−1` in *both* branches
  (its `matrix_layout` is arg 1). Our ColMajor arm passed Fortran info raw
  while the copied RowMajor ld checks were LAPACKE-numbered (+1) → the two
  arms disagreed, and `lapack/` diverged from `lapacke/`.
- **Ruling**: report Fortran indices, so the index is checkable against the
  Fortran source. Code fix — gesdd RowMajor −6/−9/−11 → −5/−8/−10; gesvd
  −7/−10/−12 → −6/−9/−11 (4 sites, verified against `SRC/dgesdd.f` /
  `SRC/dgesvd.f`, whose own branches read `INFO = -5/-8/-10`). Documented as
  a divergence from `lapacke/` (ADR-0009 §7).

### Bridge threading → undecided

- The drivers call the serial `orderchange_out_*_ix2_cpu_serial` (faithful to
  LAPACKE's serial `LAPACKE_dge_trans`). Moving to per-device/ambient
  threading is deferred. Recorded as undecided in ADR-0009; reverses
  DECISIONS #4's "replace `_cpu_serial`".

### Skill (`lapacke-translation`)

- Sections tagged with their owning ADR; bridge bullet rewritten to
  "undecided"; §3b reworded to Fortran numbering; thunk skeleton `work`/
  `rwork` types corrected to `*mut T` / `*mut T::Real`; dangling "Part-2
  items in ADR-0009" reference removed.

### Side finding — fmt drift

- `cargo +nightly fmt` reflowed two *pre-existing* committed lines (gesdd
  `lrwork` binding; svd_layout strided chain) in all five vendor copies:
  HEAD is **not** `cargo fmt --check`-clean under rustfmt 2026-10-06, and CI
  runs fmt on unpinned nightly. Reflows kept.

### Order derivation unified on `get_output_order`

- User: the LAPACK interface layer should reuse `util_layout::get_output_order`
  (the blas3 mechanism) instead of hand-rolled `match (c_prefer, f_prefer)`
  blocks; the general priority rule = explicit output first-class, then input
  votes, then device default.
- Applied to gesvd/gesdd builders:
  `get_output_order(&[], &[(a.c_prefer(), a.f_prefer())], device.default_order())`.
- **Tie-break fixed in the helper** (user: ambiguous should prefer
  `default_order`): the voted slots now count only exactly-one-prefer
  inputs — both-prefer (size-0, single-row/column) and neither-prefer are
  neutral and fall to the device default. Previously any c_prefer vote won
  outright, so a lone `(true, true)` input resolved RowMajor. All three
  blas3 callers pass `by_all = &[]`, so they are unaffected.
- blas3 is explicitly out of ADR-0010's scope (its `out=`/trans-flag interface
  is a different mechanism, though now sharing the same helper).

### Verification

- fmt `--all -- --check` clean; `clippy -p rstsr-openblas --all-targets
  -D warnings` clean; 22/22 `tests_driver_impl` pass.
