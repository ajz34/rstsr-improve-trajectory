# Campaign integration progress — 2026-09-11

Status of integrating the five campaign patches (from
[260909-campaign-outcome.md](260909-campaign-outcome.md)) into `../rstsr`,
after the first integration session (Claude Code + glm-5.3-flash).

## Patch 1 — argmax/argmin: INTEGRATED

- Applied to `../rstsr`, reviewed by the owner, four review points addressed
  (restore general closure API; check NumPy NaN semantics + draft nanarg
  proposal; clippy clean; re-verify efficiency).
- Verification: gates 110/110 lib + 290/290 entry_row_cpu, correctness gate,
  clippy — all green both configs; three paired criterion passes vs
  same-session refA baselines, no stable regression (details and numbers in
  [../2026-09-09-argmax-argmin/README.md](../2026-09-09-argmax-argmin/README.md)
  addendum).
- **Committed:** `../rstsr` branch `260910-core-efficiency`, commit `091f3e2`
  ("rstsr: speed up argmin/argmax with 8-lane contiguous fast path",
  4 files +667/−248). Final diff kept
  as `2026-09-09-argmax-argmin/proposed-v2-post-review.patch`.

## Patch 1 follow-up — nanargmin/nanargmax: INTEGRATED

- The owner then asked for NumPy-like argmin/argmax ("usual and its nan
  form"). Implemented same day (see the 2026-09-11b addendum in the argmax
  README for the full record): `nanargmin`/`nanargmax` shipped with NumPy
  nanarg semantics — **free** on NaN-free input; plain argmin/argmax keep
  NaN-skipping semantics after measurement showed NumPy's first-NaN-wins
  rule cannot be added without de-vectorizing the kernel (+55…+100% small,
  +5…+18% large at kernel level; a block-tiled attempt hit +100…+250% and
  was reverted same day).
- Gates: lib 110/110, entry_row_cpu 302/302 (12 new tests), clippy clean
  both configs; rayon/faer device path verified by a cross-device spot
  check (18/18) kept in `2026-09-09-argmax-argmin/nan-scan-variants/`.
- **Committed:** `../rstsr` commit `9c42b1f` ("rstsr: add
  nanargmin/nanargmax (NumPy nanarg semantics)", 12 files +681/−133).
- **PR #100 merged** (RESTGroup/rstsr, squash `e835173` into `master`,
  2026-09-11): both commits landed together as
  "rstsr: faster argmin/argmax kernels + nanargmin/nanargmax". CI fix
  commits on the way in: rustfmt (b3f5483), needless borrows + comment
  form (630258f + restore — CI nightly wraps comments; local rustfmt
  unwraps; device crates compile rstsr-core tests via the
  tests/core_func symlink, so workspace clippy sees them).
- This closes NANARG-PROPOSAL open question §5.1 (plain-arg NaN semantics:
  decided — keep rstsr semantics, divergence documented) and §5.2 (all-NaN:
  error, matching NumPy). §5.3/§5.4 (unraveled twins deferred; naming)
  stand as decided in the proposal.

## Patch 2 — elementwise: READY FOR REVIEW

- Applied to `../rstsr` branch `260914-elementwise`, base = `origin/master`
  `e835173` (post-PR#100). Applied clean (+346, files unchanged:
  `rstsr-native-impl/src/cpu_{serial,rayon}/op_with_func.rs`).
- Working-tree-only so far (not committed — same review-then-commit cycle as
  patch 1). Post-rustfmt diff:
  `2026-09-09-elementwise/proposed-v2-post-fmt.patch` (fmt reflowed added
  lines only; zero content changes).
- Gates: lib 110/110 + entry_row_cpu 302/302 both configs (faer-free config
  too: 94+1 ign / 302), clippy `-p rstsr-native-impl --all-targets
  -D warnings` clean both configs, correctness example 94/94 both configs.
- G1 caller enumeration done: only out-of-family callers are the reduction
  order-fixup sites (dead under default iter order; disjoint-buffer copy
  where tiled order is harmless). T8 union smoke already covered compose.
- Paired benches: 3 alternating refA(master)/cand passes × portable+native,
  full 54-bench suite — **no stable regression in any cell; all headline
  wins reproduce** (strided B serial 23.4→7.4 ms, faer16 2.06→0.96 ms, odd
  8–9×, small 64² 16×, `c += bᵀ` 0.34×; lottery cells inside documented
  bands). Evidence: `2026-09-09-elementwise/results/integration260914/`.

## Patches 3–5 — PENDING

transpose-assign (with compose-smoke's shape-identity guard amendment),
reductions, vecdot: not yet applied. Patches are mutually disjoint;
`2026-09-09-compose-smoke/combined_all_five.patch` is now stale relative to
the reviewed patch 1 (regenerate from per-dir patches when stacking the next
one).

## Still open (owner decisions, from the campaign outcome)

1. Integrate patches 2–5 (one at a time, same review+bench cycle as patch 1).
2. File the three upstream bug reports
   (`2026-09-09-fill-misc-structural/BUG-NOTES.md`): stride-0 summed-axis
   reduction mis-multiply; `inner_dot` uninit `beta*c` read; f32-vecdot
   dispatch caveat.
3. Decide plain argmin/argmax NaN semantics (keep skip-mid-NaN documented, or
   align with NumPy's first-NaN-wins — see NANARG-PROPOSAL §5).
4. Optional docs drafts (reuse-idiom + MALLOC tunables; benchmark
   conventions; `dispatch_simd` do-not-add ADR).
