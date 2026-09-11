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
  4 files +667/−248). Local only — not pushed, no PR opened. Final diff kept
  as `2026-09-09-argmax-argmin/proposed-v2-post-review.patch`.
- Follow-up artifacts: `2026-09-09-argmax-argmin/NANARG-PROPOSAL.md`
  (nanargmin/nanargmax design + open owner decisions, incl. the NumPy
  first-NaN-wins divergence of plain argmin/argmax).

## Patches 2–5 — PENDING

elementwise, transpose-assign (with compose-smoke's shape-identity guard
amendment), reductions, vecdot: not yet applied. Patches are mutually
disjoint; `2026-09-09-compose-smoke/combined_all_five.patch` is now stale
relative to the reviewed patch 1 (regenerate from per-dir patches when
stacking the next one).

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
