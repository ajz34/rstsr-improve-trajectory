---
name: rstsr-compose-smoke-t8
description: T8 compose-smoke COMPLETE — five campaign patches compose (campaign order 1→5, zero apply conflicts); union exposed a latent patch-3 bug (orderchange router missing shape-identity guard → order-changing reshape panics) fixed by compose_guard_fix.patch; all suites green, 332-check union gate PASS, zero spot regressions; rstsr restored clean.
metadata:
  type: project
---

T8 (2026-09-09-compose-smoke) results, rstsr 386948be:

- **Apply order**: campaign order 1→5 applies cleanly (git apply, no fuzz needed),
  both with the original and the AMENDED patch 3. Patches 1 (argmax) and 4
  (reductions) overlap on reduction.rs trio as claimed — hunks disjoint.
- **Union bug found in round 1, CLOSED IN PATCH 3 ITSELF**: patch 3's 2-D orderchange
  router in native-impl cpu_{serial,rayon}/assignment.rs guarded stride pattern but
  NOT shape identity; blocked kernels require shape identity, and reshape calls
  assign_arbitary_uninit with shape-CHANGING layouts (reshape.rs:157) → 3
  entry_row_cpu failures (reshape/doc_draft tests; clean tree 290/290). The
  `lc2.shape() == la2.shape() &&` guard (4 router sites) + reshape fixture were
  folded into 2026-09-09-transpose-assign/proposed.patch by its owner;
  compose-smoke/compose_guard_fix.SUPERSEDED.patch is the discovery record only.
- **Amended union (final verdict COMPOSE: YES)**: union +1462/−397; entry_row_cpu
  290/290 + lib 110/110 portable AND native with NO separate fix; 332-check
  cross-experiment gate PASS both devices both configs; spots all hold —
  amended-union re-check: strided-B add 7.499 ms (ref 7.51), transpose-B 5.885 ms
  (ref 5.99), serial native med of 3.
- **Methodology lesson**: fixed-iteration-median harness read ~6% high on strided-B
  vs criterion in round 1; origin experiment's criterion bench on the same tree
  resolved a would-be false "drift" — when a spot drifts 5-15%, re-run with the
  ORIGIN harness before believing composition loss.
- RUSTFLAGS caching gotcha: cargo treated alternating RUSTFLAGS as fresh more
  eagerly than expected ("Finished in 0.02s"); verify binary identity by md5 per
  config (touch src/ to force).
- Task brief for T8 quoted "~29 µs" for % 1e4 faer16 — that was the PRE-patch
  baseline; candidate ref is 7.19 µs (T3' tables.md).
