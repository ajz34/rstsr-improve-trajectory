# Grilling init prompt — zero-size stride policy / BLAS empty operands

Recorded 2026-10-19. The human opened a planning grill on the report in this
directory and stated the following accepted decisions:

- Accept P1 — documentation only.
- Accept P2 — return a full-zero stride when the shape has a zero dimension.
- P3 — the `shape <= 1` branch of the proposal is accepted. For the `else`
  branch, change the contract so that `ld_col` returns an error or `None` when
  `stride[1] < max(1, shape[0])`; `ld_row` symmetrically.
- P3 — early returns for empty operands are accepted as essential.
- P4 — the zero-stride test can live in a new directory and should be
  OpenBLAS-only, not affecting other devices. Using `openblas_set_xerbla` as a
  test harness is accepted as a future task, not now.
- If the agent has doubts while planning, ask in this session.
