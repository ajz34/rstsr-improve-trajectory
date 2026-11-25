---
name: rstsr-blas-lstsq-grill
description: "BLAS lstsq grill (2026-11-25): R1 written (10 questions, recommendations), waiting for owner answers; scope BLAS-only, NumPy 4-tuple, stacked, GELSD-only"
metadata:
  type: project
---

2026-11-25. Grill R1 for `lstsq` on the BLAS devices, directory
`2026-11-25-blas-lstsq/` (`GRILL-INIT-PROMPT.md`, `GRILL-R1-QUESTIONS.md`).
rstsr at `ed1f62d`, rstsr-ffi at `83efe50`, both clean; **no rstsr file
edited** — this is still planning.

Recommended frontier (answers pending): BLAS-only (no faer impl; faer's own
`Svd::solve_lstsq` asserts m>=n and has no rank truncation); NumPy's 4-output
`LstsqResult { x, residuals, rank, s }`; stacked `(...,M,N)` with SciPy-1.18
broadcast; `rcond: T::Real`, default `eps*max(M,N)`; GELSD only; SciPy-style
residuals `(...,K)` with NaN where undefined; per-slice `Tensor<usize>` rank;
destructive `a`/`b` under the QR/eig rule; NumPy's `nrhs=0` pad workaround;
one PR on `261125/blas-lstsq` after `DECISIONS.md` + `PLAN.md`.

Facts that shaped it: reference semantics in memory
`numpy-scipy-lstsq-semantics.md`; no cheat-sheet row possible (the page's test
is faer-backed); `solve_plan` in `rstsr-linalg-traits/src/stack.rs` asserts a
square `a`, so a rectangular variant is needed; residuals come free from the
GELSD `B` workspace rows `n..m`; gap-notice rows owed to skill
`lapacke-translation` (§3a gelsd query, §3b info=-4).

Next: read `GRILL-R1-ANSWERS.md` when written, then R2 or `DECISIONS.md` +
`PLAN.md`.
