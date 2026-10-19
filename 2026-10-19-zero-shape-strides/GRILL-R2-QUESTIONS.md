# Grill R2 — task split, execution protocol, docs placement

Round 2, 2026-10-19. Frontier only. Answers are user-owned.

## Background: R1 answers (agent-transcribed)

Q1, Q2, Q3, Q4, Q5, Q6, Q9, Q10 accepted as recommended.

- **Q7**: no PR this time; the agent does **not** execute the task
  automatically. Split into tasks; the user manually checks each task.
- **Q8**: `order_semantics.md` is for row/column-major, not zero-size. Docstrings
  and an ADR are accepted; no `order_semantics.md` section.

Consequences already fixed: the wave becomes a sequence of manually reviewed
tasks, not a PR; the documentation deliverable is code docstrings + one ADR.

## Q1 — task granularity

Options:

- **(a)** five layer tasks:
  1. BLAS-3 guards: `gemm` (`m/n/k == 0`), `syhemm`, `trsm` (guard in `run`
     before `flip_trans`) + OpenBLAS BLAS-3 tests;
  2. LAPACK guards: `gesv`/`sysv`/`getrf`/`getri`/`potrf` + `syev(d)`/`sygv(d)`
     + move the existing SVD `minmn == 0` guard above `lda` + OpenBLAS tests;
  3. `ld_row`/`ld_col` contract (Q2 rule) + `rstsr-core` unit tests + docstrings;
  4. P2 zero strides in `stride_f_contig`/`stride_c_contig` (both `Ix`/`IxD`) +
     `rstsr-common` unit tests + docstrings;
  5. ADR-0011 in `rstsr-book`.
- **(b)** routine-level tasks (one per BLAS/LAPACK routine): finer review, but
  many near-identical tasks.
- **(c)** three tasks: all guards, all layout changes, docs.

Each task carries its own tests, so every task leaves the tree green and is
independently verifiable. Order is guard-first so the `ld_*`/P2 changes never
leave an unguarded empty path.

**Recommendation:** (a). It matches the review boundary (BLAS-3 vs LAPACK vs
layout vs docs) and keeps each diff small enough for one manual pass.

## Q2 — execution protocol and branch/commit policy

After this plan is approved:

- **(a)** implement **one task per turn**, stop for the user's manual check,
  and start the next only after the user says so. Work on a single branch
  `261019/zero-size-stride-policy` off `main`; leave the task uncommitted until
  the user's check, then commit only on explicit instruction (rstsr's
  no-auto-commit rule). No push, no PR.
- **(b)** one branch with one commit per task, committed by the agent as soon
  as the task is implemented (needs an explicit override of no-auto-commit).
- **(c)** no branch yet; present tasks as patches.

**Recommendation:** (a). It matches "for each task I will manually check" and
keeps the no-auto-commit rule intact.

## Q3 — where the task plan lives

- **(a)** `TASKS.md` in this directory, with per-task scope, files, tests,
  exit criteria, and order; commit it (and the two GRILL rounds) in
  `rstsr-improve-trajectory` after approval; no rstsr changes.
- **(b)** same file, left uncommitted until the implementation wave is done.
- **(c)** keep it only in the conversation (not recommended; the convention
  says decisions live in the directory).

**Recommendation:** (a). The plan is the artifact the user reviews; the
trajectory repo allows autonomous commits, and nothing is touched in rstsr.

## Q4 — docstring placement and the ADR task

- **(a)** docs travel with their code task: stride rule documented on
  `stride_f_contig` / `stride_c_contig` / `stride_contig` /
  `new_c_contig` / `new_f_contig` in task 4; leading-dimension contract on
  `ld_row` / `ld_col` / `ld` in task 3; a short type-level note on `Layout`.
  Task 5 is the ADR only.
- **(b)** one dedicated docs task covering every docstring plus the ADR.
- **(c)** docstrings only; drop the ADR.

**Recommendation:** (a). The docstring is part of the change it describes and
is reviewed with it; the ADR then records the cross-cutting rationale after the
behavior is settled. Include the `Layout` type-level note.
