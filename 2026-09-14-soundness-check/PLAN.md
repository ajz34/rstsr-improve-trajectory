# Plan: rstsr soundness check campaign

- **Date**: 2026-09-14
- **rstsr base commit**: `acfa93e` (master; includes #100 argmax/argmin, #101 elementwise,
  #102 transpose-assign integrations from the 2609 campaign)
- **Origin prompt**: [./initial-prompt.md](./initial-prompt.md)
- **Repo rules**: [../AGENTS.md](../AGENTS.md) (this repo), test conventions
  `rstsr-core/tests/CONTEXT.md`, ADR-0002 (entry-binary test matrix) in rstsr-book.

## 1. Mission

Thorough soundness check of rstsr, focused on `rstsr-common`, `rstsr-core`
(incl. `device_faer`), `rstsr-native-impl`, `rstsr-dtype-traits`,
`rstsr-linalg-traits`. Soundness means four things here:

| # | Aspect | Deliverable & where it lands |
|---|--------|------------------------------|
| S1 | Theoretical correctness (index math, layout invariants, iteration order, broadcasting/reshape semantics, overflow) | Report **only in this repo** (subtask README); fixes proposed as diffs |
| S2 | `unsafe` soundness | `// SAFETY:` justification, 1–3 lines, written **into rstsr code** (patch file here, human integrates); expanded discussion for anything nontrivial in this repo |
| S3 | Column-major correctness | New integration tests following existing conventions (`entry_col_*` binary per ADR-0002); col build run of the whole existing suite; bugs found → fixes |
| S4 | Easy efficiency wins | Not the emphasis; recorded in subtask READMEs when stumbled upon |

**Out of scope**: `crates-device/*` (BLAS devices), `rstsr-tblis`, `rstsr-sci-traits`,
GPU. **In scope additionally**: device faer including its linalg functionalities.

Non-negotiables (inherited from 260908 campaign):
- Never commit in `../rstsr`, `../rstsr-book`. All rstsr-side changes land here as
  `*.patch` files; the human integrates.
- Work happens in **detached git worktrees** under `/home/a/rstsr_pack/tmp/snd-*`
  so the main checkout and the user's stash stay untouched; never `git commit`
  in any rstsr worktree (they share the main repo's object store).
- Tests in release mode when numbers are reported; correctness-only runs may use
  the default test profile unless timing is part of the claim.
- Edge shapes/strides everywhere: 0-sized and 1-sized dims, non-contiguous,
  partially contiguous, broadcast (stride-0) inputs.

## 2. Subtasks

Each subtask is a directory under this one: `README.md` (findings/report),
`*.patch` files (rstsr-side deliverables), optional `results/`.

### T1 `T1-unsafe-audit` — unsafe-soundness sweep (S2)

Sweep every `unsafe` in rstsr-common / rstsr-core (incl. `feature_rayon`,
`device_faer`) / rstsr-native-impl / rstsr-dtype-traits / rstsr-linalg-traits
(~536 occurrences today, ~32 existing safety comments). For each block:
verify the safety argument against the actual invariants (layout bound
guarantees, pointer arithmetic, aliasing); add a `// SAFETY: …` comment where
missing. Any block whose argument **fails** is a real soundness bug: minimal
fix in `fixes.patch`, expanded discussion in README.
Worktree: `snd-unsafe`.

### T2 `T2-col-major` — column-major build + tests (S3)

`col_major` is a compile-time contract feature (mutually exclusive with
`row_major`, `compile_error!` guard) but **nothing wires it**: no entry binary,
the suite has never been run in col mode. Tasks: wire `entry_col_cpu.rs`
(ADR-0002 name) + `[[test]] required-features` gate; run the entire existing
body under `--no-default-features --features "col_major,…"`, triage every
failure (library bug vs row-major-hardcoded test); author a new col-major test
track (creation/iteration order, slicing, reshape, transpose, broadcasting
limits, matmul, reductions, 0/1 shapes, non-contig strides); fix genuine bugs.
Worktree: `snd-colmajor`.

### T3 `T3-layout-theory` — layout & index-math audit (S1, parts of S3)

Audit `rstsr-common/src/layout/*` (layoutbase invariants, indexer bounds,
iterator, broadcast, reshape, slice, rearrangement, shape) and the
layout-consuming machinery in `rstsr-core/src/tensor/*` (indexing,
iterator_axes, iterator_elem, creation): signed/unsigned overflow in
offset/size products, negative-offset `get_unchecked` justifications, stride-0
broadcast, 0/1-dim degenerate cases, reshape `-1` inference, K-order
translation. Unit tests for edge cases in-crate; integration tests via the
row-major entry where behavior-level. Real bugs → `fixes.patch`.
Worktree: `snd-theory`.

### T4 `T4-faer-linalg` — device faer + linalg audit (S1/S2 on faer paths)

Audit `rstsr-core/src/device_faer/*`: conversion rstsr-layout ↔ faer
(row-major ↔ column-major stride mapping), matmul entry conditions, edge
shapes (0×n, 1×1, offset, non-contiguous/transposed views), the
`faer_ext` interop unsafe. Locate and check DeviceFaer's linalg
functionalities (solve/inv/svd/eigh/… wherever implemented — `rstsr-linalg-traits`
and/or `device_faer`), test against serial reference implementations.
Fixes → `fixes.patch`. Worktree: `snd-faer`.

## 3. Coordination rules

- Max 4 subagents, one per subtask, run in parallel; main agent (this session)
  consolidates, validates patches apply cleanly to `acfa93e`, writes the final
  cross-task report, commits this repo.
- Patches must be generated as `git -C <worktree> diff > <file>.patch` against
  clean `acfa93e`; comment-only SAFETY patch kept **separate** from behavioral
  fixes so the human can review/land them independently.
- Efficiency observations (S4) go to the subtask README "Efficiency notes"
  section; no benchmark campaign in this task.
- Any cross-cutting convention change worth an ADR → propose diff to
  rstsr-book only if genuinely necessary.

## 4. Build matrix notes

- Toolchain: system stable 1.97.1 (rstsr's `rust-toolchain.toml` does not
  propagate along path deps; known from T5 study).
- Row-major suite (default): `cargo test -p rstsr-core` (entry_row_cpu gated on
  `row_major`, in default features).
- Col-major suite: `--no-default-features --features "col_major,aligned_alloc,faer,faer_as_default"`
  (+ whatever the dev-dep umbrella `rstsr` needs — resolve during T2).
- Each worktree has its own `target/`; no shared target dirs (feature sets
  conflict).
