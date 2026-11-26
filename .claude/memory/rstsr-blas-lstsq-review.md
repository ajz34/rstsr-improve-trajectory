---
name: rstsr-blas-lstsq-review
description: "2026-11-26 two-axis review of the BLAS lstsq feature (commit 182e38d): fixes 04f7a12, owner-reverted span narrowing dc2274f, in-place consumption + LAPACKE write-back 2dc1be0 with one accepted divergence and its measured cost; owner decisions to reuse"
metadata:
  type: project
---

Branch `261126/lstsq-review-fixes` in `/home/a/rstsr_pack/rstsr` (off
`182e38d`, not pushed). Record:
`2026-11-25-blas-lstsq/IMPLEMENTATION.md` (review follow-up section).

**Commits**
- `04f7a12` review fixes: `(0, 0)` / `(m, 0)` corners return, short `b` is
  an error (shared `split_rhs`), 16 input-kind pairs, GELSD under the device
  BLAS thread policy, error messages, docstrings, tests.
- `dc2274f` owner reverted the narrowed raw-slice spans of `04f7a12`.
- `2dc1be0` owned/mutable `a` consumed in place when it needs no broadcast
  and has the call order (destroyed); row-major bridge writes the destroyed
  `a` back like `LAPACKE_dgelsd_work`.

**Owner decisions (reuse, do not re-ask)**
- Thunks follow LAPACKE literally ("direct translation from lapacke"):
  check `~/Git-Others/lapack/LAPACKE/src/*_work.c` before changing one.
- Plain `rows * ld` raw-slice spans; no span helpers ("complicate code").
- Separate commits over amend; a revert is its own commit.
- Accepted divergence: a failed call (`info != 0`) skips the output
  write-back, LAPACKE writes back unconditionally (skill §3, §6).
- Destructive entries destroy only an in-order owned/mutable input; an
  off-order one is factored from a work buffer and left untouched (no
  `clone_to_mut`), a broadcast one is copied.

**Cost of the write-back** (rough probe, release, OpenBLAS 0.3.32, row-major
bridge, view input): +3–5% for 64×32, 64×4096, 1024×512; +22% for 4096×64.
A mutable input saves a copy that roughly pays for it.

**State at the time of writing**
- rstsr working tree: uncommitted docstring fix in `map_gelsd_info` — the
  NaN → `info = -4` check is netlib master only (`3d90a787f`, 2026-11-06,
  after 3.12.1, in no release).
- rstsr-agents `main`: uncommitted `lapacke-translation` skill additions
  (gelsd rows: destructive `a`, zero-size synthesis, `b` bridge sizing,
  workspace table, NaN info mapping, failed-call write-back).
- Test pitfalls met on the way: `blas-layout-flag-logical-values-pitfall`.
