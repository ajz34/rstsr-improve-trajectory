---
name: rstsr-lapacke-layer-grill
description: 2026-10-11 grill closed on the LAPACKE-to-rust translation layer; 14 decisions in 2026-10-11-lapacke-layer/DECISIONS.md; execution awaits user go
metadata:
  type: project
---

Grill on the rstsr-blas-traits LAPACKE translation layer closed 2026-10-11
(5 rounds; 4 comprehension + 2 fact agents). Full decision set:
`2026-10-11-lapacke-layer/DECISIONS.md`; fact base `FINDINGS.md` in the same
directory. Execution NOT started — awaits explicit user go.

Key settled points: f-prefer views pass zero-copy with derived ld, everything
else clones (fully-non-contig = our extension beyond LAPACKE); hybrid output
order (fresh = device default / in-place = input buffer's order); per-device
orderchange bridge replaces `_cpu_serial` kernels in driver files; trait
`driver_xxx` signatures unchanged this campaign, per-file `pub fn xxx_work`
thunks; LAPACKE-faithful workspace table + O(1) hardening (MAX(1,·),
lwork==0→1, checked casts, f32/c64 nextafter, no fudge); raw info at thunk +
builder-side error mapping (no info−1 remap, no −1010/−1011 sentinels);
ipiv 0-based at builder (sysv fix pending); gesvd keeps superb; zero-size =
identity sub-blocks synthesized; no nan check this campaign; lapack/lapacke
stay mutually exclusive, lapack primary; OpenBLAS-only validation; tests ≤20
per side with 3-case layout matrix + zero-size.

Two parked items: (1) multi-threaded transpose policy = the most severe open
problem, user said ignore for now, agent may re-ask; (2) the user's
`_adv_` trait shape (defaulted `driver_xxx` → `driver_adv_xxx(nthreads,
nan_check, ..)`) evaluated sound but parked as a SEPARATE structural task —
this campaign is technical translation refactor only.

Deliverables when go comes: skill `lapacke-translation` (rstsr-agents) +
ADR-0009 (rstsr-book) + rstsr branch `261011/…`, each repo on its own branch,
part-by-part with review stops, no auto-commit in rstsr repos.
