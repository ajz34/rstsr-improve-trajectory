---
name: rstsr-alloc-pagefault-t7
description: T7 findings — the 32 MiB glibc mmap-threshold cap causes ~4 ms fault rider on large fresh outputs; existing reuse APIs (op_mutc_refa_refb_func, assign, fill) recover it; MALLOC_* env pair recovers ~96-97%.
metadata:
  type: project
---

T7 alloc-pagefault study (2026-09-09, dir `2026-09-09-alloc-pagefault-study`,
rstsr 386948be) durable findings:

- **Mechanism**: glibc's dynamic mmap-threshold adaptation caps at
  DEFAULT_MMAP_THRESHOLD_MAX = 32 MiB (LP64). A 2048x2048 f64 output (32 MiB
  + chunk header + 64-B align padding) sits just ABOVE the cap -> mmap/munmap
  per alloc/free -> 8193 minor faults per fresh 32 MiB output, sys-time
  bound. Medium (2 MiB) and odd (6.2 MiB) sizes adapt -> rider 0-5%. Small
  outputs: exactly 0. **The rider is a >32 MiB-fresh-output phenomenon only.**
- **Magnitude at 2048²**: ~4 ms fixed per fresh output = 66% of add contig
  serial, 77% faer16; 88% of `rt::full`; 25% (serial) / 57% (faer16) of
  transpose copy. Same fault count on faer16 (8225/iter) — rayon does not
  change it.
- **Existing reuse APIs at 386948be** (all correctness-gated, benched):
  `op_mutc_refa_refb_func(c, a, b, &mut |c,a,b| c.write(a+b))` in
  `rstsr-core/src/tensor/operators/op_with_func.rs` — public single-pass
  c<-a+b into existing storage, any layout, both devices (the reuse
  workhorse); `c.assign(&a.t())` for transpose-copy reuse; `c.fill(v)`;
  `c += &b`. No reduce-into API exists (sum_axes allocates).
- **Env remedy**: `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`
  recovers 96-97% (6.18→2.27 ms serial, 2.70→0.72 ms faer16). Trap:
  MMAP_THRESHOLD_=33554432 does nothing (chunk > threshold still mmaps);
  threshold alone only halves it (trim refaults).
- **THP trap**: THP is `madvise` on this machine; rstsr's 64-B-aligned
  buffers are NOT page-aligned (glibc memalign trim) -> madvise(MADV_HUGEPAGE)
  fails EINVAL. With 2 MiB-aligned alloc + hint, fresh-alloc fill 4.20→0.68
  ms/iter (18 faults). THP integration requires an alignment change first.
- **Denominator rule for T1-T5**: judge kernel wins on reuse-variant (kernel
  only) numbers for large allocating ops; the ~4 ms rider masks kernel
  deltas at 2048² otherwise. `rt::zeros` must stay calloc-lazy (2.4 µs).
