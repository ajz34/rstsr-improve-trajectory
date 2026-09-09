---
name: rstsr-fill-t5
description: T5 fill study at 386948be — fill_promote is NOT the creation path (vec![] is), kernel already at write floor (honest-skip), zeros laziness is size-regime dependent, rt::full rider split, rustc drift to 1.99.0; consolidation docs live in 2026-09-09-fill-misc-structural
metadata: type: project
---

T5 (2026-09-09-fill-misc-structural) PHASE 1 complete, honest-skip verdict,
no proposed.patch; rstsr tree untouched (verified clean 386948be start/end).

- **Call-chain correction (code map §2(e)/§7-8 wrong)**: `rt::full/ones/zeros`
  route to device `full_impl/ones_impl/zeros_impl` → `vec![fill; len]`
  (`device_cpu_serial/creation.rs:17-75`), NEVER through
  `fill_promote_cpu_serial` (`cpu_serial/assignment.rs:129-153`). The kernel
  serves only `c.fill(v)` (owned tensors ⇒ contig layouts), `eye`'s diagonal
  (shape [d], stride [n+1], creation.rs:829), and BLAS beta-zeroing.
- **Fill kernel is at the write floor**: reuse c.fill 2048² f64 native
  507 µs (66 GB/s w, 0.29 ins/elem) vs raw Vec loop bound 544 µs — no ≥10 %
  kernel headroom anywhere ≥ medium; small 64² gap is 0.2 µs of layout
  machinery. faer16 fill never beats serial (write BW saturates per-core;
  18 % slower at 512² where PARALLEL_SWITCH=16384 lets it parallelize).
- **rt::full rider split**: 4.65 ms = ~4.0 ms fault rider (8193 faults/iter,
  90 % sys) + ~0.5–0.6 ms broadcast kernel. T7 MALLOC tunables: 4.59 →
  0.62 ms (7.5×), beating numpy's allocating fill (1.19 ms).
- **zeros laziness is size-regime dependent** (refines "calloc-lazy" dogma):
  ≥ ~32 MiB outputs → lazy zero pages (2.3 µs @2048²); 2 MiB → calloc
  memsets explicitly (ties full); 6.2 MiB → glibc NT-memset at DRAM speed →
  zeros 2.4× SLOWER than full (107 vs 45 µs). Still "leave zeros alone".
- **Strided fill branch** (device-level only): ~2× headroom serial (iterator
  per-element cost) but no tensor-API surface → not worth a patch.
- **Toolchain identity (corrected at review)**: the campaign crates built
  with the rustup DEFAULT — stable `rustc 1.97.1 (8bab26f4f 2026-07-14)`
  recorded at T5 gate time (`results/rustc_version.txt` +
  `results/toolchain_identity.txt`). The rstsr checkout's undated nightly
  `rust-toolchain.toml` does NOT propagate along cargo path deps (rustup
  resolves from the invoking directory tree only); an earlier draft's
  "1.99.0-nightly" claim came from running rustc INSIDE the rstsr tree.
  Identity unpinned → treat cross-experiment absolute numbers as
  drift-prone; pin a dated toolchain for meta-analysis.
- Consolidation deliverables (half 2) live in the T5 dir: EDIT-GUIDE.md
  (dispatch_simd NOT-adding ADR text, symlink + fill code-map corrections,
  serial/rayon sharing assessment, MaybeUninit ergonomics, reductions reuse
  gap, THP alignment blocker, leftovers), BUG-NOTES.md (stride-0 reduce bug
  reduction.rs:226; inner_dot uninit beta serial matmul_naive.rs:233 / rayon
  :91; D8 f32-vecdot 0.57×; numpy .T.copy() anomaly), RECONCILIATION.md
  (T1–T5 hypotheses hit/miss/moot per the final summary).
- Methodology note: single perf-stat runs of 0.4–0.6 ms streaming-store
  loops swing 1.5× with frequency/order — trust criterion medians from the
  interleaved suite; use perf only for the qualitative fault/sys/ins split.
