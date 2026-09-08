---
name: rstsr-bench-context
description: Machine specs and rstsr benchmarking facts as of 2026-09-08 (no existing bench harness; criterion unused; nightly toolchain).
metadata:
  type: reference
---

Baseline facts for efficiency experiments, recorded 2026-09-08:

**Machine** (where experiments run):
- AMD Ryzen 9 9950X3D (Zen 5): 16 cores / 32 threads, 1 NUMA node, max boost ~5.75 GHz.
- Full AVX-512: f/dq/cd/bw/vl/ifma/vbmi/vbmi2/vnni (incl. BF16), GFNI, VAES, AVX-VNNI; BMI1/2.
- Cache: 768K L1d, 16M L2, 128 MiB L3 (X3D stacked, 2 instances).
- `RAYON_NUM_THREADS=16` (physical cores) is the convention in the workspace settings.

**rstsr benchmarking state** (as of 2026-09-08):
- No benchmark harness exists: no benches/, no [[bench]], criterion 0.5 is
  declared in workspace deps and rstsr-core dev-deps but completely unused.
- Only perf artifact: `rstsr-core/tests/tensor_sum.rs` — ad-hoc Instant timing,
  rstsr serial vs rayon vs ndarray sum_axis on [4, 512, 512] f64.
- rstsr uses a **nightly** toolchain pin (rust-toolchain.toml), MSRV 1.82.
- Fast verified test command for rstsr-core:
  `cargo test -p rstsr-core --test entry_row_cpu --no-default-features --features "backtrace row_major"`.
- 15-crate workspace; efficiency-relevant crates: rstsr-core (tensor/storage/device),
  rstsr-common (CPU layout iterators, non-SIMD), rstsr-native-impl (tensor add /
  reduction / layout-change, serial + rayon), rstsr-linalg-traits, plugin rstsr-tblis (contraction).
