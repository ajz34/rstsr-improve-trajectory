# Machine & tooling facts — recorded 2026-09-08

Environment facts for the efficiency campaign. Verify with a quick command if a
session is much later than 2026-09-08.

## CPU

- AMD Ryzen 9 9950X3D (Zen 5): 16 cores / 32 threads, 1 NUMA node, max boost ~5.75 GHz.
- SIMD: full AVX-512 (f, dq, cd, bw, vl, ifma, vbmi, vbmi2, vnni incl. BF16),
  GFNI, VAES, AVX-VNNI, BMI1/2. `target-cpu=native` ≈ x86-64-v4 + AVX-512.
- Peak vector throughput: 2× 512-bit FMA pipes → **32 double-precision FLOP/cycle/core**
  (FMA = 2 ops). Use as the analytic ceiling for compute-bound benches (vecdot);
  for memory-bound ops, **measure** achievable bandwidth with a triad/memcpy
  microbench instead of trusting nominal DRAM numbers.
- Actual caches: 48 kB L1d / core, 1 MB L2 / core (16 MB aggregate), 128 MiB L3
  (X3D stacked, 2 instances).
- **Design budget rule (from the task owner): optimize for ≤32 kB L1 and ≤256 kB
  L2; do not truly optimize for L3.** Cache-aware blocking is a last resort,
  often overkill for these ops.

## Toolchain

- rstsr repo pins `channel = "nightly"` (no date); observed active rustc
  `1.97.1 (2026-07-14)`. MSRV of rstsr is 1.82. Record `rustc --version` in
  every experiment README.
- Threads convention: `RAYON_NUM_THREADS=16` (physical cores) for parallel benches.

## Profiling / measurement tools

- `perf` 7.0.14 at `/usr/bin/perf` — primary tool (`perf stat -d`, `perf record/report`).
- valgrind/cachegrind: **not installed**. cargo-flamegraph: **not installed**
  (installable later if wanted; no sudo assumed).
- numpy/torch reference runs need the conda env: `conda activate torch` →
  numpy 2.5.1, torch 2.14.0+rocm7.2. (Plain `python3` has no numpy.)

## Reference sources at `~/Git-Others`

Local checkouts useful for kernel design (read C sources for blocking /
vectorization patterns):

- `numpy` (229 MB; C loops incl. einsumsimd), `scipy`
- `OpenBLAS`, `blis` — blocked/cache-aware kernel references
- also: candle, pyscf, gpu4pyscf, libxc

## lightweight-simd crate

- crates.io `lightweight-simd` v0.1.1 (2026-09-07) by ajz34 — fixed-array
  vector types relying on compiler auto-vectorization (NOT true SIMD intrinsics).
- Repo: github.com/ajz34/lightweight-simd-rs. Features: `complex`, `half`,
  `use_libm_fma`. MSRV 1.82.
- **No docs.rs build yet** — read the GitHub repo for exact type names
  (expected F64x8/F32x16-style fixed arrays) before writing dispatch code.
- Not referenced anywhere in rstsr at commit 386948be.

## Disk hygiene

- Rust `target/` directories inside **this repo** (experiment crates) are
  deleted when the experiment finishes; `Cargo.lock` stays (committed).
- `target/` inside `../rstsr` is left as-is (rebuilds of the 15-crate workspace
  are expensive); remove only under disk pressure, noting it in the README.
