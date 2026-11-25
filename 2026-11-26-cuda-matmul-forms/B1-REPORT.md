# B1: the inner dot as a GEMM (2026-11-26)

rstsr `261021/cuda`: committed as `bc4c8d9` (alignment fix) and `a7c65aa` (inner dot), on top of
`80e626d` (diff of the two combined in
`patches/b1_inner_dot_gemm.patch`, 10 files, +167/-47).

## What changed

- `matmul/geom.rs`: `lift_inner_dot` turns `[k] @ [k] -> []` into `1 x k @ k x 1 -> 1 x 1` before
  the rule table is asked (a size-1 axis on each operand and on the output, strides as a
  contiguous matrix would have them). The "single-axis contraction" refusal is gone. Host tests:
  the lifted geometry addresses both vectors at their own strides in both orders; other ranks
  are not lifted.
- No driver code. `alpha`/`beta`, the driver flags, the trace and mixed dtypes come from the chain.
- `tests/test_matmul_modes.rs::inner_dot`: `f32`, `f64`, `c32`, `c64`; Lt, `cublasEx`, cuBLAS,
  naive and the chain, each alone; both orders; contiguous and stepped vectors; a fresh rank-0
  output and the `alpha`/`beta` path into a caller's rank-0 output; reference = the CPU device.
- `tests/test_kernels.rs`: the refusal test is removed.
- Docs: `matmul/mod.rs`, `linalg.rs` (the "asymmetry with `tensordot`" section is gone),
  `drivers/naive.rs`.

## Found and fixed on the way: cuBLASLt reads 16-byte scales with aligned loads

The new `c64` cell **segfaulted inside `libcublasLt`** in a debug build and passed in release.

- Faulting instruction: `movdqa (%rdx), %xmm1`, a 16-byte aligned load, of `alpha`
  (`Complex<f64>`, 16 bytes, alignment 8).
- We passed `&alpha` of a stack local. Its 16-byte alignment was chance: yes in the release
  build and in the earlier matrix (debug), no for this caller in debug. Every `c64` cuBLASLt
  matmul was exposed; the fault is on the host, so no status could report it.
- Fix: `AlignedScale<T>` (`#[repr(C, align(16))]`) in `matmul/drivers/mod.rs`; all three cuBLAS
  drivers hand `alpha`/`beta` through it (the Ex and classic entries take them by pointer as
  well). Host test `scales_are_aligned_for_cublas` pins the alignment.
- Diagnosis trail: standalone reproduction (`lt-dot-repro` in `src/probe_forms.rs`) passed in
  release in both orders and crashed in debug; the cuBLASLt log (`CUBLASLT_LOG_LEVEL=5`) showed
  valid descriptors; `gdb` gave the instruction.

## Gates

| gate | result |
|---|---|
| `cargo test -p rstsr-cuda --features faer -- --test-threads=1` (nvcc 13.2) | 232 passed (`results/b1_full_suite.txt`) |
| `CUDARC_CUDA_VERSION=12060 cargo check -p rstsr-cuda --all-targets --features faer` | ok (`results/b1_check_12060.txt`) |
| `cargo clippy -p rstsr-cuda --all-targets --features faer` | no warning in `rstsr-cuda` |
| `cargo doc -p rstsr-cuda --no-deps --features faer` | ok, no warning (`results/b1_doc.txt`) |

Test count: 230 before; +2 host tests, +1 device test, -1 removed refusal test.

## Open

- On the naive driver the dot is one thread (a `1 x 1` tile), as for any `1 x 1` product; only
  when that driver is forced.
- The alignment fix deserves its own line in the PR description: it changes every `c64` Lt call.
