# B3: matrix, benchmark, docs, and a corrected gate (2026-11-26)

rstsr `261021/cuda` on top of `196e348` (**uncommitted**; diff in
`patches/b3_matrix_bench_docs.patch`, 8 files, about +220/-90).

## What changed

- **Layout matrix** (`tests/test_matmul_modes.rs`): two rows, `inner dot` and `syrk`, through the
  same type x driver x order matrix as the other shapes.
- **Benchmark** (`benches/bench_matmul.rs`): `syrk` cells (`a @ a.t()`; square 2048 and 512,
  `[4096, 64]`, `[4096, 1024]`) for `f32`, `f64`, `c64`, on the chain and on the gemm alone
  (classic driver off switches the form off); `dot 2^22` and `dot 2^24`; a `GB/s` column (memory
  traffic of operands and output).
- **Size gate replaced by a flag** (see below): `MatMulUseSyrk { Allow, Scientific, Disallow }`
  on `CudaRuntimeFlags::matmul_use_syrk`, default `Scientific`; no size gate.
- Module docs updated; the look-alike tests moved above the floor so that only the layout tells
  them apart from the form.

## The gate was wrong

B0 measured square operands only (`k = n`). The benchmark's `[4096, 64]` `f32` cell showed the
**form 1.7x slower** than the gemm it replaced. A sweep on a grid of `n` in 512..8192 and
`k` in 32..2048 (`results/b3_syrk_gate.txt`, two `f32` runs, cold and warm) gives:

| `f32` ratio gemm / chain | k=32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| n = 2048 | 0.44 | 0.63 | 0.78 | 0.91 | 0.97 | 1.09–1.14 | 1.58 |
| n = 4096 | 0.41 | 0.50 | 0.64 | 1.12 | 1.50 | 1.73 | 1.66 |
| n = 8192 | 0.45 | 0.56 | 0.70 | 0.90 | 1.06–1.09 | 1.27 | **0.80** |

- Below 1 the form loses. The loss on thin operands is up to 2.3x; `(8192, 2048)` repeats at 0.80.
- `f64`: wins on the whole grid, 1.5–1.9x. `c32`: 1.3–3.6x. `c64`: 1.4–1.6x.
- Small sizes (`results/b3_syrk_gate_small.txt`, `n` in 32..256): `f64` loses below 256 (0.63–0.98,
  2–5 us), `c64` loses up to `n = 256` (0.81–0.94), `c32` mostly wins by about 1 us.

The rule shipped in B2 (`f32` from `n >= 2048` on) therefore made thin `f32` products up to
2.3x **slower**.

**Settled by the maintainer: no size gate.** The types are chosen by a new flag on
`CudaRuntimeFlags`:

```rust
pub enum MatMulUseSyrk { Allow, Scientific, Disallow }   // default: Scientific
```

- `Scientific` (default): `f64`, `c32`, `c64` take the form; `f32` stays on the gemm chain.
- `Allow`: `f32` as well (the user opts into the thin-operand losses).
- `Disallow`: never.

Constants `MIN_N` / `F32_RANGE` and `syrk::gated` are gone; `syrk_applies` reads the flag.
Under the default the small-size and thin-`f32` losses are both out of the picture; `f64` and
`c64` below `n = 256` still lose a few microseconds (the mirror launch), which the maintainer
accepted.

Routing checked by trace for `n` in {511, 512, 2048, 4096} with the gate build
(`results/b3_syrk_trace.txt`); with the flag the form is taken for `f64`/complex at every size.

## Benchmark (`results/b3_bench_matmul.txt`, RTX 5070 Laptop, median of 5)

| triple | shape | chain (form) | gemm | speed-up |
|---|---|---|---|---|
| `f32` | syrk 2048 | 0.888 ms (`Allow`) | 1.431 ms | 1.61 |
| `f32` | syrk 512 | 0.060 ms (`Allow`) | 0.037 ms | 0.62 (lost) |
| `f32` | syrk 4096x64 | 0.531 ms (`Allow`) | 0.297 ms | 0.56 (lost) |
| `f32` | syrk 4096x1024 | 1.956 ms (`Allow`) | 3.531 ms | 1.81 |
| `f64` | syrk 2048 | 28.2 ms | 52.7 ms | 1.87 |
| `f64` | syrk 4096x64 | 3.82 ms | 6.53 ms | 1.71 |
| `f64` | syrk 4096x1024 | 53.6 ms | 101.0 ms | 1.88 |
| `c64` | syrk 2048 | 106.2 ms | 162.3 ms | 1.53 |
| `c64` | syrk 4096x1024 | 206.8 ms | 325.7 ms | 1.57 |

`f64` dot: `2^22` 0.030 ms (2.2 TB/s, in cache), `2^24` 0.376 ms (714 GB/s, DRAM).

`f32` rows: under the default `Scientific` the chain equals the gemm; the "form" column is the
chain under `Allow`. The losses at the two small/thin cells are the reason `f32` is not in the
default.

## Gates

| gate | result |
|---|---|
| `cargo test -p rstsr-cuda --features faer -- --test-threads=1` (nvcc 13.2) | 241 passed (`results/b3_full_suite.txt`) |
| `CUDARC_CUDA_VERSION=12060 cargo check -p rstsr-cuda --all-targets --features faer` | ok |
| `cargo clippy -p rstsr-cuda --all-targets --features faer` | no warning in `rstsr-cuda` |
| `cargo doc -p rstsr-cuda --no-deps --features faer` | ok, no warning |
