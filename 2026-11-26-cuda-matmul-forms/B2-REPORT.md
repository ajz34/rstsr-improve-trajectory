# B2: `a @ a.t()` as `?syrk` (2026-11-26)

rstsr `261021/cuda` on top of `a7c65aa` (**uncommitted**; diff in `patches/b2_syrk.patch`,
6 files, about +560/-40).

## What changed

- `matmul/geom.rs`: `CallerGeom::is_syrk` (pure): `b` is `a` with reversed strides at the same
  offset, unbatched, square output. Host test.
- `matmul/drivers/syrk.rs` (new):
  - `args(view)`: the `?syrk_v2` call from the caller's layouts, or `None` when a layout has no
    column-major expression (a stepped operand or output, a negative stride). The output may be
    presented to cuBLAS transposed (`ldc` = the row stride); since the product is symmetric that
    only swaps which triangle is written, recorded in `lower`.
  - `gated::<T>(n)`: `f32` only from `n >= 2048` (`F32_MIN_N`); `f64` and complex always.
  - `run`: one `?syrk_v2` with `beta = 0`, `FILL_MODE_UPPER`, then `rstsr_matmul_mirror`, a
    small NVRTC kernel copying the written triangle across the diagonal.
- `matmul/drivers/cublas.rs`: `CublasScalar::syrk` (the four typed `?syrk_v2` entries).
- `matmul/drivers/mod.rs`: `dispatch` tries the form first when `syrk_applies`: the layout
  condition, the same buffer (`std::ptr::eq` of the device pointers), `beta == 0`, one type,
  unbatched, the gate, and the classic driver enabled (it owns the routine). A declined form
  falls through to the chain unchanged. The trace prints `cublas (syrk)`.
- `matmul/mod.rs`: a "Symmetric products" section of the module docs.
- `tests/test_matmul_forms.rs` (new): `f32`, `f64`, `c32`, `c64` against the CPU device on the
  chain, the classic driver alone and the chain without it, both orders, a view inside a larger
  buffer; the look-alikes (`beta != 0`, `a @ a.h()`, a copy of the transpose, a batch, a stepped
  operand) against the CPU device in both orders; the `f32` gate at `n = 2047` and `2048`.

## Routing, checked with the trace (`syrk-trace` in the probe)

| type | `n = 37` | `n = 2048` |
|---|---|---|
| `f32` | cuBLASLt | cublas (syrk) |
| `f64`, `c32`, `c64` | cublas (syrk) | cublas (syrk) |

Both orders.

## Found on the way

- **Mirror direction**: the first version copied the *empty* triangle over the filled one (the
  condition was written backwards): the diagonal was right and both off-diagonal halves zero.
  Separating the steps (`syrk` without the mirror) showed which triangle cuBLAS had written.
- **Batch axis under `ColMajor`**: the rule table places the batch axis last (shape
  `[n, k, p]`, batch stride `n * k`). A test written with the batch leading is refused by
  `resolve_geom` itself. Not a bug; recorded here because it is easy to get wrong.
- `asarray(...).into_shape(shape)` keeps the logical shape and only the strides follow the
  default order: data for a column-major tensor has to be fed in column-major positions.

## Gates

| gate | result |
|---|---|
| `cargo test -p rstsr-cuda --features faer -- --test-threads=1` (nvcc 13.2) | 241 passed (`results/b2_full_suite.txt`) |
| `CUDARC_CUDA_VERSION=12060 cargo check -p rstsr-cuda --all-targets --features faer` | ok (`results/b2_check_12060.txt`) |
| `cargo clippy -p rstsr-cuda --all-targets --features faer` | no warning in `rstsr-cuda` |
| `cargo doc -p rstsr-cuda --no-deps --features faer` | ok, no warning (`results/b2_doc.txt`) |

Test count: 232 before; +4 host tests, +5 device tests.

## Open

- The mirror is a separate kernel launch (one more enqueue per product); at the sizes where the
  form pays it is negligible, at tiny sizes the `f32` gate keeps the form off anyway. `f64` and
  complex at tiny `n` pay one extra launch against a cuBLASLt call of a few microseconds.
- No batched form (cuBLAS has no batched `syrk` entry); recorded in PLAN as a follow-up.
- `F32_MIN_N = 2048` comes from the B0 probe on one GPU (RTX 5070 Laptop); the matrix (B3) will
  show whether it holds elsewhere.
