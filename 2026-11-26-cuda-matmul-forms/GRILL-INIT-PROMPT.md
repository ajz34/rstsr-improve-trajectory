# Grill: CUDA matmul special forms (inner dot, gemv, syrk)

Started 2026-11-26. Grills the follow-up of the CUDA matmul implementation
(`../2026-11-25-cuda-matmul-impl`, rstsr `261021/cuda` @ `80e626d`).

## Initial prompt (user, verbatim)

> The inner dot ([k] @ [k]) is refused by every driver; the naive kernel could serve it.
>
> Well inner dot and gemv is probably some important issue that need to be done. Also, some matmuls
> can be accelarated by syrk. I think you can plan for this feature now. You see if there is any
> important decision that need my discussion. Also, I think perhaps these features do not require
> user-interface change, but the underlying driver need to be adapted.

Earlier in the same session (answer to "should syrk for `a @ a.t()` come now?"): after B6, as a
measured pre-driver step.

Standing session rules: no subagents; rstsr commits only on an explicit go; check, probe and
baseline sources stay in this directory, never in the rstsr tree; English in code and docs.

## Facts gathered before round 1 (agent, read from the repos)

### What is refused today

- `[k] @ [k]` is refused in `src/matmul/geom.rs` (`resolve_geom`): the inner-dot rule of the rule
  table leaves `la_matmul` rank 1, and the geometry raises `UnImplemented` ("single-axis
  contraction ... belongs on the reduce kernel and is not wired to it yet"). Pinned by the host
  test `inner_dot_has_no_geometry`.
- `m == 1` / `n == 1` products are **not** refused: they are ordinary rank-2 GEMMs and run on the
  chain (cuBLASLt gemm first). No cuBLAS driver calls a gemv entry. Not benchmarked yet.
- `a @ a.t()` is an ordinary GEMM with a transposed `b` layout. No driver detects the symmetry.
  Not benchmarked.

### Kernels that already exist (`rstsr-cuda`)

- `src/kernels/reduce.rs::vecdot_launch` with the classifier
  `src/kernels/reduce_plan.rs::classify_reduce`: a multi-block, lane-vectorised contraction kernel
  for uniform `T: CudaDType`, strided contracted axes, `MaybeUninit<T>` output, reached through
  `DeviceVecdotAPI` (`src/operators.rs`). It writes the raw dot: no `alpha`, no `beta * c`.
- The naive GEMM kernel serves a `1 x 1 x k` tile with **one thread**: it is correct for the inner
  dot but serial in `k` (the user's "naive kernel could serve it" is true for correctness only).
- `trace_matmul` (`src/flags.rs`) prints driver and verdict in `drivers::dispatch`.

### cudarc 0.19.10 bindings (the version in `Cargo.lock`)

**Corrected the same day: this table under-reported the bindings (a too narrow search). The survey
`RESEARCH-cublas.md` supersedes it**: gemv has batched entries, complex dots are bound, and
`cublasDotEx` is bound. What stands is the blocking result in host pointer mode. The
"setter not bound" claim was also wrong (CUDA 13 renamed it to `cublasSetPointerMode_v2`);
see the dot section of the survey.

(The table that stood here is removed; `RESEARCH-cublas.md` is the record.)

### CPU precedent (`../2026-11-18-gemv-dispatch`, branch `261123/blas2-gemv`)

- The BLAS devices gained the BLAS-2 dispatch on 2026-11-23: `m == 1` or `n == 1` goes to
  `?gemv`; `k == 1` stays gemm (outer product); non-unit-stride vectors are copied (D28, no
  `incx`/`incy` in `driver_gemv`, D27). OpenBLAS f64 cells went from 1.04–9.42x of raw
  `cblas_dgemv` to 0.92–1.07x.

### cuBLAS semantics that matter here

- `?gemv`: `y = alpha * op(A) x + beta * y`, column-major, `op in {N, T}`, `incx`/`incy` are
  free strides (no copy needed on the GPU, unlike the CPU rule); `beta == 0` never reads `y`.
- `?syrk`: `C = alpha * op(A) op(A)^T + beta * C`, **one triangle only** (`uplo`), the other
  triangle of `C` is untouched. `cublasCsyrk` uses the plain transpose (matches `a @ a.t()`);
  `cublasCherk` is the conjugate form. To be confirmed by a test, not by the doc.
- `?syrk` does n^2 k / 2 flops; a full `?gemm` on the same shapes does n^2 k. The gain exists only
  if the second triangle is filled by something cheaper than a second syrk.
- Mirroring the triangle is wrong when `beta != 0` and the old `C` is not symmetric:
  `mirror(alpha * AA^T + beta * C_old)` gives `beta * C_old[j, i]` where `beta * C_old[i, j]` is
  due.
