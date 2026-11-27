# 2026-11-26-cuda-matmul-review

Review of the rstsr-cuda matmul implementation on `261021/cuda`: the eleven commits
`c75aecd..67a0c79`, all in `crates-device/rstsr-cuda/`. The rstsr tree was not modified.

- rstsr commit: `67a0c79d77cd4b2dd188b29c5b9531e3016b6108`
- environment: NVIDIA GeForce RTX 5070 Laptop GPU (8 GiB, sm_120), driver 595.99.02,
  CUDA 13.2 (`cuda_13.2.r13.2`), rust nightly, rstsr's rustflags
  (`target-cpu=native`, `-fp-contract=fast`), cudarc 0.19.10.
- method: two-axis review (standards, spec fidelity) by sub-agents reading the final sources
  against the rules and the plans, plus a correctness pass whose claims were reproduced with
  the probe crate in `probe/`.

## Conclusion

The driver chain, the layout algebra and the complex, batched, degenerate and strided cases
are correct on the GPU, and the crate's suite is green (10 binaries, 242 tests). Three things
need fixing before merge; details, evidence and fixes are in [REVIEW.md](REVIEW.md).

1. **Critical, FIXED 2026-11-26 (uncommitted on `261021/cuda`):** an operand broadcast along
   a matmul axis (stride 0, extent > 1) gave silently wrong results on cuBLASLt, `cublasGemmEx`,
   classic cuBLAS, the default chain and the symmetric form, because `cublas_args` clamped a
   zero leading dimension to the row count instead of refusing it. Both `args()` functions now
   refuse `stride <= 0` with extent > 1; the chain serves the shapes through tiled/naive.
2. **High, FIXED 2026-11-26 (same changeset):** in-place matmul across two streams returned
   stale data: every matmul driver called `order_output` before the launch, while the helper
   records its event at call time. All five drivers now order `c` as an input before the
   enqueue and its allocation stream behind the enqueue. Probe T5: 0 of 30 stale (was 30/30).
3. **Medium, FIXED 2026-11-27 (uncommitted on `261021/cuda`):** the symmetric form ignores the
   compute mode. Measured on the way: cuBLAS's `?syrk` entries ignore a math mode that changes
   the arithmetic (`?syrk` under TF32 is plain `f32`), so the form now declines such modes and
   the chain serves them on a `gemm`; `Default`/`Pedantic` keep the form with the mode set and
   restored around the call (`Pedantic` form 3.46 ms vs pedantic `gemm` 6.44 ms). Under
   `EmulatedFixedPoint`, `a @ a.t()` now runs the emulated `gemm` at 1.01 ms (was 3.45 ms), and
   `fixedpoint_mantissa_bits = 30` errors identically on both forms (was: a silent result).
4. **Medium, FIXED 2026-11-27 (same changeset):** `build.rs` never recognised a pinned
   `CUDARC_CUDA_VERSION` (parses six characters, cudarc spells five), so the recommended
   `=13000` build lost the math modes and the emulation. It now parses cudarc's
   `{major}0{minor}0` spelling (unit-tested by including `build.rs` into the lib test build);
   the `13000` pin yields both cfgs and passes the whole hardware suite; `tier3-cuda.yml` runs
   `--print cfg` checks for both pins via a CI matrix.
5. **Low, FIXED 2026-11-27 (uncommitted on `261021/cuda`):** `[0] @ [0]` failed with
   `INVALID_VALUE` on `cublasGemmEx` and the classic entries (the lifted `1 x 0` operand declared
   `ld = 0`; cuBLAS wants `ld >= max(1, rows)` even for an empty matrix) and a leading dimension
   over `i32::MAX` was clamped instead of refused. Both derivations now floor every declared
   `ld` at 1 and refuse strides that do not fit `i32`; probe T4 serves `dot = 0` on every driver
   set (499 `ok`, the two `k = 0` ERRs gone).
6. **Low, FIXED 2026-11-27 (same changeset):** the two batch attribute sets of the cuBLASLt
   driver dropped their status (a failure would have left a batch-1 layout — only the first
   slice written); they now decline like every sibling `SetAttribute`. `build` takes the derived
   `CublasArgs` directly, and `batch_attributes_read_back` pins the batch attributes of a built
   entry (the failure itself is not triggerable through a public path).

Test gaps that let the first two through: no cuBLAS test with a broadcast operand; complex
tests with purely real data; the symmetric form never run on a non-base offset; nothing
observes routing. The two broadcast tests and the cross-stream test now close part of that
(`broadcast_along_matmul_axis`, `inplace_cross_stream_is_ordered`, plus unit tests in
`cublas_args.rs` and `syrk.rs`); the complex-imaginary and routing gaps stay open with C3+.

## Numbers (probe T3, n = k = 1024, `f64`, mean of 5 launches)

Before (the findings) and after (`results/t3_after_c3c4.txt`):

| flags | `a @ b` | `a @ a.t()` before | `a @ a.t()` after |
|---|---|---|---|
| `EmulatedFixedPoint`, syrk `Scientific` (default) | 1.044 ms | 3.449 ms | **1.013 ms** |
| `EmulatedFixedPoint`, syrk `Disallow` | 1.045 ms | 1.010 ms | 1.010 ms |
| `Default` or `Pedantic`, syrk `Scientific` | 6.419 ms | 3.448 ms | 3.460 ms (form kept) |

## Files

- `REVIEW.md` — the findings (summary table, correctness, test coverage, API soundness,
  standards, spec fidelity, probe index).
- `probe/` — the probe crate (`tests/probe.rs`, one test per group; prints `ok`, `FAIL`, `ERR`,
  `ref` lines). Path dependencies on the rstsr checkout next to this repository.
- `results/probe.log` — probe output before the C1/C2 fixes (479 `ok`, 66 `FAIL`)
- `results/probe_after_fix.log` — after the C1/C2 fixes (497 `ok`, 0 `FAIL`, T5 0 of 30 stale)
- `results/probe_after_c3c4.log` — after the C3/C4 fixes (497 `ok`, 0 `FAIL`, same 28
  intentional forced-tiled-complex ERRs)
- `results/probe_after_c5c6.log` — after the C5/C6 fixes (499 `ok`, 0 `FAIL`: the two `k = 0`
  ERRs are now `dot = 0` on every driver set)
- `results/probe_after_c7.log` — after the C7 fix (unchanged: 499 `ok`, 0 `FAIL`, 28
  intentional forced-tiled-complex ERRs)
- `results/t3_after_c3c4.txt` — the T3 timing/error lines after the C3 fix
- `results/cfg_CUDARC_CUDA_VERSION_13000.txt` — before the C4 fix (neither cfg — the bug)
- `results/cfg_*_after_fix.txt` — after the C4 fix: `13000` → both cfgs, `12060` → neither
  (correct for 12.6), nvcc 13.2 → both
- `results/rstsr_cuda_full_suite_after_fix.log` — the gate after the fixes (109 lib + hw, green)
- `results/probe.log` — probe output (479 `ok`, 66 `FAIL` = the two broadcast findings,
  30 `ERR` = 28 expected tiled refusals of complex types + 2 `k = 0` refusals, 8 `ref`).
- `results/rstsr_cuda_full_suite.log` — the crate's suite:
  `RUST_BACKTRACE=1 cargo test -p rstsr-cuda --features faer -- --test-threads=1`.
- `results/cfg_CUDARC_CUDA_VERSION_13000.txt`, `results/cfg_nvcc_13_2.txt` —
  `cargo rustc -p rstsr-cuda --lib -- --print cfg` with and without the pin.
- `results/build_CUDARC_CUDA_VERSION_12080.txt` — the same pin, full build.

## Rerunning the probe

The probe is not in the rstsr workspace; it needs the same toolchain and rustflags to reuse
the workspace's build cache:

```sh
cd 2026-11-26-cuda-matmul-review/probe
mkdir -p .cargo && printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "llvm-args=-fp-contract=fast"]\n' > .cargo/config.toml
cp ../../../rstsr/Cargo.lock .            # optional; keeps dependency versions identical
CARGO_TARGET_DIR=../../../rstsr/target cargo test --features rstsr-cuda/faer --test probe -- --nocapture --test-threads=1
```

`target/`, `Cargo.lock` and `.cargo/` are ignored by this repository's `.gitignore`.

Pitfalls met while writing the probe, for the next one:

- Under the workspace's default features the CPU device's default order is **row-major**;
  create `DeviceCpuSerial::default()` and `set_default_order(order)` per order when the
  reference has batched shapes.
- A device value's flags are read from the **tensor's** device, at launch: set
  `trace_matmul` on the device the tensors are created with, not on a later clone.
- `CudaRuntimeFlags` is `#[non_exhaustive]`: `let mut f = CudaRuntimeFlags::default(); f.x = ..;`.
- Forced single drivers decline layouts they cannot express (the transposed output view needs
  the whole chain); such refusals are the chain's normal fall-through, not findings.
