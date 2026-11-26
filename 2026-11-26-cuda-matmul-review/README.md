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

1. **Critical:** an operand broadcast along a matmul axis (stride 0, extent > 1) gives
   silently wrong results on cuBLASLt, `cublasGemmEx`, classic cuBLAS, the default chain and
   the symmetric form, because `cublas_args` clamps a zero leading dimension to the row count
   instead of refusing it (pre-existing in `cublas_args.rs`, now reached by three drivers).
2. **High:** in-place matmul across two streams returns stale data: every matmul driver calls
   `order_output` before the launch, while the helper records its event at call time; the
   elementwise kernels call it after the launch. 30 of 30 probe iterations stale.
3. **Medium:** the symmetric form ignores the compute mode (`a @ a.t()` is 3.3x slower than
   `a @ b` under `EmulatedFixedPoint`, `Pedantic` is not applied, the documented mantissa
   error is bypassed), and `build.rs` never recognises a pinned `CUDARC_CUDA_VERSION`
   (parses six characters, cudarc spells five), so the recommended `=13000` build loses the
   math modes and the emulation.

Test gaps that let the first two through: no cuBLAS test with a broadcast operand; complex
tests with purely real data; the symmetric form never run on a non-base offset; nothing
observes routing.

## Numbers (probe T3, n = k = 1024, `f64`, mean of 5 launches)

| flags | `a @ b` | `a @ a.t()` |
|---|---|---|
| `EmulatedFixedPoint`, syrk `Scientific` (default) | 1.044 ms | 3.449 ms |
| `EmulatedFixedPoint`, syrk `Disallow` | 1.045 ms | 1.010 ms |
| `Default` or `Pedantic`, syrk `Scientific` | 6.419 ms | 3.448 ms |

## Files

- `REVIEW.md` — the findings (summary table, correctness, test coverage, API soundness,
  standards, spec fidelity, probe index).
- `probe/` — the probe crate (`tests/probe.rs`, one test per group; prints `ok`, `FAIL`, `ERR`,
  `ref` lines). Path dependencies on the rstsr checkout next to this repository.
- `results/probe.log` — final probe output (479 `ok`, 66 `FAIL` = the two broadcast findings,
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
