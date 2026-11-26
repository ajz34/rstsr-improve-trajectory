---
name: cuda-matmul-impl-grill
 description: 2026-11-25 grill closed (D1-D28), B1 tiled check done for rstsr-cuda matmul — driver chain Lt>Ex>cuBLAS>tiled>naive, output-keyed compute types, MatMulDrivers flags replacing env vars, src/matmul/ layout; impl on 261021/cuda head awaits go
metadata:
  type: project
---

Plan (batches B1-B6) in `2026-11-25-cuda-matmul-impl/PLAN.md`; decisions in `DECISIONS.md` there; B1 tiled check DONE (`TILED-CHECK.md`):
70/70 accepted cases exact in f32/f64; tiled 1.4-3.0x behind cuBLAS in f32 but 13-17% FASTER in
f64 on the RTX 5070 Laptop (both at the GeForce 0.33-0.38 TF fp64 rate); refusals are restrictions
only (lane_axis rejects stride -1; two-axis broadcast batches are non-affine). Test and timing example live ONLY in the task dir `src/` (owner: do not leave check files in the rstsr tree); B2 awaits go.

- Owner's model: naive driver accepts **every** type triple and layout; cuBLASLt/Ex/cuBLAS are only
  the accelerated drivers, tried in that order; tiled SIMT kernel sits before naive.
- Compute types are keyed by **output** type: `ComputeType32F/64F/16F`, one flags field each; FP64
  fixed-point emulation is reachable only through the classic handle (cuBLAS driver only).
- `f16 x f16 -> f32` needs no core change: `matmul_from` has no `Mul` bound; `rt::matmul`/`%`
  infer f16 from `half`'s `Mul`.
- D29 tensordot fully deferred (cuTENSOR later); D30 tiled `lane_axis` fix for stride -1 is verified
  but DEFERRED by the owner — B2 moves the tiled kernel unchanged. Check sources stay in the task
  dir `src/` only, never left in the rstsr tree.
- No environment variable for op behaviour: driver selection is `CudaRuntimeFlags.matmul_drivers`
  (bit set; forcing = one bit). Kernel-build envs (`ARCH`, `STRICT_FP`, `CACHE*`) stay.
- `matmul/mod.rs` must stay an import/re-export file; **no new FFI**, only cudarc 0.19 (its
  `cublaslt` feature carries the Lt layer). Work on the `261021/cuda` head, no new branch.
- Fact base: CuPy materializes operand promotion and keys compute type per dtype; PyTorch refuses
  mixed operand dtypes, Lt-then-GemmEx fall-through.

Update 2026-11-25 (B2 done, committed as rstsr `c75aecd`, `B2-REPORT.md`): `src/matmul/{mod,mode,flags,geom,cublas_args,inner}.rs`
+ `drivers/{mod,cublas,tiled,naive}.rs`; `MatMulDrivers` bit set replaces the env vars; `DeviceCuda::with_flags`;
`DeviceMatMulAPI` generic over `(TA,TB,TC)` with run-time `TypeId` re-typing (`CudaBuffer::view_as`) for the
single-type drivers. Pitfalls learned: under `ColMajor` the matrix axes are the LEADING axes (`geom.m/n` are
row-major roles; use `GemmGeom::caller()`), the CPU serial device refuses a strided `c` in `matmul_from` and
panics on integer overflow in debug (device wraps), `#[non_exhaustive]` flags means `default()` + field set.
Open for the owner: inner dot still refused; `bf16 -> bf16` mode family (16F vs D18's 32F) for B5.

Complex matmul (c32/c64) runs on the naive driver since B2 (did not compile on CUDA before); it is in `test_matmul_naive` and the check matrix (`naive_c32`/`naive_c64`). No accelerated complex driver until B5.

Update 2026-11-25 (B3 done, rstsr working tree awaits go, `B3-REPORT.md`): `drivers/cublas_ex.rs` (GemmEx /
GemmStridedBatchedEx, compute type per mode), classic driver takes the mode as a handle **math mode**
(set/restore inside the handle lock), table per `(triple, mode)` in a stack `Chain`. Facts learned:
- cudarc gates bindings by CUDA version: `cublasSetMathMode` >= 12.9, `CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH`
  >= 13.0, no `cublasSetFixedPointEmulationMaxMantissaBitCount` at all; a dep's features are invisible to
  `cfg`, so rstsr-cuda got a `build.rs` emitting `cfg(cublas_math_mode)` / `cfg(cublas_fixedpoint_math)`
  from `CUDARC_CUDA_VERSION` else `nvcc`. CI pins 12060 (old bindings): test both builds.
- On sm_120: FastTF32/Fast16F/Fast16BF all pick the TF32 path (same bits, 2.75e-4, ~1.5x faster);
  Pedantic == Default bits; FP64 fixed-point emulation 0.32 -> 2.55 TF/s at 3.8e-15 (classic only).
- A tensor carries its device value: flags set on a `with_flags` clone do not apply to tensors
  created before; move operands with `to_device(&flagged)`.
- `Result` in rstsr is a 1-param alias: use `core::result::Result<T, E>` for foreign error types.

Update 2026-11-25 (B3 committed as rstsr `493a42f`; B4 committed as rstsr `554f956`, `B4-REPORT.md`):
`drivers/cublas_lt.rs` (`CublasLtHandle` per ordinal in `CudaOrdinalObj::cublaslt()`, Mutex held across
heuristic + matmul, cache keyed by the whole problem incl. pointer low 8 bits, 32 MiB workspace per STREAM in
`CudaStreamObj::cublaslt_workspace`); `cublas::prepare` -> `GemmCall` shared by the three cuBLAS drivers.
Facts learned:
- cudarc 0.19.10 `cublaslt`: all needed entries bound; `cublasLtMatmulAlgoCheck` is the 8-arg (layouts-only)
  form; `cublas` and `cublaslt` carry separate copies of the shared enums (copy by value); cudarc's safe
  `CudaBlasLT` has no `matmul` in this version (only the `result` helpers are usable).
- Lt needs FOUR layouts (D mirrors C); the pointer order is cuBLAS's slot order (slot A = `geom.b`).
  `CUBLASLT_LOG_LEVEL=5` prints which descriptor is null when Lt answers INVALID_VALUE.
- Lt runs misaligned (off-16-byte) pointers correctly; the key carries the alignment class.
- Steady state Lt == Ex == classic (+3% on f32 2048^3); first call of a new problem pays the heuristic
  (58 ms on f32 2048^3, 3.6 ms on 64^3).
- Probe: `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` through Lt 13.4 on this GeForce is accepted and NEVER
  emulated (exact, 0.34 TF/s) under any emulation descriptor/strategy -> D18's "classic only" stands.
- Read-back pitfalls: `to_cpu_vec(storage())` is the whole storage (ignores a layout offset, on CPU too);
  `into_shape`/`to_owned` of a view materialize offset 0; `TensorBase::new(storage, layout)` keeps it.
- rustfmt on a `mod`-declaring file descends into the sibling modules: format only the changed files.

Update 2026-11-25 (B5 committed as rstsr `85ed399`, `B5-REPORT.md`): owner
WITHDREW D14 (`ext_matmul`): mixed operands go through `matmul_from`/`matmul_with_output` (no `Mul` bound) on
naive. `CublasPlan` = the problem (`F32/F64/C32/C64/F16/F16F32/BF16/BF16F32/None` + mode), `plan::<TA,TB,TC>`
resolves invalid modes to the family default (Fast* f32-input only; Acc16F f16 only; fixed-point real only);
`Buffers<A,B,C>` + `Problem::retyped`; `matmul/tensor.rs` `TensorCudaMatMulAPI` + `matmul_with_mode_f`.
Facts learned:
- `CUBLAS_COMPUTE_16F` needs `f16` scales (desc create answers INVALID_VALUE with `CUDA_R_32F`); `cublasGemmEx`
  serves 16F on cuBLAS 13.4; `cublasCgemm_v2`/`Zgemm_v2` bound with `cuFloatComplex`/`cuDoubleComplex` pointers.
- cudarc's `cublas` module has no `mode` sibling: import `crate::matmul::mode::..` in drivers.
- `rustfmt --edition 2024` rewrites import order and single-line `if` in this repo: use `--edition 2021` on
  changed files only, and restore single-line `if/else` the committed code keeps (repo style is not rustfmt-clean).
- Under `ColMajor` the cuBLAS pointers are GEOM-named (`geom.a` = caller's `b`): `prepare` must swap buffer AND
  element size per operand; mixing them up passes RowMajor tests and fails `*_col_major_vs_cpu`.
- An extension trait on `TensorAny<R, T, DeviceCuda, D>` must carry `R: DataAPI<Data = CudaBuffer<T>>`; method
  where-clauses live on the trait declaration.
- syrk for `a @ a.t()`: recommended after B6, as a measured pre-driver step (one-triangle output).

Update 2026-11-25 (B6 committed as rstsr `3508d3a`, `B6-REPORT.md`; all batches done): triple layout
matrix test (`triple_layout_matrix`, 464 cells) found that the naive kernel did not COMPILE for bf16 (hard-coded
`__half2float`); fixed. Bench `benches/bench_matmul.rs` follows the crate's `harness = false` style (D27 said
criterion; flagged). Numbers: before/after within +-4%; f64 tiled 14-16% faster than cuBLAS on this GeForce
(owner open point: f64 chain order); c64 Lt 19% faster than Ex/classic; f16 43 TF/s.
Facts learned:
- A ColMajor device puts the batch axis LAST (`[m,k,p]`): build operands in the device's order, or the CPU
  reference is another matrix. `DeviceCpuSerial::default()` is RowMajor (FlagOrder::C; Debug prints C/F).
- `rt::asarray` on a device lays data out in that device's default order; `to_device` keeps the layout.
- Baseline timing: `git worktree add /tmp/wt-base 9c7d49a`, drop a bench into its `benches/` with a
  `[[bench]]` entry, `CARGO_TARGET_DIR=/tmp/...`, remove the worktree; keep the source in the task dir `src/`.

Update 2026-11-26: implementation through rstsr `67a0c79` reviewed; findings and probe in
`2026-11-26-cuda-matmul-review/REVIEW.md` (memory: `cuda-matmul-broadcast-axis-wrong.md`,
`cuda-matmul-order-output-before-launch.md`, `cuda-syrk-ignores-compute-mode.md`,
`rstsr-cuda-build-rs-pin-parse.md`).
