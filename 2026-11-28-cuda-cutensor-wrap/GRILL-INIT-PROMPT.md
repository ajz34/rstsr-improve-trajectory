# Grilling: a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

Started 2026-11-28. Target: the `rstsr-cuda` crate of the `rstsr` repo (read at
`261127/cuda-prelude-f-convention`; the CUDA work lives on `261021/cuda`).

## The initial prompt (verbatim)

> You are going to see if we can build safe wrapper on cuTENSOR.
> This is a first step before we are going to implement the einsum and tensordot.
>
> - We have a not-finished grill session in rstsr-improve-trajectory, but that is going to be einsum
>   and tensordot, and I'm going to defer this until this work finished.
> - I suppose this work to be done in `rstsr-cuda/cutensor_wrap`. This is not a high-level API
>   function to user, but user can use it to perform evaluation. No trait implementation there.
>   Should be generally safe wrapper.
> - For example `cutensorCreateContraction()` and `cutensorContract()`, I'm considering that if it
>   is able to merge them to a rust function that inputs (TsrView/TsrMut, mode, op for ABCD),
>   descCompute, workspace, stream, possibly handle, desc, etc into a function `rtCutensorContract`.
>   Do this as possible as we can cover the cutensor functions. Note that I think that mode/op are
>   still raw cuTensor's raw FFI, not need to wrapping to some rust enum. Return result if possible.
>
> I'm also not very familiar with cutensor, so you may need to make fact check and try to see if we
> need to make alignments on comprehension.

## Relation to the deferred grill

`2026-11-27-cuda-einsum-cutensor` (R1 answered, R2 questions issued, R2 answers file empty) designs
the *user-facing* `rt::cuda::einsum` / `tensordot` entries on `DeviceCuda`. This grill designs the
layer underneath: a memory-safe, thin Rust skin over the cuTENSOR C API that the einsum executor
(and a user evaluating cuTENSOR) can call. Its two probes (`probe/PROBE-REPORT.md`, `probe2/results/
main.log`) are the fact base for cuTENSOR 2.7.0 behaviour and are reused here, not repeated.

## Fact base gathered before round 1 (agent-authored)

### What the crate already has (precedent for wrapping a CUDA library)

- cuBLAS / cuBLASLt are reached through cudarc 0.19 (`fallback-dynamic-loading`: `libcublas*` is
  dlopened at first use, no header at build time). Per-ordinal handle
  `CudaOrdinalObj::cublaslt()` (`OnceLock<Option<_>>`, created after `context().bind_to_thread()`),
  per-stream workspace `CudaStreamObj::cublaslt_workspace()` (32 MiB, lazy). Drivers take
  `TsrView`/`TsrMut` and derive pointer / leading dimension / `cudaDataType_t` from the view
  (`src/matmul/cublas_args.rs`). Runtime knobs live in `CudaRuntimeFlags` (`src/flags.rs`); nothing
  is read from the environment. Hardware tests skip silently without a GPU (`require_cuda!`).
- cudarc 0.19.10 `cutensor` feature (= `cutensor-02006`, 2.6 header; 2.3-2.6 supported): `cutensor::sys`
  bindgen fns + `cutensor::result` thin `Result` wrappers; **no safe layer**. The compute
  descriptors are exported `const` globals the bindgen skipped; reachable through the public
  `cutensor::sys::culib()` (`libloading::Library`) with `get::<*const cutensorComputeDescriptor_t>`.
  The dlopen candidates do not include the pip package's `libcutensor.so.2`: locally a
  `libcutensor.so` symlink on `LD_LIBRARY_PATH` is required (cuTENSOR is not installed system-wide).

### cuTENSOR 2.7.0 behaviour that a safe wrapper must not rely on (probe 1 + probe 2)

- Stride 0 (extent > 1) is **accepted** by 2.7.0 (inputs broadcast correctly) but is undocumented and
  rejected by 2.4.1; a stride-0 **output** is accepted and writes garbage (write race).
- Negative stride / zero extent -> `INVALID_VALUE` at descriptor creation (so the wrapper can also
  refuse them up front, consistently).
- `opC = CONJ` silently accepted and ignored on 2.7.0 (fixed to reject in 2.8.0).
- `alignmentRequirement` is not checked; only rule: alignment >= element size.
- **Scalar pointer type is never checked** (wrong type -> silently wrong, e.g. all zeros).
- `R32F x C32F -> C32F` passes `CreateContraction` and fails only at `CreatePlan` (`NOT_SUPPORTED`):
  a decline can arrive at any planning stage.
- **Extent mismatches are not consistently rejected**: a contracted mode with extent 3 in A and 4 in
  B passed create / estimate / plan on 2.7.0 (probe2 D7, verdict DIFFERS: garbage); reduction /
  permutation mismatches fail at plan; a duplicate mode in one operand is accepted (diagonal).
  Every D mode must occur in A or B (`INVALID_VALUE` otherwise).
- Rank-0 descriptors are accepted (pass `nullptr` extent/stride/mode arrays or non-null dummies).
- `beta = 0` does not read C (C may be anything, even NaN-filled; `C = nullptr` with the reduce entry
  is the "null" group). D may alias C (accumulate).
- Planning cost: one-time ~28 ms lazy init of the cuBLAS path, then 20-50 us per new plan (0.2 ms
  non-GEMM), 12-16 us on a repeat; the internal plan cache saves nothing measurable. The elementwise
  probe measured ~2 us per execute call. Non-GEMM f64 contractions need their 32-36 MiB workspace.
