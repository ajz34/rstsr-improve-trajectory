# GRILL-INIT — `02-kernel-nvrtc`: kernel layer + dtype traits for rstsr-cuda

Initiating prompt from the owner, 2026-10-23 (verbatim, screen message):

> In this grill, you are going to handle kernel implementation using NVRTC.
>
> You are also going to see if we need a / some DType traits for our cuda
> device. I think dtype issue should also be discussed here.
> - We may defer multi-dtype operations (such as acos(f32, f64) will be
>   implemented later or even not to be implemented). But some assignment
>   functions such as assign_arbitary are suggested to have some common type
>   conversions.
> - We are not going to use cudarc's provided trait on this task.
> - I'm not sure if frequently use dispatched dtype is proper way.

("assign_arbitary" read as assignment accepting an arbitrary (differently
typed) source — the `DTypeCastAPI`-based assign path; to be confirmed in R1.)

## Standing constraints carried in

- From `01-device-basic` (../01-device-basic/DECISIONS.md) and the cuda-grill
  doc set (../../2026-10-21-cuda-grill/): cudarc is the sole external CUDA
  dependency, as substrate only (driver/maintenance/cuda-gates/`CudaSlice`);
  its `nvrtc` feature is already enabled in the crate. Gaps go through
  cudarc's `result`/`sys` layers or our own FFI, never a second crate.
- New owner directive this round: cudarc's own trait layer (e.g. the launch
  param / dtype traits it provides) is **not** to be adopted; dtype trait
  design is ours.
- Implementation baseline: rstsr branch `261021/cuda` @ `cdadd12`
  (device-basic slice, B1–B9). This round's kernels retire the registered
  `// DEFER:` naive paths (fill/sequence, strided assign, format gather,
  non-contig `to_contig`) and open the general kernel machinery.
- Multi-dtype (mixed operand) ops may be deferred by policy; assignment-side
  common type conversions are wanted.

## Fact-finding fan-out (2026-10-23, before R1)

Five read-only agents dispatched:

1. cudarc 0.19.10 NVRTC + module/launch API surface (and its `DeviceRepr` /
   `ValidAsZeroBits` traits, to know exactly what we are declining).
2. rstsr-core dtype-dispatch idioms (`impl_uninit_dispatch!`,
   `for_each_fast_dtype!`, representative op impls, `DTypeCastAPI`).
3. rstsr-cuda current capability inventory (DEFER sites, stream/buffer
   substrate for launches, host round-trip helpers).
4. rstsr-dtype-traits trait inventory + core cast machinery.
5. Digest of cuda-grill docs 02/03/04 kernel-machinery sections (cupy
   kernel-as-data, NVRTC verdict, M0–M6, workload kernel needs).
