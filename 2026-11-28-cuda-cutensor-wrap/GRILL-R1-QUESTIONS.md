# Round 1 questions — a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

## Background: facts looked up for this round (agent-authored, three read-only reports)

### A. Crate precedent (`rstsr-cuda` at `937ce04`, branch `261127/cuda-prelude-f-convention`, one commit ahead of `261021/cuda`)

- **The precedent for cuTENSOR is cuBLASLt, not cuBLAS.** `CublasLtHandle` (`matmul/drivers/cublas_lt.rs`) is
  raw FFI through `cudarc::cublaslt::sys::*` (cudarc has no safe layer for cuBLASLt either), per ordinal:
  `CudaOrdinalObj::cublaslt()` = `OnceLock<Option<Arc<CublasLtHandle>>>` filled by a `catch_unwind`
  loadability probe, created after `context().bind_to_thread()`, a `Mutex` held across every call, every
  descriptor owning its `*Destroy` in `Drop`. (cuBLAS proper is the odd one: cudarc's safe `CudaBlas`, per
  stream.)
- **Workspace**: `CudaStreamObj::cublaslt_workspace()` = per-stream lazy `OnceLock<Option<Arc<CudaBuffer<u8>>>>`,
  32 MiB, stream-ordered free; the driver declines when it is `None`.
- **Operand plumbing** (`matmul/tensor.rs` -> `matmul/inner.rs` -> drivers): operands enter as
  `impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = D>` / `TensorViewMutAPI<...>` **by value**
  (accepts `&Tensor`, `Tensor`, views alike), then `.view()` / `.view_mut()`, `.layout()`, `.device()`.
  Pointer = `tsr.raw().read_ptr_on(exec)` -> `(*const T, SyncOnDrop)` (guard kept past the enqueue) +
  `tsr.offset()` elements; cross-stream safety by `order_input(buf, exec)` / `order_output(buf, exec)`
  (`kernel.rs:583-620`, no-op when the allocation stream is the executing stream).
- **Errors**: `rstsr_common::Error` built by `rstsr_raise!(Variant, "msg")`; no custom CUDA error. A
  `cublasStatus_t` is **never** an `Error`: it travels as a `Refusal` inside the fall-through chain and an
  `Error` (`UnImplemented`) is raised only when nothing ran. Variants in use: `DeviceMismatch`,
  `InvalidLayout`, `UnImplemented`, `MemoryError` (OOM), `DeviceError` (driver), `RuntimeError` (status after
  the enqueue). `InvalidValue` exists but is unused on the cuBLAS path.
- **Naming**: snake_case throughout; on this branch the `_f` suffix marks the twin that returns `Result`
  (`DeviceCuda::new_f`, `synchronize_f`, `CudaStreamObj::new_f`), the unsuffixed one panics via
  `.rstsr_unwrap()`. The facade forwards only `rstsr_cuda::prelude` (as `rt::cuda` / `rstsr::cuda`).
- **Tests**: hardware tests start with `require_cuda!` (silent skip without a GPU); helpers (`host_nd`,
  `assert_close`) are copy-pasted per test file; `--lib` holds the pure-function units.

### B. Core view types (`rstsr-core`, `rstsr-common`)

- `TensorView<'a, T, B, D>` / `TensorViewMut<'a, T, B, D>` (aliases `TensorRef` / `TensorMut`) are type
  aliases of `TensorBase<Storage<DataRef<'a, B::Raw> | DataMut<'a, B::Raw>, T, B>, D>`; for CUDA
  `B::Raw = CudaBuffer<T>`. Accessors: `.layout()` -> `shape().as_ref(): &[usize]`,
  `stride().as_ref(): &[isize]` (strides are **signed**), `.offset(): usize`, `.ndim()`, `.size()`;
  `.device() -> &DeviceCuda` (the view owns a cheap Arc clone of the device value).
- No runtime `DType` enum anywhere; dispatch is type-level. `rstsr_cuda::dtype::CudaQuery` is implemented
  for exactly the six cuTENSOR-capable types (`f32, f64, f16, bf16, Complex<f32>, Complex<f64>`).
- `num::Complex<T>` is `#[repr(C)] { re, im }` = ABI-identical to `cuFloatComplex` / `cuDoubleComplex`.
- `CudaBuffer<T>` has no `Drop` of its own; cudarc's `CudaSlice` frees stream-ordered (alloc stream waits on
  the recorded read/write events).

### C. cuTENSOR 2.7.0 and cudarc 0.19.10 `cutensor`

- cudarc `cutensor` = `cutensor-02006` (bindings from the **2.6 header**, vendored in
  `src/cutensor/sys/mod.rs`; there is no separate `cutensor-0200*` crate). All 56 function signatures
  match the 2.7.0 header; the only enum gap is one block-sparse attribute. `cutensor::result` wraps
  contraction + reduction only (not permutation, elementwise, trinary contraction, `GetErrorString`).
  `cutensor-version-from-build-system` **panics** on a 2.7 header ("Unsupported cuTENSOR version 2.7").
  Any `cuda-11*` cudarc feature compiles the module out.
- Loading: with `fallback-dynamic-loading` every symbol is a `libloading::Library::get` cached in a
  `OnceLock`; `culib()` **panics** when no candidate loads, `is_culib_present()` is the probe. Candidates
  for "cutensor": `libcutensor.so`, `libcutensor64.so`, `libcutensor64_13*.so`, `libcutensor.so.{13,12,11,10,9,1}`
  -- never `libcutensor.so.2` (what the pip wheel ships), so locally a `libcutensor.so` symlink on
  `LD_LIBRARY_PATH` is required. Under rstsr-cuda's `dynamic-linking` feature cudarc instead links
  `-lcutensor` at build time and `culib()` does not exist.
- `cutensorComputeDescriptor_t` = `*mut cutensorComputeDescriptor` (an opaque **pointer**); the nine
  `CUTENSOR_COMPUTE_DESC_{16F,16BF,TF32,3XTF32,32F,64F,9X16BF,8XINT8,4X16F}` are `extern const` globals
  (data symbols) absent from the bindings. Reachable under dynamic loading through the public
  `culib().get::<*const cutensorComputeDescriptor>(b"CUTENSOR_COMPUTE_DESC_64F\0")` (double deref), or
  under `dynamic-linking` through a self-declared `extern "C" { static CUTENSOR_COMPUTE_DESC_64F: _; }`.
- Pipeline (all 2.x): tensor descriptors -> operation descriptor -> plan preference (algo, JIT mode) ->
  `cutensorEstimateWorkspaceSize` (MIN / DEFAULT / MAX) -> `cutensorCreatePlan(.., workspaceSizeLimit)` ->
  execute (`cutensorContract` / `Reduce` / `Permute` / `Elementwise{Binary,Trinary}Execute` /
  `ContractTrinary`). The stream is the **trailing argument of every execute**; there is no
  `cutensorSetStream` in 2.x. `descD` must be identical to `descC` (and `modeD == modeC`); `C` and `D` may
  alias; A / B must not overlap the written D. Scalars are **host** pointers; "Its data type is determined
  by 'descCompute'" and is readable through `cutensorOperationDescriptorGetAttribute(desc,
  CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE)`. Workspace: 256-byte aligned, "recommended ... e.g., 128 MB".
- Device binding: "The device associated with a particular cuTENSOR handle is assumed to remain unchanged
  after the cutensorCreate call"; "The plan is created for the active CUDA device"; executing on another
  device returns `CUTENSOR_STATUS_ARCH_MISMATCH`.
- Thread-safety (header `\remarks`): `cutensorCreate` / `Destroy`, `CreateTensorDescriptor`,
  `Destroy*`, `Create{Contraction,Reduction,Permutation,Elementwise*}`, every execute, `GetErrorString` =
  "thread-safe"; `cutensorCreatePlan`, `cutensorEstimateWorkspaceSize`, `OperationDescriptor{Get,Set}Attribute`,
  `PlanPreference{Get,Set}Attribute`, `PlanGetAttribute` carry **no** remark; `cutensorHandleResizePlanCache`
  is the one "not thread-safe" function ("the resulting cache can be shared across threads").
- Plan cache: per handle, LRU capacity 64, on by default; `CUTENSOR_OPERATION_DESCRIPTOR_TAG` (int32)
  "enables users to distinguish two identical problems w.r.t. the sw-managed plan-cache";
  `CUTENSOR_PLAN_PREFERENCE_CACHE_MODE` {NONE, PEDANTIC}. Plans "must not be captured via CUDA graphs if
  Just-In-Time compilation is enabled".
- Operations available (2.7.0): Contraction (A*B), ContractionTrinary (A*B*C -> E, 2.1+), Reduction,
  Permutation, ElementwiseBinary (`D = opAC(alpha*opA(A), gamma*opC(C))`), ElementwiseTrinary
  (`D = opABC(opAB(alpha*opA(A), beta*opB(B)), gamma*opC(C))`); BlockSparse* (public beta, 2.3+); Logger*;
  plan-cache / kernel-cache file I/O. The library also exports seven undeclared symbols (compute-descriptor
  create/destroy/attributes, `OperationEstimateRuntime`, `OperationNumAlgos`,
  `TensorDescriptorSetAttribute`) -- not for use.

### D. Probe facts that constrain "safe" (cuTENSOR 2.7.0 on this card; `2026-11-27-cuda-einsum-cutensor`)

- Stride 0 with extent > 1: accepted on 2.7.0 (inputs broadcast correctly), **rejected on 2.4.1**; a stride-0
  output is accepted and writes garbage. Negative stride / zero extent: `INVALID_VALUE` at descriptor creation.
- `opC = CONJ` silently accepted and ignored (2.8.0 rejects it).
- **Scalar pointer type never checked** (wrong type -> silently wrong, e.g. zeros). `alignmentRequirement`
  not checked (rule: >= element size).
- **Extent mismatches are not consistently rejected**: a contracted mode with extent 3 in A and 4 in B passed
  create / estimate / plan (probe2 D7, garbage); reduction / permutation mismatches fail at plan; a mode
  listed twice in one operand is accepted (diagonal). Every D mode must occur in A or B (`INVALID_VALUE`).
- `R32F x C32F -> C32F` passes `CreateContraction`, fails only at `CreatePlan` (`NOT_SUPPORTED`).
- Rank-0 descriptors accepted (null or dummy arrays). `beta = 0` does not read C (NaN-filled C untouched;
  `C = D` with `beta = 0` fine).
- Cost: one-time ~28 ms lazy cuBLAS init inside the first cuBLAS-mapped estimate; then 20-50 us per new plan
  (0.2 ms non-GEMM), 12-16 us on a repeat; the internal plan cache saves nothing measurable; execute ~2 us
  host-side. Non-GEMM f64 contractions need their 32-36 MiB workspace.

---

❓ **Q1** - **Two layers: RAII objects, plus the merged one-shot built on them**: cuTENSOR 2.x is
plan-based (descriptors -> operation descriptor -> plan preference -> workspace estimate -> plan ->
execute), so "merge `cutensorCreateContraction` and `cutensorContract` into one function" is a function
that builds five objects and runs the sixth. Proposal: the module has **both** layers, the second built
from the first:

- **Object layer** (all RAII, `Drop` calls the matching `cutensorDestroy*`, raw pointer never public):
  `CutensorHandle`, `TensorDescriptor` (built from a view: extent, stride, dtype, alignment; **no
  pointer** stored), `OperationDescriptor` (`new_contraction` / `new_contraction_trinary` / `new_reduction`
  / `new_permutation` / `new_elementwise_binary` / `new_elementwise_trinary`), `PlanPreference` (algo,
  JIT mode), `Plan` (`::new(handle, &op, &pref, workspace_limit)`; also `required_workspace()`).
- **One-shot layer**: one free function per operation that does the whole pipeline and returns `Result<()>`,
  for evaluation and for callers that do not care about the 20-50 us planning cost per call:
  ```rust
  pub fn contract<T: CutensorDType, DA, DB, DD>(
      device: &DeviceCuda,
      a: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = DA>, mode_a: &[i32], op_a: cutensorOperator_t,
      b: impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = DB>, mode_b: &[i32], op_b: cutensorOperator_t,
      c: Option<impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = DD>>,   // None: D stands in (beta must be 0)
      d: impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = DD>, mode_d: &[i32],
      alpha: &T::Scalar, beta: &T::Scalar,
      compute: cutensorComputeDescriptor_t,
      workspace: Option<&CutensorWorkspace>,
  ) -> Result<()>
  ```
  (`mode_c` is not a parameter: cuTENSOR requires `modeC == modeD` and `descC == descD`, so a given `c`
  must have D's layout and is validated against it; the one-shot is same-dtype, the object layer takes one
  dtype per operand.)
- **Plan re-validation**: a `Plan` may be executed many times against *different* buffers (that is what
  einsum will cache), so `plan.contract(..)` takes the views again, re-derives (extent, stride, dtype,
  pointer alignment) and **refuses an operand whose layout differs from the one the plan was created for**
  (the descriptors hold no pointer, nothing else would catch it). `Plan` holds an `Arc<CutensorHandle>`
  (the plan cache is per handle; a plan must not outlive it) and plain Rust copies of the operand layouts
  for that check.

➡️ Yes: both layers, public; the one-shot is thin and the `Plan` object is the reuse tool.

---

❓ **Q2** - **Coverage: the six dense operations, nothing else**: wrap Contraction, ContractionTrinary,
Reduction, Permutation, ElementwiseBinary, ElementwiseTrinary (each: operation descriptor, plan, execute,
one-shot). **Excluded**: BlockSparse* (public beta, different descriptor type), the Logger* functions, the
plan-cache / kernel-cache file I/O, `cutensorHandleResizePlanCache` (the one "not thread-safe" entry),
and the seven undeclared exports.

- (b) Contraction and Reduction only (what einsum/tensordot need first); the rest when a caller appears.

➡️ (a). The marginal cost of an operation is ~60 lines once the pipeline exists, and the elementwise probe
showed f16/bf16 and fused `a*b+c` are capability gaps only cuTENSOR fills today.

---

❓ **Q3** - **Handle: per ordinal, plus a public standalone constructor**: the handle is device-bound
("the device ... is assumed to remain unchanged after `cutensorCreate`") and plans must be created with
that device active, so the house slot is the cuBLASLt one: `CudaOrdinalObj::cutensor()` =
`OnceLock<Option<Arc<CutensorHandle>>>`, filled by the `catch_unwind` loadability probe (cudarc panics
when `libcutensor` is not loadable), created after `context().bind_to_thread()` on the ordinal's context,
`None` cached forever. Additionally `CutensorHandle::new(stream: &Arc<CudaStreamObj>) -> Result<Self>` is
public (an owned handle for the object layer and for evaluation). The one-shot functions take **no handle
parameter**: they take `&DeviceCuda` and resolve `device.ordinal_obj().cutensor()`, refusing with a named
error when it is `None` (as the cuBLASLt driver declines on `NoCublasLt`).

- (b) Only the standalone `CutensorHandle::new`; no ordinal slot (callers pass a handle everywhere).

➡️ (a). Two entries, one creation path; the one-shot keeps its parameter list to the tensor facts.

---

❓ **Q4** - **Thread-safety discipline: lock planning, never execute**: the header marks execute,
`Create{Contraction,Reduction,Permutation,Elementwise*}` and descriptor creation "thread-safe", but
`cutensorCreatePlan`, `cutensorEstimateWorkspaceSize` and every `*Attribute` get/set carry no remark, and
the per-handle plan cache is shared by all threads. Proposal: `CutensorHandle` owns a `Mutex<()>` held
across the planning sequence (estimate + create plan) and across attribute get/set -- the
`CublasLtHandle::lock()` idiom -- and **not** across execute (documented thread-safe; serializing it would
kill multi-stream overlap for nothing). `Send`/`Sync` for the wrapper types on that basis
(`TensorDescriptor` is immutable metadata; `Plan` is immutable after creation).

- (b) No lock at all: trust "thread-safe" for the marked functions and the absence of a remark for the rest.

➡️ (a). One mutex on a path that costs 20 us anyway buys the undocumented case for free.

---

❓ **Q5** - **What "safe" refuses (and never fixes)**: memory safety comes from RAII, checked lengths
(`mode.len() == ndim`, `numModes <= 64`) and the borrows; beyond that the wrapper **refuses before calling
cuTENSOR** (with a named `InvalidValue`/`InvalidLayout` error) every case the probes showed to be accepted
with garbage or undocumented, and **never copies** (materialization is the einsum layer's job):

1. a stride `< 0`, or a stride `== 0` with extent `> 1`, in any operand including the output (stride-0
   *inputs* work on 2.7.0 and fail on 2.4.1 -- not relied on);
2. an extent `== 0` (cuTENSOR rejects it anyway; refuse with a clear message instead of its `INVALID_VALUE`);
3. `op_c != IDENTITY` on contraction / reduction / elementwise (silently ignored on <= 2.7);
4. **mode consistency**, since cuTENSOR's own checks are incomplete (probe2 D7): no duplicate mode
   within one operand; every D mode occurs in A or B (contraction) / in A (reduction, permutation);
   one extent per mode label across all operands; `c.layout() == d.layout()` when `c` is given;
5. **overlap**: the byte range of A's or B's buffer overlaps D's buffer range (cuTENSOR: A/B must not
   overlap the written D; the only allowed alias is C == D). Range check on `CudaBuffer`, so two
   disjoint views of one allocation are also refused (conservative, stated).

The wrapper does **not** validate dtype/compute combinations (cuTENSOR's table is the authority; the
`NOT_SUPPORTED` it returns is mapped as in Q10).

➡️ Yes, all five, no copies.

---

❓ **Q6** - **Raw FFI at the boundary, except the compute descriptor**: your position is that `mode` and
`op` stay raw cuTENSOR FFI. Confirmation plus one unavoidable exception:

- modes: `&[i32]` (2.x uses `int32_t`; rank 0 = empty slice, the wrapper passes a dummy pointer);
- operators: `cudarc::cutensor::sys::cutensorOperator_t` (and `cutensorAlgo_t`, `cutensorJitMode_t`,
  `cutensorWorksizePreference_t`, `cutensorStatus_t`, `cudaDataType_t`), **re-exported from
  `cutensor_wrap`** so a caller never names `cudarc` directly;
- **compute descriptor**: it is an opaque *pointer* whose only source is the library's exported globals,
  which the bindings do not carry -- a caller cannot obtain one "raw". The wrapper exposes the nine names
  as functions returning the raw pointer (`compute_desc::F32()`, `::TF32()`, `::ThreeXTF32()`, `::F16()`,
  `::BF16()`, `::FourXF16()`, `::NineXBF16()`, `::F64()`, `::EightXInt8()`), resolved once per process
  (the symbol source differs by loading mode, Q12); the functions take the raw
  `cutensorComputeDescriptor_t`. No Rust enum for it (your preference), the names are just the globals.

➡️ Yes.

---

❓ **Q7** - **Scalars: typed by the output dtype, verified against the operation, aligned storage**:
`alpha` / `beta` / `gamma` are `&T::Scalar` (associated type on the wrapper's dtype trait: `f32` for
`f16`/`bf16`/`f32`, `f64`, `Complex<f32>`, `Complex<f64>`), passed through `#[repr(C, align(16))]`
storage as the cuBLASLt driver does for its scales (the crate's `AlignedScale` lesson). Because the header
says the scalar type "is determined by descCompute" and the pointer type is never checked, the wrapper
also reads `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` right after creating the operation descriptor and
**refuses when it differs from `T::Scalar`'s `cudaDataType_t`** -- closing the silent-zeros hole without
hard-coding cuTENSOR's table.

➡️ Yes.

---

❓ **Q8** - **Workspace: an owning buffer type plus a per-stream default**: `CutensorWorkspace::new(stream:
&Arc<CudaStreamObj>, bytes: usize) -> Result<Self>` (a `CudaBuffer<u8>`, 256-byte aligned by `cuMemAlloc`,
stream-ordered free). Default: `CudaStreamObj::cutensor_workspace()` = per-stream lazy slot like
`cublaslt_workspace()`, sized by a new flag `CudaRuntimeFlags::cutensor_workspace_bytes: usize`
(default **128 MiB** per the header's recommendation, `0` = no workspace; non-GEMM f64 contractions need
32-36 MiB), `None` when the allocation fails. The one-shot takes `Option<&CutensorWorkspace>` (`None` =
the stream's default; `Some(&ws)` = yours). Planning uses `CUTENSOR_WORKSPACE_DEFAULT` for the estimate and
`CreatePlan(.., workspace.len())` as the limit; execute passes the buffer and its full length.

- (b) No default slot: the caller always passes a `CutensorWorkspace` (or an explicit "none").

➡️ (a). Evaluation and einsum both want "just work" with the header's number, and the slot is lazy
(streams that never touch cuTENSOR allocate nothing).

---

❓ **Q9** - **Stream: from the device value, ordering inside**: both layers take `&DeviceCuda` and issue on
`device.stream()` (the house rule: ops run on the device value's stream; a caller picks a stream by
making a device value on it). Inside, the cuBLASLt sequence: `order_input` on A / B and on C when
`beta != 0`, `read_ptr_on(exec)` guards kept past the enqueue, `order_output` on D after it. No bare
`cudaStream_t` / `&CudaStream` parameter anywhere in the module. (The object layer's `Plan` is not tied
to a stream: it is created on the ordinal's device and executes on any stream of that device, which is
what cuTENSOR guarantees.)

- (b) The object layer's execute takes `&Arc<CudaStreamObj>` explicitly and leaves ordering to the caller.

➡️ (a). One stream source, and the cross-stream laws the crate already paid for apply automatically.

---

❓ **Q10** - **Errors: `Result` with a status table, the stage in the message, a decline helper**: every
wrapper function returns `rstsr_common::Result<T>`; a non-SUCCESS status is raised with
`rstsr_raise!(variant, "cutensor: <stage>: <cutensorGetErrorString> (CUTENSOR_STATUS_...<context>)")` where
`<stage>` is `load library | create handle | tensor descriptor | operation descriptor | plan preference |
workspace estimate | plan | execute`. Variant by status: `NOT_SUPPORTED` -> `UnImplemented`;
`INVALID_VALUE` -> `InvalidValue`; `ALLOC_FAILED` / `INSUFFICIENT_WORKSPACE` -> `MemoryError`;
`CUDA_ERROR` / `CUBLAS_ERROR` / `EXECUTION_FAILED` / `ARCH_MISMATCH` / `INSUFFICIENT_DRIVER` ->
`DeviceError`; the rest (`NOT_INITIALIZED`, `MAPPING_ERROR`, `INTERNAL_ERROR`, `LICENSE_ERROR`, `IO_ERROR`)
-> `RuntimeError`; library not loadable -> `UnImplemented("cutensor: libcutensor is not loadable")`.
The wrapper's own refusals (Q5, Q7) are `InvalidValue` / `InvalidLayout` / `DeviceMismatch`, so a caller
that has a fallback (einsum's tiers) detects a decline with one helper `cutensor_wrap::is_decline(&Error)`
(true for `UnImplemented`, `InvalidValue`, `InvalidLayout`), instead of matching messages.

- (b) A dedicated `CutensorError { stage, status }` error type for the whole module, with `From` into
  `Error` at the one-shot boundary.

➡️ (a). One error type in the crate, and the decline question is answered by variant, not by string.

---

❓ **Q11** - **Naming and visibility**: module `rstsr_cuda::cutensor_wrap` (your name), public, with
submodules `handle`, `descriptor`, `operation`, `plan`, `workspace`, `compute`, `error` and the one-shot
functions re-exported at the module root as `contract`, `contract_trinary`, `reduce`, `permute`,
`elementwise_binary`, `elementwise_trinary` (snake_case, as the crate; `rtCutensorContract` would be the
only camelCase name in it). Not forwarded by the `rstsr` facade (an evaluation user depends on
`rstsr-cuda` directly; einsum's user surface is designed in the other grill). **No `_f` twins**: every
function of the module returns `Result` and there is no panicking variant to twin -- a stated deviation
from the branch's `_f` convention, written in the module doc.

- (b) Also re-export as `rstsr::cuda::cutensor_wrap` (one `pub use` in the facade) so facade users reach
  it without a direct dependency.

➡️ Yes (a for the facade, as written).

---

❓ **Q12** - **Cargo wiring**: add `"cutensor"` to the cudarc feature list of `rstsr-cuda` unconditionally
(compiled always, `libcutensor` dlopened at first use like cuBLAS; no rstsr-cuda feature gate, no
`build.rs` version cfg since the 2.6 bindings are used as-is); do **not** add `cutensor-version-from-build-system`
(panics on the local 2.7 header). `build.rs` gains one cfg, `cutensor_dynamic_loading`, emitted unless
rstsr-cuda's `dynamic-linking` feature is on: under it the nine compute-descriptor globals come from
`culib().get(..)`, otherwise from a self-declared `extern "C" { static .. }` block (cudarc's own cfg is
not visible to this crate -- the `cublas_math_mode` precedent). The module doc states the local
requirement: a `libcutensor.so` symlink (the wheel ships only `libcutensor.so.2`, which cudarc's candidate
list never names) on `LD_LIBRARY_PATH`.

- (b) Drop `dynamic-linking` support for cuTENSOR (dynamic loading only; the cfg is not needed).

➡️ (a). Three lines in `build.rs` keep the crate's existing feature honest.

---

❓ **Q13** - **Tests and the evaluation bench**: `tests/test_cutensor_wrap.rs` (hardware; skips with a
message when there is no GPU **or** when `libcutensor` is not loadable, so a green run says which), oracle
= the CPU device via the crate's `to_device` / `to_cpu_vec` idiom (`require_cuda!`-style guard, helpers
copied into the file as its siblings do): GEMM-shaped contraction per dtype (`f32 f64 c32 c64 f16 bf16`)
on the cuBLAS chain vs cuTENSOR, batched and strided views, `C == D` accumulate, `beta = 0` with `c =
None`, conjugated A / B, rank-0 output, reduce / permute / elementwise binary / trinary, the refusal
matrix of Q5 and Q7 (each a named error), `NOT_SUPPORTED` for `f32 x c32` at plan, a plan re-executed
against a changed layout (refused), cross-stream execute (device value on a second stream). `--lib` units
for the pure parts: alignment computation, mode/extent consistency, stride/extent refusals, overlap
ranges, status -> variant table, scalar-type table. `benches/bench_cutensor_wrap.rs` (`harness = false`
like the others): one-shot vs held-plan per operation at the probe's shapes, the planning cost, cuTENSOR
vs the cuBLAS drivers on GEMM shapes -- this is the "evaluation" entry.

➡️ Yes.

---

❓ **Q14** - **No plan cache in this module**: the one-shot plans on every call (20-50 us, plus the
one-time ~28 ms cuBLAS init; cuTENSOR's internal LRU is left at its default and saves nothing measurable)
and its doc says so; the `Plan` object is the reuse tool; a keyed cache (problem hash -> `Plan`, with
`CUTENSOR_OPERATION_DESCRIPTOR_TAG` to separate identical problems) belongs to the einsum executor that
knows the key, not to a wrapper that does not.

- (b) A per-handle `Mutex<HashMap<u64, Plan>>` in the wrapper, keyed by the operation's descriptors, modes,
  operators, compute descriptor and preference, used by the one-shot.

➡️ (a). The wrapper stays a wrapper; the cache is an einsum design item (round 3 of that grill).

---

❓ **Q15** - **Comprehension alignment**: you asked for a fact check; these are the statements the
questions rest on. Each is sourced (header 2.7.0, cudarc 0.19.10, the probes). Do they match your
understanding -- correct any that do not?

1. **"TsrView/TsrMut"** are `TensorView`/`TensorViewMut` in the code (`TensorRef`/`TensorMut` aliases);
   the wrapper's operands are `impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim = D>`.
2. cuTENSOR 2.x is **plan-based**; "merge create + contract" means building five objects and running a
   sixth. Planning costs 20-50 us per new plan and execute ~2 us; so a merged function is an *evaluation*
   convenience, and anything that repeats a problem holds the `Plan` (Q1, Q14).
3. `descCompute` is **an opaque pointer**, not an enum; its nine legal values are the library's exported
   globals, which the Rust bindings do not carry (Q6).
4. The **stream is an argument of every execute call** (no `cutensorSetStream` in 2.x); the **handle and
   the plan are device-bound** (per-ordinal handle, Q3, Q9).
5. `alpha`/`beta` are **host** pointers whose type cuTENSOR decides ("determined by descCompute") and
   never checks; for `f16`/`bf16` operands they are `f32` (Q7).
6. `descC` must equal `descD` and `modeC == modeD`; `C` may alias `D` (accumulate); `A`/`B` must not
   overlap the written `D` (Q1's `c: Option<_>`, Q5's overlap rule).
7. cudarc 0.19.10 binds the **2.6 header**; it is ABI-identical to the 2.7.0 library for every function
   the wrapper uses; the local install needs a `libcutensor.so` symlink (Q12).
8. cuTENSOR's argument validation is **incomplete** (extent mismatches can pass every stage; `opC = CONJ`
   ignored on 2.7; scalar type unchecked) -- hence Q5/Q7 refuse up front rather than "let cuTENSOR reject".
9. Coverage: six dense operations exist in 2.x (incl. a trinary contraction `A*B*C -> E` and the two
   elementwise entries); block-sparse is a public beta with its own descriptor type (Q2).

➡️ Yes -- confirm, or name the items to correct.
