# DECISIONS — a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

Decision log for this grill (rounds 1-3 answered on screen 2026-11-28; the owner's answer files
`GRILL-R1-ANSWERS.md` / `GRILL-R2-ANSWERS.md` are committed verbatim). Facts behind the decisions:
`GRILL-INIT-PROMPT.md` (fact base), `GRILL-R1-QUESTIONS.md` (crate precedent, core types, cudarc +
header), `GRILL-R2-QUESTIONS.md` (facade wiring, local versions), `GRILL-R3-QUESTIONS.md` (addendum:
version-matrix probe, `probe/versions/REPORT.md`), and the memory notes `cutensor-probe-facts`,
`cutensor-elementwise-facts`, `cutensor-version-matrix`.

Execution plan: `PLAN.md`. Target: `rstsr` crate `rstsr-cuda`, branch `261021/cuda` (at `937ce04`).

## Shape

**D1 - Two layers, plan re-validation.** An RAII object layer (`CutensorHandle`, `TensorDescriptor`,
`OperationDescriptor`, `PlanPreference`, `Plan`, `CutensorWorkspace`; `Drop` calls the matching
`cutensorDestroy*`; raw pointers never public) plus a thin one-shot layer per operation that builds
the objects and executes. A `Plan` is executed against views passed again at execute time and
**refuses an operand whose extent / stride / dtype / pointer alignment differs from what was planned**;
`Plan` holds an `Arc<CutensorHandle>` and `Layout<IxD>` copies for that check.

> owner (R1 Q1): "Your propose on both layer and plan re-validation is also sounding and better than my
> thought in my mind."

**D2 - `IxD` only, both layers.** Operands are `impl TensorViewAPI<Type = T, Backend = DeviceCuda, Dim =
IxD>` / `TensorViewMutAPI<..., Dim = IxD>`; the wrapper performs no `to_dim`; a static-dim tensor is
converted by the caller (`.to_dim::<IxD>()`), stated in the module doc.

> owner (R1 Q1): "we simply only accept IxD (dynamic dimensionality), not dispatching `DA, DB, DD`, to
> make things simpler (and anyway, cuTENSOR I think it also only accepts IxD, using some fixed
> dimensionality does not give efficiency gains)." -- confirmed: rank is a runtime `numModes`.

**D3 - Six dense operations, nothing else.** Contraction, ContractionTrinary (`A*B*C -> E`), Reduction,
Permutation, ElementwiseBinary, ElementwiseTrinary (each: operation descriptor, plan, execute,
one-shot). Excluded: BlockSparse* (public beta, own descriptor type), Logger*, plan-cache / kernel-cache
file I/O, `cutensorHandleResizePlanCache` (the one "not thread-safe" entry), the seven undeclared exports.

**D4 - No plan cache in the wrapper.** The one-shot plans on every call (20-50 us; cuTENSOR's internal
LRU saves nothing measurable) and its doc says so; `Plan` is the reuse tool; a keyed cache belongs to the
einsum executor (that grill's round 3).

## Handle, stream, threads

**D5 - Handle per ordinal, plus a public constructor.** `CudaOrdinalObj::cutensor()` =
`OnceLock<Option<Arc<CutensorHandle>>>` filled by the `catch_unwind` loadability probe (cudarc panics
when `libcutensor` is not loadable), created after `context().bind_to_thread()`, `None` cached forever;
plus public `CutensorHandle::new(&Arc<CudaStreamObj>) -> Result<Self>`. The one-shot functions take
`&DeviceCuda` and no handle parameter; they refuse by name (`UnImplemented`) when the slot is `None`.
The handle exposes `version()` (diagnostic only, see D9).

**D6 - Lock planning, never execute.** A `Mutex<()>` on the handle is held across estimate + create plan
and across attribute get/set (the header gives those no thread-safety remark); execute is not locked
(documented thread-safe; serializing it would kill multi-stream overlap). `Send`/`Sync` on that basis.

**D7 - Stream from the device value, ordering inside.** Both layers take `&DeviceCuda`, execute on
`device.stream()`, and apply the cuBLASLt sequence themselves: `order_input` on A / B and on C when
`beta != 0`, `read_ptr_on(exec)` guards kept past the enqueue, `order_output` on D after it. No bare
`cudaStream_t` / `&CudaStream` parameter. A `Plan` is not stream-tied (created on the ordinal's device,
executes on any stream of that device).

## Safety policy (what "safe" refuses; it never copies)

**D8 - Refused before cuTENSOR** with a named `InvalidValue` / `InvalidLayout` / `DeviceMismatch`:
negative stride; stride-0 **output** (extent > 1); extent 0; `op_c != IDENTITY` on contraction /
reduction / elementwise (ignored through 2.7.0, rejected only in 2.8.0 -- refused unconditionally); mode
consistency (no duplicate mode within an operand; every D mode in A or B for contraction, in A for
reduction / permutation; one extent per label across operands -- no version validates extents; `c.layout()
== d.layout()` when `c` is given); `mode.len() == ndim`; `numModes <= 64`. Materialization is the
caller's / einsum's job.

**D9 - Stride-0 inputs pass through; cuTENSOR's status is the verdict.** No Rust-side gate. 2.4.1
returns `NOT_SUPPORTED` at descriptor creation -> `UnImplemented` -> `is_decline` (a caller with a
fallback declines cleanly); 2.6.0-2.8.0 accept and compute correctly (`probe/versions/` S1/S2). The
only accepted-with-garbage behaviour found (S3) is for outputs, which stay refused by D8.

> owner (R3 Q2): "I think that, we just allow stride=0 (input-only, output should pre-check and gated
> at rust side) to return error at runtime?" -- adopted.

**D10 - No overlap check.** The cuTENSOR aliasing rule (A / B must not overlap the written D; C may
alias D) is documented, not checked.

> owner (R1 Q5): "Overlap check is probably too defensive and I think not overcheck that."

**D11 - Scalars typed, aligned, verified.** `alpha` / `beta` / `gamma` are `&T::Scalar` (associated type
on the wrapper's dtype trait: `f32` for `f16` / `bf16` / `f32`, `f64`, `Complex<f32>`, `Complex<f64>`),
passed through `#[repr(C, align(16))]` storage (the cuBLASLt `AlignedScale` lesson), and checked after
the operation descriptor is created: `CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE` must equal
`T::Scalar`'s `cudaDataType_t`, else refuse (the pointer type is never checked by cuTENSOR).

**D12 - Descriptor alignment: computed by default, overridable.** `TensorDescriptor::new(handle, &view)`
declares the base pointer's own alignment (largest power of two dividing the byte address, capped at
256, validated `>= size_of::<T>()`); `TensorDescriptor::with_alignment(handle, &view, alignment)`
declares a floor (for one plan serving every buffer of a shape regardless of offset views); plan
re-validation checks `actual pointer alignment >= declared`.

## FFI boundary

**D13 - Raw cudarc types at the boundary; no re-exports.** Signatures name
`cudarc::cutensor::sys::{cutensorOperator_t, cutensorAlgo_t, cutensorJitMode_t,
cutensorWorksizePreference_t, cutensorComputeDescriptor_t}` directly (safe enum / pointer types);
callers never call an `unsafe` cudarc fn; the wrapper calls `cudarc::cutensor::sys::*` internally.

> owner (R1 Q6): "directly call cudarc is not a problem, as long as it is not unsafe."

**D14 - Compute descriptors: nine safe accessors, no enum.** `cutensor_wrap::compute_desc::{desc_16f,
desc_16bf, desc_tf32, desc_3xtf32, desc_32f, desc_64f, desc_9x16bf, desc_8xint8, desc_4x16f}() -> 
cutensorComputeDescriptor_t`, each symbol resolved once per process (a missing symbol is an
`UnImplemented` error, not cudarc's panic); functions take the raw pointer. No Rust enum and no binding
to the output dtype now; the matmul-style enum keyed by the output dtype is deferred to the einsum
utility.

> owner (R1 Q6): "use cudarc, not wrap it now ... I still have compute descriptor wrapping in mind ...
> when we move on to some utility of `rt::cuda::einsum` then handle such kind of thing." (cudarc carries
> no descriptor item; its only route, `culib()`, is `unsafe` -- hence the minimal safe accessor.)

## Resources

**D15 - Workspace type plus per-stream default.** `CutensorWorkspace::new(&Arc<CudaStreamObj>, bytes)`
(a `CudaBuffer<u8>`, 256-byte aligned by `cuMemAlloc`, stream-ordered free);
`CudaStreamObj::cutensor_workspace()` = per-stream lazy slot sized by
`CudaRuntimeFlags::cutensor_workspace_bytes: usize` (default 128 MiB per the header; `0` = no
workspace). The one-shot takes `Option<&CutensorWorkspace>` (`None` = the stream's default); planning
estimates with `CUTENSOR_WORKSPACE_DEFAULT` and creates the plan with the buffer's length as the limit.
If the flag is > 0 and the allocation failed, the call fails with `MemoryError` (no silent no-workspace
run).

**D16 - The one-shot's `c: Option<_>`.** `None` means **C is D**: the output pointer stands in for C
at any `beta` (in-place accumulation, `d += alpha * opA(A) * opB(B)` at `beta = 1`). Amended in B2:
the original text ("None requires beta == 0") made in-place accumulation inexpressible — passing the
same tensor as `c` and `d` is two aliasing borrows, which safe Rust forbids; cuTENSOR itself defines
the C-pointer = D-pointer form (probe2 D1c/D2 verified it at `beta = 0` and `beta = 1`). `Some(c)`
must have D's layout (`descC == descD`, `modeC == modeD`); there is no `mode_c` parameter. Owner
correction (2026-11-28): `c` is `Option<impl TensorViewAPI>` **by value** — the caller passes a view
(`t.view()`), uniform with the other operands' by-value convention (R2 Q6); the B2 working tree
briefly had `Option<&impl TensorViewAPI>`, an unnecessary deviation. `alpha` / `beta` are likewise `TD::Scalar`
**by value** (owner correction, same day) — `CutensorScalar: Copy`, and `AlignedScale` copies into its
aligned stack slot anyway; the plan's scalar-type re-check against `TD::Scalar::DATA_TYPE` is
unchanged.

**D16a - `opC` has no parameter on contraction** (B2 amendment of D8's wording). The contraction
constructor pins `opC = CUTENSOR_OP_IDENTITY` and takes no argument: no cuTENSOR version honours a
non-identity `opC` (silently ignored through 2.7.0, rejected by 2.8.0), so the D8 refusal happens at
the API boundary — there is no slot to misuse. `opA` / `opB` stay raw.

## Errors

**D17 - `Result` with a status table, the stage in the message, a decline helper.** Every function
returns `rstsr_common::Result<T>`; a non-SUCCESS status is raised as `rstsr_raise!(variant, "cutensor:
<stage>: <cutensorGetErrorString> (CUTENSOR_STATUS_...<context>)")`, stage in {load library, create
handle, tensor descriptor, operation descriptor, plan preference, workspace estimate, plan, execute}.
Variant by status: `NOT_SUPPORTED -> UnImplemented`; `INVALID_VALUE -> InvalidValue`; `ALLOC_FAILED` /
`INSUFFICIENT_WORKSPACE -> MemoryError`; `CUDA_ERROR` / `CUBLAS_ERROR` / `EXECUTION_FAILED` /
`ARCH_MISMATCH` / `INSUFFICIENT_DRIVER -> DeviceError`; the rest -> `RuntimeError`; library not loadable
-> `UnImplemented`. `cutensor_wrap::is_decline(&Error)` is true for `UnImplemented`, `InvalidValue`,
`InvalidLayout` (variant match, never message match).

**D18 - No `_f` twins.** Every function of the module returns `Result` and has no panicking twin; a
stated deviation from the branch's `_f` convention, written in the module doc.

## Wiring, naming, tests

**D19 - Module and names.** `rstsr_cuda::cutensor_wrap` with submodules `handle`, `descriptor`,
`operation`, `plan`, `workspace`, `compute`, `error`; one-shots at the module root: `contract`,
`contract_trinary`, `reduce`, `permute`, `elementwise_binary`, `elementwise_trinary` (snake_case; the
crate's convention). Facade: `rstsr::cuda::cutensor_wrap` (one `pub use rstsr_cuda::cutensor_wrap;` in
`rstsr/src/cuda.rs`) and `rt::cuda::cutensor_wrap` (the same line inside `rt::cuda` in
`rstsr/src/prelude.rs`) -- the **module** as a named item, no glob.

> owner (R2 Q4): "(b). And yes, `cutensor_wrap::*` should not pollute the prelude, we only introduce
> module `cutensor_wrap` in prelude."

**D20 - Cargo wiring.** `"cutensor"` added to the cudarc feature list of `rstsr-cuda` unconditionally
(compiled always; `libcutensor` dlopened at first use like cuBLAS; no rstsr-cuda feature gate); not
`cutensor-version-from-build-system` (panics on the local 2.7 header). `build.rs` emits one cfg,
`cutensor_dynamic_loading`, unless rstsr-cuda's `dynamic-linking` feature is on: under it the nine
descriptor globals come from `cudarc::cutensor::sys::culib().get(..)`, otherwise from a self-declared
`extern "C" { static .. }` block. The module doc states the local requirement (a `libcutensor.so`
symlink on `LD_LIBRARY_PATH`; the wheel ships only `libcutensor.so.2`).

**D21 - Tests and bench.** `tests/test_cutensor_wrap.rs` (hardware; skips with a message when there is
no GPU **or** `libcutensor` is not loadable, so a green run says which; CPU oracle through the crate's
`to_device` / `to_cpu_vec` idiom; the refusal matrix of D8 / D11 as named errors; `NOT_SUPPORTED` for
`f32 x c32` at plan; plan re-executed against a changed layout refused; cross-stream execute; prints the
cuTENSOR version it ran against). `--lib` units for the pure parts (alignment, mode / extent
consistency, refusals, status -> variant table, scalar table). `benches/bench_cutensor_wrap.rs`
(`harness = false`): one-shot vs held plan per operation, planning cost, cuTENSOR vs the cuBLAS drivers
on GEMM shapes -- the evaluation entry. The Rust tests can be run against another installed version by
pointing `LD_LIBRARY_PATH` at its symlink dir (`probe/versions/lib<V>`).

**D22 - Base and commits.** Branch `261021/cuda`, fast-forwarded by the owner's instruction from
`26eab76` to `937ce04` (the convention branch `261127/cuda-prelude-f-convention` merged; it carried the
prelude / `_f` / facade wiring the wrapper uses). No commit in `rstsr` without the owner's instruction.

**D23 - Batches B1-B4**, each leaving the crate compiling and `--lib` + hardware tests green here:
see `PLAN.md`. The facade lines (D19) are part of B4.

**D24 - `cutensor_wrap/mod.rs` is a pure module map** (owner preference, 2026-11-28): module
declarations, re-exports and the crate-level docs only; every executable item lives in a leaf
file. Amended the same day (owner): **all tensor-side wrapper functions live in one leaf file**,
`one_shot.rs` — contraction today, then reduction, permutation, elementwise and trinary
contraction as B3 adds them — one place, separate from the RAII object layer. The re-export
`pub use one_shot::contract;` keeps `cutensor_wrap::contract` (and the prelude path) stable.

**D25 - The B3 families** (2026-11-28): reduction, permutation, elementwise binary and elementwise
trinary join `one_shot.rs` with the same conventions as contraction — operands by value,
`c: Option<impl TensorViewAPI>` with `None` = **C is D** (in-place accumulate/elementwise), scalars
by value, and **no `mode_c` parameter anywhere** (`descC == descD` is a cuTENSOR requirement, so
`C`'s modes are `D`'s; a differently-labelled operand in the `C` slot is expressed by the view's
own layout). The elementwise join operators (`op_ac`; `op_ab`/`op_abc`) are *functional* parameters
— the D8 elementwise clause refined: `opC` is pinned only on contraction. `OperationDescriptor`
gains a `kind` tag and the execute calls assert it (a contraction plan refuses `Plan::reduce`).
Reduction/permutation/elementwise serve the **same-dtype** path first; cuTENSOR's mixed-dtype rows
are unexplored and stay reachable only at the object layer. "Ternary contraction" from the original
B3 list needs no new code: `contract(.., Some(c_view), ..)` already serves it. Fact: `NORM1`/`NORM2`
reduction operators are gone in cuTENSOR 2.x (1.x relics); the 2.x reduction set is
ADD/MUL/MIN/MAX. The one-shot derives `descC` from `D`, so a `C`-slot view must carry `D`'s exact
layout (strided-`C` needs the object layer).

**D26 - Header-audit completion of the wrapper** (2026-11-28, owner's question: "can contract
trinary be implemented?" - it can, and now is): the full 2.7.0 header (39 status functions + 2
non-status) was audited against the wrapper. Implemented: the five operation families plus
**`cutensorCreateContractionTrinary` / `cutensorContractTrinary`** (`E = alpha*opA(A)*opB(B)*opC(C)
+ beta*opD(D)`; `d: Option` with None = D-is-E, `opD` pinned IDENTITY, `descD == descE`,
**symbol-gated** on libcutensor >= 2.1.1 with a clean `UnImplemented` - cudarc's own `load` panics
on a missing symbol, so the wrapper probes via `culib().get` first; some dtype rows are Ampere+ and
cuTENSOR declines those itself), the plan-cache management (`resize_plan_cache`,
`write/read_plan_cache_to/from_file`) and kernel-cache persistence on `CutensorHandle`, and
`cudart_version`. Deliberately out of scope, documented in the module docs: `cutensorLogger*` (x6,
no tensor semantics), block-sparse (incl. `BLOCKSPARSE_REPRODUCIBLE`; the exported
`cutensorBlockSparse*` pair likewise), `cutensorOperationDescriptorSetAttribute` (setting
`SCALAR_TYPE` would defeat the wrapper's derived-type verification), `PlanPreference` Get/Set
attributes (2.6+ autotune knobs - future). Fact: `MOVED_BYTES` is block-sparse-only on the probed
versions (INVALID_VALUE for dense ops) and is not exposed. Fact: the one-shot derives `descC`
from `D`, so a `C`-slot view must match D's declared alignment - a row-slice C (240 B offset ->
16 B alignment vs the output's 256) honestly refuses in the one-shot and is served at the object
layer with a shared `with_alignment(8)` descriptor (tested both ways).