# Round 1 questions — cuTENSOR-backed `tensordot` / `einsum` on `DeviceCuda`

Design tree root: *`tensordot` and `einsum` on `DeviceCuda` served by cuTENSOR, exposed as
`rt::cuda::einsum` / `rt::cuda::tensordot`, implemented under `rstsr-cuda/src/einsum`*.
Round 1 is the trunk: layout policy, conjugate, dtype and precision scope, API shape, tier order,
FFI route, resources, module layout, the einsum subset, path policy, delivery. Tests, bench matrix,
flags/trace knobs and the batch plan hang off these and wait for round 2 (together with the results
of the empirical probe, which is still running).

Facts referred to below are in `GRILL-INIT-PROMPT.md` (sections "How einsum/tensordot exist
today" and "cuTENSOR facts"). The two notifications you asked for first:

- **Strides**: cuTENSOR takes *arbitrary positive per-mode strides* (elements) — sub-tensor and
  transposed views need no packing, modes may be in any order, up to 64 modes. It does **not**
  take a stride of zero (a broadcast is expressed by *omitting* the extent-1 mode from the mode
  list), a negative stride, or a zero extent; and the base-pointer alignment must be declared in
  the descriptor.
- **Conjugate**: `CUTENSOR_OP_CONJ` is a per-operand flag on A and/or B of a dense contraction
  (`opA`/`opB`); `opC` must be `IDENTITY` (2.8.0 started rejecting it; 2.7.0 and older accept a
  non-identity `opC` and silently compute garbage); it is for complex dtypes only; the block-sparse
  contraction has no conjugate at all. Conj-transpose is nothing extra: CONJ plus the mode order.

---

❓ **Q1** - **Layout policy: what happens to layouts cuTENSOR cannot express**: for each input
with (a) a stride-0 axis of extent > 1 (a broadcast view), (b) a negative stride (a reversed
view), (c) a zero extent, and for extent-1 axes with any stride, and for the *output* layout:

- (a)+(b): **materialize a contiguous copy of that operand** silently (one pass over `m*k`
  elements, buys the whole contraction; this is what makes `rt::cuda::einsum` just work).
  Extent-1 axes are dropped from the mode list (stride irrelevant, so no copy). (c) is
  special-cased without calling cuTENSOR: a zero *contracted* extent gives a zero-filled output,
  a zero *free* extent gives an empty output. The output is never copied: a user-supplied output
  (`_with_output`/`_from`, `tensordot_from`) with stride-0 or negative strides is refused by name;
  the allocating entries always produce a fresh contiguous output. Alignment is declared as the
  pointer's own power-of-two alignment (capped at 256), so offset views are served too.
- (b) refuse (a) and (b) by name, as the view-only tier refuses today.

➡️ (a). The pack is the cheap half of the linalg.rs "third tier" argument, and cuTENSOR is the
expensive half that was missing.

---

❓ **Q2** - **Conjugate**: cuTENSOR conjugates A and/or B for free. Should wave 1 expose it?

- (a) No conjugate in wave 1 (CPU parity: `tensordot`, `vecdot` and `rt::tblis::einsum` have
  none; einsum has no NumPy syntax for it).
- (b) Wave 1 adds an options struct used by the `rt::cuda::*` entries only, carrying a per-operand
  `conj: &[bool]` (tensordot: two flags); the plain entries are the default options. Real operands
  ignore the flag (conj of a real is identity, mapped to `IDENTITY` locally so cuTENSOR never sees
  CONJ on a real type). No string-syntax extension to einsum.

➡️ (b). It is one enum value per operand, the physics codes pay a materialized `conj()` for it
today, and the same struct is where the compute modes (Q4) go in wave 2.

---

❓ **Q3** - **Dtype coverage of the cuTENSOR tier**:

- (a) Same-dtype operands: `f32`, `f64`, `c32`, `c64` in wave 1; `f16`/`bf16` also in wave 1
  (cuTENSOR: `R_16F`/`R_16BF` with `COMPUTE_DESC_32F` and `f32` scalars — two table rows).
  Mixed operand dtypes (`f32 x f64`, `f32 x c32`, `f64 x c64`) and integers are **not** served
  in wave 1; mixed arrives in wave 2 as *materialized promotion* to the common dtype (the array-API
  `ext_tensordot` rule; also gives cuTENSOR's only native mixed case `R_64F x C_64F` for free).
  Integers never go to cuTENSOR (it has no integer contraction).
- (b) `f16`/`bf16` deferred to wave 2 as well.
- (c) Mixed dtypes via cuTENSOR's native `R_64F x C_64F` only, everything else refused.

➡️ (a). The half types are two rows in the dtype table and the matmul chain already carries them
on the cuBLAS side, so leaving them out saves little.

---

❓ **Q4** - **Compute-type policy**: which cuTENSOR compute descriptor per dtype?

- (a) Wave 1 is strict only: `f32`/`c32`/`f16`/`bf16` -> `COMPUTE_DESC_32F`; `f64`/`c64` ->
  `COMPUTE_DESC_64F` (on Ampere+ that already runs on the FP64 tensor cores, so nothing to
  emulate). The matmul modes (`FastTF32` -> `COMPUTE_DESC_TF32`, `Fast16BF` -> `16BF`, ...) are
  wired onto the options struct in wave 2, after the probe says whether cuTENSOR's TF32 path is
  even faster on this sm_120 card (the matmul probe found TF32 only ~1.5x there).
- (b) Wire the existing `ComputeType32F`/`ComputeType64F` modes onto cuTENSOR descriptors in
  wave 1.

➡️ (a). Numerics stay identical to the CPU default, and the mode plumbing is independent.

---

❓ **Q5** - **API surface**: `rt::cuda::einsum` and `rt::cuda::tensordot` mirror `rt::tblis::*`
one-for-one — same signatures, same operand list type (`impl TensorViewListAPI<T, DeviceCuda>`),
same `optimize: impl PathOptimizer` and `memory_limit: impl Into<SizeLimitType>`:

- `einsum` / `einsum_f` (allocate), `einsum_with_output` / `_f` (write into a user output),
  `einsum_from` / `einsum_from_f` as the trait `TensorCudaEinsumFromAPI` on `TensorAny<_, T,
  DeviceCuda, IxD>` (`self` is the output; beta = 0, as tblis does);
- `tensordot` / `tensordot_f` (+ `_with_output[_f]`) with the core `axes` overload;
- all in `prelude::rstsr_funcs` / `rstsr_traits`, so the facade gives `rt::cuda::*`.
- And the **core** `DeviceTensordotAPI` impl for `DeviceCuda` (behind `rt::tensordot`,
  `a.tensordot(b, axes)`, `rt::tensordot_from`) also routes through the cuTENSOR tier, so
  `rt::tensordot` and `rt::cuda::tensordot` give the same numbers and `rt::tensordot` stops being
  the "view-only" entry on CUDA. Output dtype = operand dtype (as tblis).

➡️ Yes to all of it. `rt::cuda::tensordot` then differs from `rt::tensordot` only by the options
struct (Q2) and the `_with_output` form.

---

❓ **Q6** - **Tier order and fallback**: when libcutensor is loadable and serves the
(dtype, layout), is cuTENSOR tried *before* the existing view-GEMM tier (cuBLAS)?

- (a) **cuTENSOR first** (your stated direction: on CUDA a contraction is a cuTENSOR call, not a
  GEMM scheduling problem). The view-GEMM tier stays as the fallback for machines without
  `libcutensor`, followed by today's refusal; no new kernel in wave 1. A "pack + GEMM" tier for
  cuTENSOR-less machines and integers is wave 2.
- (b) View-GEMM first (no plan-cache lookup on the host, the proven path), cuTENSOR for the
  layouts that do not collapse.

➡️ (a). The B1 check measures the per-call host cost of a cuTENSOR plan-cache hit against a
cuBLAS launch; a flag flips the order if it ever matters (round 2).

---

❓ **Q7** - **FFI route**: cudarc 0.19.10's `cutensor` feature (= `cutensor-02006`, 2.6 bindings,
2.3-2.6 supported; `sys` + `result` layers, no safe layer) is the only new FFI surface, and the
nine compute descriptors (exported const globals, skipped by bindgen) are resolved through the
already-public `cudarc::cutensor::sys::culib()` with `get::<*const cutensorComputeDescriptor_t>
(b"CUTENSOR_COMPUTE_DESC_64F\0")` — a symbol lookup inside `einsum/cutensor.rs`, no new crate, no
new dependency, no link-time binding. The handle creation checks `cutensorGetVersion() >= 20300`
(the installed library is 2.7.0; soname `.2`; the functions we call are unchanged since 2.3).

- (b) A new `rstsr-cutensor-ffi` crate in the rstsr-ffi repo instead of cudarc's module (the
  2026-10-21 "own FFI repo" rule).
- (c) An `extern "C"` block inside rstsr-cuda (link-time; breaks the dynamic-loading setup).

➡️ (a). It is the matmul wave's "no new FFI, cudarc only" rule with one documented symbol lookup.

---

❓ **Q8** - **Resource model**: one cuTENSOR handle per **ordinal** (`CudaOrdinalObj::cutensor_slot`,
`OnceLock<Option<Arc<CutensorHandle>>>`, created after `context().bind_to_thread()`, the same
"cached: the probe works once or never" shape as `cublaslt()`); the stream is a per-call
argument of `cutensorContract`, so no per-stream handle and no mutex (cuTENSOR's handle APIs are
thread-safe by contract); rely on cuTENSOR's **built-in plan cache** (on by default) instead of
our own — descriptors are cheap host objects and the cache is keyed for us; a per-stream
workspace like `cublaslt_workspace()`, sized by a new `CudaRuntimeFlags::cutensor_workspace_bytes`
(default 128 MiB = cuTENSOR's recommendation, 0 = none).

- (b) Per-stream handle like cuBLAS plus our own plan cache keyed by problem (like the Lt driver).
- (c) No workspace in wave 1.

➡️ (a). Our own cache only if B1 measures the built-in hit as slow.

---

❓ **Q9** - **Module layout**: `crates-device/rstsr-cuda/src/einsum/` with `mod.rs` (doc include +
re-exports only, the `matmul/mod.rs` rule) and `doc.md` (markdown page), `cutensor.rs` (handle,
RAII wrappers for tensor descriptor / operation descriptor / plan preference / plan, status ->
`Error` mapping, the compute-descriptor symbols, the `cudaDataType_t` table per dtype),
`parser.rs` (subscripts -> per-operand mode ids via `opt_einsum_path::parser::parse_einsum_input`,
which already handles whitespace, ellipsis and implicit output), `path.rs` (pairwise sequence),
`layout.rs` (Layout -> extents/strides/alignment/mode omission and the Q1 materialization
decisions), `tensordot.rs` (the `DeviceTensordotAPI` impl moves here from `linalg.rs`, plus
`rt::cuda::tensordot`), `tensor.rs` (free functions + `TensorCudaEinsumFromAPI`). New dependency:
`opt-einsum-path` (already a workspace dependency for tblis; pure Rust). `linalg.rs` keeps
`outer`/`cross` and points at `einsum` for `tensordot`.

➡️ Yes.

---

❓ **Q10** - **einsum feature subset of wave 1**: served — explicit or implicit output, ellipsis,
N-ary operands through the path, any mix of batched/contracted/free modes, outer products,
Hadamard products, and single-operand permutations/copies (`ij->ji`, `i->i`) through the existing
strided-assign kernels; refused with a named error — diagonal (a label repeated within one
operand, `ii->i`), unary reductions (`ij->`, `ij->i`, `i->`; wave 2 via `cutensorReduce`, which
cudarc binds, or the existing sum kernels), and anything cuTENSOR itself refuses.

➡️ Yes. Diagonals and unary reductions have no caller in the dfmp2-style workloads and each is
its own small executor.

---

❓ **Q11** - **Contraction path and intermediates**: `opt_einsum_path::contract_path` with the
caller's `optimize` (so `true` = auto, as tblis) and `memory_limit`; intermediates are fresh
contiguous `Tensor<T, DeviceCuda, IxD>` in the device's default order, dropped at the end of the
call; the final pairwise contraction writes straight into the output (fresh or user-supplied).

- (b) Fixed left-to-right order (no optimizer) in wave 1.

➡️ (a). The optimizer is already a dependency of the tblis path and is pure host code.

---

❓ **Q12** - **Delivery**: implementation on the rstsr repo, continuing from the current head
(`261127/cuda-prelude-f-convention` @ `937ce04`, one commit ahead of `261021/cuda`; the prelude
wiring `rt::cuda::einsum` needs is in that commit), no PR until you say so; tests in
`crates-device/rstsr-cuda/tests/test_einsum.rs` (hardware-skipping, copy-and-compare against the
CPU devices) plus `--lib` units for the pure functions; `benches/bench_einsum.rs` in the crate's
`harness = false` style; the trajectory dir `2026-11-27-cuda-einsum-cutensor/` keeps the probe,
the plan and the batch reports. Local development uses the pip cuTENSOR 2.7.0 of the active conda
env through `LD_LIBRARY_PATH` (documented in the crate doc, like the tblis linking note); CI is
unchanged (no GPU runner exists today).

➡️ Yes.
