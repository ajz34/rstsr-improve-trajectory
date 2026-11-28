# Round 2 questions — cuTENSOR-backed `tensordot` / `einsum` on `DeviceCuda`

## Background: round-1 answers as received on screen (agent transcription, 2026-11-27)

- Owner declined the wave 1 / wave 2 split: **everything is designed now**; batches may order the
  implementation, not defer scope.
- Q1 accepted: inputs with stride-0 (extent > 1) or negative strides are copied; zero extents are
  special-cased; outputs are never copied.
- Q2 yes, with a different shape than proposed: explicit entries `einsum_with_conj`, and since
  `einsum_with_output` exists also `einsum_with_output_conj`; `einsum` stays conjugate-free. The
  owner was unsure whether `einsum_from` exists in `rt::tblis` (it does, see Q3 below).
- Q3: "(a) or (c); see if (c) is easy".
- Q4: explanation requested.
- Q5 accepted (plus the Q2 note). Tensordot is the secondary entry: "we will usually not use
  tensordot, and let it work correctly is good"; einsum is the general one and gets its careful
  design in the plan.
- Q6 (a), Q7 (a), Q8 ok, Q11 ok; Q9 and Q12 "approximately yes" (details go to the plan).
- Q10: "as long as opt-einsum-path accepts and cuTENSOR returns correct value, we can allow them";
  fact-check needed for reduction / transposition / diagonal (probe2 dispatched).

Two probes are running: `probe/` (contraction: strides, broadcast, alignment, CONJ, scalar types,
planning cost, cuTENSOR-vs-cuBLAS timing) and `probe2/` (reduction, permutation, diagonal-by-view,
`beta = 0` read behaviour, broadcast-by-omission). Their results are a prerequisite for round 3
(algorithm choice, autotuning, intermediate layout, the single-operand engine); nothing below
depends on them.

Facts looked up for this round:

- `rt::tblis` **has** `einsum_from`: the trait `TblisEinsumFromAPI<T, B>` (`rstsr_tblis::prelude::rstsr_traits`,
  so a trait under `rt::tblis`) with `einsum_from(subscripts, operands, optimize, memory_limit)`
  and `einsum_from_f`, implemented for `TensorAny<R, T, B, IxD>` with `R: DataMutAPI`; `self` is the
  output and is **overwritten** (`einsum_with_output(self.view_mut(), ..)`, i.e. beta = 0). The
  tblis prelude's `rstsr_funcs` exports the four free functions `einsum`, `einsum_f`,
  `einsum_with_output`, `einsum_with_output_f`; `rstsr_traits` exports `TblisEinsumFromAPI` (the
  `einsum_from` methods) and `RTToTblisTensorAPI`.
- The existing compute-mode flags: `CudaRuntimeFlags::compute_type_32f: ComputeType32F
  {Default, Pedantic, Fast16F, Fast16BF, FastTF32}` (read by `f32`/`c32`-output matmuls),
  `compute_type_64f: ComputeType64F {Default, Pedantic, EmulatedFixedPoint}`, `compute_type_16f:
  ComputeType16F {Default, Pedantic, Acc16F}`. `matmul_drivers: MatMulDrivers` is a bit set with
  `CUBLAS_LT | CUBLAS_EX | CUBLAS | TILED | NAIVE`; `trace_matmul: bool`.
- `diagonal` is a pure layout op in rstsr-core (`tensor/indexing.rs`: `diagonal_f` returns a
  `TensorView`), so a "diagonal view" needs no kernel and no copy.
- cuTENSOR mixed dtype: `R_64F x C_64F -> C_64F` and `C_64F x R_64F -> C_64F` are two rows of
  the contraction table (scalar type `CUDA_C_64F`); the descriptor data type is per operand, so
  they cost two more `DeviceTensordotAPI` impls and nothing else.

---

❓ **Q1** - **Compute types, explained and decided**: cuTENSOR lets the *internal* precision differ
from the storage dtype, chosen by the compute descriptor:

| storage | strict descriptor | reduced descriptors |
|---|---|---|
| `f32`, `c32` | `COMPUTE_DESC_32F`: every product and sum in fp32, no tensor cores | `TF32` (10-bit mantissa inputs, fp32 accumulate, tensor cores on Ampere+; the matmul probe measured ~1.5x on this card with 2.75e-4 mantissa error), `3XTF32` (three TF32 products, ~fp32 accuracy, tensor cores), `16BF` / `16F` (half-precision products; fastest, least accurate) |
| `f64`, `c64` | `COMPUTE_DESC_64F`: full double, **already on the FP64 tensor cores** on Ampere+ | `32F` (accumulate in fp32; no tensor cores either, so nothing to gain) |
| `f16`, `bf16` | `COMPUTE_DESC_32F`: accumulate in fp32 | `16F` / `16BF` compute (16-bit accumulation) |

The matmul chain already has the house mechanism: the three `compute_type_*` flags above. Should
einsum and tensordot **read the same flags**, so one setting governs the whole crate, with this
mapping: `Default`/`Pedantic` -> the strict descriptor; `FastTF32` -> `TF32`; `Fast16BF` -> `16BF`;
`Fast16F` -> `16F`; `EmulatedFixedPoint` (f64; classic cuBLAS only) -> strict `64F` (cuTENSOR has no
emulation); `Acc16F` -> strict `32F` (cuTENSOR's contraction table lists no 16-bit accumulation for
the half types)? Flags only: no per-call mode entries for einsum (matmul has them because you asked
for them there).

- (b) Strict descriptors only; the flags do not reach einsum/tensordot.

➡️ (a). A user who set `FastTF32` for `matmul` expects it everywhere, and the default stays strict.

---

❓ **Q2** - **Mixed operand dtypes**: (c) is easy: the two native rows are two more
`DeviceTensordotAPI<f64, Complex<f64>, Complex<f64>, ..>` / `<Complex<f64>, f64, Complex<f64>, ..>`
impls with the per-operand descriptor type. Should the design take (c) **and** the promotion route
for the remaining pairs (`f32 x f64 -> f64`, `f32 x c32 -> c32`, `f16 x f32 -> f32`, ...): cast each
operand to the common dtype into a contiguous temporary with the existing CUDA cast kernels
(`DTypePromoteAPI` decides the type), then cuTENSOR — that is exactly `DeviceExtTensordotAPI`
(array-API `ext_tensordot`, promotion-compatible). `einsum` stays same-dtype, as tblis: its operand
list is `TensorViewListAPI<T, B>`, and a heterogeneous list would be an rstsr-core API change.
Integers never go to cuTENSOR.

➡️ Yes: native rows for free, promotion copies for the rest, einsum same-dtype.

---

❓ **Q3** - **The entry family (naming)**: mirror tblis and add the conjugate twins:

- free functions in `rt::cuda`: `einsum[_f]` (allocate), `einsum_with_conj[_f]`,
  `einsum_with_output[_f]`, `einsum_with_output_conj[_f]`;
- trait `TensorCudaEinsumFromAPI<T>` on `TensorAny<R, T, DeviceCuda, IxD>` with `R: DataMutAPI`
  (the tblis `einsum_from` shape): methods `einsum_from[_f]` and `einsum_from_conj[_f]`, `self`
  overwritten (beta = 0, as tblis);
- the conjugate flag is `conj: impl AsRef<[bool]>`, one entry per operand in operand order,
  a length mismatch is an error; a real operand ignores its flag (mapped to `IDENTITY` locally);
  parameter order `einsum_with_conj(subscripts, operands, conj, optimize, memory_limit)`,
  `einsum_with_output_conj(subscripts, operands, conj, optimize, memory_limit, output)`;
- `output` is `impl TensorViewMutAPI<Type = T, Backend = DeviceCuda, Dim = DC>` (any rank, as the
  CUDA matmul entries take views), not tblis's `TensorMut<_, _, IxD>`;
- tensordot: `tensordot[_f]` and `tensordot_with_output[_f]` as tblis defines them, **no conjugate
  form** (a conjugated two-operand contraction is `einsum_with_conj`).

➡️ Yes.

---

❓ **Q4** - **The last tensordot tier (no waves)**: the chain for the core `DeviceTensordotAPI`
becomes cuTENSOR -> view-GEMM (today's zero-copy cuBLAS tier, kept for machines without
libcutensor and for dtypes cuTENSOR declines) -> **pack + GEMM**: materialize each operand in its
`[free.., ctr..]` / `[ctr.., free..]` order with the strided-assign kernel, reshape to 2-D, and
run the existing matmul chain (which ends in the naive kernel) into the output. That last tier is
what recovers the CPU capability on CUDA — integers, mixed dtypes through the naive kernel's
in-kernel cast, cuTENSOR-less machines — and is the "third tier" `linalg.rs` already argues for;
about a hundred lines, no new kernel.

- (b) No pack tier: refuse by name when cuTENSOR cannot serve (today's behaviour for everything
  cuTENSOR does not cover).

➡️ (a). Your bar for tensordot is "work correctly"; this is what makes that true for every
`Mul` triple the CPU serves.

---

❓ **Q5** - **einsum feature policy**: serve everything opt-einsum-path accepts **and** cuTENSOR
computes correctly, using layout rewrites where cuTENSOR has no direct form:

- diagonal (`ii->i`, `iij->j`, `iij->ij`, `iij,jk->ik`): a **diagonal view** per operand before
  the step (shape loses one axis, stride = `s_i + s_j`, extent = the common extent; unequal extents
  are an error, NumPy's rule) — no kernel, no copy; implicit output drops a label repeated inside
  one operand (so `ii` is the trace, as NumPy);
- mode-dropping (`ij->j`, `ij->`, `i->`): cuTENSOR Reduce (`OP_ADD`), written directly into the
  final output's mode order; a scalar output is a rank-0 descriptor if cuTENSOR accepts one, else a
  one-element 1-D output reshaped to rank 0;
- reorder / copy / broadcast (`ij->ji`, `i->i`, `i->ij`): cuTENSOR Permute (broadcast by omitted
  extent-1 mode);
- two-or-more-operand steps: cuTENSOR Contract (Hadamard = zero contracted modes, outer = no
  shared mode, A-only modes = contracted), along the opt-einsum-path sequence.
- Which **engine** (cuTENSOR Reduce/Permute versus the crate's existing assign and sum kernels)
  serves the single-operand cases, and the algorithm/autotune settings, wait for probe2/probe
  (round 3); this question is the policy.

➡️ Yes.

---

❓ **Q6** - **Decline versus error**: a cuTENSOR `NOT_SUPPORTED` / `INVALID_VALUE` at descriptor,
operation-descriptor or plan creation is a **decline**: in tensordot the next tier is tried, in
einsum it is an error naming the step, the subscripts and the status (einsum has no fallback
engine: when libcutensor is not loadable, `rt::cuda::einsum` errors by name rather than running a
slow path). A failure at execution (`cutensorContract`/`Reduce`/`Permute`) is an error: nothing was
enqueued and something is wrong. An operand whose storage is the output's storage is refused
(cuTENSOR forbids A/B overlapping D and we do not prove disjointness); the `_from` forms therefore
never accept `self` among the operands.

➡️ Yes.

---

❓ **Q7** - **Flags**: `CudaRuntimeFlags` gains `tensordot_drivers: TensordotDrivers` (bit set
`CUTENSOR | VIEW_GEMM | PACK_GEMM`, the `MatMulDrivers` pattern, so a test can pin a tier),
`trace_einsum: bool` (one line per step: subscripts, engine, dtype, shape), and
`cutensor_workspace_bytes: usize` (default 128 MiB, `0` = none). Compute types reuse the existing
fields (Q1). Algorithm and autotuning knobs only if round 3 finds they matter. No environment
variable for any of it.

➡️ Yes.

---

❓ **Q8** - **Test oracle and matrix**: the oracle is a **test-local naive einsum** in pure Rust
on the CPU device (loops over the subscript; supports ellipsis, implicit output, diagonal,
per-operand conjugation) plus the existing CPU `rt::tensordot` for the two-operand cases — no
tblis dev-dependency (its C library would become a test prerequisite). Matrix: dtypes
{`f32`,`f64`,`c32`,`c64`,`f16`,`bf16`} x layouts {contiguous, stepped slice, transposed, offset
view, extent-1 broadcast mode, reversed view (copied), stride-0 extent>1 (copied)} x features
{pairwise, batched, A-only-mode, Hadamard, outer, N-ary path, permute, broadcast, reduce, scalar
output, diagonal, ellipsis, implicit output, conj variants} x output forms {allocate,
`_with_output` contiguous and strided, `_from`} x both default orders; integer-valued data with
`assert_eq!` where the dtype is exact, `assert_close` otherwise (the crate's convention); `--lib`
units for the pure functions (subscript parsing, mode/stride/alignment derivation, the
materialization decision, the path-to-steps conversion, flags); hardware tests skip without a GPU.

➡️ Yes.

---

❓ **Q9** - **Acceptance and bench set** (no hard speedup target; numbers recorded): (1) every
served feature matches the oracle for all six dtypes; (2) on GEMM-shaped contractions the cuTENSOR
tier is within 10 % of the view-GEMM/cuBLAS tier in device time, and the per-call host cost of a
plan-cache hit is measured and reported; (3) non-GEMM contractions (`abc,cbd->ad`, the tblis
docstring's `μi,νa,μνκλ,κj,λb->iajb`, batched `[p,n,k] x [p,k,m]`, a multi-axis `K`) are
benchmarked against the pack tier and the numbers recorded — no GPU baseline exists for them.
Shapes: `f64`/`f32` 2048^3, `c64` 1024^3, batched 64x(256,256), 128^3 three-mode, the dfmp2
contraction at nao = 32; strided variants of each. `benches/bench_einsum.rs`, `harness = false`
like the crate's other benches.

➡️ Yes.

---

❓ **Q10** - **Intermediate layouts**: each pairwise intermediate is a fresh contiguous tensor in
the device's default order with the axis order of that step's output subscript (shape = the
step's free labels in subscript order); dropped at the end of the call (stream-ordered frees);
`memory_limit` goes to the path optimizer only. The final step writes straight into the output
layout (fresh, or the user's `_with_output`/`_from` layout).

- (b) Choose each intermediate's axis order by a heuristic (e.g. follow the largest operand's
  stride order, cuTENSOR's performance guideline).

➡️ (a); (b) only if the bench shows the intermediate layout matters.

---

❓ **Q11** - **Plan structure**: batches order the implementation without deferring scope —
B1 probe facts frozen + skeleton (handle, RAII wrappers, status mapping, flags, dtype table),
B2 tensordot: cuTENSOR tier + the three-tier chain + tests, B3 einsum: parser, path, executor,
entries, tests, B4 bench, module doc page, review. All on the rstsr repo from the current head,
no PR until told.

➡️ Yes.

---

# Addendum (2026-11-28) — aligning the cuTENSOR usage model; supersedes Q1, Q3 and Q7

Owner's position before answering round 2: (i) cuTENSOR handles one input type with output type =
input type, and the `cutensorComputeDescriptor` controls the internal precision; (ii) the
descriptor set is finer-grained than cuBLAS's compute types, so cuTENSOR gets **its own** entries
and flags, organized like matmul but standalone (Q1's "read the same flags" and Q7's reuse are
withdrawn).

## Comprehension check (facts from the 2.7.0 header tables)

- **Same input type, output = input type — right for contraction, with two exceptions**:
  `R_64F x C_64F -> C_64F` and `C_64F x R_64F -> C_64F` (f64 real times c64 complex). There is no
  `R_32F x C_32F` (descriptor accepted, plan NOT_SUPPORTED — probe 1). Every other row is
  A = B = D. Reduction and permutation *can* change the type (`R_64F -> R_32F`, `C_64F -> C_32F`,
  `R_32F -> R_16F`, ...), but einsum keeps the type, so that ability stays unused. For einsum
  (same-dtype operand list) the rule holds outright; the exception matters only for
  `ext_tensordot` (Q2).
- **Descriptor = internal precision — right, with one nuance**: it is a *minimum accuracy
  guarantee*, not an exact specification; cuTENSOR may pick a higher precision (user guide
  "Accuracy Guarantees"; probe 1: f64 data under `COMPUTE_DESC_32F` came out f64-accurate). The
  scalar type follows the output type + descriptor (c64 output -> complex `alpha`/`beta`).
- **The descriptor set per storage type (contraction table)**:

| storage | descriptors |
|---|---|
| `f16`, `bf16` | `32F` only (fp32 accumulate) — no choice |
| `f32` | `32F` (strict, no tensor cores), `TF32`, `3XTF32`, `16BF`, `16F` (Ampere+/Volta+), `9X16BF` (Hopper+), `4X16F` (Blackwell+) |
| `c32` | `32F` (no tensor cores), `TF32`, `3XTF32`, `9X16BF` (Hopper+) — no `16BF`/`16F`/`4X16F` for complex |
| `f64` | `64F` (FP64 tensor cores, Ampere+), `32F` (no tensor cores), `8XINT8` (Hopper+; the INT8-based FP64 emulation) |
| `c64` | `64F` (tensor cores, Ampere+), `32F`, `8XINT8` (Hopper+) |

  cuBLAS's modes have no 1:1 counterpart (`3XTF32`, `4X16F`, `9X16BF`, `8XINT8` are cuTENSOR-only;
  cuBLAS `Pedantic` has no cuTENSOR counterpart because the descriptor is already a floor), so
  standalone types are the honest shape. `Hopper+`/`Blackwell+` rows are expected on this sm_120
  card but were not probed (B1 measures them).

---

❓ **Q12** - **Standalone compute types**: two `#[non_exhaustive]` enums keyed by the **output**
family (the matmul convention), named after the descriptors, default = strict:

- `CutensorCompute32F { F32 (default), TF32, ThreeXTF32, F16BF, F16F, FourX16F, NineX16BF }` read
  by `f32` and `c32` outputs; a variant the output type cannot serve (`c32` with `F16BF`/`F16F`/
  `FourX16F`) resolves to `F32` silently, as matmul's 32F modes resolve for complex outputs.
- `CutensorCompute64F { F64 (default), F32, EightXInt8 }` read by `f64` and `c64` outputs.
- `CutensorCompute16F { F32 }` for `f16`/`bf16` outputs: a single-variant enum so the per-call
  entries are callable for every dtype through `<T as CutensorComputeMode>::Mode`.
- `CudaRuntimeFlags::cutensor_compute_32f` and `cutensor_compute_64f` (standalone; the matmul
  `compute_type_*` fields are not read by einsum/tensordot; half has no flag field because it has
  one descriptor).

➡️ Yes.

---

❓ **Q13** - **Per-call mode entries**: like matmul, the mode can also be given per call. Entry
family = {flags | per-call mode} x {no conj | conj} x {allocate | with_output | from}:
`einsum[_f]`, `einsum_with_conj[_f]`, `einsum_with_output[_f]`, `einsum_with_output_conj[_f]`,
`einsum_from[_f]`, `einsum_from_conj[_f]` (Q3, flags), plus `einsum_with_mode[_f]`,
`einsum_with_conj_mode[_f]`, `einsum_with_output_mode[_f]`, `einsum_with_output_conj_mode[_f]`,
`einsum_from_mode[_f]`, `einsum_from_conj_mode[_f]` (per-call; parameter order
`(subscripts, operands, [conj,] mode, optimize, memory_limit[, output])`). Mechanical and
discoverable, 12 names. Tensordot takes no mode entries (flags only).

- (b) The `_with_mode` entries also carry `conj` (the "full" form), 9 names: `einsum_with_mode`,
  `einsum_with_output_mode`, `einsum_from_mode` all take `(.., conj, mode, ..)`.

➡️ (a); (b) hides conj behind a name that does not say so.

---

❓ **Q14** - **Other cuTENSOR knobs** — algorithm (`ALGO_DEFAULT` / `DEFAULT_PATIENT` / `GETT` /
`TGETT` / `TTGT`), JIT mode, incremental autotuning — are the other fine-grained controls; they are
planning knobs rather than precision knobs. Proposal: round 3 decides them from the probe numbers
(probe 1: `ALGO_DEFAULT` plans in 20-50 us, non-GEMM at the full f64 rate), exposed as
`CudaRuntimeFlags::cutensor_algo` only if a variant measurably matters; no per-call entries for
them.

➡️ Yes: precision gets entries (Q12/Q13), planning knobs get at most a flag (round 3).
