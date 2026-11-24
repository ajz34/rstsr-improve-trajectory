# PR #4 record — what the trajectory missed

The trajectory recorded the device-basic and kernel/NVRTC decisions, the
elementwise wave and the pinned/streaming work, but **not** PR #4 (reductions,
linalg, indexing, numerics, arch matrix, owned ops) nor the owner's review
directives. This file is that missing record.

- PR: [ajz34/rstsr#4](https://github.com/ajz34/rstsr/pull/4), title `261021/cuda`
- Author: `atlaschan1310` (Yang Qian)
- Base: `261021/cuda`; merged 2026-11-23T06:35:46Z as `6d17d04`
- Reviewed head: `a4d717a` on `261119/pr-4-cuda`
- Rebased final history: `6d17d04` with 30 commits from `451c2a9`; the PR body
  says "14 commits" because it was written mid-review. Seven of the 30 are
  `DEBUG:` vecdot commits that were folded by the rebase; the final history has
  18 commits after the vecdot work.
- No reviews and no inline comments; three issue comments from the owner
  (`ajz34`), quoted in §3.
- One core change (`rstsr-core/src/tensor/reduction.rs`); everything else is in
  `crates-device/rstsr-cuda`.

---

## 1. What PR #4 added, by component

### Reductions (the largest block)

| File / module | What it introduced |
|---|---|
| `kernels/reduce_plan.rs` | Host-side classifier: `lanes`, `outputs_per_block`, `n_contract`, `step_a/b`, `n_out`, `out_step_a/b`; `MIN_BLOCKS = 512` policy constant; lane placement on the unit-stride axis; shared-memory stage above 32 lanes; refusal with named errors. Pure functions, tested without a GPU. |
| `kernels/reduce_unary.rs` | Single-accumulator fold family: sum, prod, mean, l2_norm, min, max, all, any, count_nonzero; `AccSpec` (C type name + ±inf spellings). |
| `kernels/reduce_pair.rs` | Two-slot `(Σx, Σx²)` for var/std; `scale_pair_output` applies `correction`. |
| `kernels/reduce_arg.rs` | `(value, index)` accumulator with lexicographic combine; cannot split into independent slots. |
| `kernels/scan.rs` | `cumsum`/`cumprod` with chunk carry and `include_initial`, no plan (every prefix is an output). |
| `reduce_ops.rs`, `arg_ops.rs`, `scan_ops.rs` | Device-trait impls, dtype variants, bool sum, `nanarg*`. |

Design decisions the PR made explicit (and the review accepted):

1. The reduction plan is **data**, a pure function of shape/strides/order —
   never a device or occupancy query. The summation tree is reproducible for a
   shape and a binary.
2. A kernel family gets its own file when the accumulator changes shape
   (`reduce_unary` one register, `reduce_pair` two, `reduce_arg` a struct,
   `scan` no plan).
3. The policy is **refuse rather than fall back**; there is no generic
   slow path for unclassified layouts. `vecdot` exposed a case where the plan
   was accepted but the two operands enumerated the contraction in different
   memory orders; `a4d717a` refuses it instead of returning a wrong result.

### Linalg

| Component | What it introduced |
|---|---|
| `kernels/gemm.rs` | Shared `resolve_geom` (matrix axes, transpose flags, batch folding) and a tiled GEMM kernel; `RSTSR_CUDA_MATMUL=tiled` forces it. |
| `kernels/cublas_gemm.rs` | cuBLAS typed `gemm`/`gemm_strided_batched`; column-major mapping; operand swap; per-stream handle + mutex; `RSTSR_CUDA_DEBUG_GEMM`. |
| `linalg.rs` | `outer`, `cross`, `matmul`/`matmul_uninit`, `tensordot` (view-only tier), `vecdot` (two-stage shuffle). |

The batch-stride bug is worth remembering: cuBLAS's strided-batched API takes
strides in **elements**, the code initially passed bytes. `batchCount == 1`
hides it completely; on a batched f64 matmul it wrote the first `m*n` outputs
correctly and scribbled elsewhere. `batch_strides()` and a unit test
(`batch_strides_are_element_counts`) now make the unit explicit.

### Indexing

`index_select` (gather by host index slice) and `index_put` (scatter,
last-write-wins order) in `adv_index_ops.rs` + `kernels/adv_index.rs`. This is
the pair the workload needed; the rest of the indexing family (mask, fancy
indexing, along-axis) was left out.

### Core change: `var`/`std` correction

`var_with_args`/`std_with_args`/`var_with_dtype` applied the array-API
`correction` rescale with `mapv(|x| x * factor)`, so their `where` clauses
required `Op_MutA_RefB_API<..., dyn Fn(&mut T, &T)>`. A host closure cannot
exist on CUDA, and a `where` clause is checked even when the branch is not
taken, so `var_with_dtype(x, None)` did not compile on CUDA.

The PR first replaced the closure with `OpMulAPI` and a new `rescale_var`
helper. The owner then made the more thorough change on the review branch
(`7b901ff`): `correction: Option<f64>` moved into `OpVarAPI`/`OpStdAPI`/
`OpVarDtypeAPI`/`OpStdDtypeAPI`, the tensor layer kept only the reduction
trait bound, and the CUDA `scale_pair_output` was updated. The public
`rt::var*`/`rt::std*` signatures and semantics are unchanged.

This is the concrete case behind the owner's general rule: **a tensor-tier
trait corresponds to one device trait; the dtype bounds live in the device
impl, not in core.** `mapv` stays host-only and is now documented as such.

### Numerics, arch matrix, owned ops, tests

| Change | Detail |
|---|---|
| Numerics contract | `lib.rs` three tiers: bit-identical elementwise, deterministic-but-differently-ordered reductions, different-implementation transcendentals; `RSTSR_CUDA_STRICT_FP=1` disables FMA contraction. |
| Arch matrix | `RSTSR_CUDA_ARCH=compute_NN`, `examples/arch.rs`, `scripts/arch-matrix.sh`; four arches on one card, disk cache keyed by arch. |
| Owned `a + b` | Consuming-operand binaries (`OpLConsume*`/`OpRConsume*` for add/sub/mul/div) so `a + b` on owned tensors reuses the buffer. |
| Tests | `test_kernels.rs` (reductions/scan/arg/vecdot), `test_ops_wave.rs` (elementwise/dtype/complex), `test_matmul_tiled.rs` (tiled path), host-side plan/argument tests. |
| Guard | cuBLAS `beta == 0` contract, batch strides, operand enumeration order, `beta`-zero empty-contraction. |

### What it did **not** add

- No sort/search/set; no mask/nonzero/array_index; no `tril`/`triu`; no `neg`;
  no `isclose`/`allclose`.
- No cuSOLVER, no cuBLASLt, no TF32 consumption, no complex/integer/half GEMM.
- No allocator framework, DLPack, graphs, multi-GPU peer copy.
- No throughput numbers for the new families; the PR body says the benchmark
  harness is maintained outside and only elementwise had one by then.
- No facade `rt::cuda` wiring.

---

## 2. Corrections to the PR body (if reused as a report)

1. **Commit count**: 14 at the time of writing; the final PR had 30 commits
   (`451c2a9..a4d717a`), 18 after the rebase folded the vecdot debug commits.
2. **cuBLAS handle scope**: the body says "one registry object per ordinal →
   one stream → one cuBLAS handle" and later "the registry shares one handle
   per ordinal". The code has one handle **per stream**, lazily created in
   `CudaStreamObj` and shared under a mutex. Per-stream is the correct design
   (a handle is bound to a stream); the body is stale.
3. **`allclose`**: the body says it is absent because `allclose_axes` is
   unimplemented on the CPU. That is true of `allclose_axes`, but user-facing
   `allclose` uses `allclose_all`/`OpAllCloseAPI`, which the CPU implements.
   `allclose` is missing on CUDA because `OpAllCloseAPI` was not implemented.
4. **`take_along_axis`/`put_along_axis`**: the body says they take a device
   index buffer. The tensor layer actually requires `Raw = Vec<usize>`, a host
   index tensor; the CUDA blocker is that core gate, not a read-back policy.
5. **Throughput**: "not measured" was true for PR #4 at merge; the elementwise
   table (2026-10-24) and the streaming table (2026-11-24) now exist, but there
   is still no reduction/linalg/indexing benchmark.

---

## 3. Owner review directives (2026-11-19)

Quoted from the three issue comments on `ajz34/rstsr#4`; Chinese original
abridged, English is the owner's.

**(a) Acceptance**

> Broadly, I'll accept all of the changes as they are. For a few smaller
> decisions I'd adjust, I'll push directly to your branch so the PR content is
> updated. For the parts that are numerically correct today but need a larger
> structural change, I'll take the PR as it is and we can discuss and refactor
> them separately afterwards.

Implication for the roadmap: small fixes may be pushed directly; structural
work (cuBLASLt, cuSOLVER, linalg trait tier) is explicitly deferred to a
separate refactor, not rejected.

**(b) `var`/`std` and the core design rule**

> The tensor trait interface of rstsr-core generally corresponds to one device
> trait, and generally only that device trait is used for computation. The trait
> conditions the data type must satisfy are given by the device impl, not
> imposed in rstsr-core. Some special traits such as Send/Sync/Clone/Copy are
> exceptions, but they should be used as sparingly as possible. … `mapv` can
> hardly be implemented on a CUDA backend.
>
> correction is pushed down to the device op trait: the methods of `OpVarAPI` /
> `OpStdAPI` / `OpVarDtypeAPI` / `OpStdDtypeAPI` gain a `correction: Option<f64>`
> parameter; the tensor layer keeps only the reduction trait bound. … Your CUDA
> implementation has been updated in the same change (a new `scale_pair_output`;
> the kernel needs no change).

This is the template for every future accelerator-facing core fix: move the
policy/value into the device op, keep the tensor tier generic, never require a
closure.

**(c) ReducePlan**

> For ReducePlan … it performs the reduction per its rules on some key cases —
> those where each part can be written as a single affine walk (good
> continuity, but not limited to it); and it refuses the reduction when it meets
> a case unsuitable for handling.
>
> Therefore, unlike the CPU implementation, the current CUDA implementation
> prioritizes efficiency (and prefers to refuse), and for now has no fallback
> implementation that supports all tensor layouts.
>
> The fix in `a4d717a` avoids the case where … vecdot (as a binary tensor
> reduction) gives a wrong result under a wrongly-accepted reduction plan.

The refusal policy is **accepted for now**, not a defect to "fix" immediately;
a generic fallback is future work (roadmap W5), not a review blocker.

**(d) matmul/tensordot direction**

> For matmul (gemm) and tensordot, the future direction is use cublaslt and
> cutensor wrapper. cublaslt should provide TF32 support, which is probably
> needed for medium-low precision matmul in post-SCF evaluation. Also, cublaslt
> and cusolver wrapper will try to be of similar way in rstsr-blas/linalg-traits.
> I may merge the current code on matmul and tensordot; just note they require
> large refactor or reimplementation now or in future.

This is the strongest signal in the review: **do not polish the current
`cublas_gemm` into the final design.** Plan for cuBLASLt + cuTENSOR (owner
wrote `cutensor`) and wrappers shaped like `rstsr-blas-traits` /
`rstsr-linalg-traits`; treat the current matmul/tensordot as a numerically
correct interim.

**(e) The rest**

> For other implementation in this PR, I will trust currently. May revisit
> testing, benchmarking and fallback kernel implementation in future.

So: no review debt on the other families, but testing/benchmarking/fallback
are explicitly named as future work — which is why they are P1 here.

---

## 4. Where this record lives

- The six directories under `2026-10-21-cuda-decisions/` and
  `2026-11-23-*` carry the design decisions and benchmarks up to the streaming
  work. This directory carries the post-PR state and the gap list.
- The owner's directives above should be read before any matmul/tensordot
  refactor; they decide the target shape (cuBLASLt/cuSOLVER wrappers), not just
  the next step.
- The var/std core change is the worked example for the "no host closure in a
  tensor-tier bound" rule; a future core PR that reintroduces a `dyn Fn` in a
  `where` clause will recreate the CUDA breakage.
