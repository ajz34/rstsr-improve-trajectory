# 2026-11-28 — cuTENSOR safe wrapper (`cutensor_wrap`)

Safe wrapper over the cuTENSOR library in `rstsr-cuda`, the machine layer under the future
einsum/tensordot implementations. Not a high-level API: RAII objects over the cuTENSOR pipeline
(tensor descriptors → operation descriptor → plan preference → workspace estimate → plan → one
execute call), one one-shot function per operation family in `one_shot.rs`, raw cuTENSOR FFI types
(modes, operators, compute descriptors) at the boundary, every unsafe inside the module.

Made against rstsr `937ce04` (*refactor: CUDA prelude surfaces and fallible twins, wired into the
facade*), branch `261021/cuda`. Environment: NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), CUDA
13.2 (nvcc), driver 13020, cudarc 0.19.10 (cutensor = cutensor-02006 bindings), libcutensor 2.7.0
pip wheel (+ 2.4.1 / 2.8.0 in the probe tree for cross-version runs), rustc nightly 1.12.0-nightly
(fmt), Linux x86_64. Bench harness: hand-rolled median-of-repeats (`benches/bench_cutensor.rs`),
following the crate's `bench_matmul` harness rather than criterion (one binary, CSV + markdown
out).

## Commits (branch `261021/cuda`)

| Commit | Batch | Content |
|---|---|---|
| `618fd38` | B1 | Foundation: handle slot per ordinal, per-stream workspace slot, runtime flag, build cfg |
| `8a84c9d` | B2 | Contraction: descriptors, operation, plan, one-shot; the libloading deref fix |
| `08f4b41` | B3 | Reduction, permutation, elementwise binary/trinary, trinary contraction (symbol-gated), plan/kernel-cache management, header-audit coverage |
| `dbe6a5c` | B4 | Facade wiring (`rstsr/cutensor` feature → `rstsr::cuda::cutensor_wrap`, `rt::cuda::cutensor_wrap`), `bench_cutensor`, docs |
| `5f1cae6` | review | Review fixes: `dynamic-linking` build, `resize_plan_cache` lock, `finish` descriptor leak, workspace guard, `beta == 0` C ordering, temporary names, dedup, version print, 4 new hardware tests |
| `e80cf67` | review | Multi-stream overlap microbenchmark (`overlap` family: contraction vs D2H/H2D on a second stream and thread) |

## Conventions (decisions D1–D26, `DECISIONS.md`)

- Operands enter as views **by value** (`t.view()` / `t.view_mut()`); `c: Option<impl TensorViewAPI>`
  with `None` = **C is D** (in-place accumulation at any beta — two aliasing borrows are not
  expressible in safe Rust); scalars **by value** (`CutensorScalar: Copy`).
- No `mode_c`/`mode_d` parameters where cuTENSOR itself requires `descC == descD`.
- The wrapper **refuses before calling cuTENSOR** everything the library accepts with wrong results
  (stride-0 outputs, extent mismatches, foreign mode sets) and verifies what it never checks
  (derived scalar type, execute-time operand re-validation against the plan).
- Header audit complete (39 + 2 functions): five operation families + trinary contraction wrapped;
  logger and block-sparse out of scope; attribute setter refused by design; `MOVED_BYTES` is
  block-sparse-only; `NORM1`/`NORM2` gone in 2.x.
- The trinary contraction is symbol-gated on libcutensor ≥ 2.1.1 (clean `UnImplemented` decline —
  cudarc's own symbol loader panics, so the wrapper probes first).

## Bench (benches/bench_cutensor.rs)

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (sm_120), CUDA 13.2, driver 13020,
libcutensor 2.7.0, `--release`, median of 5 repeats, 0.2 s budget.

| family | dtype | shape | path | median ms | TFLOP/s or GB/s |
|---|---|---|---|---|---|
| contraction | f32 | 1024x1024x1024 | one-shot | 0.183 | 11.7 |
| contraction | f32 | 1024x1024x1024 | held plan | 0.181 | 11.8 |
| contraction | f32 | 1024x1024x1024 | cublas chain | 0.184 | 11.7 |
| contraction | f32 | 2048x2048x2048 | one-shot | 1.327 | 12.9 |
| contraction | f32 | 2048x2048x2048 | held plan | 1.966 | 8.7 |
| contraction | f32 | 2048x2048x2048 | cublas chain | 1.321 | 13.0 |
| contraction | f64 | 1024x1024x1024 | one-shot | 6.589 | 0.3 |
| contraction | f64 | 1024x1024x1024 | held plan | 6.366 | 0.3 |
| contraction | f64 | 1024x1024x1024 | cublas chain | 6.592 | 0.3 |
| contraction | f64 | 2048x2048x2048 | one-shot | 52.883 | 0.3 |
| contraction | f64 | 2048x2048x2048 | held plan | 50.480 | 0.3 |
| contraction | f64 | 2048x2048x2048 | cublas chain | 53.084 | 0.3 |
| contraction | c64 | 1024x1024x1024 | one-shot | 20.568 | 0.4 |
| contraction | c64 | 1024x1024x1024 | cublas chain | 20.640 | 0.4 |
| ceiling (measured) | f64 | 4096x4096x4096 | cublas chain | 402.087 | 0.3 |
| ceiling (measured) | f32 | 8192x8192x8192 | cublas chain | 72.448 | 15.2 |
| ceiling (measured) | c64 | 4096x4096x4096 | cublas chain | 1292.927 | 0.1 |
| trinary contraction | f64 | 512x512x512x512 | one-shot | 1.768 | 0.3 |
| reduction | f64 | 4096x4096 -> 4096 | one-shot | 0.414 | 324.3 GB/s |
| elementwise | f64 | 2048x2048 | one-shot | 0.342 | 294.0 GB/s |

Ceilings and theory for this machine (RTX 5070 Laptop, consumer Blackwell, ~4600 CUDA cores):

| dtype | datasheet theory (approx.) | measured ceiling (large cuBLAS GEMM) | best wrapper contraction |
|---|---|---|---|
| f64 | FP32/64 = ~0.3 TFLOP/s | **0.3 TFLOP/s** (4096^3) | 0.3 - at the ceiling |
| f32 | ~20 TFLOP/s shader rate | **15.2 TFLOP/s** (8192^3) | 12.9 (85% of measured) |
| c64 | ~2x f64 real work, ~0.15 TFLOP/s as 4-flop/MAC | **0.1 TFLOP/s** (4096^3) | 0.4 at 1024^3 (small-problem cache effects; scales to the ceiling at 4096^3) |

Reading: the wrapper's contractions sit **at parity with the cuBLAS chain** for GEMM-shaped
problems on every dtype (cuTENSOR dispatches to cuBLAS-backed kernels there): f64 at the measured
ceiling, f32 at ~85% of it, c64 scaling to its ceiling at the larger size. The one-shot's planning
overhead is 0-6% (f32 2048^3 shows a held-plan outlier: the plan fixes its kernel choice at
creation, while the per-call one-shot can re-plan differently — a real effect worth knowing, not
noise). The streaming families run at 290-325 GB/s. The trinary metric counts its two chained
products (`2·ijk + 2·ikl`), not the naive four-loop count.

### Multi-stream overlap (post-review microbenchmark, `e80cf67`)

Execute is deliberately unlocked (D6) so that contractions on one stream overlap with work on
another. The `overlap` family measures it: an f64 1024^3 contraction on the default stream from
one thread against a 256 MiB transfer on a second stream and thread, through the **same handle**;
each sample times the pair sequentially (transfer, sync, contraction) and concurrently. RTX 5070
Laptop, cuTENSOR 2.7.0, `RSTSR_BENCH_REPEAT=7`:

| transfer | contract alone | transfer alone | sequential | concurrent | 100·seq/par | ideal (= transfer alone) |
|---|---|---|---|---|---|---|
| D2H 256 MiB | 6.5 ms | 50.1 ms | 62.7 ms | **53.1 ms** | 118 | 50.1 ms |
| H2D 256 MiB | 6.5 ms | 66.0 ms | 79.1 ms | **73.2 ms** | 108 | 66.0 ms |

Reading: in both directions the contraction is hidden under the transfer (concurrent ≈ transfer
alone + 3-7 ms, versus the serialized sum of 56.6 / 72.5 ms). The transfers run at 5.3 / 4.0 GB/s
(pageable host memory through `to_cpu_vec` / `to_device`; pinned buffers would be faster). The
hardware test `concurrent_execute_on_one_handle` covers the stronger claim — four threads
executing through one handle, each on its own stream, all results verified (green on 2.4.1 /
2.7.0 / 2.8.0).

## Usage (facade)

```rust
use rstsr::prelude::*; // feature "cuda"; the wrapper needs feature "cutensor" (CUDA >= 12)
use rstsr::cuda::cutensor_wrap; // also rt::cuda::cutensor_wrap

let mut d = rt::zeros(([m, n], &dev));
cutensor_wrap::contract(
    &dev,
    a.view(), &[0, 1], ident_op,
    b.view(), &[1, 2], ident_op,
    Some(c.view()),        // None = C is D (in-place accumulate)
    d.view_mut(), &[0, 2],
    alpha, beta,           // by value; T::Scalar
    cutensor_wrap::compute_desc::desc_64f().unwrap(),
    None,                  // workspace: the per-stream slot by default
)?;
```

## Key findings

- **libloading 0.9 `Symbol<T>` deref yields the dlsym address itself** ("additional reference
  level") — data globals need one further dereference. The missing deref passed the address of the
  compute-descriptor globals where the descriptor pointer belongs and cuTENSOR answered
  `NOT_INITIALIZED` at operation creation; isolated with a minimal C dlopen repro
  (`probe/versions/dl_test.c`). Memory note: `.claude/memory/libloading-symbol-deref.md`.
- cuTENSOR validates very little itself: stride-0 outputs and extent mismatches produce garbage on
  every probed version, `opC != IDENTITY` is silently ignored through 2.7.0 (rejected by 2.8.0),
  the scalar pointer type is never checked — hence the wrapper's pre-FFI refusal matrix.
- Per-thread current-context matters for the raw FFI: every entry binds the handle's context
  (`CUTENSOR_STATUS_NOT_INITIALIZED` on worker threads otherwise).

## Review (post-B4, two-axis: standards + spec)

Fixed in `5f1cae6` / `e80cf67` (all gates re-run): the `dynamic-linking` build break (`has_symbol`
used cudarc's `culib()` unguarded), the unlocked `resize_plan_cache` (the one header-`not
thread-safe` entry), the descriptor leak on the scalar-type refusal, the workspace guard dropped
before the enqueue, the unconditional `C` ordering at `beta == 0`, temporary decision/probe labels
in code, the ×23 status-check and ×5 `descC == descD` duplication, the missing version print, the
missing stride-0-*input* test, the crate-doc rows, the docs.rs feature and the facade path check.

Facts established during the review:

- **Header remarks (2.4.1 `cutensor.h`)**: `cutensorCreate`/`Destroy`, `cutensorPermute`, the
  elementwise executes and the plan/kernel-cache file entries are "thread-safe";
  `cutensorHandleResizePlanCache` is "not thread-safe"; `cutensorContract`, `cutensorReduce`,
  the `cutensorCreate*` operation entries, `cutensorCreatePlanPreference`, `cutensorCreatePlan`,
  `cutensorEstimateWorkspaceSize` and the attribute get/set entries carry **no remark**. The
  unlocked execute therefore rests on cuTENSOR's general thread-safety, not on a per-entry
  remark; the owner's decision (2026-11-28): keep it unlocked, the multi-stream feature needs it —
  the overlap cell above and the concurrent-execute test are the evidence.
- The scalar type cuTENSOR derives follows the **output dtype** for `f64`+`32F` and `f16`+`16F`
  (both compute successfully on 2.7.0), so the one-shot can never reach the scalar guard; only the
  object layer can (a free scalar-type parameter on fixed descriptors) — tested there.
- Stride-0 inputs: declined (`NOT_SUPPORTED`) on 2.4.1, accepted and correct on 2.7.0 / 2.8.0 —
  now a version-gated hardware test.

Deferred by the owner's instruction ("do not fix others"): bench cells (batched 64×(256,256),
`abc,cbd->ad`, planning cost, `permute`, held-plan rows beyond contraction); history rewrites
(trailer casing on the four B-commits, `(B2)`-style subjects); DECISIONS amendments for the
facade feature / `cutensor_bindings` cfg; design choices left as built (`registry.rs` `Err` →
permanent `None`, `Plan`/`OperationDescriptor` `!Send`, raw `op_reduce`); judgement-call
refactors (execute-skeleton dedup, `rstsr_assert!` conversion, `duplicate_item`, operand bundle).

## Gates (as of `e80cf67`)

25 hardware tests green on cuTENSOR 2.4.1 / 2.7.0 / 2.8.0 · `--lib` 134 passed · clippy
`--all-targets` 0 · `cargo doc -D warnings` clean (rstsr-cuda and the facade) · touched files
fmt-clean · facade `test_cuda_prelude` green with and without the `cutensor` feature ·
`cargo check -p rstsr-cuda --features dynamic-linking` compiles.
