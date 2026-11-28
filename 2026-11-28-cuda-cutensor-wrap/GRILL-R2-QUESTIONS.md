# Round 2 questions — a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

## Background: round-1 answers as received on screen (agent transcription, 2026-11-28)

- Followed the recommendation: **Q2** (six dense operations), **Q3** (per-ordinal handle + public
  constructor), **Q4** (lock planning, never execute), **Q7** (typed + verified + aligned scalars), **Q8**
  (workspace type + per-stream default, 128 MiB flag), **Q9** (stream from the device value, ordering
  inside), **Q10** (`Result` + status table + `is_decline`), **Q12** (`"cutensor"` feature, `build.rs`
  cfg for the descriptor symbol source), **Q13** (tests + bench), **Q14** (no plan cache here), **Q15**
  (comprehension alignment confirmed).
- **Q1** accepted (both layers, plan re-validation: "sounding and better than my thought"), with one
  change: the **one-shot layer accepts `IxD` only** -- no `DA, DB, DD` generics -- "cuTENSOR also only
  accepts IxD; fixed dimensionality gives no efficiency gains".
- **Q5** reshaped: (i) a **stride-0 input** (extent > 1) is **allowed under a runtime cuTENSOR version
  gate** ("gate on cuda/cutensor version at runtime"); a stride-0 **output stays refused**; (ii) the
  **overlap check is dropped** ("probably too defensive"); (iii) the rest as recommended (extent 0,
  `opC != IDENTITY`, mode consistency refused; never copies).
- **Q6** reshaped: callers may name `cudarc` types directly ("not a problem, as long as it is not
  unsafe") -- so **no sys re-exports**; "for compute descriptor, use cudarc, not wrap it now"; the
  matmul-style enum bound to the output dtype is kept in mind for the `rt::cuda::einsum` utility, not
  now.
- **Q11**: **(b)** -- also reachable as `rstsr::cuda::cutensor_wrap` through the facade.

## Facts looked up for this round (agent-authored)

- **Five cuTENSOR versions are installed locally**: 2.0.1 (`/scratch/miniconda3-bak/envs/pyscf/...`,
  CUDA-11 build), 2.4.1 (`/scratch/miniconda3-bak/pkgs/cutensor-2.4.1.4-*/lib`), 2.6.0
  (`~/.cache/uv/archive-v0/hQ6viqq5PutHCWkI/cutensor/lib`), 2.7.0 (pip wheel in `torch-gpu4pyscf`), 2.8.0
  (`/home/a/miniconda3/envs/ruaa-dev/...`, py3.13). The stride-0 boundary (2.4.1 rejects, 2.7.0 accepts)
  can therefore be probed exactly; a version-matrix probe is running in `probe/versions/` (S1 stride-0
  input contraction, S2 stride-0 reduced mode, S3 stride-0 output, S4 `opC = CONJ`, S5 duplicate mode,
  S6 unchecked alignment, S7 extent mismatch) across all five. Its result is the prerequisite of the
  **gate threshold** question (round 3); nothing below depends on it.
- **cudarc does not export the compute descriptors** (grep over `cudarc-0.19.10/src`: zero
  `CUTENSOR_COMPUTE_DESC` matches; they are `extern const` data symbols the bindgen skipped). The only
  cudarc route is `cudarc::cutensor::sys::culib()` -- a **`pub unsafe fn`** returning the `libloading`
  library, with which the global is fetched by `get::<*const cutensorComputeDescriptor>(b"..\\0")`. So
  "use cudarc, not wrap it" is possible only as *unsafe* caller code; a safe accessor has to live in the
  wrapper (or the caller declares `extern "C" { static .. }` under `dynamic-linking`).
- **Static dims vs `IxD`**: `TensorViewAPI` has `type Dim = D` as an associated type, so a
  `Tensor<T, DeviceCuda, Ix<2>>` does not match `impl TensorViewAPI<.., Dim = IxD>`; the caller converts
  with `to_dim::<IxD>()` / `into_dim` (two `Vec` allocations, ~100 ns; the common creation paths
  `rt::asarray((vec, &dev))`, `rt::zeros(([..], &dev))`, `into_shape([..])` already give `IxD`).
  cuTENSOR's rank is a runtime `numModes` -- the user's premise is right: no static-rank gain exists.
- **Facade wiring** (`rstsr/src/cuda.rs`, `rstsr/src/prelude.rs`): `cuda.rs` is `pub use
  rstsr_cuda::prelude;` (so the paths are `rstsr::cuda::prelude::X`); `rt::cuda` is globbed from the
  prelude's three namespaces `rstsr_traits` / `rstsr_structs` / `rstsr_funcs` only. A facade re-export of
  the wrapper is one line in `cuda.rs`: `pub use rstsr_cuda::cutensor_wrap;` -> `rstsr::cuda::cutensor_wrap`.
- **Git**: `rstsr` is at `937ce04` on `261127/cuda-prelude-f-convention`, clean tree, one commit ahead of
  `261021/cuda` (that commit introduced the `_f` twins and the prelude namespaces the wrapper follows).
  The main repos prohibit auto-commits (AGENTS.md); branch naming `YYMMDD/<slug>` (CLAUDE.local.md).

---

❓ **Q1** - **Compute descriptor: a safe accessor is unavoidable; keep it minimal**: "use cudarc, not
wrap it" cannot be followed literally -- cudarc carries no `CUTENSOR_COMPUTE_DESC_*` item, and the only
cudarc route (`culib()`) is an `unsafe fn`, which your own rule excludes for callers. Proposal, in the
spirit of "not wrapped":

- `cutensor_wrap::compute_desc` holds **nine safe functions** returning the raw
  `cudarc::cutensor::sys::cutensorComputeDescriptor_t`, named after the globals: `desc_16f()`,
  `desc_16bf()`, `desc_tf32()`, `desc_3xtf32()`, `desc_32f()`, `desc_64f()`, `desc_9x16bf()`,
  `desc_8xint8()`, `desc_4x16f()`; each resolves its symbol once per process (panic-free: a missing
  symbol is an `UnImplemented` error, not a panic as in cudarc's own loader). No Rust enum, no binding
  to the output dtype; every wrapper function takes the raw pointer.
- The matmul-style enum keyed by the output dtype (`CutensorCompute32F { F32, TF32, .. }`, the einsum
  grill's R2 addendum) is **deferred to the einsum utility**, as you suggest.
- No sys re-exports: signatures name `cudarc::cutensor::sys::{cutensorOperator_t, cutensorAlgo_t,
  cutensorJitMode_t, cutensorWorksizePreference_t, cutensorComputeDescriptor_t}` directly (safe enum /
  pointer types; callers never call an `unsafe` cudarc fn).

➡️ Yes.

---

❓ **Q2** - **`IxD` only, on both layers**: the one-shot takes `impl TensorViewAPI<Type = T, Backend =
DeviceCuda, Dim = IxD>` / `TensorViewMutAPI<..., Dim = IxD>` (as you decided); the **object layer too**:
`TensorDescriptor::new(&handle, &view)` with `view: TensorView<'_, T, DeviceCuda, IxD>`, and `Plan`
stores `Layout<IxD>` copies for the re-validation. The wrapper performs **no** `to_dim` conversion
itself: a static-dim tensor (`Tensor<T, DeviceCuda, Ix<2>>`) is passed by the caller as
`.to_dim::<IxD>()` (documented in the module doc with the reason: no cuTENSOR gain, fewer generics).

➡️ Yes.

---

❓ **Q3** - **Descriptor alignment: computed from the pointer by default, overridable**: cuTENSOR's
`alignmentRequirement` is the base pointer's byte alignment (base = buffer pointer + `offset` *
element size). The probe: it is not checked against the pointer, only `>= element size` is a rule; the
perf effect of the declared value is unmeasured. Proposal:

- `TensorDescriptor::new(handle, &view)` declares **the pointer's own alignment** (largest power of two
  dividing the byte address, capped at 256), validated `>= size_of::<T>()`;
- `TensorDescriptor::with_alignment(handle, &view, alignment: u32)` declares a **floor** of the caller's
  choice (power of two, `>= element size`) -- what einsum will use to make one plan serve every buffer of
  a shape regardless of offset views (`size_of::<T>()` is always true for a valid pointer);
- plan re-validation (Q1 of round 1) checks `actual pointer alignment >= declared`, so a plan made for
  256-byte-aligned operands refuses an 8-byte-aligned one, and a plan declared at element size never
  refuses.

- (b) Always declare `size_of::<T>()` (always valid, never blocks reuse, no second constructor); the
  computed value is never used.

➡️ (a). Honest by default, and the override is one line for the reuse case.

---

❓ **Q4** - **Facade re-export: `rstsr::cuda::cutensor_wrap` only**: one line in `rstsr/src/cuda.rs`
(`pub use rstsr_cuda::cutensor_wrap;`), giving `rstsr::cuda::cutensor_wrap::contract(..)` next to
`rstsr::cuda::prelude`. Not added to the `rt::cuda` namespaces (those are the prelude's three
namespaces, i.e. the user surface the wrapper deliberately stays out of).

- (b) Also inside `rt::cuda` (`rt::cuda::cutensor_wrap`), one more `pub use` in `prelude.rs`.

➡️ (a).

---

❓ **Q5** - **Branch**: `261128/cuda-cutensor-wrap`, created from the current head `937ce04`
(`261127/cuda-prelude-f-convention`) -- the wrapper follows that commit's `_f` and prelude conventions,
and it is a strict superset of `261021/cuda`. No commit without your instruction (main-repo rule); the
grill record in `rstsr-improve-trajectory` is committed by the agent as usual.

➡️ Yes.

---

❓ **Q6** - **Implementation batches** (each leaves the crate compiling, `--lib` green, and the
hardware tests green on this machine before the next starts; the local `libcutensor.so` symlink is
set up in B1 and documented):

- **B1 wiring and foundation**: `"cutensor"` cudarc feature; `build.rs` cfg `cutensor_dynamic_loading`;
  module skeleton (`handle`, `descriptor`, `operation`, `plan`, `workspace`, `compute`, `error`);
  `CutensorHandle` (+ `CudaOrdinalObj::cutensor()` slot, runtime `cutensorGetVersion` stored on the
  handle for the gate, planning mutex); the status -> variant table + stage messages + `is_decline`;
  `compute_desc` accessors; the `CutensorDType` trait (`type Scalar`, `const DATA_TYPE`) for the six
  types; `CutensorWorkspace` + the `cutensor_workspace_bytes` flag + `CudaStreamObj::cutensor_workspace()`;
  `--lib` units for all of it.
- **B2 contraction end to end**: `TensorDescriptor` (alignment policy, stride-0 gate, extent checks),
  `OperationDescriptor::new_contraction` (+ the `SCALAR_TYPE` verification), `PlanPreference`, `Plan`
  (+ re-validation, `required_workspace`), `Plan::contract` (ordering, guards), the `contract` one-shot;
  the contraction half of `tests/test_cutensor_wrap.rs` (GEMM shapes per dtype vs the cuBLAS chain,
  batched / strided, `C == D`, `beta = 0` with `c = None`, conjugates, rank-0 output, refusals,
  `f32 x c32` -> `NOT_SUPPORTED` at plan, layout-mismatch refusal, cross-stream).
- **B3 the other five operations**: `contract_trinary`, `reduce`, `permute`, `elementwise_binary`,
  `elementwise_trinary` (object + one-shot) and their tests (incl. the mode-consistency refusals per
  operation kind).
- **B4 bench and docs**: `benches/bench_cutensor_wrap.rs` (one-shot vs held plan per operation, planning
  cost, cuTENSOR vs cuBLAS drivers on GEMM shapes); `lib.rs` module list + "What is implemented" row +
  "Running the tests" line; the module docstring (pipeline, cost model, local symlink requirement, the
  no-`_f` deviation, the stride-0 gate and its threshold).

➡️ Yes.

---

❓ **Q7** - **Default workspace failure is an error, not a silent slowdown**: when the one-shot's
`workspace` argument is `None`, it uses the stream's default slot; if `cutensor_workspace_bytes > 0` and
the allocation failed (`None` from the slot), the call fails with `MemoryError` (as an OOM would);
`cutensor_workspace_bytes = 0` is the explicit "no workspace" switch (null pointer, size 0; cuTENSOR
documents the workspace as optional).

➡️ Yes.

---

## Pending for round 3 (prerequisite: the version-matrix probe)

- **Stride-0 input gate threshold**: the exact lowest cuTENSOR version that accepts a stride-0 input and
  computes it correctly (2.4.1 rejects, 2.7.0 accepts; 2.6.0 and 2.8.0 unknown until the probe reports),
  and whether 2.8.0's tightened checks (`opC = CONJ` rejected at create) change any refusal.
