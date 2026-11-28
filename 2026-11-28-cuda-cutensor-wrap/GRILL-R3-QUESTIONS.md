# Round 3 questions — a safe cuTENSOR wrapper in `rstsr-cuda` (`src/cutensor_wrap`)

## Background: round-2 answers as received on screen (agent transcription, 2026-11-28)

- Followed the recommendation: **Q1** (nine safe `compute_desc::*` accessors, raw pointer, no enum, no sys
  re-exports), **Q2** (`IxD` only on both layers, caller converts static dims), **Q3** (alignment computed
  from the pointer by default, `with_alignment` floor, re-validation `actual >= declared`), **Q6**
  (batches B1-B4), **Q7** (default-workspace allocation failure is a `MemoryError`; `0` = explicit no
  workspace).
- **Q4**: **(b)** -- "yes, `cutensor_wrap::*` should not pollute the prelude, we only introduce module
  `cutensor_wrap` in prelude": the module as a named unit (`rt::cuda::cutensor_wrap::contract`), no glob.
- **Q5**: "Still at `261021/cuda`" -- the base is the CUDA work branch, not `261127/cuda-prelude-f-convention`.

## Fact looked up for this round (agent-authored): what `261021/cuda` has, and what it does not

`261021/cuda` tip = `26eab76`. Verified with `git show 261021/cuda:<path>`:

- `crates-device/rstsr-cuda/src/` has **no `prelude.rs` and no `prelude_dev.rs`**: `lib.rs` carries the
  inline `pub mod prelude_dev { .. }` ("Device-internal development prelude ... Internal modules glob this
  for one-line imports") and `pub mod funcs { .. }`. There is **no user-facing `prelude`** module.
- The `rstsr` facade has **no cuda wiring**: `rstsr/Cargo.toml` has no `cuda` feature, `rstsr/src/lib.rs`
  and `rstsr/src/prelude.rs` have no `cuda` line, `rstsr/src/cuda.rs` does not exist. The dev-dependency is
  `rstsr = { path = "../../rstsr", default-features = false }` (the CPU facade, for `rt::matmul` and
  friends in tests). Tests import `rstsr_cuda::prelude_dev::*`, `rstsr_cuda::funcs`,
  `rstsr_cuda::matmul::{..}`.
- Everything Q4 (b) relies on -- `rstsr_cuda::prelude` with its three namespaces, `rstsr/src/cuda.rs`,
  `rt::cuda` -- is the **single commit `937ce04`** on `261127/cuda-prelude-f-convention` (one commit ahead
  of `261021/cuda`, a strict superset).
- Everything else the wrapper needs is on `261021/cuda`: `CudaOrdinalObj::cublaslt()` (registry.rs),
  `CudaStreamObj::cublaslt_workspace()` (stream.rs), `order_input` / `order_output` (kernel.rs),
  `read_ptr_on` (buffer.rs), `CudaRuntimeFlags` (flags.rs), the `catch_unwind` loadability probe idiom.

---

❓ **Q1** - **Base and facade wiring, reconciled**: "introduce the module `cutensor_wrap` in the
prelude" cannot be done on `261021/cuda` -- there is no prelude and no `rt::cuda` / `rstsr::cuda` there.
Proposal that honors both answers:

- **B1-B4 land on `261021/cuda`** (the CUDA work branch, as before); the public path is
  `rstsr_cuda::cutensor_wrap` (the `pub mod`), with `pub use crate::cutensor_wrap;` added to the inline
  `prelude_dev` in `lib.rs` as a **named module** (so `rstsr_cuda::prelude_dev::cutensor_wrap::contract`
  resolves for the crate's own tests, no glob) -- nothing else.
- **The facade lines are a follow-up on `261127/cuda-prelude-f-convention`**, done when that branch lands
  or on request: `pub use rstsr_cuda::cutensor_wrap;` in `rstsr/src/cuda.rs` (-> `rstsr::cuda::cutensor_wrap`)
  and the same line inside the facade prelude's `rt::cuda` module (-> `rt::cuda::cutensor_wrap`), both
  as a module, no glob. Recorded in the plan as **B5 (two lines, on the convention branch)**.

- (b) Base B1-B4 on `261127/cuda-prelude-f-convention` instead (it is a strict superset; Q4 (b) done in
  B4), accepting that the wrapper then depends on the convention branch landing first.
- (c) Base on `261021/cuda` and add the facade wiring there too (duplicates the convention branch's work;
  conflicts at merge).

➡️ (a).

---

## Pending (prerequisite: the version-matrix probe, still running)

- **Stride-0 input gate threshold** -- the lowest cuTENSOR version accepting a stride-0 input correctly
  (2.0.1 / 2.4.1 / 2.6.0 / 2.7.0 / 2.8.0 matrix), and whether 2.8.0 changes any other refusal.

---

# Addendum (2026-11-28) — the stride-0 gate threshold, from the version-matrix probe

Probe: `probe/versions/` (`vs.cpp` compiled once against the 2.7.0 header, run per version through
`LD_LIBRARY_PATH=lib<V>`; `REPORT.md`, `run-<V>.log`). Versions: 2.0.1, 2.4.1, 2.6.0, 2.7.0, 2.8.0.
**2.0.1 is unusable on this machine**: it needs `libcublasLt.so.12` and, with one supplied,
`cutensorCreate` still fails (`INTERNAL_ERROR`) on sm_120 / driver 595 -- S1-S7 could not run for it.

| case | 2.4.1 | 2.6.0 | 2.7.0 | 2.8.0 |
|---|---|---|---|---|
| S1 stride-0 **input** (`A[i,j]` strides `(0,1)`, `ij,jk->ik` f64) | NOT_SUPPORTED at descriptor | ok | ok | ok |
| S2 stride-0 on a **reduced** mode (`ij->j`, `ij,j->j`) | NOT_SUPPORTED | ok | ok | ok |
| S3 stride-0 **output** (variants a-e) | NOT_SUPPORTED (all) | accepted with garbage when C shares D's layout; NOT_SUPPORTED only when C is packed | same | same |
| S4 `opC = CONJ`, c64, `beta = 0.3` | silently ignored | silently ignored | silently ignored | **NOT_SUPPORTED at CreateContraction** |
| S5 duplicate mode `[i,i]` | diagonal, accepted | same | same | same |
| S6 alignment 256 declared on an 8-byte-offset pointer | accepted, correct | same | same | same |
| S7 extent mismatch (contracted / batch mode) | `INTERNAL_ERROR` at EstimateWorkspaceSize | same | same | same (not tightened) |

Consequences: (1) the lowest installed version that accepts a stride-0 input **and computes it
correctly** is **2.6.0**; 2.4.1 rejects it; 2.5.x is not installed, so the true boundary lies in
(2.4.1, 2.6.0]. (2) No version honours `opC != IDENTITY` (older ones ignore it, 2.8.0 rejects it), so the
wrapper refuses it unconditionally. (3) Extent mismatches and stride-0 outputs are never validated by any
version -- the wrapper's own checks (round 1, Q5) stand for all versions. (4) The 2.x ABI is stable: one
binary ran all five libraries.

❓ **Q2** - **Stride-0 input gate: `cutensorGetVersion() >= 20600`**: the gate reads the version once
at `CutensorHandle` creation (`cutensorGetVersion()` = major*10000 + minor*100 + patch) and
`TensorDescriptor::new` refuses a stride-0 (extent > 1) **input** below the threshold, accepts it from
2.6.0 on; stride-0 **outputs** are refused on every version (S3). The constant carries the probe facts in
its comment (2.4.1 rejects, 2.6.0-2.8.0 verified correct, 2.5.x unverified). The cuTENSOR version alone
gates it -- the cudart build version (2.8.0 is built against 13.4, the others 13.1; all ran on the 13.2
runtime) plays no role.

- (b) `>= 20500`: assumes 2.5.0 already accepts (unverified; 2.5.x is not installed).
- (c) No version gate but a capability probe at handle creation (try a tiny stride-0 descriptor +
  plan, cache the verdict): robust to the unknown boundary, but "accepted" is not "computed correctly"
  (S3 shows accepted-with-garbage exists), and it costs a plan at every handle creation.

➡️ (a). The lowest version seen to be *correct*, not merely accepted.

---

## Resolution of Q1 (owner, on screen, 2026-11-28): merged first

> owner: "Oh, I forgot that this branch 261127/cuda-prelude-f-convention not merged into 261021/cuda.
> You can do it for me."

Fast-forwarded by the agent at the owner's instruction: `261021/cuda` `26eab76 -> 937ce04` (the convention
branch was exactly one commit ahead and `261021/cuda` had nothing it lacked, so no merge commit was
created; working tree checked out on `261021/cuda`, clean; not pushed). The facade wiring
(`rstsr/src/cuda.rs`, `rstsr-cuda/src/prelude.rs` + `prelude_dev.rs`, the facade's `cuda` feature) is
therefore on `261021/cuda`. **Consequence**: B1-B4 all land on `261021/cuda`; the Q4 (b) lines
(`pub use rstsr_cuda::cutensor_wrap;` in `rstsr/src/cuda.rs` and inside `rt::cuda` in the facade prelude,
module, no glob) are part of **B4**; the "B5 on the convention branch" of option (a) is void.
