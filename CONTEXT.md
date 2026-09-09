# rstsr-improve-trajectory

Scratch repo tracking the trajectory of improving rstsr's efficiency: one
self-contained experiment directory per task, each recording its own baseline,
numbers, and proposed patches against a pinned rstsr commit.

## Language

### Repo conventions

**Experiment directory**:
A self-contained `YYYY-MM-DD-<task>` directory at repo root holding one
experiment's code, benchmarks, README, and proposed patch; unrelated to other
directories.
_Avoid_: task folder, bench dir

**Proposed patch**:
A git diff against the pinned rstsr commit, recorded inside an experiment
directory and never applied to rstsr from this repo.
_Avoid_: PR, fix, commit (for the diff artifact)

**Edit guide**:
A written refactor proposal for when the efficiency lever is code structure
rather than a kernel; delivered instead of (or alongside) a proposed patch.

**Baseline**:
Measurement of unmodified rstsr at the pinned commit under the standard
benchmark configuration, taken before any optimization work in an experiment.

**Peak reference**:
The hardware ceiling an experiment compares against — measured memory
bandwidth (triad microbench) for streaming ops, analytic FLOP/cycle for
compute-bound ops.

### Process

**Task (T-numbering)**:
One unit of optimization work targeting an op or op family; numbered T0, T1, …
in plan order, one experiment directory each.

**G1 / G2 gates**:
The two mandatory review checkpoints per task — G1 approves the task plan
before implementation, G2 validates results and benchmark honesty before the
work is accepted.

**Main agent / Code agent / Review agent**:
The three-role workflow: main splits and assigns, code does all writing,
review audits at the gates and relays summaries back to main.

### rstsr domain

**Transpose view**:
A layout-only permutation of a tensor's shape/strides with no data movement
(`t()`).
_Avoid_: transpose copy, transpose (alone, when a copy is meant)

**Transpose copy**:
The actual data movement that materializes a permuted layout into new storage;
implemented by the assign kernel, not by the transpose view.

**Assign kernel**:
The generic element-copy primitive behind layout changes (`to_contig`,
`change_layout_f`, assignment traits).

**Vecdot**:
Batched inner-product reduction contracting summed axes across one or more
tensors.

**Reduction**:
An op collapsing axes into accumulators (sum/min/max/mean/var/norm/argmin/…),
either over the whole tensor or over selected axes.

**Lane**:
A fixed-width group of elements processed as one unit by fixed-array (SIMD-style)
kernels; lane count is fixed at compile time regardless of target CPU.

**Fault rider**:
The glibc page-fault overhead (~4 ms at ≥32 MiB fresh outputs, sys-time-bound)
that dominates alloc-inclusive timings of allocating ops; measured by T7,
remedied at environment/API level, not by kernels.

**Reuse variant (B-variant)**:
A benchmark variant writing into preallocated output (existing reuse APIs or an
emulated kernel bound), isolating kernel time from allocation; the primary
denominator for judging large-output kernel wins.

**Paired A/B**:
Back-to-back clean-vs-candidate runs in the same session used to refute
apparent gate regressions; the campaign's standard against build-layout
lottery (±5% on small cells).

**dispatch_simd**:
The (proposed) cargo feature gating dtype-dispatched fixed-array kernels built
on the lightweight-simd crate; off by default.

**dispatch_dim_layout_iter**:
An existing rstsr cargo feature that runtime-dispatches layout iterator
monomorphization on tensor rank; off by default.
