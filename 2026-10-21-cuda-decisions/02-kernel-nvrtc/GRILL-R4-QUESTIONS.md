# GRILL R4 — `02-kernel-nvrtc`: final fork + close

## Agent-authored background — R3 answers as received on screen (2026-10-23)

Transcribed from the owner's on-screen replies to GRILL-R3-QUESTIONS.md
(answers file remains user-owned):

- **R3-Q1/Q2: Okay. You can try that.** — `same_device` drops the
  stream clause entirely (ordinal + flags-as-stored + order); ops order
  cross-stream inputs with event record/wait; cross-ordinal stays an
  explicit `change_device` copy; `DeviceMismatch` names both devices.
- **R3-Q3/Q4/Q5: Okay.** — pre-flight list as written; trait name
  `CudaDType`; completion record (DECISIONS.md, PLAN.md, memory update,
  side-ledger; no rstsr commits without owner instruction).

Settled so far (R1–R3): scope; launcher builder-as-mechanics u8-wise;
strict compile defaults + additive flags later; PTX + runtime arch;
two-tier cache; `CudaDType` 16-type set, monomorphization-as-dispatch,
no runtime tag; assign cast kernels + half↔float `DTypeCastAPI`
extension (try-first); v1 dtype split; async ops, host reads sync;
crate-local copy-and-compare tests; same branch `261021/cuda`, no PR;
fix-forward strided-assign; per-specialization lazy modules;
ordinal-only `same_device` + event-wait ordering.

## Questions

### R4-Q1 — v1 kernel indexing scheme

How kernels consume layouts in v1:

- (a) **Runtime-stride flat kernels**: every kernel takes the layout as
  runtime arrays (`const int64_t* shape/strides`, offset, rank, plus
  size), body is a flat grid-stride loop decomposing the linear index
  to a multi-index per array. One compiled binary per (family, dtype,
  variant); zero rank/contiguity specialization; ~1 extra indirection
  per element. Simple, correct, rank-agnostic (IxD-ready, matches the
  doc01 R20 escape from rank monomorphization; signed-stride math per
  R21 trivially).
- (b) **CuPy-style specialization now**: compile-time ndim +
  c_contiguity + 32-vs-64-bit index as template params (their
  `CArray`/`CIndexer` model), dim-squash canonicalization before
  codegen, cache keyed on the shape class. Faster per element, at the
  cost of implementing the specialization/canonicalization machinery
  and multiplying cache entries in this very round.

v1 = (a), with a **contiguous fast branch** inside the flat kernel
(`if (contig) linear copy` — the only specialization v1 gets, checked
at runtime); (b) is a measured follow-up only if the strided-path
numbers demand it (doc03 §4.6 gap (d) is exactly this measurement).

➡️ Recommend (a) + runtime contig branch. Matches "first try implement
and see if this way works"; defers CuPy-machinery complexity until
numbers justify it.

### R4-Q1 amendments (owner, on screen after R4 issued)

1. "I may think that we need both runtime and specialization. We persue
   efficiency on elementwise kernels, and maybe we can specialize on
   some common layouts/dimensionalities with efficiency implementation,
   but leave other cases to also supported." — settles R4-Q1 as a
   **specialization ladder** (both tiers), not option (a) alone.
2. "Well note that we also eager for partially strided (not fully
   contiguous, but have at least one axis that is contiguous, and
   offset not zero)." — the middle tier is explicitly in scope.

Consolidated ladder (elementwise families):

- **T0 generic**: runtime-stride flat kernel, always compiled, any
  rank/layout/offset — the correctness floor; never unsupported.
- **T1 inner-contiguous run (partially strided)**: 2-level decomposition
  — outer grid-stride over the non-contiguous part, inner linear loop
  over the trailing contiguous run (house precedent: CPU serial
  kernels' contiguity-switch walk, `CONTIG_SWITCH` in
  rstsr-native-impl op_with_func); nonzero offset passed as a kernel
  arg. Covers row-slices/column slices, broadcast rows, padded rows.
- **T2 fully contiguous**: C/F-order whole-buffer linear path (v1 also
  accepts nonzero offset by pointer arithmetic at launch).
- Selection at launch time inspects layouts and picks the highest
  applicable tier; dtype/rank remain runtime (no CuPy dim-squash
  machinery in v1). Efficiency effort concentrated on elementwise;
  copy/fill/cast/gather get T0+T2 only.

### R4-Q2 — Close confirmation

With R4-Q1 answered, the frontier is empty. On close I write:
`DECISIONS.md` (D1..Dn, canonical), `PLAN.md` (pre-flight P1-P5,
batches, tests, naive-path register, delivery: same branch, no PR),
update the memory notes (close this grill; supersede the 01-device-basic
"next grills" pointer), and keep the side-ledger of ride-along fixes.
No rstsr-repo commits without explicit owner instruction.

➡️ Recommend yes — close after this round.
