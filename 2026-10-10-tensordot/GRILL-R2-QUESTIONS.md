# GRILL-R2 — tensordot (architecture + axes API)

Agent-authored. Answers go in `GRILL-R2-ANSWERS.md` (user-owned).

---

## Background — R1 answers (agent transcription, 2026-10-10; agent-authored record)

Given on screen; transcribed here as background (never into an answers file).

- **Preamble.** TBLIS is a *learning reference only*; `rstsr::tblis::tensordot` stays a
  standalone plugin — **no plugin work, no shared-helper migration** in this task.
  Base the work on rstsr `main` **`#133`**.
- **Q1.** Prefer hybrid **(c)** *only if* you are confident you can implement it;
  otherwise **(a)**. Hard constraint: **if a reshape for the GEMM path would clone
  data, do not use matmul — use the naive kernel** (avoiding allocation is the first
  priority; rstsr has utilities to test copy-free reshape). **No einsum** (external
  lib). Naive permute+reshape may fail for interleaved patterns; iterating outer axes
  and doing per-slice `matmul` (unit-stride inner ⇒ GEMM-able) is acceptable; two
  contracted indices are trickier. So (b) is *not* obviously less code or faster.
- **Q2.** Agreed — NumPy-superset forms.
- **Q3.** You believe vecdot uses `(0,1)` = a-axis0 vs b-axis1 to distinguish
  `[0,1]` = a{01} vs b{01}; asked to verify.
- **Q4.** Agreed — vecdot-parity variant/method surface.
- **Q5.** Agreed, except: `axes=None` is *not* an array-API/NumPy form (default is
  `2`); asked to verify.
- **Q6/Q7.** Yes: two commits, split by files; core first, **no commits to rstsr
  until manual check**.

## Facts verified since R1

- **`axes=None`:** NumPy 2.5.1 raises `TypeError: bad operand type for unary -:
  'NoneType'`. Array-API signature is `tensordot(x1, x2, /, *, axes=2)` — `None` is
  not an accepted form. Q5's suspicion is correct.
- **`[0,1]` vs `(0,1)`:** NumPy treats them **identically** (both = pair: a-axis 0 vs
  b-axis 1) — verified, both give the same shape-mismatch. rstsr's `AxesPairIndex`
  treats them **differently** (`[0,1]` → same-axes-both), with an explicit code
  comment defending the split. So Q3's reading of **rstsr's** convention is right,
  but it is an **rstsr-only divergence** from NumPy, not a NumPy convention.
- **Base:** rstsr `main` already at `#133` `d7056ba` (in sync with origin). Workspace
  moved off `e7cdc6a (#117)` onto branch **`261010/tensordot` @ `d7056ba`**.
- **`AxesPairIndex` forms** confirmed (`axes_pair_index.rs`): `Val(T)` from an int;
  `Pair` from a 2-tuple `(X1,X2)`; same-axes `Pair` from a bare `Vec`/`[X;N]`/slice;
  `None` from `Option<T>::None`. There is **no** `()` overload.

---

## Round 2 questions

### Q8 — Architecture package: how deep, and where does the GEMM dispatch live?

Proposed **layered** design:

- **L0 — naive device op (always, the correctness core).** New
  `DeviceTensordotAPI` (mirrors `DeviceVecdotAPI`): serial kernel
  `rstsr-native-impl/src/cpu_serial/tensordot.rs` + a hand-written rayon twin
  `cpu_rayon/tensordot.rs`, exposed via `feature_rayon/auto_impl/tensordot.rs` +
  the 6 symlinks. Handles *any* axes / *any* layout / *any* free-axis interleaving.
  This is the "CPU-serial + rayon-auto-impl" deliverable and the universal fallback.
- **L1 — view-only GEMM fast path (tensor layer).** In
  `tensor/linalg/tensordot.rs`: normalize axes → permute A to `(free_a…, ctr…)`,
  B to `(ctr…, free_b…)`; **if both reshapes to `(M,K)` / `(K,N)` are copy-free
  views**, do one `rt::matmul`; else fall through to L0. Covers matrix-matrix,
  tensor-matrix, `axes=0` (K=1), and the whole `axes=int N` family on contiguous
  inputs (A's contracted axes are already the trailing block, B's the leading block).
- **L2 — blocked / staged GEMM (deeper, optional).** Iterate outer free axes so each
  slice is a unit-stride 2D GEMM even when a full reshape would copy (your
  "iterate `ab`, then `cd,de->ce`" trick); multi-axis contracted axes handled by
  staged contraction. This is where the boundary cases (and the bugs) concentrate.

Dispatch location: the **tensor layer** orchestrates (it can call `rt::matmul` and
test reshape feasibility), with the naive device op as the fallback — so the device
op stays device-local and simple, and the GEMM route is device-generic through the
existing BLAS/faer matmul.

➡️ **Implement L0 + L1 now; defer L2 to a measured follow-up.** Hard rule confirmed:
**never materialize a copy to enable GEMM** — if the reshape is not a view, use L0.
Rationale: L0+L1 already converts the common tensordot shapes to GEMM with zero
copies, while L2's arbitrary-interleaving handling is high-risk and the payload
("naive is okay") doesn't require it. Confirm the depth — or say L2 is required now.

### Q9 — The `axes` type: reuse `AxesPairIndex`, or a dedicated `TensordotArgs`?

- **(a) Reuse `AxesPairIndex<isize>`** (vecdot's type): It already encodes
  `{None, int, (seq,seq)}` exactly. `Val(n)`'s meaning becomes site-specific
  ("last `n` of A vs first `n` of B" — like the old plugin), and the bare-collection
  shorthand stays (`[0,1]` = same-axes-both), which **diverges from NumPy** — but
  `[0,1]` is not an array-API form anyway (array API wants `int` or `(seq,seq)`).
- **(c) Dedicated `TensordotArgs`** (house pattern) faithful to NumPy: `int`,
  `(A,B)` with each side int-or-seq; no same-axes shorthand; `Val` not overloaded.

➡️ **(a)**, given your Q3 remark. It keeps a single axes-pair type in core and the
array-API-valid forms (`int`, `(seq,seq)`) are correct either way; only the
rstsr-only `[0,1]` shorthand differs from NumPy, and it is documented as an
extension. Say if you'd rather be strictly NumPy-faithful on the bare-list form.

### Q10 — The Rust default token for `axes=2`

NumPy/array-API have a default of `2` but reject `None`; Rust has no default args,
so a token is needed. Options: **(a)** accept `None` → 2 (via the existing
`From<Option<T>>`; matches how `vecdot` takes its default) and also `()` → 2 (house
rule: if `()` is valid, `None` must be); **(b)** only `()` → 2; **(c)** require an
explicit `2`.

➡️ **(a)** — `None` is the Rust-API analogue of omitting `axes`, even though Python
callers never pass it; the shim keeps `axes=2`. This is a Rust-only convenience, not
an array-API statement. Confirm, or pick (b)/(c).

*(The kernel-level round — the contraction loop, free/contracted split, rayon parallel
axis and `PARALLEL_SWITCH`, exact bound placement — follows once Q8's depth and
dispatch location are fixed.)*
