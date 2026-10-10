# GRILL-R3 — tensordot (kernel / device-layer contract)

Agent-authored. Answers go in `GRILL-R3-ANSWERS.md` (user-owned).

## Background — R2 answers (agent transcription, 2026-10-10; agent-authored)

- **Q8.** L0+L1, L2 deferred — **agreed**, with one correction: the view-only GEMM
  fast path belongs at the **device layer**, not the tensor layer. The tensor layer
  stays a thin wrapper: normalize arguments, create the output tensor buffer + layout,
  call the device-layer trait function. So `DeviceTensordotAPI::tensordot` itself
  decides GEMM-vs-naive.
- **Q9.** Reuse `AxesPairIndex<isize>` — (a) "if possible" (it is; details below).
- **Q10.** (a) — `None`→2 and `()`→2.

## Settled so far

- Core only; TBLIS plugin untouched. Base `#133`; branch `261010/tensordot`.
- `axes` = `AxesPairIndex<isize>`: `None`→default 2; `Val(n)`→contract last `n` of A
  with first `n` of B, in order; `Pair(a,b)`→explicit per-side axes; bare collection
  → same-axes-both (rstsr extension). NumPy-superset forms accepted.
- Output layout: non-contracted A axes (original order) ++ non-contracted B axes
  (original order); fully-contracted → 0-d; no broadcasting; no conjugation.
- Two commits (core, then shim); no rstsr commits until manual check.

---

## Round 3 questions

### Q11 — Device-op contract: signature and who expands `Val`/default

➡️ Mirror `DeviceVecdotAPI`. Proposed:

```rust
pub trait DeviceTensordotAPI<TA, TB, TC, DA, DB, DC>
where
    DA: DimAPI, DB: DimAPI, DC: DimAPI,
    Self: DeviceAPI<TA> + DeviceAPI<TB> + DeviceAPI<MaybeUninit<TC>>,
{
    fn tensordot(
        &self,
        c: &mut <Self as DeviceRawAPI<MaybeUninit<TC>>>::Raw, lc: &Layout<DC>,
        a: &<Self as DeviceRawAPI<TA>>::Raw, la: &Layout<DA>,
        b: &<Self as DeviceRawAPI<TB>>::Raw, lb: &Layout<DB>,
        axes_a: &[isize], axes_b: &[isize],
    ) -> Result<()>;
}
```

The **tensor layer** resolves the `AxesPairIndex` into two already-normalized,
non-negative `axes_a`/`axes_b` slices (expanding `Val(n)` → `(ndimA-n..ndimA)` /
`(0..n)`, resolving `Pair` via `normalize_axes_index`), computes `lc`, allocates via
`uninit_impl`, calls the op — exactly the `vecdot_f` shape. The device op never sees
`AxesPairIndex`. Confirm this split (vs. passing the enum into the device op).

### Q12 — The device-layer GEMM trigger, and how it delegates

Proposed rule. With normalized `axes_a`/`axes_b` (len `k`), `M = prod(free_a)`,
`N = prod(free_b)`, `K = prod(contracted)`:

- The op may use GEMM **iff** it can view `A` as `(M,K)` and `B` as `(K,N)` as
  **copy-free layout views** — i.e. after reordering A to `[free_a…, ctr…]` and B to
  `[ctr…, free_b…]`, each layout is stride-compatible with the 2-D merge — **and**
  `C` as `(M,N)` is likewise reshape-compatible. Then delegate to the device's own
  **`self.matmul(...)`** (`DeviceMatMulAPI`), so BLAS/faer devices hit real GEMM and
  `DeviceCpuSerial` hits the naive matmul.
- For `k ≥ 2`, require the contracted axes to be **mergeable** (adjacent +
  contiguous-compatible on both operands) so `K` is a single factor; otherwise
  **fall back to the naive kernel** (staged one-axis-at-a-time GEMM is L2, deferred).
- Layout-only: the trigger and the reordering are pure `Layout` operations
  (`dim_split_axes`-style + a copy-free-reshape test); **no data is touched**.

➡️ As above. This keeps the device op device-generic (each device inherits the right
matmul automatically: BLAS→BLAS, faer→faer, serial/rayon→naive) and guarantees the
"never copy to enable GEMM" rule. Confirm the mergeability restriction for `k ≥ 2`
(vs. wanting staged contraction now), and confirm delegating to `self.matmul` rather
than the op re-implementing a GEMM call.

### Q13 — Naive kernel shape and the rayon twin

➡️ Mirror `vecdot`:

- SPLIT: `la.dim_split_axes(axes_a)` → contracted vs free layouts, same for `lb`;
  assert the paired contracted sizes match.
- LOOP: iterate the **output (free) multi-index**; for each, a sequential reduction
  over the contracted multi-index — one general loop handling any `k` (no staging).
- RAYON TWIN: parallelize over the output/free elements via
  `layout_col_major_dim_dispatch_par_*` + `AtomicPtr`-hoisted output pointer, with
  `PARALLEL_SWITCH = 512` serial fallback (copy `vecdot_naive_cpu_rayon`).
- BOUNDS: element bounds on the **device impl** — `TA: Clone, TB: Clone,
  TC: Clone + Zero, TA: Mul<TB, Output=TC>, TA: ExtNum` (+ `Send + Sync` on the rayon
  twin) — **no `ext_conj`**. Tensor layer carries only
  `B: DeviceTensordotAPI<…> + DeviceAPI<…> + DeviceCreationAnyAPI<TA::Output>`.
- FILES: `operators/linalg.rs` (+`DeviceTensordotAPI`); `device_cpu_serial/linalg/tensordot.rs`;
  `rstsr-native-impl/src/cpu_serial/tensordot.rs` + `cpu_rayon/tensordot.rs`;
  `feature_rayon/auto_impl/tensordot.rs` + 6 symlinks; `tensor/linalg/tensordot.rs`;
  registration in the respective `mod.rs`/`prelude_dev.rs`/`prelude.rs`.

→ Confirm the bounds set (in particular whether `ExtNum` is needed without
conjugation, or a weaker `Zero + Mul + Add` suffices) and `PARALLEL_SWITCH = 512`.

### Q14 — Edge cases and errors

➡️ `axes=0` (k=0, outer product) takes the GEMM path with `K=1` (view-only when the
inputs are contiguous) and also works naive — no special case. Errors: mismatched
contracted sizes; duplicate axis in a side; axis out of range; `n > ndimA` or
`n > ndimB`; negative `n`; device mismatch (`a`/`b` on different devices). Confirm
the list, and whether any should be a panic (`rstsr_assert!`) vs a `Result` error
only (the `_f` twin returns `Result`; the panicking form unwraps).

*(If these land, the design tree is settled and I'll write `PLAN.md` — the file
inventory, the free/contracted split math, the GEMM trigger pseudocode, the test/doc
deliverables, and the two-commit split — for your review before any implementation.)*
