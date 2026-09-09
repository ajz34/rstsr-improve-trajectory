# PLAN.md — T1' phase-2 edit plan (transpose copy / generic strided assign)

Base: rstsr `386948be819baa334b8da02232f3a1944e5447d5` (clean tree verified
before baselining; phase 1 made NO edits). All numbers cited here are in
[results/tables.md](results/tables.md) and
[results/perf/](results/perf/) (native primary, portable secondary).

## 1. What phase 1 established (the evidence base)

1. **Baseline reproduces T0/T7.** Transpose copy 2048² f64: A (allocating
   idiom `a.t().to_contig(RowMajor).into_owned()`) 23.2 ms serial native /
   3.33 ms faer16; B (reuse `c.assign(&a.t())`) 17.2 / 1.45 ms. The glibc
   fault rider (T7) is exactly as carried forward: 26% serial / 56% faer16 of
   A; **B is the primary judge for the large class.** Variant C
   (A + `MALLOC_MMAP_THRESHOLD_=67108864` + `MALLOC_TRIM_THRESHOLD_=134217728`)
   recovers the rider (17.0 / 1.56 ms) as in T7; f32 has no rider (A ≈ B ≈ C).
2. **The kernel is instruction-bound iterator machinery, confirmed on the
   B variant** (perf stat, serial native, 2048²): **107.8 ins/elem, 22.4
   cyc/elem, IPC 4.8, 2.9% L1d-miss, ~0 steady-state page faults** (T0's
   122 ins/elem was the A profile: 119.7 ins/elem here with 8.4k faults/iter).
   The generic strided branch drives two `IterLayoutColMajor::next` per
   element (end check + multi-axis index update + `try_into().unwrap()`)
   plus `MaybeUninit::write` + `T::clone()`
   (`rstsr-common/src/layout/iterator.rs:288-300`).
3. **The dormant blocked kernel is the winner, measured directly**
   (benches/kernels_probe.rs, 2048² f64): `orderchange_out_c2r_ix2_cpu_serial`
   **5.70 ms** (native) vs 15.24 ms for the generic kernel on identical
   layouts — **2.7×** — and its rayon twin `0.542 ms` vs 1.18 ms — **2.2×**.
   At odd 1000×777: 0.296 ms serial / 97.9 µs rayon16 (vs 0.446 / 0.212 ms
   generic). Loop-orientation probes: the kernel's own orientation (inner
   loop walks the OUTPUT-fast axis: strided reads, contiguous
   write-combined stores) is right — the swapped orientation is **3×
   slower** (17.4 ms). Raw-pointer writes vs bounds-checked indexing:
   ≈0-5% — **not the lever**.
4. **Wiring costs are visible but secondary.** Tensor-API B (17.16 ms)
   runs ~1.9 ms above the bare generic kernel (15.24 ms) — layout
   translation + broadcast plumbing in the `assign` path; the A idiom
   (23.2 ms) adds ~8 ms of allocation + fault rider + Cow plumbing on top
   of the kernel.
5. **Gap to anchors is largest away from large**: serial B vs ndarray
   `.t().to_owned()`: small 64² 14.1 µs vs 0.27 µs (**52×** — pure iterator
   overhead), medium 512² 0.816 vs 0.0285 ms (**29×**), odd 1000×777
   2.42 vs 0.0727 ms (**33×**), large 3.9/2.9×. The blocked kernel collapses
   the odd/medium/small gaps with the same mechanism (tile-resident hops).
6. **Call-chain anatomy** (all file:line at 386948be):
   - A: `TensorBase::t()` (view, `rstsr-core/src/tensor/manipulation/transpose.rs:438`)
     → `to_contig(RowMajor)` → `change_contig_f`
     (`to_contig.rs:8-45`; not c-contig → new layout) → `change_layout_f`
     (`to_layout.rs:8-38`: `uninit_impl` alloc + `device.assign_arbitary_uninit`
     at :33) → `DeviceCpuSerial::assign_arbitary_uninit`
     (`device_cpu_serial/assignment.rs:15-25`) → **4th `#[duplicate_item]`
     variant `assign_arbitary_uninit_promote_cpu_serial`**
     (`rstsr-native-impl/src/cpu_serial/assignment.rs:6-67`; same-dtype ⇒
     `into_cast` is identity). Contig branch skipped (`la.c_contig()` false)
     → `translate_to_col_major_unary(lc/la, C)` (reverse axes) →
     `layout_col_major_dim_dispatch_2diff` → 2 × `IterLayoutColMajor` zip.
     In col-major iterator space: output walks sequentially, input hops
     16 KiB — the pattern T4' showed costs ~161 ins/elem with 3 iterators,
     ~108 with 2.
   - B: `c.assign(&a.t())` → `TensorAssignAPI::assign_f`
     (`rstsr-core/src/tensor/assignment.rs:119-140`) → `broadcast_layout_to_first`
     → `OpAssignAPI::assign` → `assign_promote_cpu_serial` (**3rd variant of
     the second duplicate family**, `cpu_serial/assignment.rs:69-119`) →
     `translate_to_col_major(&[lc, la], K)` (greedy) → size_contig = 0 < 16 →
     per-element `layout_col_major_dim_dispatch_2`. Rayon twins:
     `cpu_rayon/assignment.rs` (PARALLEL_SWITCH = 16384).
   - faer16: `device_faer/rayon_auto_impl/assignment.rs:15-26` →
     `*_cpu_rayon` kernels with `get_current_pool()`.
   - The dormant kernel: `cpu_serial/transpose.rs:11-48` (serial, BLOCK_SIZE=64,
     bounds-checked `c[dst_idx]`/`a[src_idx]` indexing), c2r wrapper :54-61;
     rayon `cpu_rayon/transpose.rs:15-67` (raw-pointer stores,
     nested block `into_par_iter`, serial fallback `size < 16·64² = 65536`),
     pool-aware wrapper :73-90. Only callers: LAPACK drivers
     (svd/gesvd, potrf) — unused by the tensor path.
7. **Guards the dormant kernel already enforces** (`rstsr_assert_eq!` → Err):
   shape identity; r2c: `la.stride()[1] == 1 && lc.stride()[0] == 1`; c2r:
   the reversed pair (input axis-0 fast, output axis-1 fast). Slow-axis
   strides are arbitrary (`lda`/`ldc`), **including negative** — direct-kernel
   correctness with `la = [n,m] stride [1, -n], offset (m-1)·n` (a
   `flip(0).t()` view) PASSES (examples/correctness.rs).

## 2. Design options

### Option A — route 2-D order-changes to the dormant blocked kernels inside BOTH assign families (RECOMMENDED)

In `rstsr-native-impl/src/cpu_serial/assignment.rs` and
`cpu_rayon/assignment.rs`, in each of the two duplicate families
(`assign_arbitary_*`, `assign_*`), insert a 2-D dispatch **before** the
layout-translation else-branch, operating on the ORIGINAL layouts:

```rust
// pseudo-guard (both families; serial + rayon twins)
if let (Ok(lc2), Ok(la2)) = (lc.to_dim::<Ix2>(), la.to_dim::<Ix2>()) {
    let (sc, sa) = (lc2.stride(), la2.stride());
    if sa[1] == 1 && sc[0] == 1 {
        return orderchange_out_r2c_ix2_cpu_serial(c, &lc2, a, &la2);
    }
    if sa[0] == 1 && sc[1] == 1 {
        return orderchange_out_c2r_ix2_cpu_serial(c, &lc2, a, &la2);
    }
}
// fall through: existing translate + iterator path, byte-identical
```

Guard set (mirrors the elementwise acceptance contracts):

- `ndim == 2` exactly (anything else falls through unchanged);
- fast-axis strides must be **exactly +1** — this rejects zero strides
  (broadcast fast axis), negative fast strides (flip on the fast axis), and
  sliced fast axes (`a[:, ::2]` → stride 2) in one comparison each;
- slow-axis strides: arbitrary isize — zero (broadcast slow axis: the
  kernel re-reads one row/column; arithmetically correct — phase-2
  correctness adds the `bcast.t()` case), negative (flip on slow axis —
  verified correct by direct-kernel check), or padded `lda`/`ldc` all pass
  the guard and are in-kernel correct;
- offsets: `isize` sums, `usize` cast only at the indexing site (the kernel
  already does this); debug_assert bounds contracts if we switch the serial
  stores to raw pointers (measured ≈0-5% — do it only as cleanup, not for D3);
- aliasing: A path writes freshly allocated storage (`change_layout_f`);
  B path (`c.assign(&src)`) cannot alias through the borrow checker (same
  argument as T4' PLAN §3);
- overlap: impossible through safe API for the same reason (B writes `c`
  while borrowing `src` immutably);
- thresholds: serial — none (blocked ≥ iterator at every measured size;
  64² should improve from ~14 µs toward ~1-2 µs); rayon — keep the kernel's
  built-in `size < 16·BLOCK²` serial fallback;
- dtype-generic (`T: Clone` already required by both families); visit order
  changes only within the copy — a copy has no stateful closure, so the
  T4'-style visit-order note does not even apply (result is bit-identical
  unconditionally).

Blast radius: ~30-50 lines across `cpu_{serial,rayon}/assignment.rs`; no
trait/API/feature changes. Covers BOTH user idioms (A `to_contig` and
B `assign`) because both families route through it.

Expected (probe-anchored): B serial large 17.2 → ~7-8 ms (≈2.3×; kernel
5.7 + ~1.9 ms wiring), B faer16 1.45 → ~0.8 ms (≈1.8×), odd serial
2.42 → ~0.4-0.5 ms (≈5-6×), medium 0.82 → ~0.15 ms, small 14.1 µs → ~1-2 µs.
D3 (≥10% large, both configs) exceeded ≥10× margin.

### Option B — generic any-rank blocked strided assign (elementwise-tile style) in the assign families

Port the T4' blocked-tile recipe to `assignment.rs` (guards: no common
f-contig prefix, ndim ≤ some bound, size ≥ floor, no zero strides anywhere).
Would ALSO accelerate sliced views whose fast axis is strided (e.g.
`a[::2, ::2]` assigns) — cases Option A's guard rejects. BUT for the pure
transpose it cannot beat the specialized kernel: it pays per-axis generic
index arithmetic per element where r2c/c2r uses two strided adds, and the
measured specialized kernel (5.7 ms) already achieves what a 64×64 tile
scheme achieves (11.3 GB/s serial). More complex guards, more code, same or
worse transpose numbers. **Kept as a follow-up (T5-class) if strided sliced
assigns ever show up in a profile** — the baseline for them (8.75 ms
`sliced_t_large`, 6.40 ms `sliced_large` serial native) is now recorded in
`assign_gates`.

### Option C — dispatch at the rstsr-core layer (`change_layout_f` / tensor assign)

Rejected: requires a new device-API method (surface area), needs separate
wiring to cover `c.assign` (B variant), and the elementwise precedent
(T4') put the fast path inside the native kernels so ALL callers benefit.
Consistency argues for native-impl.

### dispatch_simd verdict: SKIP (per plan §5)

The cost is iteration machinery + hop pattern, not per-element arithmetic a
lane type could vectorize: strided loads are a gather (T4' showed native ≈
portable on the same pattern), and the plain blocked kernel already reaches
2.7× serial with zero new dependencies. No `dispatch_simd` proposal from T1'.

**Recommendation: Option A.** Option B stays available as a later,
independent extension (its guard set is disjoint from Option A's; both can
eventually coexist — Option A catches stride-1 fast axes first, Option B
the rest, with Option B's guard excluding what Option A took).

## 3. Phase-2 bench matrix & accept criteria

- Re-run THIS crate's suite unchanged after the rstsr edit (path dep picks
  it up): `./reproduce.sh correctness portable native portable_c native_c perf`,
  then archive as `candidate_*` (the `candidate` stage documents this).
- Primary judge: **B variant, large 2048², ≥10% improvement in BOTH
  configs** (probe-anchored target: ≥1.7× serial, ≥1.5× faer16).
- Secondary: A ≥20% large serial; odd/oddT/medium/small B improvements
  reported (expected 5-30×); C reconfirms rider recovery.
- Gates (no >2-3% regression, BOTH configs): `assign_gates` contig small/
  large (slice-copy path untouched), `sliced`/`sliced_t` large/odd (guard
  must REJECT stride-2 fast axes → fall-through canary), ndarray anchors
  untouched, f32 A/B rows.
- Correctness 100% both configs — existing gate already covers: both
  orientations, degenerate 1×7 / 7×1, flip(0)/flip(1) transposed views,
  zero-stride broadcast + transposed broadcast, sliced views, 3-D
  reverse-axes fall-through, i32/f32, direct-kernel checks incl. negative
  slow-axis stride. Phase 2 must additionally verify the `bcast.t()`
  tensor-level case actually routes through the kernel (stride-0 slow axis)
  — it is already in the gate and must stay PASS.
- perf stat re-run on tb_serial / tb_faer16: expect ins/elem ≪ 108
  (blocked kernel ≈ 10-15 ins/elem class) and GB/s ≫ 4.

## 4. Risks & fallbacks

| risk | mitigation / fallback |
|---|---|
| Small-size regression from kernel dispatch overhead | 64² gate in transpose + assign_gates; if regressed, add a size floor (e.g. route only when size ≥ 4096) |
| `to_dim::<Ix2>()` cost on every strided assign | it is a shape-length check + copy (ns); the gates cover hot contig/small paths |
| Zero-stride slow axis mis-routed | arithmetic re-reads one row — correctness gate has `bcast.t()` through BOTH to_contig and assign |
| Negative fast axis (flip) mis-routed | guard rejects (`stride != 1`); correctness gate has flip(0)/flip(1).t() |
| faer16 rayon overhead at odd sizes | kernel's own `16·64²` serial fallback retained; odd faer gate watched |
| Wiring overhead masks kernel win at large (B) | B target still ≥1.7× with wiring included (probe-anchored); if not, look at the 1.9 ms translation cost as follow-up |
| Regression hides behind A fault rider | judge on B (T7 rule); C recorded as secondary |

## 5. Interaction with the T4' elementwise patch (verified)

T4' `proposed.patch` touches exactly
`rstsr-native-impl/src/cpu_rayon/op_with_func.rs` and
`rstsr-native-impl/src/cpu_serial/op_with_func.rs`
(functions `op_mutc_refa_refb_func_cpu_*`, `op_muta_refb_func_cpu_*`,
`blocked_2d_*` helpers). T1' Option A touches
`rstsr-native-impl/src/cpu_{serial,rayon}/assignment.rs` (the
`assign_arbitary_*` and `assign_*` duplicate families) and, only if kernel
tuning happens, `cpu_{serial,rayon}/transpose.rs`. **Different files,
different functions — the patches are line-disjoint and can land in either
order.** Semantic check: the elementwise tile path executes only inside the
`op_with_func` drivers, which pure copy/assign never calls
(`change_layout_f` → `assign_arbitary_uninit`; tensor `assign` →
`OpAssignAPI::assign`); conversely T1's new path never executes for
arithmetic ops. No double-dispatch, no shared mutable state.

## 6. Phase-2 execution checklist

1. `git -C ../rstsr status` clean + HEAD 386948be (re-verify).
2. Implement Option A (serial first, then rayon twins).
3. `./reproduce.sh correctness` (both configs) — 100% PASS required.
4. `./reproduce.sh portable native portable_c native_c perf`; archive
   outputs as `candidate_*` in results/; update tables.md.
5. Gates per §3; if any gate fails >2-3%, dissect before proceeding.
6. If D3 PASS: `git -C ../rstsr diff > proposed.patch` (do NOT commit);
   then `git -C ../rstsr checkout -- .` to restore.
7. If FAIL: revert, document honest negative; consider Option B.

---

## 7. PHASE-2 OUTCOME (executed after G1 PASS; D3 verdict: PASS)

### 7.1 Dtype handling decision (G1 review item 1 — MANDATORY)

**Chose the `into_cast()` generalization branch, NOT the TypeId guard.**
The new kernels are `orderchange_out_{r2c,c2r}_ix2[_uninit]_promote_cpu_*`
with `c[dst] <- a[src].clone().into_cast()`:

- `DTypeCastAPI::into_cast` is identity-inlined for `TC == TA` — the same
  contract the existing promote assign families already run on every
  same-dtype assign (their contiguous branch is memcpy-class in phase-1
  measurements with the same `into_cast` in the loop). Zero same-dtype cost.
- A `TypeId::of::<TC>() == TypeId::of::<TA>()` guard would require either
  monomorphizing over a CLOSED dtype list (rstsr's dtype set is open:
  i16/u32/Complex/half all flow through the generic kernels today) or an
  unsafe slice transmute to a common type that cannot be named. The
  generalization avoids both.
- Consequence (beyond the review's non-goal minimum): cross-dtype cast
  assigns now route through the blocked path CORRECTLY (into_cast applied
  per element) instead of being excluded. Same-dtype remains the benched
  configuration; cross-dtype correctness is covered by the promote-family
  contract and the f32/i32 tensor-level cases run the same-dtype path —
  cross-dtype (e.g. i32 -> f64 assign) was NOT specifically re-gated
  through the 2-D path and is recorded as a known, accepted limitation of
  the correctness matrix (the code path is shared with same-dtype; only
  the cast differs).

### 7.2 What was implemented

- `cpu_serial/transpose.rs`: +promote/uninit duplicate variants of the r2c
  kernel + c2r wrappers (original T-kernels untouched for LAPACK callers).
- `cpu_rayon/transpose.rs`: +rayon promote/uninit variants + c2r pool
  wrappers (raw-pointer stores now with `debug_assert!` bounds contracts;
  original kernels untouched).
- `cpu_{serial,rayon}/assignment.rs`: both duplicate families gained
  per-variant `oc_r2c`/`oc_c2r` columns; the non-contiguous branch of each
  gained the 2-D guard (`to_dim::<Ix2>()` + fast-axis strides exactly +1,
  r2c first, then c2r) with a fall-through comment. Everything not matching
  reaches the iterator path byte-identical; the contiguous branches are
  unreachable to the guard (family 1: explicit contig check first; family 2:
  only `size_contig < CONTIG_SWITCH` enters the guard branch — a
  fully/partially contiguous pair is normalized by the greedy translation
  into the contig branch first).
- MaybeUninit justification (G1 item 2): documented on the kernels — the
  blocked loop covers every (i, j) exactly once, so writing
  `MaybeUninit<TC>` slots directly is sound; isize offset sums with the
  `usize` cast only after the full sum (elementwise-precedent contract).

### 7.3 Results (full tables: results/tables.md; raw: results/candidate_*)

Reuse variant B (primary judge) before -> after:

| case | serial native | serial portable | faer16 native | faer16 portable |
|---|---|---|---|---|
| large 2048² | 17.155 -> 5.989 ms (**2.86×**) | 17.546 -> 5.958 ms (**2.94×**) | 1.450 -> 0.547 ms (**2.65×**) | 1.504 -> 0.562 ms (**2.68×**) |
| odd 1000×777 | 2.425 -> 0.251 ms (**9.7×**) | 2.447 -> 0.248 ms (**9.9×**) | 0.360 -> 0.106 ms (**3.4×**) | 0.367 -> 0.101 ms (**3.6×**) |
| oddT 777×1000 | **8.1×** | **8.8×** | **8.8×** | **8.8×** (0.370 -> 97 µs) |
| medium 512² | **2.6×** | **3.5×** | **2.6×** | **3.8×** |
| small 64² | **6.5×** (14.1 -> 2.2 µs) | **6.2×** | **6.4×** | **6.4×** |

Secondary: A large serial 23.2 -> 10.35 ms (**2.2×**, rider shared); f32 B
large **2.9×/3.1×**; C large **2.9×/2.75×**; review rows: r2c orientation B
**2.93×/2.68×**, bcast-slow-axis-0 B **8.4×/3.7×** (bcast_t odd A/B ~10×).
perf (B serial native): ins/elem **107.8 -> 13.2**, cyc/elem 22.4 -> 8.6,
GB/s 4.0 -> 11.1, L1d-miss 2.9% -> 49.9% (instruction-mountain ->
memory-bound: the correct regime for a copy).

### 7.4 Gates & dissection

All gates within ±3% except two single-suite cells, dissected per the T4'
protocol with re-runs (results/gates_rerun_{clean,candidate}.txt + dissect
runs): `sliced_t_large faer16-native` (+6.1% in the suite; candidate re-runs
861-868 µs vs baseline 916 — does not reproduce) and `contig_large
faer16-portable` (+5.5% suite; interleaved clean/candidate runs overlap:
clean 1.354-1.398 ms, candidate 1.232-1.474 ms, mean +1.4%, and the contig
branch is provably unreachable to the new guard — greedy translation
normalizes contiguous pairs into the contig branch, `ndim_of_f_contig`
analysis). Both recorded as build/run noise, no mechanism; no >3%
reproducible regression anywhere. ndarray anchors unchanged (their own
tree-independent spread this session was ±8%: 5.29-6.14 ms large native).

### 7.5 Non-goals (recorded per G1)

- Cross-dtype cast assigns: routed correctly via `into_cast` generalization
  (see 7.1 — beyond non-goal), but not separately re-gated end-to-end.
- Sliced fast axes (stride k ≠ 1, e.g. `a[:, ::2]`): guard rejects; they
  stay on the generic iterator path (canary gates `sliced`/`sliced_t`).
- 3-D+ order changes: fall through unchanged (correctness 3-D case).
- BLOCK_SIZE retuning / unrolling: not attempted (2.7× measured with the
  in-tree BLOCK=64; a sweep is a possible follow-up, not needed for D3).
- dispatch_simd: skipped (§2/§5 rationale unchanged).

### 7.6 Patch & tree state

`proposed.patch` (+434/-28 lines, 4 files, 2 kernel files + 2 wiring files)
captured from the working tree, `git apply --check` verified on the clean
`386948be` checkout, and the tree restored (`git checkout -- .`, status
empty, HEAD 386948b). The `candidate` stage of reproduce.sh re-runs
everything end-to-end on a patched tree.
