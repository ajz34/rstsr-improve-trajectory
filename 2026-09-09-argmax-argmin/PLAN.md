# PLAN.md — T6 phase 2: argmax/argmin contiguous fast path (for G1 review)

Status: **IMPLEMENTED (phase 2 complete)** — this document is the G1-approved
design; deviations from it during implementation, all G1-compatible:

- The fast path is held in a `#[inline(never)]` dedicated function (serial +
  rayon) instead of inlining into the dispatching `pub fn`: inlining both
  paths perturbed the untouched strided fallback's codegen (+2.3–3.8% native);
  out-of-lining restored it (paired A/B ≈ +1%).
- The lane seeding changed after the probe caught a real bug in the first
  draft (a NaN at positions 1–8 blocked its whole lane and could swallow a
  later same-lane max): the poisoning first element is now detected up front
  via the generic `xs[0] == xs[0]` NaN idiom, and all lanes are seeded with
  that guaranteed-comparable element (`arg_contig_seeded_cpu_serial`,
  `usize::MAX` sentinel). Only global index 0 may poison — exact semantics.
- Rayon combine collects per-chunk partials **in order** and folds them
  sequentially (instead of a pairwise rayon `reduce`): fully deterministic,
  thread-count-independent, exactly serial-equivalent — also resolves the G1
  review item 2 (pre-patch NaN pairing instability; probe transcripts in
  `results/probe_nan_{clean,patched}_tree.txt`).
- `device_faer/rayon_auto_impl/*` turned out to be **symlinks** into
  `feature_rayon/auto_impl/*` (one file, not a dead copy) — the wiring edits
  therefore appear under the `feature_rayon/` path in the diff.

Results and the D3 verdict: see README.md ("Phase 2 — candidate kernel").
Patch: [proposed.patch](proposed.patch); `../rstsr` reset to clean `386948be`.

---

Original design (as reviewed at G1):

Goal: replace the per-element instruction flood (280 ins/elem, 0.85 GB/s) in
rstsr's argmin/argmin reduce-all kernels with a contiguous 8-lane-unrolled
(value, index) fast path, preserving the EXACT current semantics documented in
README.md ("Current semantics"). Realistic target 5–10x at 1e7 serial
(93 ms → ≤ 19 ms; stretch ≤ 4.7 ms ≈ DRAM floor), while strided/small paths
stay untouched.

---

## 1. Exact files/functions to change (rstsr @ 386948be)

### 1.1 Native kernels (the real work)

`rstsr-native-impl/src/cpu_serial/reduction.rs`

| item | current lines | change |
|---|---|---|
| `reduce_all_unraveled_arg_cpu_serial` | 427–479 | **Primary edit.** Add c-contig fast path at the top (after the existing size>0 assert at 439); keep the existing `IndexedIterLayout` fold as the strided fallback, unchanged. |
| `reduce_all_arg_cpu_serial` | 533–553 | No code change (delegates to the above; fast path applies automatically). |
| `reduce_axes_unraveled_arg_cpu_serial` | 481–531 | No code change; its per-output-element calls to `reduce_all_unraveled_arg_cpu_serial` (line 521) pick up the fast path whenever the inner (reduced-axes) layout is contiguous — e.g. `argmax_axes(-1)` on a row-major tensor. |

`rstsr-native-impl/src/cpu_rayon/reduction.rs`

| item | current lines | change |
|---|---|---|
| `reduce_all_unraveled_arg_cpu_rayon` | 442–512 | **Primary edit.** After the `size < PARALLEL_SWITCH` serial fallback (458–460), add a c-contig parallel branch: slice split into contiguous chunks → serial fast kernel per chunk → index-aware combine. Keep the existing `IndexedIterLayout` fold/reduce as strided fallback. |
| `reduce_all_arg_cpu_rayon` / `reduce_axes_arg_cpu_rayon` | 579–600 / 602–633 | No code change (delegates). |

### 1.2 Device wiring (only because of the API change in §2.1)

`rstsr-core/src/device_cpu_serial/reduction.rs`

| item | current lines | change |
|---|---|---|
| `OpArgMinAPI for DeviceCpuSerial` (`argmin_axes`, `argmin_all`) | 323–372 | Replace the 8-line `f_comp`/`f_eq` closure pairs with `ArgCmp::Min` (new enum, §2.1). |
| `OpArgMaxAPI for DeviceCpuSerial` (`argmax_axes`, `argmax_all`) | 374–423 | Same with `ArgCmp::Max`. |
| `OpUnraveledArgMinAPI` / `OpUnraveledArgMaxAPI` impls | 519–615 | Same replacement (4 fns). |

`rstsr-core/src/device_faer/rayon_auto_impl/reduction.rs` (DeviceFaer = the
`DeviceRayonAutoImpl` alias, `device_faer/device.rs:10`)

| item | current lines | change |
|---|---|---|
| `OpArgMinAPI` / `OpArgMaxAPI` impls | 356–466 | Same closure→enum replacement (4 fns). |
| `OpUnraveledArgMinAPI` / `OpUnraveledArgMaxAPI` impls | 572–673+ | Same. |

Tensor-level API: **no change** (`rstsr-core/src/tensor/reduction.rs:196–197`
calls `device().argmax_all/…` — signatures there are trait-stable; operator
trait definitions in `rstsr-core/src/operators/reduction.rs:11–43` unchanged).

Not touched: `rstsr-core/src/feature_rayon/auto_impl/reduction.rs` (dead code —
byte-identical duplicate not declared in `feature_rayon/mod.rs`; left alone per
T5 scope), `cpu_serial/assignment.rs`, iterators, `rstsr-common`.

Call chain recap (serial): `rt::argmax` → `tensor/reduction.rs:197` →
`device_cpu_serial/reduction.rs:405 argmax_all` → `cpu_serial/reduction.rs:533
reduce_all_arg_cpu_serial` → `:427 reduce_all_unraveled_arg_cpu_serial`
(fast path here). Rayon: `device_faer/rayon_auto_impl/reduction.rs:411
argmax_all` → `cpu_rayon/reduction.rs:579 reduce_all_arg_cpu_rayon` → `:442
reduce_all_unraveled_arg_cpu_rayon` (fast path here).

## 2. New kernel design

### 2.1 Direction enum replaces the (Fcomp, Feq) closures

The fast path must not call a closure per element, but the current kernel API
is generic over `Fcomp: Fn(Option<T>, T) -> Option<bool>` / `Feq` — the
min/max direction is *inside* an opaque closure. Audit of every call site
(4 fns × 2 device files, lines above): `f_comp` is always exactly `y < x`
(argmin) or `y > x` (argmax); `f_eq` is always `y == x`. No other
instantiation exists in the workspace (grep-verified for
`reduce_all_unraveled_arg|reduce_axes_unraveled_arg|reduce_all_arg_cpu|reduce_axes_arg_cpu`;
these `pub` fns are only called from the two device-wiring files — re-verify at
implementation with a workspace grep before finalizing).

Change: introduce (in rstsr-native-impl, e.g. top of `cpu_serial/reduction.rs`,
re-exported where needed)

```rust
#[derive(Clone, Copy, Debug)]
pub enum ArgCmp { Min, Max }
```

and change the arg* kernel signatures from `(a, la, f_comp, f_eq[, order,
pool])` to `(a, la, cmp: ArgCmp[, order, pool])`. All 8 wiring sites updated
mechanically. Semantics identical by construction: the old closures encoded
exactly Min/Max.

*Fallback option B (if G1 prefers zero API churn)*: keep the closure
signatures for the strided fallback and add `cmp: ArgCmp` as an additional
parameter used only by the fast path. More params, same behavior — default
plan is the clean enum swap.

### 2.2 Serial contiguous fast path (exact-semantics argument)

Condition: `la.c_contig()` (`LayoutBase::c_contig`, layoutbase.rs:202) —
covers 1-D contiguous and row-major contiguous N-D, which is the case where
the hardcoded RowMajor `IndexedIterLayout` visit order equals ascending buffer
order. Otherwise: existing fold, untouched.

Kernel over `let xs = &a[la.offset() .. la.offset() + la.size()];`
(no_std-compatible, no Option, no closure):

```rust
// seed lanes 0..8 unconditionally from the first 8 elements
// (size < 8 handled by a plain scan loop variant)
let mut vs: [T; 8] = [xs[0].clone(), xs[1].clone(), ..., xs[7].clone()];
let mut idx: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
for (base, ch) in (8..).step_by(8).zip(xs[8..].chunks_exact(8)) {
    for l in 0..8 {
        // argmax shown; argmin flips the comparison
        if ch[l] > vs[l] { vs[l] = ch[l].clone(); idx[l] = base + l; }
    }
}
// remainder (< 8 elements): fold into lane by position%8 with the same
// strict comparison (ascending index ⇒ strict > keeps the earlier/smaller one)
// combine lanes with the FULL two-way rule:
let (mut bv, mut bi) = (vs[0], idx[0]);
for l in 1..8 {
    if vs[l] > bv || (vs[l] == bv && idx[l] < bi) { bv = vs[l]; bi = idx[l]; }
}
```

Why this is exact vs the current fold (README "Current semantics"):

- **Seed**: current fold accepts element 0 unconditionally; lane 0 holds
  `xs[0]` and the combine starts from lane 0 accepting later candidates only
  on strict-greater — identical decision sequence.
- **Ties**: within a lane, indices ascend, strict `>` keeps the earlier
  (smaller) index; across lanes, the combine's `vs[l] == bv && idx[l] < bi`
  term picks the smaller global index (needed because lane order ≠ global
  index order for equal values, e.g. lane1 idx=801 vs lane2 idx=2).
- **NaN**: comparisons against NaN are false, so NaN never replaces an
  accumulator, exactly like `f_comp`/`f_eq` returning `Some(false)`; a NaN at
  flat 0 poisons lane 0 and nothing ever beats it → result 0; all-NaN → 0.
- **Empty**: the existing `rstsr_assert!(la.size() > 0, …)` stays first —
  same `Err(InvalidLayout)`.
- **Return value**: current kernel returns the row-major-unraveled `D` of the
  winning position. For a c-contig layout the buffer position `bi` IS the
  row-major flat index; construct `D` by c-order unraveling `bi` against
  `la.shape()` (small ndim-loop; use existing `DimAPI` shape accessors /
  constructor; exact helper chosen at implementation — plumbing only, no
  public API change). Callers (`reduce_all_arg_cpu_serial`) then ravel it
  back to the same flat index; `unraveled_argmax_all` gets the same `D` as
  today.

dtype generics: bound stays **`T: Clone + PartialOrd`** (kernel uses only
`>`, `==`, `.clone()`); no `ExtNum`, no `Copy` requirement, no new trait
dependency. f64/f32/ints/all PartialOrd dtypes keep working.

### 2.3 Layout branches (explicit)

| layout | path | expected effect |
|---|---|---|
| c-contig (1-D contiguous, row-major N-D contig) | new fast path | the win |
| f-contig | **unchanged generic fold in v1** | zero risk; v2/stretch candidate (below) |
| strided / broadcast / t-view | unchanged generic fold | must not regress (only added `c_contig()` check) |

f-contig stretch (only if v1 lands cleanly and time permits): scan the buffer
in storage order with the two-way update rule
`x > v || (x == v && flat_rm(p) < idx)` where `flat_rm` is the row-major flat
index of storage position p — exactness proof: with explicit min-flat-index
tie-breaking and seeding from storage position 0 (which IS the row-major-first
element for any contiguous layout), any scan order yields min-flat-index
among maxima, or 0 if the flat-0 element is NaN — identical outcome to the
visit-order fold. Costs one extra compare per element. NOT in the v1 accept
criteria; the 64²/2048² t-view benches gate whatever we do here to no-regress.

### 2.4 Rayon twin

`reduce_all_unraveled_arg_cpu_rayon`, after the `size < PARALLEL_SWITCH`
(1024) serial fallback (unchanged):

- c-contig branch: split `xs` into `P` contiguous chunks
  (P ≈ threads × 4 via rayon `(0..P).into_par_iter()`; chunk boundaries at
  multiples of 8 so each chunk runs the same 8-lane serial kernel with
  `base_flat = la.offset() + chunk_start`); each chunk returns `(T, usize)`
  via the serial fast kernel; combine pairwise with the two-way rule
  `v > bv || (v == bv && i < bi)`.
- Exactness vs today's rayon combine (cpu_rayon/reduction.rs:493 `sum_func`):
  today partial accs combine through `fold_func`, which accepts the second
  candidate on strict-greater or on equality-with-smaller-index — i.e. the
  same two-way rule resolving ties by lower global index (verified against
  the gate's duplicated-extreme fixtures at n=4099 on the faer device).
  Per-thread folds today walk RowMajor visit order = ascending flat for
  c-contig, so chunk-local results are the same partial answers the current
  code would produce; combining order is irrelevant under the two-way rule.
- Non-contig branch: existing `IndexedIterLayout` fold/reduce untouched.
- `reduce_axes_unraveled_arg_cpu_rayon` (line 514+) calls the all-kernel per
  output element (line 562) — inherits the fast path for contiguous inner
  layouts automatically.

### 2.5 dispatch_simd (lightweight-simd): **SKIP — decided, not deferred**

Per plan §5 the order of attack is "plain fixed-size batching first; reach for
lightweight-simd only if batching alone doesn't meet D3". Assessment for arg*:

- A SIMD argmax needs per-lane (value, index) vectors plus a horizontal
  argmax: numpy does this with explicit compare-select trees and index-blend
  shuffles over 4 unrolled vector accumulators
  (`numpy/_core/src/multiarray/argfunc.dispatch.c.src:30–100`: `m_ba/m_dc/
  m_dcba` masks, `npyv_select` on values AND indices AND an index-scale
  vector for tie correction). That is real intrinsic-style data shuffling.
- lightweight-simd is autovectorization-based fixed arrays, not intrinsics;
  expressing the index-blend/horizontal-argmax through it gives LLVM's
  autovectorizer a job it reliably declines (data-dependent per-lane index
  updates defeat vectorization), and even under `target-cpu=native` the
  scalar-cmov 8-lane loop is already expected to be memory-bound at large
  sizes. T0 also showed native-vs-portable ≈ 1.00x for exactly this class of
  loop structure.
- Therefore: no `dispatch_simd` variant is proposed for T6. If the scalar
  fast path lands below D3 thresholds in BOTH configs (not expected), the
  honest report will say so rather than add an SIMD feature for it.

## 3. Bench matrix and accept criteria (plan D3)

Matrix = phase 1's, unchanged (`benches/arg.rs` + `benches/anchors_ndarray.rs`):
1-D f64 {64, 1000, 1e6, 1e7} + f32 1e7, 2-D whole {64², 512², 2048²}, strided
t-view {64², 2048²}; serial + faer16; portable + native; ndarray anchors.
Phase 2 re-runs identical benches; per-config comparison against the
`--save-baseline` criterion baselines saved by `reproduce.sh baseline`
(phase-1 numbers are committed under `results/`).

- **Accept (D3, memory-bound class)**: ≥10% improvement at large inputs —
  1e7 1-D AND 2048² 2-D whole — in BOTH configs (portable and native), for
  argmax AND argmin; no >2–3% regression on the gate cases below.
- **Regression gates** (small/strided): 1-D `small_64`, `small_odd_1000`;
  2-D whole `small_64x64`; strided `small_64x64-tview` and
  `large_2048x2048-tview` (the strided fallback is untouched — these verify
  no wiring/check overhead); faer16 `small_64` (dispatch overhead unchanged).
  (arg* has no broadcast bench: broadcast inputs walk the unchanged generic
  fold; correctness covers behavior.)
- **Realistic expectation** (framing per §3.8): serial 1e7 0.85 GB/s →
  ~5–10x ⇒ ≤ 19 ms (≥ 4.2 GB/s); anything ≤ 4.7 ms would be at the 17 GB/s
  L3-assisted class; L1-scale (n=64) should drop from 774 ns to tens of ns
  (per-element overhead removal, ~17x vs ndarray's 45 ns becomes parity or
  better). faer16 should scale roughly proportionally (fold over 8-lane
  chunks): 11.9 ms → ~1.5–3 ms at 1e7. ndarray anchor 7.11 ms and numpy
  context 1.32 ms remain the honest comparators.
- **Correctness gate must pass under both configs before any perf claim**
  (it already validates the current tree; after the patch it validates the
  fast path: odd 999_983 lane-tail, all-equal ties, NaN placements incl.
  all-NaN, broadcast, axes reductions, t-view, f32, both devices — plus the
  in-gate duplicate-extreme n=4099 faer combine case). Additionally re-run
  rstsr's own test suite subset: `cargo test -p rstsr-core --test entry_row_cpu
  --no-default-features --features "backtrace row_major"` and the arg tests in
  `tensor/reduction.rs` (`cargo test -p rstsr-core argmin argmax`).

## 4. Risks and fallbacks

| risk | mitigation / fallback |
|---|---|
| LLVM emits branches instead of cmovs in the 8-lane update; adversarial ascending data updates every element and mispredicts | fixtures are random (updates ~O(log n) expected, well predicted); check codegen on the ascending-data correctness fixture (already in gate as all-equal/arange-like cases); fallback: write updates branchlessly (`let upd = x > v; v = if upd { x } else { v };`) which LLVM lowers to cmp+cmov; worst case keep the loop but note in report |
| `[T; 8]` accumulator requires `Clone` not `Copy`; move/drop overhead for non-Copy types | lanes updated via assignment (drop of old value is the same cost as today's clone-through-fold); f64/f32 are Copy and dominate real use; `Clone` bound unchanged from today |
| Enum refactor (`ArgCmp`) touches 8 wiring sites + kernel signatures | mechanical; grep-verified caller set is closed (2 device files; `feature_rayon/auto_impl` is dead code — leave untouched, note in report); fallback option B keeps closure signatures and adds the enum alongside (§2.1) |
| D-index construction from flat index adds plumbing complexity | tiny ndim loop over `la.shape()`; no public API change; if a `DimAPI` constructor is awkward, build via `IxD` then `to_dim()` (one allocation on the final result only, outside the hot loop) |
| Behavior drift in edge cases (empty, NaN, ties, f-contig, broadcast) | gate already encodes all of them and must pass before perf; rayon combine exactness covered by n=4099 > PARALLEL_SWITCH duplicate-extreme fixtures on the faer device |
| f-contig temptation / scope creep | explicitly out of v1 accept criteria; stretch only, gated to no-regress by t-view benches |
| perf variance between runs (L3-warm X3D box) | same harness, same fixture, same iteration policy as phase 1; criterion medians; both configs; state rustc version in the report |

## 5. Phase-2 work sequence (after G1)

1. Verify `../rstsr` clean at `386948be`; `git -C ../rstsr status --short` documented.
2. Implement serial fast path + `ArgCmp` enum (§2.1–2.2); update serial wiring.
3. Gate + `cargo test -p rstsr-core` subset → must pass.
4. Implement rayon twin (§2.4); update faer wiring; gate again.
5. Run full bench suite portable + native with `--save-baseline candidate`;
   `make_tables.py`-style comparison vs `baseline`; perf stat on argmax 1e7.
6. `git -C ../rstsr diff > proposed.patch`; `git -C ../rstsr checkout -- .`;
   write README results tables + conclusion; leave `../rstsr` clean.
