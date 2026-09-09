# PLAN.md — T5 half 1: fill/creation kernel — phase-2 decision record

Status: **phase 1 complete. VERDICT: HONEST-SKIP — no proposed.patch.**
Base: rstsr `386948be819baa334b8da02232f3a1944e5447d5`, clean tree verified
before baselining and untouched since (`git -C ../rstsr status --short`
empty). Phase 1 numbers: [results/tables.md](results/tables.md),
[results/perf/derived_summary.txt](results/perf/derived_summary.txt).

This document records the design options the brief asked to be evaluated and
why the measured evidence rejects a patch on all of them.

## 1. What phase 1 established (call chain — including a code-map correction)

**The plan/code-map premise is half wrong: `fill_promote_cpu_serial` is NOT
the creation path.** At 386948be:

- `rt::full` / `rt::ones` / `rt::zeros` (and `*_like` twins) go
  `tensor/creation.rs` → device `full_impl` / `ones_impl` / `zeros_impl`
  (`rstsr-core/src/device_cpu_serial/creation.rs:17-75`) →
  **`vec![fill; len]`** — the std broadcast/`calloc` constructor. The rstsr
  fill kernel is never called. (faer delegates to the same serial impls:
  `feature_rayon/auto_impl/creation.rs:19,62,68` — creation is
  single-threaded by design.)
  Code map §2(e) "used by zeros/ones creation" and §7 item 8's framing are
  therefore corrected: `full_impl`'s `vec![v; n]` is already a
  vectorized broadcast (with std's all-zero-bits → `calloc` specialization
  for `zeros`), and cannot be improved by touching `fill_promote`.
- `fill_promote_cpu_serial`
  (`rstsr-native-impl/src/cpu_serial/assignment.rs:129-153`) is reached only
  from:
  1. `c.fill(v)` on an existing tensor (`tensor/assignment.rs:171-267` →
     `OpAssignAPI::fill` → `fill_promote_cpu_serial`; faer twin
     `fill_promote_cpu_rayon`, `cpu_rayon/assignment.rs:181-222`, which
     parallelizes only the outer non-contiguous iteration and runs the
     contiguous part serially with raw pointers);
  2. `eye`'s diagonal (`tensor/creation.rs:829`, layout shape `[d]`,
     stride `[n+1]` — the strided `else` branch);
  3. BLAS backend beta-zeroing (`crates-device/*/matmul_impl.rs`, via
     `fill_cpu_rayon`).
- An owned tensor's layout is always contiguous, so the hot user surface of
  the kernel is exactly the contiguous branch
  `for i in 0..size_contig { c[idx_c + i] = fill.clone(); }` plus the
  `translate_to_col_major` + `layout_col_major_dim_dispatch_1` machinery
  around it.

## 2. Design options evaluated (measured, not argued)

### Option (i) — contiguous branch via slice-fill write

`c[idx_c..idx_c + size_contig].fill(fill.clone())` in place of the index
loop. `slice::fill` (stable since 1.50) is **Clone-based** (`T: Clone`,
clones `value` into each slot via `clone_from`), so it is semantically
identical for every dtype, including non-Copy Clone types where
`clone_from` may reuse allocations (never worse).

Measured disposition: the current index loop **already compiles to a
vectorized broadcast store loop** — perf on the reuse fill: 0.29 ins/elem,
IPC 0.55, L1d-miss 46 %, 66 GB/s write (native, 32 MiB). Criterion
kernel-only medians (native): rstsr fill 507.4 µs vs raw
`iter_mut().for_each` bound 543.9 µs at 2048² — **rstsr is at/above the
naive bound** (64-B-aligned rstsr buffer vs 16-B `Vec`); ties at odd,
2.7× at 64² where the absolute gap is 0.2 µs of layout machinery that a
slice-fill would NOT remove (the translate + dispatch must stay for the
general layout contract). D3 outcome class: **< 5 % — note and move on**.
A slice-fill change would be pure code clarity with ~0 performance effect.

### Option (ii) — rayon twin check

Measured: faer16 `c.fill` ≈ serial `c.fill` at large (529.8 vs 507.4 µs) —
parallel fill never wins because a single Zen 5 core already saturates write
bandwidth (T7 RQ4 consistent); at medium (512², 262 k elems >
PARALLEL_SWITCH=16384) the parallel path is 18 % SLOWER (16.4 vs 13.8 µs).
Micro-observation for the edit guide: the fill/assignment PARALLEL_SWITCH
could sit higher; the effect is ~µs-class, no patch.

### Option (iii) — strided `else` branch

Stride-2-both-axes spot: 409 µs serial for 1.05 M elements (20.5 GB/s) —
the per-element layout-iterator anatomy known from T1/T4'. faer16 187 µs
(2.2×, outer-parallel). A stride-based walk would plausibly recover ~2×
serial. But **no tensor-API user can reach this branch**: owned tensors are
contiguous; `eye` fills 2048 diagonal elements in 0.7 µs; BLAS zeroing is
contiguous/f-contig. A patch with no reachable surface fails the
cost/benefit test — EDIT-GUIDE note only.

### The 88 % rider question (T7)

`rt::full` 2048² = 4.65 ms of which ~4.0 ms is the page-fault rider
(8193 faults/iter, 90 % sys — T7 reproduced exactly). The remaining
0.5–0.6 ms broadcast kernel is at the write floor: with the T7 env tunables,
`rt::full` drops to **0.62 ms (~7.5×)**, beating numpy's allocating fill
(1.19 ms) and tying the reuse fill (0.51 ms). The kernel is exonerated; the
actionable guidance is T7's (reuse `c.fill` idiom + env tunables docs), not
a kernel patch.

## 3. Accept criteria (set before measurement, per the brief's "expected honest outcome")

- D3 on kernel-only denominators (reuse fill vs raw bound, large, BOTH
  configs): fill wins are 0–7 % — **far below the ≥10 % patch bar** in both
  configs → no patch.
- Correctness gate (41 checks, f64/f32, odd/degenerate shapes, strided +
  broadcast device-level layouts, both devices, BOTH RUSTFLAGS configs):
  ALL PASS (`results/correctness_{portable,native}.txt`) — locks CURRENT
  behavior; relevant for any future fill change.

## 4. What would change the verdict (for future re-evaluation)

- A maintainer decision to give `c.fill` a c-contig fast path anyway
  (skip `translate_to_col_major` for `lc.c_contig()` layouts) would shave
  ~0.1–0.2 µs/call at L1-scale fills (3× at 64×64, absolute nanoseconds) —
  a convenience/clarity change, not an efficiency one.
- If `full_impl` were ever moved off `vec![]` (no reason found to), the
  replacement must preserve the all-zero-bits → calloc behavior: T7/T0's
  "never make zeros touch" has a size-regime caveat (zeros is calloc-lazy
  only ≥ ~32 MiB outputs; below the cap calloc memsets — measured
  2.4× slower than `full` at 6.2 MiB because glibc's non-temporal memset
  runs at DRAM speed while the broadcast loop stays cache-resident) — but
  the absolute numbers (tens of µs) still say "leave it alone".
- The strided branch becomes worth patching only if a user-facing strided
  fill/masked-assign API is added upstream; the T4'-style recipe (blocked or
  stride walk) transfers directly.

## 5. Edit-guide outline (consolidation half; full text in EDIT-GUIDE.md)

(a) dispatch_simd final ADR-candidate text (NOT adding the feature);
(b) symlink correction to code map §5 (+ §2(e) correction from §1 above);
(c) serial/rayon kernel duplication vs shared blocking helpers;
(d) MaybeUninit closure API ergonomics at the operator layer;
(e) reductions lack a reuse path;
(f) THP alignment blocker;
(g) leftovers: `arg_contig_seeded` visibility, fill/assignment
PARALLEL_SWITCH, zeros size-regime doc note, ndarray/numpy anchor caveats.
