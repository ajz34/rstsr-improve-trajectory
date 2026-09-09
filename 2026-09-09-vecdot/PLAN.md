# PLAN.md — T3' phase 2: batched vecdot (serial) + the inner_dot path

Status: **phase 2 COMPLETE — D3 PASS.** Implemented exactly as designed
below (A1+A2+B1+B2), with the four G1 review items folded in (bound-widening
PR-note, bit-exactness comment on the dropped `0 + val`, honest serial-vs-
faer framing, both uninit-hazard shapes named). Results, D3 verdict and
PR-notes: [README.md](README.md) + the PHASE 2 section of
[results/tables.md](results/tables.md). Patch: [proposed.patch](proposed.patch);
`../rstsr` restored to clean `386948be` after measurement.
This document retains the original phase-1 (G1 review) content below.

---
Base: rstsr `386948be819baa334b8da02232f3a1944e5447d5` — clean tree verified
before baselining (`git -C ../rstsr status --short` empty, HEAD `386948b`);
**phase 1 made NO edits to rstsr** (re-verified at the end, see README).
All phase-1 numbers: [results/tables.md](results/tables.md), raw criterion
logs under `results/`, probes `results/probe_*.txt`, layout branch probe
`results/branch_probe.txt`, perf `results/perf/`, numpy anchors
`results/numpy_anchors_*.txt`.

Phase-1 scope note (deviation from plan-T3 wording, per the T3' brief): the
1-D dot is a **gate, not a target** (T0 showed it memory-bound at 56.9 GB/s ≈
2.3× the DRAM-bound kernel; 3.9%-of-peak is a misleading framing for it). The
phase-1 bench matrix therefore adds the batched axis-0 case, the strided case,
Complex<f64> (D5), and the previously-unbenched `%`/inner_dot surface.

## 1. What phase 1 established

### 1.1 inner_dot API discovery (what user-level op reaches `inner_dot_naive_cpu_rayon`)

```
&v1 % &v2  (both operands 1-D, output 0-D)   [also rt::matmul with 1-D args]
  → TensorAny::matmul: tensor/linalg/matmul.rs:149 op_refa_refb_matmul(a, b, TC::one())
    → allocates fresh output (empty), beta = TC::zero()   (:283-ish op_mutc_refa_refb_matmul(&mut c, &a, &b, 1, 0))
  → DeviceFaer::matmul:            device_faer/matmul.rs:216-238 → matmul_row_major_faer (RowMajor)
    → rule (1,1,0): device_faer/matmul.rs:114-122 → inner_dot_naive_cpu_rayon
  → DeviceCpuSerial::matmul:       device_cpu_serial/linalg/matmul.rs:9-41 → matmul_naive_cpu_serial
    → rule (1,1,0): cpu_serial/matmul_naive.rs:32-39 → inner_dot_naive_cpu_serial
```

- **`%` on 1-D is the ONLY route to inner_dot** (`(1,1,0)` rule). Matvec-like
  `(m,n)·(n,)` goes through the broadcasted rules into gemm/gemv — out of T3'
  scope. The five BLAS backends (mkl/openblas/blis/aocl/kml) call the same
  `inner_dot_naive_cpu_rayon` fallback at their `matmul.rs:124` — but those
  crates are out of campaign scope (D1); the fix transfers automatically.
- **Under DeviceFaer, 1-D `%` NEVER reaches faer** — for every dtype,
  including f64, it runs rstsr's own rayon fold. `alpha` is always `1`,
  `beta` always `0` on this surface (fresh output), so an `alpha==1 &&
  beta==0` guard covers the entire user surface.
- **Latent hazard found (report, not fix)**: both twins read `c` before
  writing (`beta * c[idx].clone()` serial, `c.clone() * beta` rayon) even
  though `c` is a fresh `empty` (uninit) allocation. With beta=0 this is
  0×uninit-bits: benign when the bits are finite-or-zero, NaN-poisoning when
  the allocator hands back NaN-pattern bits. Our gate passes because fresh
  pages are zero-filled. Phase-2's fast path (write without reading when
  `beta==0`) incidentally removes the hazard for the guarded case; the
  unguarded fallback keeps current behavior. Flag to the human.

### 1.2 vecdot branch anatomy at 386948be (file:line) — which branch serves which case

`vecdot_naive_cpu_serial` (`rstsr-native-impl/src/cpu_serial/vecdot.rs:6-164`):

- `:44-68` builds the branch flags with `get_axes_composition`
  (`rstsr-common/src/layout/rearrangement.rs:336`): `flag_contig_s` = summed
  axes of a AND b share a common contiguous prefix; `flag_contig_m` =
  remaining axes of a, b AND c share one.
- **Branch 1** (`flag_contig_s`, `:74-104`): per remaining position —
  `layout_col_major_dim_dispatch_3` closure → `layout_col_major_dim_dispatch_2`
  over the DISCONTIGUOUS-summed layouts (0-dim ⇒ 1 iteration when all summed
  axes are contiguous) → `unrolled_binary_reduce`
  (`cpu_serial/reduction.rs:96-140`, ndarray-style 8-lane) over the contiguous
  run → single `c[idx_c].write(val_c)`. **Serves**: 1-D dot AND batched
  axis-(-1) (T0's primary case) AND odd/broadcast-remaining cases
  (branch_probe.txt).
- **Branch 2** (`flag_contig_m`, `:105-138`): remaining contiguous, summed
  not. Initializes `c` band by band (CHUNK=64), then per band walks the summed
  layouts (`layout_col_major_dim_dispatch_2` **inside the band loop**), doing
  per element per summed step `c.write(c.assume_init_read() + val)` — **the
  MaybeUninit read-modify-write** — plus one `dispatch_2` machinery walk per
  band. **Serves**: batched axis-0 contraction (and stride-0-summed broadcast).
- **Branch 3** (general, `:139-163`): per output, `las.ndim()==1` fast-ish
  stride fold (`idx_m + i*step` per element), else per-element
  iterator fold. **Serves**: strided/transposed-b and f-contiguous cases.
- Rayon twin `vecdot_naive_cpu_rayon` (`cpu_rayon/vecdot.rs:9-188`) mirrors
  the three branches with `*_par_*` dispatch; `PARALLEL_SWITCH=512` serial
  fallback at `:31-33`.

**Correction to T0/code-map**: T0's README attributed the batched
`(4096,512)·(4096,512)` ax-1 case to the "contiguous-remaining (RMW)"
branch. The branch probe (`results/branch_probe.txt`, replicating :44-68
exactly) shows that case hits **branch 1** (contiguous-SUMMED, no RMW). The
RMW branch serves the **axis-0 contraction** geometry — which is why
`batched_axis0` (same data, 512×4096 ax0) costs 962-1614 µs serial while
`batched_am1` costs 444-465 µs.

### 1.3 perf findings: why 466 µs serial vs 148 µs faer16 (batched_am1)

`perf stat -d` (native, `results/perf/derived_summary.txt`):

- batched_am1 serial: **1.23 cyc/elem, 3.68 ins/elem, IPC 3.0, L1d-miss
  14.3%** — a tight, mostly load-bound loop, NOT an instruction flood.
  463 µs/iter at 72 GB/s (L3-assisted; DRAM triad 40 GB/s).
- The gap to the machine is NOT in the contraction loop: probe **A0** (the
  branch-1 body alone — same `unrolled_binary_reduce` shape on plain slices)
  runs the identical math in **0.292 ms native / 0.350 ms portable**
  (115/96 GB/s, 0.77 cyc/elem). The ~1.6× delta is per-row dispatch
  machinery: `layout_col_major_dim_dispatch_3`'s `IterLayoutColMajor::next`
  × 3 per row + nested `dispatch_2` closure plumbing — consistent with T2''s
  measured 18-79 cyc per `next()` call, ~4096 rows per op.
- faer16 (148 µs wall) burns **2.43 ms of core-time per iteration — 5.2× the
  serial core-time — at 10.3 ins/elem**: the same per-row machinery, spread
  over 15.4 CPUs. The parallel path is ~3.1× wall because per-core
  efficiency collapses, not because the work splits badly. Both facts say:
  remove the per-row machinery and BOTH columns improve; the faer16 column
  must simply not regress (it gets the same restructure).
- batched_axis0 serial (branch 2): **IPC 1.48, 3.17 ins/elem, 2.15 cyc/elem**
  — dependency-chained RMW accumulation plus a per-band re-walk of the summed
  layouts (dispatch_2 per 64-element band ⇒ 64 bands × 512 rows). Probe B0
  (RMW replica, no iterators) already costs 1.76 ms ≈ the real kernel;
  probe **B1** (full 32 KiB vacc, single summed walk, row walks) runs the
  same math in **0.446-0.515 ms** — 3.4-3.9×. This is the single largest
  serial vecdot win available, bigger than the branch-1 case T0 pointed at.
- batched_strided serial (branch 3): 13.6 ins/elem, 7.6 cyc/elem, 11.8 GB/s —
  the b-side gather (stride 4096 doubles) is latency-bound; 8-lane
  restructure does nothing (C0=C1, 2.97-3.00 ms both configs). numpy einsum
  on the same geometry: **25-78 ms** — rstsr is already 8-26× ahead. Honest
  negative: no anchor pressure, leave branch 3 alone (plan hypothesis (d)
  resolved: not worth it).

### 1.4 inner_dot findings (the untested #2 from the code map)

- serial `%` 1e7: **4136 µs (portable) / 4068 µs (native), 12.25 ins/elem,
  IPC 5.18** — a pure instruction flood (per-element `index_uncheck`
  address arithmetic + `alpha.clone() *` + clone-through-fold), running
  1.4× SLOWER than the same math through `rt::vecdot` (2934 µs). The
  instruction flood IS the problem (probe D1: a plain sequential zip fold on
  slices is equally slow, 3.98 ms — the sequential dependency chain caps at
  ~40 GB/s; probe D2: the 8-lane unrolled form reaches 2.80-2.86 ms =
  56-57 GB/s, matching rt::vecdot/T0).
- faer16 `%` 1e7: 1819-1839 µs wall but **25.7 ms core-time/iter (15.4 CPUs),
  11.76 ins/elem, IPC 0.84** — a per-element rayon fold with
  `index_uncheck`, re-associated by `reduce_with`. Same pathology, parallel.
- Small-n `%` under DeviceFaer: **10.4 µs at n=64, 29.4 µs at n=1e4** vs
  0.34/3.9 µs on the serial device — `inner_dot_naive_cpu_rayon` has NO
  PARALLEL_SWITCH serial fallback (its vecdot twin has one at 512).
- Rounding contract (verified in the gate): serial `%` folds strictly
  sequentially (bit-equal to naive); faer16 re-associates. **`%` has no
  cross-device bit-exact contract today**, unlike f64 vecdot branch-1 (whose
  per-output summation order we preserve bit-exactly in the design below).

### 1.5 D5 secondary dtype (Complex<f64>) evidence

- rstsr batched_am1 c64 serial: 1383/1248 µs (port/native) — ties numpy
  einsum c64 (1.35 ms single-thread). Probe A5 (plain c64 fold, the shape
  the restructure would produce) = 1.27-1.36 ms: **the c64 time is in the
  arithmetic loop, not dispatch overhead** (unlike f64).
- c64 mul-add does NOT auto-vectorize in fold shape (A5: 9.9 GFLOP/s native
  = 5.4% of peak); an 8-lane `[Complex<f64>;8]` variant does not decisively
  help (A6: 1.39 ms portable, 1.45 ms native — worse under native).

## 2. Design options for phase 2

### Option A — vecdot branch-1 + branch-2 restructure (RECOMMENDED)

**A1 (branch 1, `cpu_serial/vecdot.rs:74-104` + rayon twin `:84-116`):**
after the existing `translate_to_col_major` of the remaining layouts, add a
flat-path guard: all three translated remaining layouts (`lc`, `lam`, `lbm`)
are 1-D **and** the summed-discontiguous layouts are 0-dim (i.e. `asc` covers
all summed axes ⇒ exactly one iteration of the inner `dispatch_2`). Then
replace the `dispatch_3` per-position closure with:

```rust
// bases from the translated layouts' offsets; steps from their (single) strides
for i in 0..n_rem {
    let idx_c = base_c + i * step_c;          // isize strides, as IterLayoutColMajor forms them today
    let idx_a = base_a + i * step_a;
    let idx_b = base_b + i * step_b;
    c[idx_c].write(unrolled_binary_reduce(
        &a[idx_a..idx_a + n_contig], &b[idx_b..idx_b + n_contig],
        || TC::zero(), |acc, (x, y)| acc + x.ext_conj() * y, |acc, v| acc + v));
}
```

`n_rem` = the remaining layouts' size; `n_contig` already computed at :76.
Semantics: per output, the contraction is the same `unrolled_binary_reduce`
over the same contiguous run — **bit-exact vs current** (the inner
`dispatch_2` iterated exactly once; its `idx_a/idx_b` formation
(`idx_m + idx_s - offset`) equals `base + i*step` by construction of the
translated 1-D layouts). Guards/fall-through: any non-1-D remaining layout
(3-D+ batched vecdot), any non-0-dim discontiguous-summed part, or a
0-size case falls through to the existing `dispatch_3` body **unchanged**.
isize contract: strides are the translated layouts' own `isize` strides used
exactly as `IterLayoutColMajor` uses them today (offset + i*stride); no new
arithmetic class. Negative strides: layouts carry offsets that keep indices
in-bounds today; the flat path forms the same indices (gate covers t-views,
f-contig, broadcast).

**A2 (branch 2, `cpu_serial/vecdot.rs:105-138` + rayon twin `:117-154`):**
when the contiguous remaining run fits the L1 design budget —
`n_contig * size_of::<TC>() <= 32_768` (the axis-0 bench case: 4096 f64 =
32 KiB) — allocate ONE local accumulator `Vec<MaybeUninit<TC>>` of length
`n_contig` (zero-init once, like the per-band init today), walk the summed
layouts **once** (not once per band), and per summed position accumulate
`acc[j] += x.conj()*y` row-walk style (probe B1 shape), then write the whole
accumulator into `c` at the end (single pass, output order). Kill both
pathologies: the MaybeUninit RMW into `c` per element per step, and the
per-band `dispatch_2` re-walk. Semantics: per output element, additions
occur in summed-position order — identical sequence to today ⇒ **bit-exact
for f64/f32 adds** (today: `for band { for summed { for j } }` gives per-j
order summed-0..K; new: `for summed { for j }` gives per-j order summed-0..K).
Budget guard: larger remaining runs (e.g. reduce 4096 rows of a 1e7-wide
matrix) fall through **unchanged** to the existing band code (bounds the
local buffer; same guard discipline as T2' Option A's scratch cap).
Complex: the same restructure applies (64 KiB c64 vacc at 4096 exceeds the
budget — budget is in BYTES, so c64 falls back above 2048 outputs; fine —
c64 has no dispatch overhead to save anyway, §1.5).

Rayon twins: same restructures inside the `*_par_3` closures (A1: flat path
per parallel chunk; A2: local vacc per parallel task — task-local allocation,
`f` bounds unchanged). **T2' caution**: T2' reverted its rayon-twin rewrite
after a codegen lottery; phase 2 implements the serial side first, gates it,
then attempts the rayon twin with the same gates and reverts it if faer16
regresses — faer16 batched am1 (141-149 µs) and axis0 (121-178 µs) are
explicit no-regress gates.

### Option B — inner_dot fast paths (RECOMMENDED, independent of A)

Files: `cpu_serial/matmul_naive.rs:207-242`, `cpu_rayon/matmul_naive.rs:57-93`
only. No wiring changes (call chains in §1.1 unchanged).

**B1 (serial twin):** guard `la.stride()[0] == 1 && lb.stride()[0] == 1
&& alpha == TC::one() && beta == TC::zero()` →

```rust
let s = unrolled_binary_reduce(&a[la.offset()..la.offset()+n], &b[lb.offset()..lb.offset()+n],
                               || TC::zero(), |acc, (x, y)| acc + x.clone() * y.clone(), |acc, v| acc + v);
c[idx_c].write(s);
```

else fall through to the existing loop **unchanged** (covers strided
row-views, scaled GEMM-style inner calls). The guard covers the ENTIRE
user-level `%` surface (§1.1: alpha=1, beta=0 always).
Semantic note (must be recorded in the phase-2 README): the 8-lane
reassociation changes serial `%` rounding vs today (bit-exact sequential).
This is accepted because (a) faer16 `%` already re-associates — no
cross-device bit-exact contract exists; (b) the gate compares with tolerance
vs naive/ndarray; (c) the win is the entire point (probe D1 shows any
order-preserving fold is chain-latency-bound at 4.0 ms, no win).
`ExtNum`/`Mul` bounds unchanged; `Zero` already required. Also removes the
uninit `beta*c` read on the guarded path (§1.1 hazard).

**B2 (rayon twin):** two changes.
(i) `PARALLEL_SWITCH`-style serial fallback for `n < 512` (mirror
`vecdot_naive_cpu_rayon:31`): call the serial twin (which after B1 takes its
fast path) — fixes the 10.4 µs @n=64 / 29.4 µs @n=1e4 rayon overhead.
(ii) For contiguous inputs, replace the per-element `(0..n).into_par_iter()
.fold(...)` with a chunked fold over `n/CHUNK` contiguous sub-slice pairs
(`CHUNK` ≈ 4096, ≥ 2× threads), each chunk running the same 8-lane reduce on
its slices, combined with `reduce_with(|a, b| a + b)` — keep `reduce_with`
(association stays as non-deterministic as today's rayon combine; no new
contract). Expected: ins/elem 11.76 → ~1-2; wall 1.82 ms → ~0.3-0.6 ms at
1e7 (aggregate-bandwidth-bound). Strided case: keep the existing per-element
fold (unchanged) — row-view dots are rare and correctness-neutral.

### Option C — dispatch_simd lane accumulators (per plan §5: only if batching insufficient)

**Not needed — the verdict is SKIP for T3** (reasoning in §3).

**Recommendation: A1 + A2 + B1 + B2, in that implementation order**, each
gated independently; B does not touch any file A touches (§4), so B can land
separately if A's rayon twin trips gates.

## 3. dispatch_simd verdict + complex decision (D6 — required output)

**Verdict: do NOT propose the `dispatch_simd` cargo feature for T3.**

Per plan §5's order of attack, plain fixed-size batching must miss D3 in
either config before lane machinery is justified. The probes say batching
clears D3 with margin in BOTH configs:

- branch-1: the in-tree `unrolled_binary_reduce` inside the flat A1 loop
  auto-vectorizes where it matters (probe A0: native 0.292 ms vs current
  0.465 — and native-vs-portable for A0 is 115 vs 96 GB/s, i.e. ISA width
  already pays through plain code; a `[f64;8]` local-lane variant A2 is
  SLOWER than A0 in both configs — 0.43 ms — because LLVM schedules the
  slice-based while-loop better than index-arithmetic lane arrays).
- branch-2: B1's win is structural (kill RMW + re-walk), not lane width
  (B2's 8-lane j-split is 1.7× slower than B1's plain row walk — the
  dependency is across rows, which lanes cannot widen).
- inner_dot: same story (D2 = the ubr shape, no new lane types needed).

Feature-cost side of the honest report: `dispatch_simd` would add a cargo
feature + `lightweight-simd` dependency to rstsr-native-impl + rstsr-core,
compile-time and binary-size cost, and a TypeId dispatch layer — for a gap
the measurements say is closed by plain restructuring. Same conclusion as
T2' (§5 of its PLAN) and T6 (§2.5), now confirmed on a third kernel family:
**fixed-lane autovectorization types add nothing that plain
slices+`chunks_exact`+8-lane unrolls don't already get from LLVM.**

**Complex-in-dispatch_simd (D6 decision, evidence in §1.5): complex is NOT
included in dispatch_simd v1 — and since the feature is not proposed at all,
complex has no dispatch path.** Reasons: (a) c64's cost is arithmetic, not
dispatch (restructure ≈ no-op for c64, probes A5 ≈ current rstsr c64);
(b) `Complex<f64>` mul-add does not auto-vectorize in fold shape, and
fixed-lane complex types did not help in the probe (A6 ≈ A5, worse under
native) — a real complex kernel needs interleaved re/im SIMD handling
(intrinsics-class work), outside lightweight-simd's autovectorization model;
(c) c64 is the D5 secondary only, and current c64 already ties numpy einsum.
Deferred as a possible future task ("hand-written complex dot kernel"),
not silently dropped.

## 4. Composeability with the existing patches (file-level, all vs clean 386948be)

Verified by listing `diff --git` targets of all four existing proposed
patches (2026-09-09-{argmax-argmin, reductions, transpose-assign, elementwise}):

| file T3' will touch | argmax | reductions | transpose | elementwise |
|---|---|---|---|---|
| `rstsr-native-impl/src/cpu_serial/vecdot.rs` | – | – | – | – |
| `rstsr-native-impl/src/cpu_rayon/vecdot.rs` | – | – | – | – |
| `rstsr-native-impl/src/cpu_serial/matmul_naive.rs` | – | – | – | – |
| `rstsr-native-impl/src/cpu_rayon/matmul_naive.rs` | – | – | – | – |

Existing patches touch: `cpu_{serial,rayon}/{reduction.rs, assignment.rs,
transpose.rs, op_with_func.rs}` + `rstsr-core/.../{device_cpu_serial,
feature_rayon/auto_impl}/reduction.rs`. **Zero overlap; also none of them
modifies `unrolled_binary_reduce`** (grep-verified, 0 hits), which A1/B1
reuse. **Verdict: T3' composes with all four, in any order, cleanly.**
(The rayon `reduction.rs` hunk-set differs between argmax and reductions
patches — that is THEIR adjacency, unchanged by T3'.)

## 5. Phase-2 bench matrix & accept criteria (D3/D7; matrix = phase 1's, unchanged)

Comparison: `--save-baseline candidate` vs the committed `portable`/`native`
baselines; faer16 cells judged on repeated runs (±8% band, tables.md note).

- **Primary accept (vecdot)**: batched serial improves ≥15% in BOTH configs:
  - `batched_am1_4096x512-ax-1-serial` (443.8 µs port / 464.7 nat → target
    ≤ 380/400 µs; probe ceiling 292-350) — stated-%-of-peak framing: from
    5.2% to ≥ 6.5% of the 32-DP-FLOP/cyc core peak (probe ceiling 7.9%);
  - `batched_axis0_512x4096-ax0-serial` (1613.9 port / 962.4 nat → target
    ≤ 650/820 µs; probe ceiling 446-515).
- **Primary accept (inner_dot)**: `%` 1e7 serial improves ≥15% BOTH configs
  (4136/4068 → target ≤ 3.5 ms both; probe ceiling 2.80-2.86 ms); faer16
  `%` improves ≥15% (1819/1839 → target ≤ 1.55 ms; stretch 0.6 ms);
  faer16 small-n `%` (64, 1e4) improves ≥2× (10.4/29.4 µs → ≈ serial
  fallback values).
- **Regression gates (no >2-3%, both configs, both devices, repeated runs
  for faer16)**: 1-D dot 1e3/1e5/1e7 (f64) — the memory-bound gate; 1-D f32
  1e7 (D8-canary context); batched_am1 faer16 f64 (141-149 µs, the faer16
  headline); batched_am1 (8192,256) both devices; odd (1000,777) both
  devices; small (64,64) both devices; batched_strided both devices (branch 3
  untouched — its cells double as codegen-canary for the same function);
  innerdot strided row-view spot (fallback path untouched); c64 batched am1
  both devices (A-restructure must not hurt c64; expected ≈ no-change).
- **Correctness gate 100% PASS both configs BEFORE any perf claim** — the
  43-check gate including zero-stride broadcast (remaining + summed axes),
  odd sizes, NaN locks, f-contig, strided view semantics, complex
  conjugation (vecdot conj / `%` no-conj), both devices. The serial
  `innerdot-f64-contig` bit-exact lock intentionally CHANGES to tolerance
  (documented §2 Option B) — the gate diff is part of the phase-2 README.
- **Deliverables**: proposed.patch, README with before/after tables +
  `perf stat -d` re-run on batched_am1/axis0 + innerdot (expect ins/elem
  12.25 → <2 on innerdot serial; IPC/cyc/elem on am1 → probe levels),
  updated memory + CONTEXT.md.

## 6. Risks & fallbacks

| risk | mitigation / fallback |
|---|---|
| A1 flat path shifts codegen of the untouched fall-through (same fn) | gates: 3-D+ batched case absent from matrix — add a spot (2,2,4)³-shaped case? Not needed: branch-3 cells + odd/small cells gate the function; if drift shows, `#[inline(never)]` the flat path (argmax-precedent) |
| A1 guard misses a case the dispatch_3 path served (e.g. 2-D remaining with contiguous runs) | fall-through-unchanged discipline: only exact 1-D-remaining + 0-dim-summed takes the flat path; everything else identical code |
| A2 vacc alloc per call costs at small n_contig | budget guard only above ~512 elements; below → existing per-band path (which inits the same zeros anyway); gate small_64x64 |
| A2 changes summation order? | NO — per-element order is summed-0..K in both shapes (§2 A2); gate's f64 exact checks hold; f32 tolerance covers nothing new |
| Rayon twin lottery (T2' precedent) | serial first, gate, then rayon twin behind the same gates; revert twin if faer16 regresses >3% after 3 runs; A lands without the twin if needed |
| B1 reassociation changes serial `%` rounding | accepted + documented (§2 B1); conservative fallback: keep sequential slice fold (NO win — probe D1 — so this fallback equals "report honest negative for serial inner_dot") |
| B1 guard excludes alpha/beta callers | guard covers 100% of the `%` surface (verified §1.1); exotic internal callers keep the old loop |
| B2 chunk boundary changes faer16 combine order | rayon combine order is ALREADY non-deterministic today; gate is tolerance-based; no new contract |
| uninit `beta*c` read (§1.1) — behavior change if we stop reading c when beta==0 | current behavior on garbage bits is already undefined-ish (0×NaN=NaN); phase-2 removes the read on the guarded path only; flag to human in README |
| criterion/run variance on faer16 cells (±8%) | accept criteria use repeated runs (3×) for faer16 gates |

## 7. Phase-2 execution checklist (after G1)

1. Re-verify `../rstsr` clean at `386948be`.
2. A1 serial (cpu_serial/vecdot.rs branch-1 flat path) → gate + bench →
   check accept vs portable/native baselines.
3. A2 serial (branch-2 budget vacc) → gate + bench.
4. B1 (serial inner_dot fast path) → gate (note the lock change) + bench.
5. Rayon twins A1t/A2t/B2t one at a time, gated after each; revert any that
   trip the faer16 gates.
6. perf stat re-runs; README (results + framing + decisions/deviations);
   `git -C ../rstsr diff > proposed.patch`; `git -C ../rstsr checkout -- .`;
   leave tree clean.
