# T3' phase-1 baseline tables (rstsr 386948be, clean tree)

Environment: AMD Ryzen 9 9950X3D (Zen 5, AVX-512), 16 threads
(`RAYON_NUM_THREADS=16`), rustc 1.97.1 (2026-07-14) nightly, criterion 0.5.1
(2 s + 0.7 s warm-up), stock release profile. `portable` = no RUSTFLAGS;
`native` = `-C target-cpu=native`.

Raw outputs: `bench_vecdot_portable.txt`, `bench_innerdot_portable.txt`,
`bench_native.txt` (both benches), reruns `bench_vecdot_portable_rerun.txt`,
probes `probe_portable.txt` / `probe_native.txt` / `probe_c64.txt`,
`branch_probe.txt`, perf `perf/perf_*.txt` + `perf/derived_summary.txt`,
numpy `numpy_anchors_st.txt` / `numpy_anchors_blas.txt` (also
`../numpy_ref/results.txt` snapshot).

## rt::vecdot — µs (criterion median, output allocation included)

| case (branch hit) | serial port | serial nat | faer16 port | faer16 nat |
|---|---|---|---|---|
| dot1d 1e3 f64 (B1) | 1.61 | 1.58 | 4.70 | 4.55 |
| dot1d 1e5 f64 (B1) | 12.82 | 13.00 | 15.48 | 15.67 |
| dot1d 1e7 f64 (B1) | 2934.3 | 2933.8 | 2964.7 | 2884.8 |
| dot1d 1e7 f32 (B1) | 1273.8 | 1268.9 | 1283.7 | 1258.0 |
| dot1d 1e5 Complex f64 (B1) | 52.4 | 34.8 | 55.8 | 37.6 |
| small (64,64) ax-1 f64 (B1) | 4.87 | 5.00 | 28.73 | 28.03 |
| odd (1000,777) ax-1 f64 (B1) | 124.8 | 131.5 | 84.7 | 81.1 |
| **batched_am1 (4096,512) f64 (B1)** | **443.8** | **464.7** | **148.9** | **141.4** |
| batched_am1 (4096,512) c64 (B1) | 1383.2 | 1248.3 | 288.5–370.8* | 248.0 |
| batched_am1 (8192,256) f64 (B1) | 561.1 | 593.4 | 177.0 | 166.5 |
| **batched_axis0 (512,4096) f64 (B2 RMW)** | **1613.9** | **962.4** | 178.2* | 120.8 |
| **batched_strided (4096,512)·(512,4096).t (B3)** | **3053.8** | **2820.3** | 321.8 | 324.7 |

\* faer16 cells have a repeat-run band of roughly ±8% (axis0 faer16 155–178 µs,
am1 c64 faer16 289–371 µs across three runs); serial cells repeat within
±2.7%. Phase-2 gates on faer16 must use repeated runs.

B1 = branch-1 contiguous-summed (unrolled_binary_reduce, one write per
output); B2 = branch-2 contiguous-remaining (MaybeUninit RMW per element per
contraction step); B3 = branch-3 general (per-element stride/index fold).
Branch determination: `branch_probe.txt` (replicates vecdot.rs:44–68 logic).

Note: T0's README attributed the batched_am1 case to the "contiguous-remaining
branch" — corrected here: batched_am1 hits branch-1; the RMW branch serves the
axis-0 contraction geometry (`batched_axis0`), which is 3.6× slower than am1
on the same data (serial portable).

### Efficiency framing (batched_am1 f64, 33.55 MB inputs, 4.194 MFLOP)

| engine | ms | GB/s (L3-assisted) | GFLOP/s | % of 32-DP-FLOP/cyc peak |
|---|---|---|---|---|
| rstsr serial portable | 0.444 | 75.6 | 9.45 | 5.2% |
| rstsr serial native | 0.465 | 72.1 | 9.02 | 5.0% |
| rstsr faer16 portable | 0.149 | 225 | 28.2 | 15.5% (aggregate) |
| probe A0 (branch-1 body alone, no dispatch) | 0.292–0.350 | 96–115 | 12.0–14.4 | 6.6–7.9% |
| numpy einsum ij,ij->i (1 thread) | 0.349 | 96.1 | 12.0 | 6.6% |
| numpy einsum ij,ij->j (1 thread) | 0.432 | 77.7 | 9.7 | 5.3% |

The 1-D dot (2.93 ms @1e7, 57 GB/s) stays memory-bound as T0 concluded —
the 3.9%-of-peak framing is misleading for it; it is a GATE, not a target.
Batched cases are L3-fed on this box (32 MiB working set vs 128 MiB L3), so
GB/s above the 40 GB/s DRAM triad ceiling is L3 assist, identical for numpy.

## `%` on 1-D (inner_dot) — µs

| n | serial port | serial nat | faer16 port | faer16 nat | numpy einsum (1thr) | np.dot BLAS (16thr) |
|---|---|---|---|---|---|---|
| 64 | 0.34 | 0.36 | 10.39 | 10.62 | ~1 µs | ~0 µs |
| 1e4 | 3.91 | 3.94 | 29.38 | 30.28 | ~3 µs | ~2–3 µs |
| 1e6 | 373.7 | 371.0 | 86.0 | 87.4 | 193 | ~30 µs* |
| 1e7 | 4136.1 | 4067.7 | 1839.1 | 1818.5 | 2791 | 949 |
| 1e5 Complex f64 | 88.5 | 62.7 | 48.6 | 46.7 | — | — |

\* np.dot at 1e6/1e4 not captured in single-thread pass; BLAS-threaded pass
covers 1e3/1e5/1e7 (see numpy_anchors_blas.txt).

Key facts:

- **The same math via rt::vecdot (2934 µs) is 1.4× faster than via `%`
  (4068–4136 µs) on the serial device** — `inner_dot_naive_cpu_serial` folds
  strictly sequentially with `index_uncheck` arithmetic + `alpha.clone()` per
  element (12.25 ins/element, perf-derived).
- **faer16 `%` burns 25.7 ms of core-time per 1e7 iteration (15.4 CPUs) to get
  a 2.5× wall win over serial** — 11.76 ins/elem, IPC 0.84: a per-element
  rayon fold with `index_uncheck`, re-associated by `reduce_with`.
- **faer16 `%` at small n is rayon-dispatch-dominated**: n=64 → 10.4 µs vs
  0.34 µs serial (30×); n=1e4 → 29.4 vs 3.9 µs (7.5×). There is no
  PARALLEL_SWITCH serial fallback in `inner_dot_naive_cpu_rayon` (vecdot has
  one at 512).
- Rounding: serial `%` is strictly sequential (bit-comparable to naive);
  faer16 re-associates. **There is no cross-device bit-exact contract for `%`
  today** — relevant license for the phase-2 fast path (see PLAN §design).

## perf stat -d (native; derived in perf/derived_summary.txt)

| op | ms/iter | GB/s | cyc/elem | ins/elem | IPC | L1d-miss% |
|---|---|---|---|---|---|---|
| batched_am1 serial | 0.468 | 71.7 | 1.23 | 3.68 | 2.99 | 14.3 |
| batched_axis0 serial | 0.810 | 41.4 | 2.15 | 3.17 | 1.48 | 14.9 |
| batched_strided serial | 2.839 | 11.8 | 7.64 | 13.62 | 1.78 | 17.5 |
| dot1d 1e7 serial | 2.884 | 55.5 | 1.64 | 2.64 | 1.61 | 24.2 |
| innerdot 1e7 serial | 4.227 | 37.8 | 2.37 | 12.25 | 5.18 | 8.3 |
| batched_am1 faer16 (core-time) | 2.425 | 13.8* | 6.04 | 10.28 | 1.70 | 5.8 |
| innerdot 1e7 faer16 (core-time) | 25.747 | 6.2* | 14.03 | 11.76 | 0.84 | 10.5 |

\* faer rows: task-clock aggregates all threads, so GB/s here is
core-time-equivalent, not wall bandwidth (wall: batched_am1 ≈ 225 GB/s
aggregate, innerdot ≈ 99 GB/s aggregate).

## Probe anchors (plain Vec<f64>, serial core; probe_portable/probe_native)

| probe | portable | native |
|---|---|---|
| A0 branch-1 body alone (ubr per row) @ (4096,512) | 0.350 ms | 0.292 ms |
| A1 per-row plain fold | 0.741 ms | 0.748 ms |
| A2 per-row 8-lane [f64;8] | 0.431 ms | 0.434 ms |
| A3 two-rows interleaved | 1.036 ms | 0.981 ms |
| A2 @ (8192,256) | 0.437 ms | 0.426 ms |
| A5 c64 plain fold | 1.361 ms | 1.274 ms |
| A6 c64 8-lane | 1.388 ms | 1.450 ms |
| B0 branch-2 RMW replica (bands, no iterator) | 1.758 ms | 1.757 ms |
| B1 full-vacc row walks (32 KiB L1) | 0.515 ms | 0.446 ms |
| B2 full-vacc 8-lane j split | 0.804 ms | 0.799 ms |
| C0 general-branch replica (stride fold) | 2.965 ms | 2.999 ms |
| C1 8-lane stride fold | 2.970 ms | 3.001 ms |
| D0 inner_dot serial replica @1e7 | 3.992 ms | 3.988 ms |
| D1 zip fold @1e7 | 3.976 ms | 3.989 ms |
| D2 8-lane unrolled @1e7 | 2.797 ms | 2.858 ms |
| D0/D1/D2 @4096 | 90/90/180 GB/s | 90/90/180 GB/s |

Readings in PLAN.md §2/§5.

---

# PHASE 2 — candidate results (proposed.patch applied to 386948be)

Raw logs: `bench_vecdot_candidate_{portable,native}.txt`,
`bench_innerdot_candidate_{portable,native}.txt`, repeated trials in
`bench_vecdot_candidate_native_rerun{1,3,4,5,6}.txt` (native) and the
portable trials recorded in `candidate_faer_trials.md`; perf under
`perf_candidate/` (derived in `perf_candidate/derived_summary.txt`).
Correctness: `correctness_candidate_{portable,native}.txt` — 49 checks,
ALL PASS both configs (43 phase-1 checks + 6 new fast-path/fall-through
checks; the serial `%` bit-exact lock became tolerance-based as
pre-declared in PLAN §2-B and G1).

## vecdot — clean tree vs candidate (µs; medians of repeated trials)

| case | serial port base→cand | Δ | serial nat base→cand | Δ | faer16 port base→cand | Δ | faer16 nat base→cand | Δ |
|---|---|---|---|---|---|---|---|---|
| dot1d 1e7 f64 (gate) | 2977→2955 | −0.7% | 2940→2955 | +0.5% | 2983→2920 | −2.1% | 2972→2920 | −1.8% |
| dot1d 1e5 / 1e3 f64 (gates) | 12.8→12.7 / 1.60→1.57 | −1% | 12.8→12.7 / 1.62→1.57 | −1..−3% | 15.7→15.4 / 4.7→4.6 | −2% | 15.4→15.4 / 4.5→4.6 | ~0 |
| dot1d 1e7 f32 (canary) | 2129*→1262 | (lottery) | 1265→1262 | −0.2% | 2166*→1265 | (lottery) | 1248→1265 | +1.3% |
| dot1d 1e5 c64 | 52.1→52.2 | +0.3% | 34.7→34.7 | 0% | 55.6→55.3 | −0.4% | 37.7→37.6 | −0.3% |
| **batched_am1 (4096,512) f64** | **466→363** | **−22%** | **450→352** | **−22%** | **150→140** | **−5%** | **141→140** | −1% |
| batched_am1 c64 | 1585*→1233 | −11..−22%* | 1061/1248→989 | −7..−20% | 305→287 | −6% | 316→242 | −10..−23% |
| batched_am1 (8192,256) f64 | 573→359 | −37% | 555→359 | −35% | 177→155 | −12% | 167→155 | −7% |
| **batched_axis0 (512,4096) ax0** | **1497→538** | **−64%** | **706→538** | **−24%** | **163→131** | **−20%** | 114→118 (band 110–130) | ~0 (band) |
| batched_strided (untouched path) | 3141→3112 | −1% | 2959→3046 | +2.9% | 344→313 | −9% | 301→312 | +1.5–4% (band) |
| odd (1000,777) | 128→103 | −20% | 123→103 | −16% | 86→82 | −5% | 81→82 | +1% |
| small (64,64) | 4.81→2.74 | −43% | 4.73→2.74 | −42% | 28.9→28.3 | −2% | 27.8→28.3 | +2% |

\* tainted baseline cells: the clean-tree "portable" baseline build hit the
codegen lottery on f32 1e7 (+67% vs the phase-1 run-1 build) and
batched-am1-c64 (+14.6%); for those cells the honest reference is the
phase-1 run-1 clean build too (see candidate_faer_trials.md). f64 primary
cells are lottery-stable (±5%).

## `%` (inner_dot) — clean tree vs candidate (µs)

| n | serial port | Δ | serial nat | Δ | faer16 port | Δ | faer16 nat | Δ |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.35→0.34 | −4% | 0.36→0.34 | −6% | 10.43→**0.41** | **−96%** | 10.64→**0.41** | **−96%** |
| 1e4 | 3.94→1.27 | −68% | 3.94→1.27 | −68% | 29.3→**7.19** | −76% | 30.3→**7.19** | −76% |
| 1e6 | 377→135.7 | −64% | 373→135.7 | −64% | 86.5→**23.5** | **−73%** | 87.4→**23.5** | −73% |
| 1e7 | 4143→**2925** | **−29%** | 4075→**2925** | **−28%** | 1851→1825 | −1.4% † | 1845→1825 | −1.1% † |
| 1e5 c64 | 89.0→51.3 | −42% | 63.4→51.3 | −19% | 48.3→16.6 | −66% | 46.7→16.6 | −64% |

† faer16 `%` at 1e7 is DRAM-bound (160 MB working set streams from DRAM:
1825 µs = ~89 GB/s aggregate, vs the 40 GB/s single-core triad), so the
~4× instruction-collapse (11.76 → 2.95 ins/elem core-time, perf) does not
move wall time — the D3 target at this one cell is honestly MISSED while
1e4–1e6 (the latency/overhead-bound range) improve 64–96%.

## perf stat -d, before → after (native; perf/ vs perf_candidate/)

| op | ms/iter | cyc/elem | ins/elem | IPC |
|---|---|---|---|---|
| batched_am1 serial | 0.468→0.359 | 1.23→0.95 | 3.68→2.67 | 2.99→2.81 |
| batched_axis0 serial | 0.810→0.564 | 2.15→1.51 | 3.17→**0.90** | 1.48→0.60 |
| innerdot 1e7 serial | 4.227→3.077 | 2.37→1.75 | **12.25→2.72** | 5.18→1.56 |
| innerdot 1e7 faer16 (core-time) | 25.75→24.68 | 14.03→13.55 | **11.76→2.95** | 0.84→0.22 |
| batched_am1 faer16 (core-time) | 2.425→2.277 | 6.04→5.66 | 10.28→8.67 | 1.70→1.53 |

## D3 verdict

- **Primary accept PASS**: batched_am1 serial −21.7% (native) / −21…−24%
  (portable); batched_axis0 serial −23.8% (native) / −64% (portable) —
  both ≥15% in BOTH configs. innerdot serial 1e7 −28/−29% both configs.
- %-of-peak movement (batched_am1 serial, native): 5.0% → **6.5%** of the
  32-DP-FLOP/cyc core peak (9.0 → 11.8 GFLOP/s; 72 → 95 GB/s L3-assisted),
  now at numpy-einsum parity (0.349 ms single-thread).
- faer16 `%` 1e7 wall: MISSED (−1.3%, DRAM-bound; instruction collapse
  delivered in core-time). faer16 `%` ≤1e6: −73…−96% (PASS by an order of
  magnitude over the 2× target).
- Regression gates: PASS — 1-D f64 gates within ±2%; faer16 batched am1
  improved (−1…−5%); strided (untouched path) within build bands
  (+1.5–4% native faer16, see candidate_faer_trials.md); c64 settled at
  baseline (34.7 = 34.7) or improved; small/odd cells improved 16–43%.
