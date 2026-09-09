# T2' — reductions (value reductions: sum/min/max/mean/var/norm + sum_axes)

Campaign task T2' (plan T2 narrowed to **value** reductions) in
`rstsr-native-impl/src/cpu_serial/reduction.rs` + the min/max wiring closures
in `rstsr-core/src/device_cpu_serial/reduction.rs` and
`rstsr-core/src/feature_rayon/auto_impl/reduction.rs` (= the `device_faer`
symlink). argmin/argmax are NOT touched (T6 argmax patch; composeability
proven below). Phase 1 = baseline + anatomy + design ([PLAN.md](PLAN.md));
phase 2 = the implementation described there (options A + A2).

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5`.
  Phase 1 made no edits; phase 2 edited 3 files, captured in
  [proposed.patch](proposed.patch) (286 lines), then `git checkout -- .` —
  **verified `git -C ../rstsr status --short` empty, HEAD `386948b` at the
  end.** The patch `git apply --check`s cleanly against a fresh `386948be`.
- Harness copied from the T0 pattern; measurement contract per plan §3.

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores, AVX-512; 128 MiB L3 (X3D) |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)` (nightly, via rstsr's rust-toolchain.toml) |
| criterion | 0.5.1 (2 s + 0.7 s warm-up) |
| perf | 7.0.14; annotated profiles used `-Cdebuginfo=2` builds |
| RAYON_NUM_THREADS | 16, asserted by the harness |
| Configs (D7) | `portable` (no RUSTFLAGS) / `native` (`-C target-cpu=native`) |
| Cargo profile | stock release defaults (explicit in Cargo.toml) |

## What was changed (phase 2)

Diff: [proposed.patch](proposed.patch) — no signature changes, no new
features, no public-API breaks beyond one bound widening.

1. **A2 — strict-compare min/max wiring closures**
   (`OpMinAPI`/`OpMaxAPI` × both device wiring files; 8 closure bodies +
   4 impl bound widenings `T: ExtReal` → `T: ExtReal + PartialOrd`):
   `acc.ext_min(x)` → `if x < acc { x } else { acc }` (max symmetric).
   Value-identical for every input on the ±`ext_*_value` seeds: NaN
   candidates never pass the strict comparison (same NaN-skipping as
   `f64::min`), all-NaN keeps the seed. For integers the form IS
   `Ord::min/max`. The old `minnum`-based shape defeated auto-vectorization
   of `unrolled_reduce` under `native` (see mechanism below).
   - **±0.0 sign-bit nuance (G1 item 1)**: on ±0.0 ties the strict-compare
     fold keeps the incumbent (earlier-seen zero); the old `f64::min` form
     returned the LATER operand on ties (measured pre-patch,
     [results/signbit_pre_patch.txt](results/signbit_pre_patch.txt):
     `min_all([-0,+0])` was `+0`, now `-0`; `max_all([+0,-0])` was `-0`,
     now `+0`). The values are equal (`== 0.0`); only the sign bit of a
     zero result can differ — same class of note as T4''s visit-order note.
     Locked by new gate spots (see correctness below).
2. **A — iterator-free band walk in `reduce_axes_cpu_serial`'s
   `size_mc > 1` branch (serial only)**: the summed-axes offsets are
   materialized once per call into a scratch `Vec<usize>` and walked as a
   plain slice, replacing one `IterLayoutColMajor::next` per summed-axes
   position PER 48-element chunk band (88k `next()` calls per op at
   2048² axis0). Visit order and accumulation order are unchanged, so FP
   results are bit-identical (including the broadcast-duplication
   finalize, see the bug note below). The walk lives in a dedicated
   `#[inline(never)]` helper (`reduce_axes_band_walk_cpu_serial`) so the
   rest of the function compiles as before (argmax-precedent code-layout
   hygiene); the original iterator walk is kept verbatim as a second
   out-of-line helper behind two guards:
   - **cap**: summed space > 128k positions (1 MiB of offsets) — the
     offsets buffer could outsize the streamed input there;
   - **floor**: summed×remaining work < 16384 elements — the one-off
     offsets allocation is not worth it below that (small tensors stay on
     the old path).
   **Coverage note (G1 item 2)**: the *cap* fallback branch is NOT
   exercised by the bench matrix (no bench has a summed space > 128k
   positions) — it has no benchmark-derived coverage; correctness of that
   path rests on it being the byte-identical pre-rewrite code. The *floor*
   fallback IS exercised (small_64x64 benches take it).

Not changed: the rayon twin `reduce_axes_cpu_rayon` (an equivalent rewrite
was tried and **reverted** — see deviations), `unrolled_reduce`,
`reduce_all_*`, arg* kernels, sum/mean/var/norm/prod closures.

## Results (D3 verdict: PASS)

Full per-cell table (two full-suite runs per config, criterion `change` vs
the saved phase-1 baselines): [results/tables_candidate.md](results/tables_candidate.md).
Headline (f64, 2048² unless noted, criterion mid-estimates; r1/r2 are the
two runs):

| case | baseline | candidate r1/r2 | change |
|---|---|---|---|
| sum_axis0 serial portable | 1.435 ms | 1.205 / 1.195 ms | **−15.8% / −16.6%** |
| sum_axis0 serial native | 1.994 ms | 1.365 / 1.228 ms | **−31.6% / −38.4%** |
| min_axis0 serial portable | 2.935 ms | 1.195 / 1.196 ms | **−59.1%** |
| min_axis0 serial native | 2.244 ms | 1.479 / 1.483 ms | **−34.0%** |
| min_all 1e7 serial portable | 2.148 ms | 1.236 / 1.258 ms | **−41.8%** |
| min_all 1e7 serial native | 10.421 ms | 1.221 / 1.227 ms | **−88.2%** |
| max_all 1e7 serial native | 10.423 ms | 1.267 / 1.269 ms | **−87.8%** |
| min/max_all 1e7 faer16 native | 0.368/0.372 ms | 0.151 / 0.152 ms | **−59%** |
| sum_axis0 faer16 portable | 0.4025 ms | 0.2756 ms (A/B) | **−31% back-to-back** |
| medium 512² sum_axis0 serial portable | 51.9 µs | 40.3 µs | **−22.2%** |
| sum_axis0 f32 serial portable (large) | 0.434 ms | 0.292/0.329 ms | **−29%** |

- **D3 (≥10% on large in BOTH configs) holds** for the primary op with
  margin; native is no longer slower than portable for sum_axis0 (1.228 vs
  1.195 ms — parity) — the T0-flagged native regression is gone.
- **Stretch target NOT met, honestly**: PLAN §4 aimed for ≤1.0 ms serial
  sum_axis0 in both configs; the candidate lands at 1.20 (portable) /
  1.23–1.36 (native) ms. The local probe without any driver overhead
  (0.57/0.82 ms) is not reachable through the full generic driver: the
  residual is the per-group `vacc` alloc+finalize pass, the
  `it_md`/`it_od` outer iterators, and generic-closure instantiation
  overhead spread over 87k slab-adds. D3 (the campaign's binding
  threshold, plan §2) is the criterion this task is accepted on.
- **Gates (no >2–3% regression)**: PASS, with an important measurement
  caveat. Single full-suite runs showed apparent regressions on untouched
  paths (portable sum_axis1 +5.4%, sum_all +3.2%, faer16 sum_axis0 +10%) —
  but back-to-back A/B (patched vs `git stash`ed original, same minutes)
  shows the ORIGINAL code measures SLOWER on every one of those cells
  (sum_axis1: 486.7 µs original vs 483.1 µs patched; sum_all: 475.2 vs
  459.6; faer16 sum_axis0: 360.1 vs 275.6 µs; faer16 medium min_axis0:
  60.2 vs 55.6 µs). The morning-saved baselines simply caught the machine
  fast; these ~0.5 ms streaming cells carry a ±5% build-layout/session
  lottery (T4' precedent). Every gate cell is ≤ original when measured
  simultaneously. Small/odd cells: small sum_axis0 −0.9/−3.4%, odd axis0
  −13/−19%, odd/medium faer16 cells within noise or improved.
  Committed raw evidence (G2 follow-up): an interleaved A-B-A re-run of
  the most load-bearing cell — faer16 sum_axis0 2048², patched vs
  original, all four runs tee'd under
  [results/ab_logs/](results/ab_logs/README.md) — shows ORIGINAL {294.6,
  302.5 µs} vs PATCHED {337.0, 273.2 µs}: a tie within a ±11%
  within-state swing, with ALL runs 25–33% below the stale saved baseline.
  This confirms the cell cannot resolve layout-level differences and that
  saved-baseline `change`% for it is session drift, not code. The three
  other pairs quoted above were interactive-session measurements without
  tee'd logs (commands identical, substitute the filter:
  `cargo bench --bench reduce -- "<cell filter>" --baseline portable` with
  `git -C ../rstsr apply|checkout` between runs); they are superseded as
  committed evidence by the ab_logs A-B-A.

## Mechanism (G1 item 4 — careful phrasing)

What is *measured* (load-bearing):

- `perf stat -d` before/after (serial, results/perf/): sum_axis0 native
  1.828 → 1.232 ms (−33%), ins/elem 2.56 → 1.38; portable 1.383 → 1.287 ms.
- `perf record` (debuginfo builds) self-time attribution, phase-1 native
  binary: fold-body self-time 0.652 s portable / 0.357 s native, while
  `IterLayoutColMajor::next` self-time was 0.167 s portable / 0.731 s
  native of total wall — i.e. ~20% vs ~66% of the op's time was inside the
  layout-iterator `next()` at 88 064 calls per op (43 chunk bands × 2048
  summed-axes rows).
- Probe deltas (examples/probe_manual.rs, plain Vec, no driver): the band
  shape with explicit offsets and no iterator runs 0.570 ms native /
  0.824 ms portable vs rstsr's 1.99/1.43 ms — i.e. removing the iterator
  walk accounts for ~1.3 ms of the native gap and ~0.6 ms of the portable
  one.
- A2: the strict-compare min shape vectorizes (probe: 8-lane strict-compare
  0.443 ms native vs 3.16 ms for the `f64::min` shape; plain `f64::min`
  scalar fold 0.49 ms — the intrinsic shape blocks vectorization only in
  the unrolled form). After the patch, native min_all 1e7 runs at 0.43
  ins/elem (perf) — consistent with vminpd.

What is *interpretive* (stated as such): WHY the outlined
`IterLayoutColMajor::next` costs ~18 cycles/call in one build and ~79 in
another at identical machine code (jump-table dispatch on ndim, scalar
index update in both binaries — objdump side-by-side). Candidate
explanations are microarchitectural (call/return + indirect-branch
behavior at very high call frequency, latency-bound at IPC ≈ 1); the
machine cannot be probed further without cycle-level counters this study
does not have. Similarly, the residual ±5% cell wobble is attributed to
build code-layout lottery because (a) the flagged cells execute no
changed instructions at all and (b) same-path sibling cells move in the
opposite direction — this is an inference from evidence, not a
counter-measured fact.

## Correctness gate

`examples/correctness.rs` — **142 checks, ALL PASS under BOTH configs and
BOTH devices** (results/correctness_{portable,native}.txt), including all
124 phase-1 checks plus (G1 items 1+3):
- **signbit locks** (new behavior): min/max on ±0.0 ties keep the
  incumbent zero (value-equal; sign may differ from the pre-patch minnum
  behavior — documented above);
- **i32 min/max spots** (min_axes/max_axes/min_all/max_all vs naive
  Ord::min/max folds on 64×97) — required by G1, passes exactly;
- all prior NaN locks (sum/mean/var/norm poison; min/max skip NaN;
  all-NaN → ±MAX seed), broadcast/t-view/3-D/odd shapes, f32 spots.
- rstsr's own suite on the patched tree: `cargo test -p rstsr-core` —
  **585 tests, 0 failures** (entry_row_cpu 290, lib 110, tensor_sum 2,
  remaining 183).

## Upstream bug found (for the human — NOT fixed here)

Reducing over a stride-0 **summed** axis multiplies by the wrong count at
386948be: `size_s0` (cpu_serial/reduction.rs:226) indexes the REMAINING
layout's shape with the SUMMED layout's broadcast-axis indices.
`sum_axes(0)` of a `[1,n]→[m,n]` `broadcast_to` view yields **n×v** where
numpy yields m×v; `mean_axes` yields n/m×v; min/max are insensitive
(idempotent). The phase-2 rewrite preserves this behavior bit-for-bit (the
gate locks it as CURRENT-BEHAVIOR); fixing it is a one-line but
semantics-changing decision that should be its own PR.

## Composeability with the argmax patch

The T6 argmax patch and this patch share four files but disjoint
functions/hunks. Verified mechanically: each patch `git apply --check`s on
a fresh 386948be, **and each applies on top of the other** (both orders
tested on the real tree, then restored). No signature interactions (T2'
changes no signatures; A2 only widens two impl bounds the argmax patch
never touches).

## Deviations from the brief / from PLAN.md

1. **The rayon twin rewrite was REVERTED.** The first candidate rewrote
   `reduce_axes_cpu_rayon`'s band walk the same way; single runs showed
   faer16 large sum_axis0 at +10–40%. After the rayon revert, back-to-back
   A/B shows the patched tree FASTER than the original on that cell
   (275.6 vs 360.1 µs) — the regression readings were build-layout/session
   lottery, but with the revert the rayon-side risk is zero and the faer16
   min/max wins (−59%) flow entirely from A2. PLAN §2-A's "same restructure
   in the rayon twin" is therefore withdrawn; a future task may revisit it
   with back-to-back A/B methodology from the start.
2. The ≤1.0 ms stretch target for sum_axis0 was not reached (see Results);
   acceptance rests on D3.
3. G1 review items folded in: ±0.0 sign-bit documentation + gate spots
   (item 1); cap-fallback coverage caveat (item 2); i32 min/max gate spots
   (item 3); mechanism phrasing separated into measured vs interpretive
   (item 4).
4. Tools: `--save-baseline` and `--baseline` are mutually exclusive in
   criterion 0.5, so candidate runs compare via `--baseline` only and the
   two-runs-per-config protocol quantifies build lottery;
   `results/make_compare.py` regenerates the before/after table.

## Reproduce

```bash
cd 2026-09-09-reductions
./reproduce.sh correctness  # 142-check gate, both configs
./reproduce.sh candidate    # phase-2 flow: gates + 2x bench/config + compare + perf
./reproduce.sh portable     # phase-1-style full portable suite (saves baseline)
./reproduce.sh native       # phase-1-style full native suite (saves baseline)
./reproduce.sh perf         # perf stat -d all profile ops, both configs
./reproduce.sh probe        # design-anchor probes (also demonstrate the A2 mechanism)
./reproduce.sh layout       # layout-branch probe
./reproduce.sh numpy        # numpy anchors
```

Note for re-review: apply [proposed.patch](proposed.patch) to a clean
`386948be` before `candidate`; `git -C ../rstsr checkout -- .` afterwards.
Criterion state per config lives in `results/criterion_{portable,native}/`
(`CRITERION_HOME`); the saved `portable`/`native` baselines there are the
phase-1 references the candidate numbers compare against.
