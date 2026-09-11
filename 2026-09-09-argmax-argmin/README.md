# T6 — argmax/argmin (phase 1 baseline + phase 2 candidate kernel)

Campaign task T6. **Phase 1** (no patch) measured the clean-tree baseline,
locked down current arg* semantics with a correctness gate, and proposed the
phase-2 design in [PLAN.md](PLAN.md) (G1-approved). **Phase 2** implemented the
contiguous 8-lane fast path in `../rstsr` (working tree only), re-ran the
identical gate + suite as the `candidate` baseline, and produced
[proposed.patch](proposed.patch). The rstsr tree was reset afterwards:
`git -C ../../rstsr status --short` verified empty and HEAD `386948be`.

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5` (path dep
  `../../rstsr/rstsr-core`; phase 2 edits were made in the working tree, diffed
  to `proposed.patch`, then reverted).
- Campaign plan: [../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
  (§3 measurement contract, §5 dispatch_simd constraints, §9 pitfalls).
- Crate pattern copied from T0 (`../2026-09-09-bench-harness-baseline/`).

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores / 32 threads, AVX-512; 48 kB L1d / 1 MB L2 per core, 128 MiB total L3 (X3D) |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)` (nightly, matches rstsr's pinned channel) |
| criterion | 0.5.1 |
| ndarray / ndarray-stats | 0.16.1 / 0.6.0 (argmax/argmin anchors; ndarray 0.16 has no built-in argmax) |
| perf | 7.0.14 |
| Parallelism | `RAYON_NUM_THREADS=16`, asserted by the harness at every bench/example start |
| Target configs (D7) | `portable` = no RUSTFLAGS (x86-64 SSE2 baseline); `native` = `-C target-cpu=native` |
| Cargo profile | stock release defaults, stated explicitly in Cargo.toml (mirrors T0) |

## Directory layout

```
Cargo.toml              standalone crate (no workspace); rstsr-core path dep with
                        default features mirrored explicitly (T0 pattern)
src/lib.rs              harness: devices, thread assert, criterion config,
                        deterministic fixtures, size classes
benches/arg.rs          rstsr argmax/argmin: 1-D {64, 1000, 1e6, 1e7} f64,
                        f32 1e7 spot, 2-D whole-tensor {64², 512², 2048²},
                        strided = transpose view {64², 2048²}; serial + faer16
benches/anchors_ndarray.rs  ndarray-stats anchors, same sizes
examples/correctness.rs semantics gate vs naive scalar reference + ndarray
                        cross-check (ties / NaN / empty / 2-D / strided /
                        broadcast / axis reductions / f32+f64 / both devices)
                        — phase-2 regression gate incl. NaN-lane + NaN-chunk
                        fixtures
examples/profile_ops.rs fixed-iteration runner for `perf stat`
examples/probe_nan_corner.rs  sweep for NaN-at-fold-split instability
                        (demonstrates the pre-patch rayon corner; 0 hits post-patch)
results/                phase-1 baseline (criterion portable/native + perf),
                        candidate/ (phase-2 runs), perf/perf_*_candidate.txt,
                        paired-A/B t-view transcripts, probe transcripts,
                        tables.md (regenerable via make_tables.py)
numpy_ref/              optional numpy context anchor (conda `torch` env);
                        argmax 1e7 = 1.34 ms, 1e6 = 0.084 ms (L3-assisted context)
PLAN.md                 the PHASE 2 design (kernel patch); G1-approved
proposed.patch          the phase-2 diff against 386948be (D3 met)
reproduce.sh            self-contained rerun: correctness → portable → native
                        → perf; `candidate` stage for patched-tree reruns
```

## Current semantics (established empirically + from source, locked by the gate)

All verified by `examples/correctness.rs` on **both** devices and **both**
RUSTFLAGS configs (`results/correctness_{portable,native}.txt`):

- **Call chain** (serial): `rt::argmax(&t)` → `argmax_all_f`
  (`rstsr-core/src/tensor/reduction.rs:197`, macro-generated) →
  `DeviceCpuSerial::argmax_all` (`rstsr-core/src/device_cpu_serial/reduction.rs:405`)
  → `reduce_all_arg_cpu_serial` (`rstsr-native-impl/src/cpu_serial/reduction.rs`)
  → `reduce_all_unraveled_arg_cpu_serial`. Rayon: `DeviceFaer` impls call
  `reduce_all_arg_cpu_rayon` → `reduce_all_unraveled_arg_cpu_rayon`
  (`PARALLEL_SWITCH=1024` serial fallback).
- **Symlink finding (corrects the T0 code map)**: `device_faer/rayon_auto_impl/*`
  are **symlinks** to `feature_rayon/auto_impl/*` — the "dead duplicate" and
  the live DeviceFaer impls are THE SAME FILE. The directory
  `feature_rayon/auto_impl/` is the physical storage of the live
  `device_faer/rayon_auto_impl/` path; `git diff` therefore shows the wiring
  changes under the `feature_rayon/` path. There is no separate dead copy to
  preserve.
- **Tie-breaking**: FIRST occurrence wins — the lowest row-major flat index
  among equal extremes. Mechanism: strict comparison accepts only `y > x`
  (`y < x` for argmin); on equality `f_eq` keeps the smaller index
  (serial fold visits in ascending row-major order; the rayon fold/reduce
  combine (`cpu_rayon/reduction.rs:493`) explicitly resolves ties by the
  smaller global index). Verified with all-equal and duplicated-extreme
  fixtures, including n=4099 > `PARALLEL_SWITCH` on the faer device.
- **NaN handling**: the first row-major element seeds the accumulator
  unconditionally; afterwards a NaN never replaces it (both `>` and `==` are
  false with NaN). Consequences — which coincide with numpy's documented
  argmax/argmin NaN behavior — all verified:
  - NaN at the **end/middle** of real data: the real extreme wins;
  - NaN at the **front**: poisons the accumulator → returns flat index 0;
  - **all-NaN** input: returns flat index 0 (no error, no NaN propagation).
- **Empty input**: `argmax_f`/`argmin_f` return
  `Err(InvalidLayout)` — "empty sequence is not allowed for reduce_arg."
  (raised at `cpu_serial/reduction.rs:439` serial,
  `cpu_rayon/reduction.rs:455` rayon); the infallible `rt::argmax` **panics**
  via `rstsr_unwrap` with that message. Verified with `catch_unwind`.
- **Flat index contract**: whole-tensor arg* on any-rank input returns the
  flat **row-major** index (raveled with a RowMajor pseudo-layout
  regardless of input layout / device default order), and 2-D whole argmax ==
  1-D argmax of the raveled tensor (verified explicitly).
- **Strided views** are supported by the same generic path
  (`IndexedIterLayout`, RowMajor) — used as the bench's strided case.
- **Parallel NaN corner (behavior intentionally refined by phase 2)** — see
  the phase-2 section below for the full story. The pre-patch rayon fold
  seeded each thread-split independently and combined partials pairwise, so a
  NaN at a fold-split boundary could suppress the true global max depending
  on pairing order (demonstrated on the clean tree by
  [results/probe_nan_clean_tree.txt](results/probe_nan_clean_tree.txt): e.g.
  n=1025, NaN@896, real max@1024 → clean tree answers 294). The patched
  kernel is deterministic and matches serial exactly: NaN anywhere except
  flat index 0 never wins. Locked by a gate fixture (NaN at
  chunk-start-like offsets, faer, n=10_003 > PARALLEL_SWITCH) and
  [examples/probe_nan_corner.rs](examples/probe_nan_corner.rs)
  ([results/probe_nan_patched_tree.txt](results/probe_nan_patched_tree.txt)):
  with `proposed.patch` applied and the crate freshly recompiled, the probe
  completes with `probe done, 0 unstable cases found` (the pre-fix draft
  kernel DID fail this probe — see the phase-2 section; the clean tree fails
  it too, under the probe's per-run `INSTABILITY:` label).

## Phase 2 — candidate kernel (G1-approved design, implemented)

What changed in `../rstsr` (full diff: [proposed.patch](proposed.patch),
4 files, +393/−296 lines):

1. `rstsr-native-impl/src/cpu_serial/reduction.rs`: new `ArgCmp {Min, Max}`
   enum replacing the `(Fcomp, Feq)` closure pair (all in-tree callers were
   exactly min-or-max); new `arg_contig_cpu_serial` (8-lane unrolled scan) +
   `arg_contig_seeded_cpu_serial` (seeded core, `usize::MAX` sentinel) +
   `unravel_c_order` helper; `reduce_all_unraveled_arg_cpu_serial` gains a
   `la.c_contig()` fast path and keeps the original closure fold as the
   strided/broadcast fallback (fast path held out-of-line via `#[inline(never)]`
   so the fallback's codegen is unchanged).
2. `rstsr-native-impl/src/cpu_rayon/reduction.rs`: rayon twin — contiguous
   chunks scanned per-thread with the serial 8-lane kernel, partials
   **collected in order and combined sequentially** (greater value wins, ties
   keep the smaller index, NaN never wins) — deterministic, scheduling- and
   thread-count-independent, exactly serial-equivalent. `PARALLEL_SWITCH`
   fallback and strided fold preserved.
3. Wiring (`device_cpu_serial/reduction.rs` + the symlinked
   `feature_rayon/auto_impl/reduction.rs` = `device_faer/rayon_auto_impl/`):
   8 impl functions now pass `ArgCmp::Min/Max` instead of building closures.

**API-break note (G1 review item 1)**: the `reduce_*_arg_*` functions are
`pub` in the published `rstsr-native-impl` crate (`cpu_serial::reduction` and
`cpu_rayon::reduction` are public modules) — swapping `(Fcomp, Feq)` for
`ArgCmp` **is a breaking signature change** of that crate. The tensor-level
API in `rstsr-core` (`rt::argmax`, `Tensor::argmin_axes`, traits, etc.) is
unchanged; downstream breakage is limited to direct users of the
rstsr-native-impl kernels (in-tree callers: the two device wiring files only).
Flag this in the PR description when integrating.

**A real bug the probe caught mid-phase-2** (kept here as a lesson): the
first draft seeded the 8 lanes from `xs[0..8]`; a NaN at positions 1–8 then
blocked its whole lane (`x > NaN` is false), so a true max landing later in
the same lane was silently lost (found by `examples/probe_nan_corner.rs`:
n=8841, NaN@8696 ≡ chunk 63's lane 2, max@8840 → wrong answer on faer).
Fix: the poisoning first element is detected generically (`xs[0] == xs[0]`
is false iff NaN; non-float types are reflexive) and, only after that check,
all lanes are seeded with the guaranteed-comparable first element. The gate
now carries the exact regression fixture (NaN at k ∈ 1..8 with the max at
k+8, same lane).

**Correctness**: gate ALL PASSED on the patched tree under both RUSTFLAGS
configs, both devices (`results/correctness_{portable,native}.txt` — now
including the NaN-at-chunk-start fixture, the NaN-lane fixtures, broadcast,
axis reductions); rstsr's own suites pass (`cargo test -p rstsr-core --lib`:
110 passed; `entry_row_cpu`: 290 passed); `cargo check` clean for
`rstsr-native-impl --no-default-features` (no_std), `rstsr-core` defaults,
and `rstsr-core` no-default feature combos.

### Candidate vs baseline (criterion medians; x = speedup; full tables in results/tables.md)

1-D f64:

| case | serial portable | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|
| n=64 argmax | 69.8 ns (11.1x) | 67.4 ns (11.6x) | 84.4 ns (9.3x) | 90.2 ns (8.8x) |
| n=1000 argmax | 249.8 ns (38.1x) | 200.0 ns (47.7x) | 256.3 ns (36.4x) | 206.5 ns (46.0x) |
| 1e6 argmax | 164.0 µs (56.7x) | 132.6 µs (70.5x) | 24.4 µs (55.1x) | 21.3 µs (65.3x) |
| **1e7 argmax** | **1.82 ms (50.8x)** | **1.55 ms (60.3x)** | **308.5 µs (38.6x)** | **250.6 µs (49.6x)** |
| 1e7 argmin | 1.82 ms (52.2x) | 1.55 ms (60.7x) | 281.0 µs (43.0x) | 251.3 µs (50.4x) |
| 1e7 argmax f32 | 2.09 ms (44.7x) | 1.46 ms (64.1x) | 205.4 µs (57.9x) | 137.1 µs (90.7x) |

2-D whole-tensor f64 (flat row-major index): 64² 51–68x serial / 5.3–5.5x
faer16 (the ~10 µs parallel-dispatch floor dominates); 512² 58–73x / ~32x;
2048² 50–61x / 65–79x. GB/s at 2048²: 0.8 → ~40–48 serial.

perf stat -d (argmax 1e7 f64 serial native, 150 iters):

| | ms/iter | GB/s | ins/elem | cyc/elem | IPC | L1d-miss% |
|---|---|---|---|---|---|---|
| baseline | 93.77 | 0.85 | 280.3 | 51.5 | 5.45 | 0.1 |
| **candidate** | **1.63** | **48.98** | **3.7** | **0.9** | **4.00** | **12.1** |

72x fewer instructions per element; the kernel is now memory-exposed
(12.1% L1d-miss, IPC 4.0, 47–55 GB/s depending on wall/task-clock) — reading
80 MB per iteration at 47–55 GB/s is L3-assisted on this 128 MiB X3D part
(the honest DRAM ceiling is T0's triad ≈ 40 GB/s; numpy's 1.34 ms context
number is the same L3 effect). The op went from the worst GB/s in the
campaign survey (1/47 of memcpy) to at/above the streaming ceiling.

### Regression gates (D3: no >2–3% on small/strided)

- Small 1-D (n=64, n=1000) and 2-D whole (64², 512²): 5.3x–73x FASTER — no
  regression possible; the small faer16 2-D case sits at the ~10 µs rayon
  dispatch floor (was 56 µs).
- Strided t-view (untouched fallback path), measured two ways:
  - Suite-vs-phase-1-baseline (different sessions): −5.8% to +3.6%, with the
    same binary drifting up to ±2.4% between sessions on the 42 ms strided op.
  - **Paired same-session A/B** (`results/tview_paired_{clean,candidate}_portable.txt`,
    2 interleaved runs each): large serial +1.0–1.5% portable (native
    +0.2–1.4%), small serial +0.5–1.7% portable / −0.1–0.5% native, faer16
    −1.1% to +0.8% (large faer16 portable is 5.5% FASTER).
  - Verdict: within the ≤2–3% gate; the isolated +3.6% suite sample
    (argmin 2048² t-view portable) is cross-session drift, not a codegen
    regression — the fallback code is byte-identical logic and the paired A/B
    bounds the real effect at ≈ +1–1.5%. The `#[inline(never)]` out-of-line
    fast path was added precisely to keep it this way (an early draft that
    inlined both paths showed +2.3–3.8% under native).
- **Reading note for `results/candidate/*.txt`**: any criterion `change:`
  lines inside those files compare against whatever criterion state existed
  in `target/criterion` at that moment (in-session earlier runs, sometimes
  across configs) — they are NOT a baseline-vs-candidate comparison. The
  authoritative gate comparison is the README/tables.md median table plus the
  paired same-session A/B transcripts above.

### Follow-ups (not in this patch)

- `arg_contig_seeded_cpu_serial` could be `pub(crate)` instead of `pub`
  (only `arg_contig_cpu_serial` needs crate-wide visibility; the seeded
  variant is an implementation detail of the serial + rayon fast paths).

### D3 verdict: PASS

Memory-bound class: ≥10% on large inputs in BOTH configs (measured 3800–9400%
on the 1e7/2048² classes, argmax AND argmin, portable AND native, serial AND
faer16) with no >2–3% regression on the gate cases (paired A/B: worst +1.7%
portable small-strided; large-strided +1.0–1.9%). `proposed.patch` is
therefore included.

## Baseline (clean 386948be, criterion medians; full tables in results/tables.md)

### 1-D f64 (argmax / argmin)

| case | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| n=64 argmax | 773.9 ns | — | 781.5 ns | — | 782.2 ns | — | 793.3 ns | — |
| n=1000 argmax | 9.51 µs | 0.8 | 9.55 µs | 0.8 | 9.33 µs | 0.9 | 9.49 µs | 0.8 |
| n=1e6 argmax | 9.30 ms | 0.9 | 9.34 ms | 0.9 | 1.34 ms | 6.0 | 1.39 ms | 5.8 |
| **n=1e7 argmax** | **92.62 ms** | **0.9** | **93.68 ms** | **0.9** | **11.89 ms** | **6.7** | **12.43 ms** | **6.4** |
| n=1e7 argmin | 95.11 ms | 0.8 | 94.01 ms | 0.9 | 12.08 ms | 6.6 | 12.66 ms | 6.3 |
| n=1e7 argmax f32 | 93.41 ms | 0.4 | 93.82 ms | 0.4 | 11.90 ms | 3.4 | 12.43 ms | 3.2 |

Anchors (same fixtures, single-threaded): ndarray-stats argmax 1e7 = **7.11 ms**
(13.0x faster than rstsr serial; ~1.7x faster than faer16), argmin 7.15 ms,
f32 spot 6.1–6.2 ms, 64-elem 45 ns vs rstsr 774 ns (**17x at L1 scale** —
pure per-element overhead, no memory excuse), 2-D 2048² whole 3.77 ms.
numpy context (`numpy_ref/results.txt`, L3-assisted): argmax 1e7 = 1.34 ms
(59.9 GB/s), argmin 1.25 ms, f32 0.67 ms, 1e6 = 0.084 ms (95 GB/s, fully
L3-resident).

### 2-D whole-tensor f64 (flat row-major index)

| case | serial portable | GB/s | serial native | faer16 portable | faer16 native |
|---|---|---|---|---|---|
| 64x64 argmax | 39.97 µs | 0.8 | 40.28 µs | 56.75 µs | 56.52 µs |
| 512x512 argmax | 2.52 ms | 0.8 | 2.55 ms | 446.4 µs | 445.4 µs |
| 2048x2048 argmax | 40.28 ms | 0.8 | 40.83 ms | 5.14 ms | 5.23 ms |
| 2048x2048 argmin | 41.25 ms | 0.8 | 40.64 ms | 5.21 ms | 5.24 ms |
| 2048x2048 t-view argmax (strided) | 41.80 ms | — | 42.62 ms | 5.65 ms | 5.78 ms |

### perf stat -d (argmax 1e7 serial native, 150 iters; results/perf/)

| op | ms/iter | GB/s | ins/elem | cyc/elem | IPC | L1d-miss% |
|---|---|---|---|---|---|---|
| argmax | 93.77 | 0.85 | **280.3** | 51.5 | **5.45** | 0.1 |
| argmin | 95.78 | 0.84 | 282.2 | 52.6 | 5.36 | 0.1 |
| argmax 1e6 | 9.31 | 0.86 | 278.7 | 51.1 | 5.46 | 0.1 |

**Confirms T0's instruction-flood picture exactly** (T0 measured 283 ins/elem,
53.6 cyc/elem, IPC 5.29): the kernel executes ~280 instructions per element at
IPC 5.4 with a 0.1% L1d-miss rate — the machine is never waiting on memory.
The pathology is per-element closure calls (`f_comp`/`f_eq`), an
`Option<(D, T)>` accumulator cloned through both closures every element
(`acc.as_ref().map(|(_, val)| val.clone())`), a `D`-tuple index yielded per
element by `IndexedIterLayout` (a `Vec` clone per element for `IxD`), and no
contiguous fast path / no unrolling. Config-invariant: portable == native
within noise (nothing to autovectorize).

### Efficiency framing (plan §3.8)

- T0's honest single-core DRAM streaming ceiling on this machine: **~40 GB/s**
  (triad 39.6 / memcpy 42.9). argmax 1e7 serial runs at **0.85–0.9 GB/s = 1/47
  of the memcpy ceiling** — the worst GB/s of any op in the campaign survey.
- Large-class caveat (§3.6): 1-D 1e7 f64 = 80 MB, 2048² = 32 MB — both fit in
  this chip's 128 MiB X3D L3, so a *good* kernel here will be partially
  L3-fed and may legitimately exceed 40 GB/s (numpy's 1.32 ms = 60 GB/s is
  exactly that). The L3 effect applies equally to rstsr and anchors.
- A memory-bound 8-lane-unrolled scalar kernel should land in the
  1.3–2.5 ms class at 1e7 serial (L3-assisted; ~2.0 ms at the 40 GB/s DRAM
  floor), i.e. the realistic target is **5–10x at 1e7, 30–60x at L1-scale
  sizes** (where per-element overhead dominates). faer16 already extracts
  ~6.7 GB/s with the SAME per-element kernel (thread-level parallelism only);
  fixing the serial kernel should lift faer16 roughly proportionally
  (fold/reduce over 8-lane chunks instead of per-element closures).

## What was benched (measurement contract)

1. Release mode via `cargo bench` (bench profile = stock release), portable and
   native configs from separate cargo builds via explicit RUSTFLAGS.
2. Correctness gate ran and passed under BOTH configs BEFORE any benchmarking
   (`results/correctness_{portable,native}.txt`).
3. Anti-cheat: inputs and outputs through `std::hint::black_box`; fixtures from
   a deterministic integer hash (no const-foldable data). arg* returns a scalar
   index — no output allocation anywhere (allocation policy: none on either
   side, baseline and phase-2 candidate alike).
4. Both devices: `DeviceCpuSerial` explicit, `DeviceFaer::new(0)` (= default
   device under `faer_as_default`), pool asserted = 16 threads.
5. Sizes: `small` (64) and `small_odd` (1000) are the D3 regression gates
   (per-element overhead + around the `PARALLEL_SWITCH=1024` boundary);
   `medium` (1e6) and `large` (1e7) are the headline sizes; 2-D whole-tensor
   needs no reshape — the API takes any-rank input and returns the flat
   row-major index; strided case = `a.t()` transpose view (documented above).
6. Statistics: criterion defaults, 100 samples, 2 s + 0.7 s warm-up per bench.

## Deviations / decisions

- **numpy anchors run as context only** (`reproduce.sh numpy`,
  `numpy_ref/results.txt`): argmax 1e7 = 1.34 ms reproduces T0's 1.32 ms
  standing number; formal reference columns remain ndarray-stats.
- **2-D whole-tensor** needed no reshape/view workaround — `rt::argmax(&a_2d)`
  works natively (returns flat row-major usize). Benched directly.
- **Strided case** uses a transpose view of a 2-D tensor (2048² and a 64²
  gate) — the 1-D strided-slice alternative would exercise the same
  `IndexedIterLayout` fallback with a smaller Cartesian surface; the t-view is
  the realistic user shape.
- **API break of rstsr-native-impl** (G1 item 1): `(Fcomp, Feq)` → `ArgCmp`
  changes the `pub` kernel signatures of the published crate; documented in
  the phase-2 section above and to be flagged in the integration PR.
- **Parallel NaN corner refined** (G1 item 2): pre-patch rayon behavior was
  pairing-order dependent (probe transcript demonstrates real wrong answers
  on the clean tree); the patched kernel is deterministic and serial-exact.
  Documented in the semantics section; gate + probe fixtures added.
- **Symlink discovery**: the T0 code map's "dead duplicate" in
  `feature_rayon/auto_impl/` is physically the same file as the live
  `device_faer/rayon_auto_impl/` (symlinks); the patch necessarily appears
  under the `feature_rayon/` path in `git diff`.
- **`#[inline(never)]` on the fast path**: an early draft inlined both paths
  and cost the untouched strided fallback +2.3–3.8% under native; holding the
  fast path out-of-line restored the fallback's codegen (paired A/B: ≈ +1%).
- **Probe + lane fixtures added beyond the brief**: the NaN-lane gate
  fixtures (NaN at 1..8 with same-lane max) were added after the probe caught
  a genuine bug in the first draft kernel — kept as permanent regression
  coverage.
- `results/make_tables.py` (regenerates `results/tables.md` from raw
  criterion text) and `examples/probe_nan_corner.rs` are tooling beyond the
  strict brief, following the T0 convention.

## Reproduce

```bash
cd 2026-09-09-argmax-argmin
./reproduce.sh              # everything: correctness → portable → native → perf (~25 min)
./reproduce.sh correctness  # gate only, both configs (~3 min)
./reproduce.sh portable     # criterion suite, portable (~8 min)
./reproduce.sh native       # criterion suite, native (~8 min)
./reproduce.sh perf         # perf stat pass (~2 min)
./reproduce.sh candidate    # patched-tree rerun (same 4 stages, candidate baselines)
./reproduce.sh numpy        # optional numpy context anchor (conda torch)
```

Requires: nightly rustc 1.97.1 active, `RAYON_NUM_THREADS=16` (set by the
script), `perf` for the perf stage, conda `torch` env only for the numpy
stage. `target/` is gitignored and may be deleted when the experiment
finishes; `Cargo.lock` is kept for reproducibility.

## Phase status

- **Phase 1** (baseline + semantics): complete, this README's baseline tables.
- **Phase 2** (candidate kernel): complete — patch implemented, gate passed,
  D3 met, [proposed.patch](proposed.patch) produced, `../rstsr` reset to
  clean `386948be` (verified). Phase-2 numbers: tables above +
  `results/candidate/`.

## Addendum 2026-09-11: integration review applied, committed in rstsr

The owner reviewed [proposed.patch](proposed.patch) and authorized
integration of this patch (first of the campaign's five) into `../rstsr`.
Review follow-ups were applied and verified before committing.

**What changed vs [proposed.patch](proposed.patch):**

1. *General closure API restored.* The original closure-based
   `reduce_{all,axes}_{,unraveled_}arg_cpu_{serial,rayon}` signatures
   (`f_comp`/`f_eq`) are kept verbatim as the general API for future
   non-standard arg-reductions; the specialized kernels are renamed
   `reduce_*_arg_cmp_cpu_{serial,rayon}` (+ `ArgCmp`) and documented as the
   argmin/argmax-only fast path. Device wiring calls the `_cmp_` functions.
   Net effect vs `386948be`: **publicly additive only** — this supersedes
   the API-break note in the phase-2 section above.
2. *Clippy clean* (both feature configs): targeted `#[allow(clippy::eq_op)]`
   with justification on the generic `x == x` NaN check (owner confirmed
   keep `==`; `is_nan` would widen public bounds via `ExtNum`), and
   `while_let_on_iterator` fixed by real `for ch in chunks.by_ref()`
   rewrites (equivalent iterator protocol).
3. *No new helper:* the patch's `pub fn unravel_c_order` was replaced by
   `DimShapeAPI::unravel_index_c` from rstsr-common (exact duplicate;
   f-order twin also exists for any future f-order-native variant).
   Column-major conventions re-checked: the arg pipeline's visit order is
   explicitly `RowMajor` everywhere (feature-independent), so c-order
   unraveling stays correct under the `col_major` feature; `c_contig()`
   gating is a strides property, also feature-independent. (Col-major has
   no test binary yet — guarded by construction, not by tests.)

**Verification** (transcripts in `results/refactor_track/`, criterion
baselines `refA_*` kept):

- Gates: rstsr-core lib 110/110 + entry_row_cpu 290/290, correctness gate
  ALL PASSED, both configs, after every change.
- Benches: refA baseline reproduced the phase-2 numbers (1e7 argmax 1.82 ms
  portable / 1.54 ms native); then **three paired passes** post-refactor.
  No case slower than refA by >2% in all passes of a config; pass-1 flags
  did not replicate (tview small argmin portable +2.6% → −0.7%; faer16 1e7
  argmin native +3.2% → +1.3%). Persistent improvements are code-layout
  luck (serial f32 1e7 argmax −10% in all passes; faer16 1e7 argmax native
  −24…−29%), not algorithmic. Small/faer cells swing ±5–20% between
  same-binary reruns — only replicated deltas mean anything.

**NaN semantics correction (important):** the phase-2 section above claims
rstsr's NaN behavior "coincide[s] with numpy's documented argmax/argmin NaN
behavior". Verified against NumPy 2.5.2 source: NumPy returns the **first
NaN at any position** (kernel breaks at the first NaN; `argmax([1,nan,3])`
= 1), while rstsr skips mid-stream NaNs (`[1,nan,3]` argmax = 2). The claim
holds only for NaN-at-front and all-NaN inputs. A nanargmin/nanargmax
proposal (incl. the open decision on plain-arg NaN semantics) is in
[NANARG-PROPOSAL.md](NANARG-PROPOSAL.md).

**Landed as:** `../rstsr` branch `260910-core-efficiency`, commit `091f3e2`
("rstsr: speed up argmin/argmax with 8-lane contiguous fast path",
4 files +667/−248) — local commit, not pushed, no PR yet.
Final diff also kept here as
[proposed-v2-post-review.patch](proposed-v2-post-review.patch).
