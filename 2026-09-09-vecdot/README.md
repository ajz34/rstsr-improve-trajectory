# T3' — vecdot + inner_dot (phase 2 COMPLETE: D3 PASS)

Campaign task T3' of the rstsr CPU-efficiency campaign. Phase 1 (baseline +
anatomy + design) and **phase 2 (implementation + measurement)** are done.
The candidate patch is [proposed.patch](proposed.patch) against clean
`386948be819baa334b8da02232f3a1944e5447d5`; the rstsr tree was restored to
clean HEAD after measurement (`git status` empty, verified).

- **Design + G1 review record**: [PLAN.md](PLAN.md) (options A/B, guards,
  composeability, dispatch_simd SKIP verdict, D6 complex decision).
- **Baseline numbers**: [results/tables.md](results/tables.md) (phase 1) and
  the PHASE 2 section of the same file (candidate results + D3 verdict).
- **Volatile-cell evidence**: [results/candidate_faer_trials.md](results/candidate_faer_trials.md).

## What the patch changes (see `git apply --stat proposed.patch` for the authoritative per-file tally; 5 files)

1. `rstsr-native-impl/src/cpu_serial/vecdot.rs` — **A1**: flat row loop for
   the contiguous-summed + 1-D-remaining case (branch 1), replacing the
   per-row `layout_col_major_dim_dispatch_3`/`_2` machinery; fall-through
   unchanged otherwise. **A2**: single local accumulator (≤32 KiB L1 budget)
   + one summed walk for the contiguous-remaining case (branch 2), killing
   the MaybeUninit RMW and the per-band dispatch re-walk; fall-through
   unchanged. Bit-exactness argued in-code (review item 2: the dropped
   `0 + val` is a no-op — zero-seeded lanes never produce −0.0; NaN
   unchanged).
2. `rstsr-native-impl/src/cpu_rayon/vecdot.rs` — **A1t**: hoists the 0-dim
   summed walk out of the parallel row loop (keeps `dispatch_par_3`
   parallelism). **A2t**: summed walk materialized once (1 MiB scratch cap,
   fall-through preserved) + band-local accumulators — note this deviates in
   shape from the plan's "single local vacc" wording: the rayon twin
   accumulates into CHUNK-local (per-band-task) `TC` buffers, one write to
   `c` per band at the end, because per-position single vaccs would
   serialize the parallel band split; per-output accumulation order is
   preserved bit-exact (summed positions `0..K` in order within each band
   task, same as serial).
3. `rstsr-native-impl/src/cpu_serial/matmul_naive.rs` — **B1**: contiguous
   `alpha==1 && beta==0` fast path through `unrolled_binary_reduce` for
   `inner_dot_naive_cpu_serial` (the whole `%` surface); strided/scaled
   callers keep the original loop. **Bound widening (PR-note, review item
   1)**: `TC: Zero + One + PartialEq` added to `inner_dot_naive_cpu_serial`
   and propagated through `matmul_naive_cpu_serial` — an API break on a
   `pub fn` of the published crate, same category as T6's `ArgCmp` change;
   all in-tree instantiations satisfy the bounds.
4. `rstsr-core/src/device_cpu_serial/linalg/matmul.rs` — propagation of the
   same widening to the `DeviceMatMulAPI for DeviceCpuSerial` impl
   (impl-level; PR-note in code).
5. `rstsr-native-impl/src/cpu_rayon/matmul_naive.rs` — **B2** (no bound
   changes): sequential fold below `PARALLEL_SWITCH=512` (was 10–30 µs of
   pure rayon overhead at small n), chunked contiguous parallel fold
   (`par_chunks(8192)` × 8-lane reduce) for large n, per-element fold kept
   as strided fallback.

## Results (medians of repeated trials; full tables in results/tables.md)

| target | portable | native |
|---|---|---|
| batched_am1 (4096,512) serial | 444–466 → **363 µs (−22%)** | 450–466 → **352 µs (−22%)** |
| batched_axis0 (512,4096) serial | 1497–1614 → **538 µs (−64%)** | 706–962 → **538 µs (−24%)** |
| batched_am1 (8192,256) serial | −37% | −35% |
| odd (1000,777) serial | −20% | −16% |
| small (64,64) serial | −43% | −42% |
| `%` 1e7 serial | 4143 → **2925 µs (−29%)** | 4075 → **2925 µs (−28%)** |
| `%` 1e6 / 1e4 / 64 — faer16 | −73% / −76% / **−96%** | −73% / −76% / **−96%** |
| `%` 1e7 — faer16 | **−1.3% (DRAM-bound, honest miss)** | −1.1% |
| batched_am1 faer16 (gate) | −5% (improved) | −1% (band) |
| 1-D f64 gates (1e3/1e5/1e7) | ±2% | ±2% |

Efficiency framing (batched_am1 serial, native): 5.0% → **6.5% of the
32-DP-FLOP/cyc core peak** (9.0 → 11.8 GFLOP/s; 72 → 95 GB/s L3-assisted) —
numpy-einsum parity (0.349 ms single-thread). **Honest framing (review item
3)**: serial batched am1 at 352–363 µs is still ~2.5× behind the faer16 wall
(140 µs) — the serial path is now within ~15% of its own contraction loop's
probe ceiling (292–350 µs) and at einsum parity, but the remaining serial
gap is per-core load bandwidth, not removable dispatch cost. perf evidence:
innerdot serial instructions 12.25 → 2.72 per element; axis0 3.17 → 0.90
(RMW gone); faer16 `%` core-time instructions 11.76 → 2.95 (wall capped by
DRAM at 1e7, hence the honest miss there).

## D3 verdict: PASS (with one honest sub-target miss)

- Primary accepts (≥15% BOTH configs): batched_am1 serial ✓ (−22/−22),
  batched_axis0 serial ✓ (−64/−24), `%` 1e7 serial ✓ (−29/−28).
- %-of-peak alternative framing: 5.2→6.5% stated above.
- Misses, stated: faer16 `%` at 1e7 wall (−1.3%; DRAM-bound — the core-time
  instruction collapse of 4× is delivered but cannot move wall when the op
  streams 160 MB from DRAM). faer16 `%` at 1e4–1e6 passed by 73–96%.
- Regression gates: PASS — see tables (untouched strided path within its
  build band +1.5–4% native faer16; everything else improved or flat).

## PR-notes for the human (fold into the eventual rstsr PR)

1. **API break**: bound widening `TC: Zero + One + PartialEq` on
   `inner_dot_naive_cpu_serial` + `matmul_naive_cpu_serial` + the
   `DeviceMatMulAPI for DeviceCpuSerial` impl (T6 `ArgCmp` precedent).
2. **Reassociation**: serial `%` summation order changes (8-lane);
   pre-declared at G1 — `%` has no cross-device bit-exact contract (faer16
   always re-associated). vecdot bit-exactness is preserved on both devices.
3. **Uninit-read hazard (review item 4 — both shapes named)**: at
   386948be, the SERIAL `inner_dot_naive_cpu_serial` reads `beta * c[idx]`
   from the fresh UNINIT output BEFORE the loop, and the RAYON twin reads
   `c.clone() * beta` AFTER its fold — both on memory never written by this
   op (`empty` allocation). With `beta = 0` (the only value the `%` surface
   ever passes) this is 0 × uninit-bits: benign when the bits are finite,
   NaN when the allocator hands back NaN-pattern bits. The B1 fast path no
   longer reads `c`; the B2 small-n fallback and the strided/scaled
   fallbacks PRESERVE the old behavior (no semantic fix smuggled in) — the
   hazard on those paths is reported here for a separate correctness
   decision by the human.
4. dispatch_simd: NOT proposed (PLAN §3); complex not included (D6 decision
   recorded: c64 cost is arithmetic not dispatch; probes A5/A6).
5. Composeability: zero file overlap with the four existing campaign
   patches (argmax, reductions, transpose-assign, elementwise);
   `unrolled_binary_reduce` untouched by them (grep-verified in PLAN §4).

## Correctness

49 checks, ALL PASS, BOTH configs, BOTH devices
(`results/correctness_candidate_{portable,native}.txt`): all 43 phase-1
checks (NaN locks, zero-stride broadcast remaining+summed, odd sizes,
strided view semantics, f-contig, complex conjugation) + 6 new
(B1 n=300 sequential-fallback boundary, A2 large-`n_contig` fall-through,
A1 3-D fall-through × 2 devices). The serial `%` bit-exact lock became
tolerance-based exactly as pre-declared at G1. rstsr's own suites pass:
`cargo test -p rstsr-core --lib` (110), `entry_row_cpu` (290, incl. the
alpha=1.5/beta=2.0 matmul tests that exercise the B1 fallback).

## Layout / reproduce

Crate layout as phase 1 (see git history of this directory), plus
`proposed.patch` and `examples/debug_b1.rs` (routing probe used to verify
the B1 fast path fires: `%` 2.78 ms ≈ vecdot 2.82 ms at 1e7).

```bash
cd 2026-09-09-vecdot
./reproduce.sh portable native      # clean-tree baselines (named, saved)
./reproduce.sh candidate            # apply patch -> gate -> bench -> perf
./reproduce.sh candidate-restore    # reset ../rstsr to clean 386948be
```

Requires rustc 1.97.1 nightly, `RAYON_NUM_THREADS=16`, conda `torch` for the
numpy stage, `perf` for profiling. `target/` gitignored and left in place;
`Cargo.lock` committed.
