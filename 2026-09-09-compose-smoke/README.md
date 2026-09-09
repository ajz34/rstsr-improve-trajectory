# 2026-09-09-compose-smoke (T8)

Compose smoke test for the rstsr CPU-efficiency campaign: do the FIVE proposed
patches apply **together**, pass rstsr's own test suites **as a union**, and
hold their **headline wins** in the combined build?

- rstsr base: `386948be819baa334b8da02232f3a1944e5447d5` (clean before, clean after)
- Date / machine: 2026-09-09, 16-core (9950X3D class), AVX-512,
  `RAYON_NUM_THREADS=16` convention for the default DeviceFaer.
- Toolchain (reconciled; evidence in `results/toolchain.txt`): **rustup default at
  gate time, stable 1.97.1 recorded** — the smoke crate's cwd has no toolchain
  file, so its cargo resolves the rustup default, and rstsr's
  `rust-toolchain.toml` (`channel = "nightly"`) does **not** propagate along path
  deps (T5 finding, re-confirmed here empirically). Consequence: rstsr's OWN test
  suites were run from inside the rstsr dir and therefore used the pinned
  `rustc 1.99.0-nightly`, while the gate/spot example binaries in this crate were
  built by `rustc 1.97.1 stable`. Both resolutions are shown in the committed
  artifact.
- Configs (campaign D7 convention): **portable** = no RUSTFLAGS;
  **native** = `-C target-cpu=native`. Release profile = stock cargo
  (opt-level 3, no LTO), mirrored explicitly in `Cargo.toml`.
- Crate: standalone path-dep on `../../rstsr/rstsr-core` with rstsr-core's
  default features mirrored explicitly (T0 harness pattern).

> **FINAL VERDICT (re-verification round): COMPOSE: YES** with the AMENDED
> patch 3. Round 1 found a latent patch-3 bug via the union (order-changing
> reshape panicked; fixed here temporarily by `compose_guard_fix.patch`,
> now **SUPERSEDED**: patch 3's own dir carries the amended
> `proposed.patch` with the guard at all 4 router sites + a reshape
> fixture). Everything below was re-verified end-to-end against the
> amended union. Round-1 numbers kept in place for the record.

## The five patches (campaign order, applied 1→5; patch 3 = AMENDED)

| # | patch (path relative to repo root) | files |
|---|---|---|
| 1 | `2026-09-09-argmax-argmin/proposed.patch` | ArgCmp enum + arg contiguous fast path (core serial + rayon auto_impl + native-impl reduction.rs) |
| 2 | `2026-09-09-elementwise/proposed.patch` | blocked 2-D tile path in op_with_func (native-impl cpu_{serial,rayon}) |
| 3 | `2026-09-09-transpose-assign/proposed.patch` — **AMENDED**: shape-identity guard at all 4 orderchange router sites + reshape fixture | blocked orderchange routing in assign families (native-impl cpu_{serial,rayon} assignment.rs + transpose.rs) |
| 4 | `2026-09-09-reductions/proposed.patch` | branch-2 band walk + strict-compare min/max (core serial + rayon auto_impl + native-impl cpu_serial reduction.rs) |
| 5 | `2026-09-09-vecdot/proposed.patch` | vecdot branches + inner_dot fast paths (native-impl {vecdot,matmul_naive}.rs + core device_cpu_serial matmul.rs bound widening) |

## 1. Apply-order verdict

**Campaign order 1→5 applies cleanly end-to-end — no reordering needed**
(re-confirmed with the amended patch 3). `git apply` succeeded for every
hunk of every patch on the first attempt; cumulative `git diff --stat`
after each step (amended round / round-1 in parens):

| after | files | insertions | deletions |
|---|---|---|---|
| 1 argmax-argmin | 4 | 393 (393) | 296 |
| 2 elementwise | 6 | 739 (739) | 296 |
| 3 transpose-assign (amended) | 10 | **1036** (1016) | 296 |
| 4 reductions | 10 | **1171** (1151) | 332 |
| 5 vecdot | 15 | **1462** (1442) | 397 |

The predicted overlap (patches 1 and 4 both touch reduction.rs in three
variants) held: hunks are disjoint as claimed, no context drift.

`combined_all_five.patch` is the verbatim union diff of the five patches
(including the amended patch 3) (**generated artifact; machine
round-trip-tested via stash/apply; NOT independently reviewed**;
2529 lines, +1462/−397).

## 2. Union test-suite result — history + amended-union result

### Round 1 (original patch 3): one real bug found

`entry_row_cpu` on the round-1 union **FAILED 3 of 290 tests** (clean tree
passes 290/290 — verified by stashing the union and re-running):

```
core_func::manipulation::test_reshape::numpy_reshape::multiarray
doc_draft::manipulation::test_reshape::doc_reshape::elaborated_diff_row_col
doc_draft::manipulation::test_reshape::doc_reshape::reshape_with_args
```

Error: `rstsr-native-impl/src/cpu_serial/transpose.rs:82: This function
requires shape identity : sc [0] = 3 not equal to sa [0] = 2`.

**Root cause (patch 3, latent, not a composition conflict):** the new 2-D
order-change router in `assignment.rs` guarded on stride pattern only
(`sa[1]==1 && sc[0]==1` → r2c, `sa[0]==1 && sc[1]==1` → c2r) but not on
shape identity. The blocked orderchange kernels require `shape(c) ==
shape(a)`. `reshape` legitimately calls `assign_arbitary_uninit` with
shape-CHANGING layouts (`rstsr-core/src/tensor/manipulation/reshape.rs:157`)
whose stride pattern still matches the transpose form (e.g. c-contig
[2,3] → f-contig [3,2]) → panic. T1' captured its patch "awaiting G2"; its
own correctness gate never exercised order-changing reshape, so this was
never seen. None of the other four patches touch these files.

**The fix (same shape as `compose_guard_fix.SUPERSEDED.patch`, now folded
into the amended patch 3):** add the shape-identity precondition to the
router:

```rust
if lc2.shape() == la2.shape() && sa[1] == 1 && sc[0] == 1 { ... r2c ... }
else if lc2.shape() == la2.shape() && sa[0] == 1 && sc[1] == 1 { ... c2r ... }
```

Shape-mismatched assigns fall through to the generic iterator path exactly
as before the patch. All transpose-copy traffic that motivated patch 3 has
identical shapes on both sides, so the win is untouched (spot #3 below).

### Amended union (final): all suites green WITHOUT any separate fix

Raw `cargo test` outputs of this run are committed:
`results/cargo_test_amended_union_{portable,native}.txt` (each contains the
lib + entry_row_cpu + native-impl runs for its config). The **round-1
failure log was not saved to a file**; its content — the three failing test
names and the `transpose.rs:82` shape-identity error message — is quoted
verbatim above from the session transcript.

| suite | clean 386948be | round-1 union (as proposed) | amended union (final) |
|---|---|---|---|
| `rstsr-core --lib` (default feats), portable | — | 110 pass / 0 fail / 3 ign | **110 pass / 0 fail / 3 ign** |
| `entry_row_cpu` (backtrace+row_major), portable | 290 / 0 | **287 / 3 FAIL** | **290 pass / 0 fail** (3 reshape tests pass with the amended patch 3 alone) |
| `rstsr-core --lib`, native | — | — | **110 pass / 0 fail / 3 ign** |
| `entry_row_cpu`, native | — | — | **290 pass / 0 fail** |
| `rstsr-native-impl` (both configs) | — | — | 0 test targets exist (nothing to run) |

## 3. Fresh combined correctness gate

`examples/correctness.rs` ports the ESSENTIAL fixtures from all five
experiments' correctness suites into one gate: argmax/argmin ties + NaN-lane
+ NaN-at-chunk-start + empty (Err + panic); tile-path strided add/mul incl.
flip views, zero-stride broadcast, odd 1000x777; transpose r2c/c2r
orientations + bcast slow-axis + sliced stride-2 fall-through + degenerate
1x7/7x1; reductions sum_axis0 values + min/max NaN-skip + all-NaN seed +
signbit locks (±0 ties keep incumbent) + the documented stride-0 sum
upstream-bug lock + i32; vecdot batched axis0/axis1 + `%` fast path +
alpha/beta fallback (rt::matmul_from, α=2 β=0 and α=2 β=0.5 accumulate) +
c64 spots. All vs naive references; f64/f32/i32/Complex<f64> where the
source fixtures had them; BOTH devices (DeviceCpuSerial, DeviceFaer default
with 16 threads asserted).

**Result on the AMENDED union: 332/332 checks PASS, portable AND native**
(`results/correctness_{portable,native}_amended.txt`; round-1 runs in
`results/correctness_{portable,native}.txt`). The gate found one bug in my
own port (f-contig r2c reference) before going green in round 1 — fixed in
the gate, not the tree.

## 4. Spot benches (combined tree)

Method: fixed-iteration medians with `black_box` (serial: 30-40 iters for
ms-scale cases, 200 for vecdot, 2000 for `%`; warmup 3-50). Three runs per
config; medians of the per-run medians below. Full logs in `results/`.

| case (serial device unless noted) | combined portable | combined native | individual candidate ref (source) | verdict |
|---|---|---|---|---|
| argmax 1e7 f64 [T6] | 1.485 ms | 1.489 ms | 1.82 ms port / 1.55 ms nat — criterion (T6 README, results/tables.md) | **holds** (≤ ref) |
| strided add 2048² REUSE `op_mutc_refa_refb_func` [T4'] | 7.973 ms (r1) | 8.002 ms (r1) / **7.499 ms (amended, med of 3: 7.499/7.787/7.407)** | 7.62 ms port / 7.51 ms nat — criterion (T4' README headline) | **holds** — round-1 harness read ~6% high; T4's own criterion bench on the round-1 combined tree: **7.50 ms** = exact match; amended-union rerun lands on the reference directly |
| transpose copy 2048² REUSE `c.assign(&a.t())` [T1'] | 5.987 ms (r1) | 5.896 ms (r1) / **5.885 ms (amended, med of 3: 5.885/5.897/5.867)** | 5.96 ms port / 5.99 ms nat — criterion (T1' README phase-2 table) | **holds** (±2%) |
| sum_axis0 2048² [T2'] | 1.225 ms | 1.160 ms | 1.205/1.195 ms port, 1.365/1.228 ms nat — criterion r1/r2 (T2' README headline) | **holds** (−5.5% native, +1.7% portable) |
| batched vecdot (512,4096) axis-0 [T3'] | 0.544 ms | 0.473 ms | 538 µs both configs (T3' README) | **holds** |
| batched vecdot (4096,512) axis-1 [T3'] | 0.348 ms | 0.334 ms | 363 µs port / 352 µs nat (T3' README) | **holds** |
| `%` 1-D 1e4, default device faer16 [T3'] | 6.91 µs | 6.28 µs | 7.19 µs (T3' results/tables.md "29.3→7.19, −76%") | **holds** |

**Zero regressions.** Round-1 strided-add drift (>5% under my harness) was
cross-checked with the origin experiment's criterion harness on the same
combined tree: 7.50 ms = exactly the individual candidate number
(methodology offset, not composition loss). After the patch-3 amendment the
two patch-3/4-adjacent headline cases were re-checked 3× (native): strided
add 7.499/7.787/7.407 ms (median 7.499 ≈ 7.51 ref) and transpose 5.885/
5.897/5.867 ms (median 5.885 vs 5.99 ref) — nothing shifted.

NOTE on the task brief's "expect ~29 µs" for `%` 1e4 faer16: 29.38 µs is the
PRE-patch baseline (T3' tables.md line 67); the patched candidate is
7.19 µs. The combined tree measures 6.3–6.9 µs, consistent with (slightly
better than) the candidate.

### Cross-patch interaction check

No negative interactions observed. The five patches touch disjoint
mechanisms (arg fold / op_with_func tile / assign routing / reduction band
walk / vecdot branches); the only shared files (reduction.rs trio, patches
1+4) merged at independent hunks. The one union bug was internal to patch 3
(and is now fixed inside patch 3 itself).

## 5. Conclusion

**COMPOSE: YES with the amended patch 3.** The five patches apply together
in campaign order without conflicts, pass rstsr's full entry_row_cpu suite
(290/290, including the three reshape tests) and lib suite (110/110) in
both RUSTFLAGS configs with **no separate fix needed**, pass a fresh
332-check cross-experiment correctness gate on both devices, and all six
headline wins hold in the combined build (each within noise of its
individual candidate measurement; several slightly better). The amended
patch 3 is safe to integrate as-is; the reshape hazard found in round 1 is
closed by the shape-identity guard now carried inside the patch.

`compose_guard_fix.SUPERSEDED.patch` is kept as the discovery record of the
round-1 union failure and the minimal fix; do not apply it.

## 6. Deviations from the brief

1. **Round-1 guard fix (headline deviation, now superseded):** the round-1
   union failed 3 tests, so the router guard was diagnosed, applied
   out-of-band, and handed to the patch-3 owner — who amended patch 3
   itself (same guard + reshape fixture). The separate fix is retained only
   as `compose_guard_fix.SUPERSEDED.patch`.
2. **Brief's `%` expectation** quoted the pre-patch baseline (29 µs); the
   correct candidate reference is 7.19 µs (see note above).
3. rstsr-native-impl has **no test targets at all** — "if time permits"
   item ran and found nothing to run (both configs).
4. Harness methodology: fixed-iteration medians (brief) vs criterion
   (references); one cross-check with the origin harness was needed to
   resolve a round-1 drift (strided add) — resolved as methodology offset.

## 7. Reproduce

See `reproduce.sh` (apply → test → gate → spot → restore; the guard-fix
step is gone since patch 3 is amended). Requires: `RAYON_NUM_THREADS=16`,
rstsr checkout at `386948be` with a clean worktree. Toolchain: whatever
rustup resolves per directory — rstsr's own suites run under its pinned
nightly (1.99.0-nightly at gate time), this crate's builds under the rustup
default (stable 1.97.1 at gate time); see `results/toolchain.txt`.
