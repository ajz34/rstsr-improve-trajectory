# T1' — transpose-assign (PHASE 1 baseline + PHASE 2 candidate implemented)

Campaign task T1' (transpose copy and the generic strided assign behind it).
**Phase 1**: baseline on the CLEAN rstsr tree, raw-kernel probes, perf
evidence, phase-2 plan ([PLAN.md](PLAN.md)). **Phase 2** (after G1 PASS):
implemented PLAN.md Option A in `../rstsr`, re-gated, re-benched, captured
[proposed.patch](proposed.patch). **D3 PASS** (see the phase-2 section
below); the rstsr working tree was restored to clean at HEAD 386948b after
patch capture.

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5` (path dep
  `../../rstsr/rstsr-core`; verified clean + HEAD 386948b before baselining;
  the tree was never modified — `git status` re-verified at the end).
- Campaign plan §3 contract:
  [../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
  (T1 section), code map §2(a)/§7 item 5.
- Context reused: T0 harness pattern + baseline
  ([../2026-09-09-bench-harness-baseline/](../2026-09-09-bench-harness-baseline/README.md)),
  T7 reuse-denominator rule
  ([../2026-09-09-alloc-pagefault-study/](../2026-09-09-alloc-pagefault-study/README.md)),
  T4' elementwise tile-path learnings
  ([../2026-09-09-elementwise/](../2026-09-09-elementwise/README.md)).

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores, AVX-512; 48 kB L1d / 1 MB L2 / 128 MiB X3D L3 |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)` (nightly via rstsr's rust-toolchain.toml) |
| criterion | 0.5.1 (2 s + 0.7 s warm-up per bench) |
| ndarray | 0.16.1 (anchor) |
| perf | 7.0.14 |
| RAYON_NUM_THREADS | 16 (asserted by the harness for every faer run) |
| Target configs | `portable` (no RUSTFLAGS) and `native` (`-C target-cpu=native`) |
| Cargo profile | stock release defaults (explicit in Cargo.toml) |
| Allocation policy | A = allocation included (T0 idiom); B = pre-allocated, pre-warmed output via `c.assign(&a.t())` (kernel-only; PRIMARY judge per T7); C = A under glibc tunables. Identical across devices/configs per variant. |

## Directory layout

```
Cargo.toml / src/lib.rs    standalone crate (T0 harness pattern; A/B/C protocol);
                           path-deps on BOTH rstsr-core (default features
                           mirrored) and rstsr-native-impl (rayon) so the raw
                           kernels can be called directly
benches/transpose.rs       transpose copy A/B (5 sizes incl. oddT, f64+f32,
                           both devices) + assign_gates fall-through canaries
benches/anchors_ndarray.rs ndarray a.t().to_owned() anchors
benches/kernels_probe.rs   direct raw-kernel calls: today's generic assign
                           kernel vs the DORMANT blocked orderchange kernels,
                           loop-orientation + raw-pointer probes
examples/correctness.rs    gate vs naive + ndarray; both orientations, flips,
                           sliced views, zero-stride broadcast, 3-D, i32/f32,
                           degenerate 1x7/7x1, direct-kernel checks incl.
                           negative slow-axis stride; BOTH RUSTFLAGS configs
examples/profile_ops.rs    fixed-iteration runner for perf stat (ta/tb modes)
results/                   raw logs, criterion data, tables.md, perf/
reproduce.sh               correctness -> portable -> native -> portable_c ->
                           native_c -> perf; `candidate` = phase-2 placeholder
PLAN.md                    phase-2 edit plan (options, guards, accept criteria)
```

## Headline baseline (full tables: [results/tables.md](results/tables.md))

Transpose copy 2048×2048 f64 — the T7 denominators, both configs:

| variant | serial native | serial portable | faer16 native | faer16 portable |
|---|---|---|---|---|
| A allocating idiom | 23.159 ms | 23.613 ms | 3.325 ms | 3.461 ms |
| **B reuse `c.assign` (primary)** | **17.155 ms** | **17.546 ms** | **1.450 ms** | **1.504 ms** |
| C = A + MALLOC tunables | 17.015 ms | 16.885 ms | 1.564 ms | 1.519 ms |

Fault rider (A−B)/A = 26% serial / 56% faer16 — T7 reproduced; C recovers it.
f32 has no rider (A ≈ B, 15.5 ms serial / 1.30 ms faer16 native).

All sizes, serial B (native): small 64² **14.1 µs**, medium 512² **0.816 ms**,
odd 1000×777 **2.42 ms**, oddT 777×1000 **2.43 ms** — vs ndarray
`.t().to_owned()`: 0.27 µs / 28.5 µs / 72.7 µs / 72.9 µs (**52× / 29× / 33× /
33× gaps**; the two non-large gaps are pure iterator-machinery overhead).

## perf baseline (the number phase 2 beats)

B variant 2048² serial native (results/perf/): **107.8 ins/elem, 22.4
cyc/elem, IPC 4.8, 2.9% L1d-miss, ~0 steady-state faults, 4.0 GB/s** (2×
bytes). T0's "122 ins/elem" was the A profile (119.7 ins/elem here, 8.4k
faults/iter). faer16 B: 1.534 ms wall, 15.4 CPUs, 116 ins/elem aggregate.

## Direct-kernel probes (the phase-2 bound; benches/kernels_probe.rs)

| probe (2048² f64, native / portable) | time | vs generic |
|---|---|---|
| P_generic_serial (today's kernel, bare) | 15.24 / 15.06 ms | 1× |
| **P_blocked_c2r_serial (DORMANT kernel)** | **5.70 / 5.78 ms** | **2.7×** |
| P_generic_rayon16 | 1.18 / 1.11 ms | 1× |
| **P_blocked_c2r_rayon16** | **0.542 / 0.545 ms** | **2.2×** |
| P_blocked_swap_serial (loops swapped) | 17.41 / 17.57 ms | 0.33× — in-tree orientation is right |
| P_blocked_rawptr_serial (bounds checks removed) | 5.42 / 5.82 ms | ≈1× — checks are not the lever |

Odd 1000×777: blocked 0.296 ms serial / 97.9 µs rayon16 (generic: 0.446 ms /
212 µs). The blocked kernel is ≥2× at every probed size — no size floor
needed on the serial side.

## Call-chain anatomy (file:line at 386948be) — the short form

- **A idiom**: `t()` view (`tensor/manipulation/transpose.rs:438`) →
  `change_contig_f` (`to_contig.rs:8-45`) → `change_layout_f`
  (`to_layout.rs:8-38`; `uninit_impl` + `assign_arbitary_uninit` at :33) →
  `assign_arbitary_uninit_promote_cpu_serial` — the 4th `#[duplicate_item]`
  variant of family 1 (`rstsr-native-impl/src/cpu_serial/assignment.rs:6-67`).
  Not c-contig → `translate_to_col_major_unary(·, C)` per layout →
  `layout_col_major_dim_dispatch_2diff` → two `IterLayoutColMajor` zipped
  per element (`rstsr-common/src/layout/iterator.rs:288-300`).
- **B reuse**: `c.assign(&a.t())` → `TensorAssignAPI::assign_f`
  (`tensor/assignment.rs:119-140`) → `OpAssignAPI::assign` →
  `assign_promote_cpu_serial` — 3rd variant of family 2
  (`cpu_serial/assignment.rs:69-119`), K-order greedy translation,
  size_contig = 0 → same per-element iterator zip. Rayon twins:
  `cpu_rayon/assignment.rs` (PARALLEL_SWITCH 16384), faer wiring
  `device_faer/rayon_auto_impl/assignment.rs`.
- **Dormant kernel**: `orderchange_out_r2c_ix2_cpu_serial`
  (`cpu_serial/transpose.rs:11-48`, BLOCK_SIZE=64, bounds-checked stores),
  c2r wrapper :54-61; rayon twin `cpu_rayon/transpose.rs:15-67` (raw-pointer
  stores, nested block-parallel, serial fallback < 16·64²). Only callers
  today: LAPACK svd/gesvd/potrf drivers.
- Kernel guards (Err on violation): shape identity; r2c: `la.stride()[1]==1
  && lc.stride()[0]==1`; slow-axis strides arbitrary isize — **including
  negative** (flip-view case direct-tested PASS) and zero (broadcast slow
  axis: re-reads one row; correct).

Full anatomy with the design options, guard sketch, accept criteria, and the
elementwise-patch interaction statement is in [PLAN.md](PLAN.md).
**Recommendation**: route 2-D order-changes through the dormant kernels
inside BOTH assign families (native-impl level, consistency with T4');
`dispatch_simd` skipped per plan §5; patches touch
`cpu_{serial,rayon}/assignment.rs` (+maybe `transpose.rs`) — disjoint from
T4''s `op_with_func.rs`, so they compose in any order.

## PHASE-2 result (blocked 2-D order-change routing) — D3 PASS

Patch: [proposed.patch](proposed.patch) (4 files in `rstsr-native-impl`:
`cpu_{serial,rayon}/{assignment,transpose}.rs`). New promote/uninit variants
of the dormant blocked kernels (`into_cast` write — identity for same dtype,
see the dtype decision in PLAN.md §7.1) routed from the non-contiguous branch
of BOTH assign families via a 2-D guard: `ndim == 2` + one layout fast on
axis 0, the other on axis 1, fast-axis strides exactly **+1**. Everything
else falls through byte-identical (zero/negative fast strides = broadcast/
flip, sliced fast axes, ndim > 2, all dtypes via generics). MaybeUninit
outputs are justified by full (i, j) write-once coverage (documented on the
kernels, per G1 item 2); isize offset sums, usize cast after the full sum,
debug_assert on the rayon raw-pointer stores.

Before -> after, reuse variant B (primary judge; full tables in
[results/tables.md](results/tables.md)):

| case (f64) | serial native | serial portable | faer16 native | faer16 portable |
|---|---|---|---|---|
| large 2048² | 17.16 -> 5.99 ms (**2.86×**) | 17.55 -> 5.96 ms (**2.94×**) | 1.45 -> 0.55 ms (**2.65×**) | 1.50 -> 0.56 ms (**2.68×**) |
| odd 1000×777 | 2.43 -> 0.25 ms (**9.7×**) | 2.45 -> 0.25 ms (**9.9×**) | 0.36 -> 0.106 ms (**3.4×**) | 0.37 -> 0.101 ms (**3.6×**) |
| oddT 777×1000 | 2.43 -> 0.30 ms (**8.1×**) | 0.361 -> 0.097 ms (**3.7×**) | 2.64 -> 0.30 ms (**8.8×**) | 0.370 -> 0.097 ms (**3.8×**) |
| medium 512² | 0.816 -> 0.315 ms (**2.6×**) | 0.204 -> 0.058 ms (**3.5×**) | 0.817 -> 0.313 ms (**2.6×**) | 0.214 -> 0.057 ms (**3.8×**) |
| small 64² | 14.1 -> 2.2 µs (**6.5×**) | 14.0 -> 2.3 µs (**6.2×**) | 14.1 -> 2.2 µs (**6.4×**) | 14.2 -> 2.2 µs (**6.4×**) |

- **D3 (≥10% large reuse-variant BOTH configs): PASS with ~26× margin**
  (2.65-2.94× on all four large cells). Stretched targets met: ≥1.7× serial,
  ≥1.5× faer16.
- Secondary: A large serial 23.2 -> 10.4 ms (**2.2×**, fault rider shared);
  f32 B large **2.9×/3.1×**; C large **2.9×/2.75×**; review rows: r2c
  orientation B **2.93×/2.68×** (to_fcontig), zero-stride slow-axis B
  **8.4×/3.7×** (bcast_t), bcast_t odd ~10×; A small/odd/oddT **8-10×**
  (no rider below 32 MiB).
- **perf stat** (B serial native, before -> after): ins/elem **107.8 ->
  13.2**, cyc/elem 22.4 -> 8.6, IPC 4.8 -> 1.5, L1d-miss 2.9% -> 49.9%
  (instruction mountain -> memory-bound streaming — the correct regime);
  GB/s (2× bytes) 4.0 -> 11.1. faer16 aggregate 43.8 -> 122.5 GB/s.
- **Gates**: contig/sliced canaries all within ±3% except two single-suite
  cells dissected with re-runs and recorded as noise (sliced_t faer16-native
  +6.1% not reproduced — candidate re-runs 861-868 µs vs baseline 916;
  contig faer16-portable +5.5% — interleaved clean/candidate runs overlap,
  mean +1.4%, and the guard provably never runs on that path). ndarray
  anchors unchanged (their tree-independent session spread was ±8%).
- **Correctness**: ALL checks PASS under BOTH RUSTFLAGS configs (154 after
  the post-G2 reshape fixture; 149 before) — both
  orientations (incl. new r2c tensor-level cases), flip views, zero-stride
  broadcast plain + transposed, sliced fall-through, 3-D fall-through,
  degenerate 1×7/7×1, negative slow-axis direct-kernel check, i32/f32,
  direct kernels serial+rayon, ndarray cross-check.
- **Non-goals** (per G1): cross-dtype casts (handled correctly via
  into_cast but not separately end-to-end gated — PLAN §7.1), sliced fast
  axes (guard rejects, canaried), 3-D+ (falls through), BLOCK_SIZE sweep,
  dispatch_simd (skipped per plan §5).
- **Disjointness vs the T4' elementwise patch** (re-verified): T4' touches
  only `cpu_{serial,rayon}/op_with_func.rs`; this patch touches only
  `cpu_{serial,rayon}/{assignment,transpose}.rs`. No shared functions; pure
  copy/assign never enters op_with_func drivers and arithmetic ops never
  enter the assign kernels. The patches compose in either order.
- **Tree state**: patch captured, `git apply --check` OK on clean 386948be,
  tree restored (status empty, HEAD 386948b).

## Post-G2 amendment (defect found by compose-smoke) — FIXED

**Defect**: the compose union failed rstsr's `entry_row_cpu` 3/290 —
order-changing reshapes panicked in the blocked kernel. Root cause: the
router guarded the stride pattern (Ix2 + fast-axis stride +1) but NOT shape
identity; reshape (`change_shape_f`, `rstsr-core/src/tensor/manipulation/reshape.rs:147-162`)
calls `assign_arbitary_uninit` with shape-CHANGING layouts, and an
order-changing one (e.g. a [4,3] f-contig source into a [3,4] c-contig
target) matched the stride guard. The orderchange kernels assert shape
identity, so the router returned `InvalidLayout` where the generic path
would have served the assign — a panic at the tensor-API level.

**Fix**: shape-identity guard `lc2.shape() == la2.shape() &&` added at all 4
router sites (both families x serial/rayon), inside the `to_dim::<Ix2>()`
block. Anything shape-changing now falls through unchanged.

**Gate gap**: our phase-1/2 gate never exercised reshape (transpose and
same-shape assigns only). Closed with an explicit fixture:
`order-changing reshape 4x3f->3x4c` ([4,3] f-contig -> [3,4] c-contig via
`to_layout`, the exact bug geometry, 5 sizes x both devices) asserting the
legacy fall-through mapping (verified identical on a clean 386948be tree
before the patch: storage-verbatim for this layout pair).

**Re-verification (patched tree with the fix)**:
- `examples/correctness.rs`: ALL PASS both RUSTFLAGS configs (154 checks,
  incl. the new reshape fixture).
- `cargo test -p rstsr-core --test entry_row_cpu --no-default-features
  --features "backtrace row_major"`: **290/290 PASS** (the 3 reshape tests
  now pass).
- Headline bench re-check (B reuse 2048x2048 f64, serial native, shape
  guard in place, 3 runs): **6.00 / 5.93 / 5.43 ms** vs 6.03 ms
  pre-amendment — the extra `to_dim::<Ix2>()` + shape compare costs nothing
  measurable (a length check + 4-element array compare against a ~6 ms
  kernel).

**Amended patch**: `proposed.patch` (+297 lines, 4 files, git apply --stat
wording below); `git apply --check` verified on a fresh 386948be checkout;
tree restored clean (status empty, HEAD 386948b).

```
 rstsr-native-impl/src/cpu_rayon/assignment.rs  |   43 +++++++++
 rstsr-native-impl/src/cpu_rayon/transpose.rs   |  110 ++++++++++++++++++++++++
 rstsr-native-impl/src/cpu_serial/assignment.rs |   50 +++++++++++
 rstsr-native-impl/src/cpu_serial/transpose.rs  |   94 +++++++++++++++++++++
 4 files changed, 297 insertions(+)
```

## Deviations from the brief (phase 2)

- G1 review items folded in: dtype via into_cast generalization (option 2 of
  item 1 — rationale in PLAN §7.1); MaybeUninit write-once justification
  comment in-kernel (item 2); broadcast slow-axis + r2c-orientation bench
  rows added as `orderchange_extra` (item 3); elementwise contracts kept
  (item 4) with the addition of debug_assert bounds contracts on the new
  rayon stores.
- The candidate re-runs of `orderchange_extra` baseline rows required one
  extra clean-tree pass (group added after the phase-1 baseline); the
  patch was extracted, tree cleaned, baseline extras + gate re-runs taken,
  patch re-applied for candidate re-runs — all raw logs committed
  (results/gates_rerun_*.txt).
- `results/<tag>/criterion/` JSON archives exist for portable/native tagged
  baselines only (partial); txt logs are the complete authoritative record.

### Phase-1 deviations

- Added `rstsr-native-impl` as a second direct path dep (rayon feature, as
  rstsr-core enables it) to make the raw-kernel probe bench possible; the
  brief explicitly allowed/requested this ("do it if cheap").
- The brief's size list is covered exactly, plus `oddT` 777×1000 (both
  orientations of the odd footprint — the blocked kernel's guard cares
  which axis carries stride 1).
- Variant C implemented as the T4'/T7 protocol: env re-run of the A filter,
  not a separate Rust variant.
- ndarray anchor caveat discovered and verified: `.t().to_owned()` PRESERVES
  f-order layout in ndarray 0.16 (content is the logical transpose;
  `as_slice()` is None) — documented in results/tables.md; anchor kept for
  T0 comparability.
- Extra gates beyond the required matrix (`assign_gates`: contig assign,
  sliced-view assign with stride-2 fast axis) — they are the phase-2
  fall-through canaries required by the brief's "no regression on
  contig-assign cases" + generic-path-preservation requirement.
- First correctness run caught two harness bugs (ndarray layout-preservation
  semantics; an invalid hand-built r2c layout with ldc = n instead of m) —
  both fixed; final gates PASS 129/129 checks under both RUSTFLAGS configs.
- No `proposed.patch` in phase 1 by design; rstsr tree untouched
  (re-verified: `git status --short` empty, HEAD 386948b).

## Reproduce

```bash
cd 2026-09-09-transpose-assign
./reproduce.sh              # everything: ~25-30 min
./reproduce.sh correctness  # gate only (both configs, ~2 min)
./reproduce.sh portable     # criterion suite, portable (~5 min)
./reproduce.sh native       # criterion suite, native (~5 min)
./reproduce.sh portable_c   # A-filter under MALLOC tunables (~2 min)
./reproduce.sh native_c     # A-filter under MALLOC tunables, native (~2 min)
./reproduce.sh perf         # perf stat -d on ta/tb modes (~1 min)
./reproduce.sh candidate    # PHASE 2: gate + suite + perf on the patched tree
```

Requires nightly rustc (via the path dep's rust-toolchain.toml),
`RAYON_NUM_THREADS=16` (set by the script), `perf` for the perf stage.
`target/` left in place for reviewer re-runs; `Cargo.lock` committed.

## Status

PHASE 1 + PHASE 2 COMPLETE. Phase 2 implemented PLAN.md Option A after G1
PASS; D3 PASS; proposed.patch captured and `git apply --check`-verified on a
fresh 386948be checkout; the rstsr working tree is restored clean
(`git status --short` empty, HEAD 386948b). Awaiting G2 review; the main
rstsr repo receives the patch only inside a PR opened with explicit user
permission.

### Raw-data note

`results/<tag>/bench_*.txt` are the complete, authoritative per-bench records
(full criterion estimate brackets per id). Tagged criterion baseline JSONs
(`estimates.json`) are additionally archived for the portable/native stages
(`results/<tag>/criterion/**/<tag>/estimates.json`, partial coverage); the
C-stage JSONs were not retained (txt logs cover them).
