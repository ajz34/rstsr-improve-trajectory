# T5 — fill-misc-structural (PHASE 1: fill study + campaign consolidation)

Campaign task T5, the consolidation task. **PHASE 1 ONLY** — this phase adds
**no patch to rstsr** and ends at the G1 review gate. Two halves:

1. **Half 1 — fill/creation kernel study** (bench crate in this directory):
   baseline benches, kernel-vs-rider split, patch-design decision.
   **Verdict: honest-skip** — no `proposed.patch` (design record:
   [PLAN.md](PLAN.md), numbers: [results/tables.md](results/tables.md)).
2. **Half 2 — campaign consolidation** (documents in this directory):
   - [EDIT-GUIDE.md](EDIT-GUIDE.md) — campaign-wide structural edit guide
     (dispatch_simd ADR-candidate text; code-map corrections incl. the
     symlink finding; serial/rayon duplication assessment; MaybeUninit
     closure ergonomics; reductions reuse gap; THP alignment blocker;
     leftovers).
   - [BUG-NOTES.md](BUG-NOTES.md) — upstream issues for the human
     (stride-0 reduce mis-multiply; inner_dot uninit-beta hazard both
     shapes; D8 f32-vecdot 0.57× caveat; numpy `.T.copy()` anomaly;
     smaller flags).
   - [RECONCILIATION.md](RECONCILIATION.md) — plan-vs-outcome notes for the
     final campaign summary (which T1–T5 hypotheses hit/missed/moot).

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5` (path
  dep `../../rstsr/rstsr-core`; verified `git status` empty + HEAD
  `386948be` before baselining, **never edited**, re-verified at the end).
- Campaign plan §3 contract:
  [../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
  (T5 §6 + §5 dispatch_simd + §7 item 8); code map
  [../2026-09-08-plan-prompt/rstsr-386948b-code-map.md](../2026-09-08-plan-prompt/rstsr-386948b-code-map.md)
  (§2(e) **corrected here**, see below).
- Context: T7 study + EDIT-GUIDE
  ([../2026-09-09-alloc-pagefault-study/](../2026-09-09-alloc-pagefault-study/README.md)),
  the five accepted experiment READMEs (argmax-argmin, elementwise,
  transpose-assign, reductions, vecdot), T0 harness
  ([../2026-09-09-bench-harness-baseline/](../2026-09-09-bench-harness-baseline/README.md)).

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores, AVX-512; 48 kB L1d / 1 MB L2 / 128 MiB X3D L3 |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)`, **stable** — the rustup DEFAULT. The rstsr path dep's `rust-toolchain.toml` (nightly) does NOT apply to this crate: rustup resolves the toolchain from the invoking directory tree only, not along cargo path deps (evidence: `results/toolchain_identity.txt`; gate-time record: `results/rustc_version.txt`). Toolchain identity is unpinned across the campaign's crates — treat cross-experiment absolute comparisons as drift-prone (EDIT-GUIDE g.7) |
| criterion | 0.5.1 (2 s + 0.7 s warm-up, 100 samples) |
| perf | 7.0.14 |
| RAYON_NUM_THREADS | 16 (asserted by the harness for every faer run) |
| Target configs | `portable` (no RUSTFLAGS) and `native` (`-C target-cpu=native`) |
| Cargo profile | stock release defaults (explicit in Cargo.toml) |
| Allocation policy | creation = alloc-inclusive (it IS the op); reuse fill = pre-allocated pre-warmed tensor mutated in place (kernel-only); bound = pre-warmed `Vec` raw loop; identical across devices/configs per variant |

## Directory layout

```
Cargo.toml / src/lib.rs   standalone crate (T0 harness pattern); rstsr-core path
                          dep, default features mirrored explicitly
benches/fill.rs           creation (full/zeros/ones × 4 sizes × f64 + f32 spot),
                          reuse_fill, fill_bound, strided_fill (diag + stride-2);
                          serial + faer16
examples/correctness.rs   41-check gate vs naive refs: creation values, c.fill
                          re-fill/fill(0.), device-level fills on diag /
                          stride-2 / broadcast layouts; f64+f32; both devices
examples/profile_ops.rs   fixed-iteration runner for perf stat (11 modes incl.
                          medium/odd zeros/full regime runs)
reproduce.sh              correctness → portable → native → perf → tunables
results/                  raw logs, tables.md, perf/ + derived_summary.txt,
                          tunables/, correctness_{portable,native}.txt
PLAN.md                   fill patch decision = honest-skip (options + evidence)
EDIT-GUIDE.md / BUG-NOTES.md / RECONCILIATION.md   half-2 consolidation docs
```

## Half-1 headline results (full tables: results/tables.md)

**Call-chain discovery (corrects the code map):** `rt::full`/`ones`/`zeros`
do NOT use `fill_promote_cpu_serial` — they go through device
`full_impl`/`ones_impl`/`zeros_impl` → `vec![fill; len]`
(`device_cpu_serial/creation.rs:17-75`). The rstsr fill kernel serves only
`c.fill(v)` (contiguous, since owned tensors are contiguous), `eye`'s
diagonal (2048 elems, 0.7 µs), and BLAS beta-zeroing.

Kernel-vs-rider split at 2048² f64 (native):

| measurement | value |
|---|---|
| `rt::full` (alloc-inclusive) | 4.65 ms criterion; perf: 8193 faults/iter, 90 % sys (T7 reproduced) |
| `rt::full` + T7 MALLOC tunables | **0.62 ms** (~0 steady faults) — 7.5×, beats numpy's allocating fill (1.19 ms) |
| `c.fill(v)` reuse (the kernel) | 507 µs serial / 530 µs faer16 (66 GB/s write; 0.29 ins/elem) |
| raw slice-fill bound | 544 µs |
| `rt::zeros` | 2.3 µs (calloc-lazy at ≥32 MiB) |

**Fill kernel verdict: honest-skip.** The contiguous clone-per-element loop
already compiles to a vectorized broadcast store loop and ties/beats the
raw bound at every size ≥ medium (D3 <5 % class in BOTH configs);
`slice::fill` would be code clarity only. faer16 fill never beats serial
(write bandwidth saturates per-core; 18 % slower at medium). The strided
branch is ~2× slower than it could be but has no tensor-API surface. The
88 % rider on `rt::full` is T7's documented domain; the broadcast kernel
behind `vec![v; n]` is at the same write floor as `c.fill`.

New nuance found (refines T0/T7): **`rt::zeros` laziness is
size-regime dependent** — calloc-lazy zero pages only at ≥ ~32 MiB outputs;
at 2 MiB calloc memsets explicitly (ties `full`); at 6.2 MiB glibc's
non-temporal memset runs at DRAM speed, making **zeros 2.4× slower than
full** (107 vs 45 µs). Absolute cost tiny → "leave zeros alone" stands.

Correctness gate: 41 checks ALL PASS under BOTH RUSTFLAGS configs and both
devices (`results/correctness_{portable,native}.txt`), including device-level
fills on diagonal, stride-2, and zero-stride broadcast layouts.

## Deviations / decisions

- **No `proposed.patch`** (half 1 verdict) — recorded with the full option
  analysis in [PLAN.md](PLAN.md), including what would change the verdict.
- Criterion is the authoritative timing source; single `perf stat` runs of
  the 0.4–0.6 ms streaming-store loops proved frequency/order-sensitive, so
  perf is used for the qualitative split (faults, sys share, ins/elem) per
  the T7 method (`results/perf/derived_summary.txt` documents one observed
  swing).
- The medium/odd zeros-vs-full regime investigation went beyond the brief
  (2 profile modes + 4 perf runs) — it sharpened an honest finding into a
  documentation-grade statement.
- Toolchain identity: an earlier draft of this README claimed T5 ran
  1.99.0-nightly — that sighting came from running `rustc` inside the rstsr
  tree, whose `rust-toolchain.toml` overrides nightly there only; this crate
  actually built with the rustup default, stable 1.97.1 (committed evidence:
  `results/toolchain_identity.txt`). No drift was observed; identity is
  simply unpinned, so cross-experiment absolutes stay drift-prone.
- `target/` left in place for reviewer re-runs (D14 deviation, same note as
  T1'/T4'); `Cargo.lock` is committed.

## Reproduce

```bash
cd 2026-09-09-fill-misc-structural
./reproduce.sh              # everything: correctness → portable → native → perf → tunables (~15 min)
./reproduce.sh correctness  # 41-check gate, both configs (~2 min)
./reproduce.sh portable     # criterion suite, portable (~4 min)
./reproduce.sh native       # criterion suite, native (~4 min)
./reproduce.sh perf         # perf stat on the 11 profile modes (native build)
./reproduce.sh tunables     # rt::full with/without the T7 MALLOC env pair
```

Requires: any recent rustc — the crate builds with the rustup default
(stable 1.97.1 on this machine); the path dep's `rust-toolchain.toml` does
not apply to this directory (`results/toolchain_identity.txt`),
`RAYON_NUM_THREADS=16` (set by the script), `perf` for the
perf/tunables stages.

## Phase status

- **Phase 1 COMPLETE** (this directory): fill baselines + rider split +
  honest-skip decision + consolidation documents. `../rstsr` verified clean
  at HEAD `386948be` (never touched).
- **Phase 2** (after G1 review): only if the review overturns the
  honest-skip (a fill patch would then follow the T1'/T4' protocol:
  patch tree → re-gate → candidate benches → proposed.patch → restore).
  Otherwise phase 2 is the final campaign-summary/bookkeeping pass, which
  per the plan happens in separate directories with owner permission.
