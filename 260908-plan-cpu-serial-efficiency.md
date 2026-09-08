# Plan: rstsr CPU efficiency improvement campaign

- **Date**: 2026-09-08 (planning session, grill-with-docs; user-approved)
- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5`
- **Status**: initial plan; task list is deliberately re-orderable and extensible
- **Audience**: a future **main agent session** (fresh context) that spawns and
  coordinates a code agent and a review agent. This file + the companions below
  are the entire brief — nothing from the planning conversation is assumed.

**Companion documents** (read them first):

- [2026-09-08-plan-prompt/initial-prompt.md](2026-09-08-plan-prompt/initial-prompt.md) — the task owner's original prompt (authoritative intent).
- [2026-09-08-plan-prompt/rstsr-386948b-code-map.md](2026-09-08-plan-prompt/rstsr-386948b-code-map.md) — where every hot op lives at this commit, dispatch machinery, ranked targets. **Read before assigning any task.**
- [2026-09-08-plan-prompt/machine-and-tools.md](2026-09-08-plan-prompt/machine-and-tools.md) — CPU/cache/SIMD facts, conda `torch` env for numpy references, `~/Git-Others` sources, tool availability.
- Repo [AGENTS.md](AGENTS.md) (rules for this repo) and [CONTEXT.md](CONTEXT.md) (glossary; extend it as terms crystallize).

## 1. Mission, scope, non-goals

**Mission**: substantially improve runtime efficiency of rstsr's own CPU code —
serial (`DeviceCpuSerial`) and rstsr-level parallel (`DeviceFaer` path, i.e.
`*_cpu_rayon` kernels) — at commit `386948be`, measured in release mode, with
honest reports whether or not each idea wins.

**In scope**: crates `rstsr-common`, `rstsr-native-impl`, `rstsr-core` (the
kernels live in rstsr-native-impl; core/common are wiring and layout
machinery). Also in scope: rstsr's own `inner_dot_naive_cpu_rayon` (it is
rstsr-native-impl code, not faer's). Structural findings: where code
organization itself caps efficiency, deliver an **edit guide** instead of /
alongside a kernel patch.

**Out of scope**: faer's own gemm/matmul internals; crates-device (BLAS ffi);
rstsr-tblis; GPU; compile-time/binary-size optimization beyond the
`dispatch_simd` gating requirement; integer and half dtypes.

**Non-negotiables from the task owner**:
- Never commit anything in `../` (rstsr, rstsr-agents, rstsr-book). Diffs are
  *proposed* only; the human integrates.
- Test in **release mode** always.
- Cache design budget: assume ≤32 kB L1, ≤256 kB L2, do not truly optimize L3;
  cache-aware blocking only when clearly justified.
- Deep dig with profiling > mindless trying. Be patient; aim for substantial wins.

## 2. Settled decisions (from the grilling session)

| # | Decision |
|---|----------|
| D1 | Scope = rstsr-common + rstsr-native-impl + rstsr-core wiring (see §1). |
| D2 | Op set is open-ended; the T-list below is the starting set, reorderable and extensible at the main agent's judgment (with G1 review). |
| D3 | Diff threshold: memory-bound ops ≥10% on large inputs with no >2–3% regression on small/strided/broadcast cases; vecdot ≥15% or a stated % of peak. Both target configs must hold (D7). 5–10%: report only. <5%: note and move on. Exact numbers adjustable per experiment with README justification. |
| D4 | Experiment crates depend on `../rstsr` via path deps; optimization work temporarily edits the `../rstsr` working tree; the proposed patch is `git -C ../rstsr diff > proposed.patch`. |
| D5 | dtypes: f64 primary everywhere; one secondary representative per experiment — f32 or Complex<f64/f32>, chosen by op relevance. |
| D6 | `dispatch_simd`: TypeId runtime dispatch, f32/f64 only, scalar fallback otherwise; fixed lanes (F64x8 / F32x16 — verify names against the lightweight-simd repo); feature declared on rstsr-native-impl + rstsr-core, **default off**, forwarding like `rayon`; complex support decided at T3. |
| D7 | Target-CPU matrix: exactly two configs — `native` (`RUSTFLAGS=-C target-cpu=native`) and `portable` (default x86-64 SSE2 baseline). Add `x86-64-v3` only if results diverge confusingly. |
| D8 | Bench feature config: rstsr **default features** = the baseline; also record a defaults+`dispatch_dim_layout_iter` secondary column. Diffs are justified against default features. |
| D9 | Sizes: three classes — small ≈L1 (e.g. 64×64 f64), medium L2-resident (512×512), large ≫L2 streaming (2048×2048, ~32 MB) — × {contiguous, strided/transpose-view, broadcast} + one odd size (e.g. 1000×777). Cap runtime at minutes-scale per criterion run. |
| D10 | References: measured triad bandwidth (honest ceiling), ndarray (dev-dep), numpy via `conda activate torch` (einsum for vecdot, `.T.copy()`, `.sum(axis)`). Peak vecdot math: 32 DP FLOP/cycle/core. |
| D11 | Profiling: `perf` (`stat -d`, `record`/`report`); any other toolset at judgment; no sudo installs. |
| D12 | Agents: main (planning/splitting only) + code (all writing) + review (G1/G2 gates, summary relay). See §7. |
| D13 | Naming: experiment dirs `YYYY-MM-DD-<task>`; this plan keeps its `260908-plan-*` name per the original prompt. |
| D14 | Hygiene: delete an experiment's `target/` when it finishes (keep `Cargo.lock`); leave `../rstsr/target/` alone unless disk-pressured (then note it in README). |
| D15 | Commits in this repo: one per experiment dir after G2 passes + this planning commit; always with the `git-commit-coauthor` trailer block. |

## 3. Benchmark standards (the measurement contract)

Every experiment adheres to:

1. **Release mode** (`cargo bench` / `cargo run --release`), profile settings
   stated in the README (default cargo release profile unless justified).
2. **Correctness gate before any perf claim**: result compared to a scalar
   reference within tolerance, including broadcast (zero-stride) and odd-size
   cases. A fast-but-wrong kernel is an automatic fail at G2.
3. **Anti-cheat**: `std::hint::black_box` / criterion handling on inputs and
   outputs; allocation policy identical between baseline and candidate
   (either timed for both or excluded for both — state which).
4. **Both target configs** (D7) via explicit RUSTFLAGS in the reproduce script;
   the script is self-contained (sets RUSTFLAGS, RAYON_NUM_THREADS=16, conda
   activation where needed) so README copy-paste reproduces the numbers.
5. **Both devices** where relevant: `DeviceCpuSerial` explicitly, and default
   device (DeviceFaer/rayon, RAYON_NUM_THREADS=16).
6. **Streaming honesty**: large benches use sizes ≫L2 and/or rotate input
   buffers so L2 residency doesn't flatter the number; state the choice.
7. **Statistics**: criterion default sampling unless a bench is unstable;
   record `rustc --version` and note nightly drift if any.
8. **Efficiency framing**: report GB/s vs triad for streaming ops, FLOP/s vs
   analytic peak for vecdot, plus the relative % vs baseline. Include `perf
   stat -d` evidence (bandwidth, IPC, cache misses) for each win/loss
   conclusion.

## 4. Repo, dependency & hygiene workflow

Each experiment directory is a **standalone crate** (no repo-root workspace;
dirs stay independent per AGENTS.md):

```toml
[dependencies]
rstsr-core = { path = "../../rstsr/rstsr-core", default-features = false,
               features = ["row_major", "aligned_alloc", "faer", "faer_as_default"] }
# + dev-deps: criterion 0.5 (soft default), ndarray, etc. as needed
```

(Mirror default features explicitly so the bench config is visible in the
manifest. Add `"dispatch_dim_layout_iter"` for the D8 secondary column.)

Per-experiment lifecycle:

1. **Start**: review agent verifies `git -C ../rstsr status` is clean and HEAD
   is `386948be` (or a documented successor base if the owner says so).
2. **Baseline** numbers taken on the unmodified tree, stored in the README.
3. **Patch phase**: code agent edits `../rstsr` in place; experiment crate
   path-deps pick changes up on rebuild.
4. **Finish**: `git -C ../rstsr diff > <exp-dir>/proposed.patch`; then
   `git -C ../rstsr checkout -- .` to reset for the next experiment; delete the
   experiment's `target/`; commit the dir in this repo (D15).
5. `../rstsr/target/` persists across experiments (D14).

## 5. `dispatch_simd` design constraints (settled D6)

- Cargo feature `dispatch_simd` on **rstsr-native-impl** and **rstsr-core**,
  default off, feature-forwarding mirroring `rayon`; adds the
  `lightweight-simd` dependency (crates.io ≥0.1.1).
- Runtime dtype dispatch by TypeId over f32/f64 following the
  `gemm_faer_ix2_dispatch` pattern (`rstsr-core/src/device_faer/matmul.rs`);
  all other dtypes take the existing scalar path.
- Lanes are fixed arrays regardless of target CPU (lightweight-simd is
  auto-vectorization, not intrinsics): F64x8 / F32x16 class types — confirm
  exact names/API from github.com/ajz34/lightweight-simd-rs (no docs.rs yet).
- Order of attack per kernel: **plain fixed-size batching first** (chunks /
  local accumulators / `chunks_exact`); reach for lightweight-simd only if
  batching alone doesn't meet D3 thresholds. Many kernels won't need it even
  under `native`.
- Propose the feature in a patch only if D3 thresholds hold in **both** target
  configs; otherwise the experiment reports honestly and the feature stays
  out. Binary-size/compile-time cost is part of the honest report.
- Complex support: decide at T3 with evidence; not in v1 dispatch.
- If `dispatch_simd` ends up proposed for rstsr, that proposal is a candidate
  ADR in the rstsr repo (the human's call) — note it in the experiment README.

## 6. Task sequence

Starting order T0→T5; the main agent may reorder after T0 evidence and add
T6+ (any new op, structure, or follow-up) with G1 review. Every task gets its
own `YYYY-MM-DD-<task>` directory (date = day work started).

### T0 — `bench-harness-baseline` (no patch)
- **Goal**: reusable harness pattern + baseline table + profile for all
  candidate ops; triad bandwidth microbench; numpy/ndarray anchors.
- **Do**: standalone criterion crate covering: transpose copy (`.t().to_contig`),
  2-D sum_axis(0)/sum_axis(1) + full sum, vecdot (1-D dot + batched),
  elementwise add (contig + broadcast + strided), fill/zeros, argmax; standard
  size/dtype/config matrix (D5, D7, D8, D9); `perf stat` pass on the large
  cases; numpy reference numbers via conda env.
- **Deliver**: README with baseline tables + perf evidence + the reusable
  harness snippet later dirs copy; no patch.
- **Accept**: stable numbers, correctness gates pass, profiling summary that
  either confirms or re-ranks the targets in the code map.

### T1 — `transpose-assign`
- **Files**: `rstsr-native-impl/src/cpu_serial/{assignment.rs,transpose.rs}` + rayon twins; wiring in `rstsr-core/src/tensor/manipulation/to_layout.rs`.
- **Hypotheses**: (a) route 2-D orderchange through the existing-but-unused
  64×64 blocked kernel (`orderchange_out_r2c_ix2_*`) inside `change_layout_f`;
  (b) contiguous assign via slice copy instead of per-element iterators;
  (c) tiled/blocked generic strided assign; (d) `dispatch_simd` lane variant
  for small/odd rows.
- **Bench**: transpose copy on the D9 matrix; GB/s vs triad; OpenBLAS/blis
  transpose routines as design references (`~/Git-Others`).
- **Accept**: D3 (memory-bound class).

### T2 — `reductions`
- **Files**: `rstsr-native-impl/src/cpu_{serial,rayon}/reduction.rs`.
- **Hypotheses**: (a) sum_axis(0)/axis(1) accumulate into local fixed buffers
  or lane accumulators instead of clone-through-closure CHUNK=48; (b) write in
  output order to kill the final order-fixup full copy; (c) verify
  `unrolled_reduce` autovectorization quality under both configs before adding
  anything; (d) argmin/argmax contiguous fast path with index tracking.
- **Bench**: 2-D sum axis 0/1, full sum/mean/var/norm spot checks, argmax on
  1e6/1e7; ndarray + numpy anchors.
- **Accept**: D3.

### T3 — `vecdot`
- **Files**: `rstsr-native-impl/src/cpu_{serial,rayon}/vecdot.rs`, `cpu_rayon/matmul_naive.rs` (`inner_dot`).
- **Hypotheses**: (a) per-output local accumulator killing the MaybeUninit
  read-modify-write in the contiguous-remaining branch; (b) lane accumulators
  (`dispatch_simd`) for contiguous contraction; (c) `inner_dot` routed through
  the contiguous unrolled-binary-reduce path; (d) stride-based inner loop for
  the general branch.
- **Bench**: 1-D dot {1e3, 1e5, 1e7}; batched vecdot (outer × contraction);
  secondary representative = Complex<f64> (D5); numpy `einsum('i,i->',…)` and
  `dot` anchors; % of 32 DP FLOP/cycle/core.
- **Accept**: D3 (compute-bound class: ≥15% or stated % of peak).
- **Decide here**: complex in `dispatch_simd` v1 (D6).

### T4 — `elementwise`
- **Files**: `rstsr-native-impl/src/cpu_{serial,rayon}/op_with_func.rs`; closure layer `rstsr-core/src/device_cpu_serial/operators/*` and `device_faer/rayon_auto_impl/*`.
- **Hypotheses**: (a) contiguous branch → `chunks_exact` slice zips (kill
  index arithmetic + bounds checks); (b) reduce `MaybeUninit`/closure overhead
  for the common ops (direct kernels or closures over `&mut T`); (c) lane
  variant behind `dispatch_simd`; (d) strided-branch blocking.
- **Bench**: `c = a + b` (contig / broadcast / strided) on the D9 matrix; GB/s
  vs triad; profiling check that closures actually inline.
- **Accept**: D3. Highest-traffic kernel in the library — most invasive diff,
  hence later in the sequence.

### T5 — `fill-misc-structural`
- **Scope**: `fill_promote` slice-fill; creation paths; argmin leftovers;
  anything T0–T4 flagged but not finished. Plus: consolidate structural
  findings into **edit guides** (e.g. the dead `feature_rayon/auto_impl` copy,
  serial/rayon kernel duplication vs shared blocking helpers, MaybeUninit
  closure API ergonomics for the operator layer).

### Final phase — bookkeeping diffs (only on real wins)
- If user-facing features land in proposed patches: draft diffs for
  **rstsr-book** (feature-flag docs, e.g. the installation page's backend
  guidance) and **rstsr-agents** (benchmark conventions skill) — each in its
  own separate directory here, never applied.

## 7. Agent workflow & gates

- **Main agent** (fresh session, briefed by this file): splits work into
  tasks, assigns to the code agent, never writes code itself; keeps token
  usage low by working from review-agent summaries, not raw dumps.
- **Code agent**: per task — detailed plan → implementation in `../rstsr` +
  experiment crate → run benches → write README (+ proposed.patch or honest
  negative) → hand to review.
- **Review agent**, two gates per task:
  - **G1 (before implementation)**: task plan vs this plan's scope/standards;
    catch drift (wrong op, wrong config, threshold games).
  - **G2 (before acceptance)**: correctness gate really ran; benches measure
    the intended op (e.g. transpose copy, not the view; allocation policy
    consistent; black_box present; streaming sizes honest); both configs and
    both devices present; README complete; patch applies cleanly to a fresh
    `386948be` checkout; no `../` commits. Then relay a compact summary to main.
- Cadence: sequential tasks (each needs a clean `../rstsr` tree); commit per
  task dir after G2 (D15); memory + CONTEXT.md updated as durable facts and
  terms crystallize.

## 8. Deliverables & reporting format

Each experiment directory contains:

1. `README.md` — rstsr base commit; environment (CPU, `rustc --version`,
   RUSTFLAGS config, RAYON_NUM_THREADS, criterion version); baseline table;
   post-change table (both configs, both devices, efficiency framing §3.8);
   conclusion (win/loss/honest-negative + why, with perf evidence); decisions
   deviating from this plan and why.
2. `reproduce.sh` (or equivalent) — self-contained: env vars, RUSTFLAGS,
   conda activation, exact commands.
3. `proposed.patch` — **only if** D3 thresholds are met; otherwise explicitly
   absent with the README saying so (honest negatives are valuable).
4. Source: the standalone bench crate (committed with `Cargo.lock`,
   `target/` removed).

## 9. Pitfalls checklist (bench-writing time)

- `.t()` is a view — force the copy in transpose benches.
- Zero-stride broadcast inputs must appear in correctness cases.
- Don't let LLVM const-fold or DCE the bench body; black_box everything.
- Same allocation policy baseline vs candidate; note which.
- Odd sizes catch lane-tail and block-edge bugs.
- criterion + rayon: one shared pool; don't nest; state thread counts.
- Nightly is unpinned — record the compiler version per README.
- row_major is the benched layout config (mirrors rstsr defaults).
- `git -C ../rstsr status` clean before baselining.

## 10. Open items (resolve during execution)

- Exact lightweight-simd type names / ops (read the GitHub repo at T0/T1).
- Whether the T0 profile re-ranks the task order (main agent + G1 call).
- Complex in `dispatch_simd` (decide at T3).
- `x86-64-v3` third config (only on confusing divergence, D7).
- Whether numpy anchors graduate from "context numbers" to "formal reference
  columns" (T0 outcome).
- ADR candidacy for `dispatch_simd` in rstsr (if proposed).
