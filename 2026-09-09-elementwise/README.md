# T4' — elementwise (PHASE 1 baseline + PHASE 2 candidate implemented)

Campaign task T4' (elementwise kernel family: add/sub/mul/div/scale/map).
**Phase 1**: baseline on the CLEAN rstsr tree, kernel-structure
investigation, perf evidence, and the phase-2 edit plan ([PLAN.md](PLAN.md)).
**Phase 2** (below): implemented the PLAN.md Option-A blocked 2-D strided
kernel in `../rstsr`, re-gated, re-benched, and captured
[proposed.patch](proposed.patch). D3 **PASS**; the rstsr working tree was
reset to clean after the patch capture.

- **rstsr base commit**: `386948be819baa334b8da02232f3a1944e5447d5` (path dep
  `../../rstsr/rstsr-core`; verified clean + HEAD 386948b before baselining,
  before patching, and restored to clean after `proposed.patch` capture).
- Campaign plan §3 contract:
  [../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md](../2026-09-08-plan-prompt/260908-plan-cpu-serial-efficiency.md)
- T0 context: [../2026-09-09-bench-harness-baseline/README.md](../2026-09-09-bench-harness-baseline/README.md)
- T7 context: [../2026-09-09-alloc-pagefault-study/README.md](../2026-09-09-alloc-pagefault-study/README.md)
  (+ EDIT-GUIDE.md; its carry-forwards are implemented here — see §Reuse
  below).

## Environment

| item | value |
|---|---|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16 cores, AVX-512; 48 kB L1d / 1 MB L2 / 128 MiB X3D L3 |
| OS / kernel | Linux 7.0.0-31-generic x86_64 |
| rustc | `rustc 1.97.1 (8bab26f4f 2026-07-14)` (nightly via rstsr's rust-toolchain.toml) |
| criterion | 0.5.1 (2 s + 0.7 s warm-up per bench) |
| ndarray | 0.16.1 (anchors) |
| perf | 7.0.14 |
| RAYON_NUM_THREADS | 16 (asserted by the harness for every faer run) |
| Target configs | `portable` (no RUSTFLAGS, SSE2 baseline) and `native` (`-C target-cpu=native`) |
| Cargo profile | stock release defaults (explicit in Cargo.toml) |
| Allocation policy | A = allocation included; B = pre-allocated, pre-warmed output (kernel-only); C = A under glibc env tunables. Identical across devices/configs per variant. |

## Directory layout

```
Cargo.toml / src/lib.rs    standalone crate (T0 harness pattern; A/B/C protocol baked in)
benches/elementwise.rs     add contig (small/medium/large/odd) + broadcast + strided
                           (large/odd/small) + strided-first + add-assign + mul + scale;
                           variants A/B; f64 + f32; both devices
benches/anchors_ndarray.rs ndarray owned (alloc incl.) + zip_prealloc (kernel bound)
benches/kernels_probe.rs   local loop-shape probes (contig codegen + strided patterns)
examples/correctness.rs    gate: naive + ndarray refs; odd/zero-stride/strided/flip views/
                           reuse/i32/complex; f64+f32; both devices (BOTH RUSTFLAGS configs)
examples/profile_ops.rs    fixed-iteration runner for perf stat (add_b/add_strided_b/add_a)
results/                   raw logs + criterion JSONs + tables.md + perf/ (derived summary);
                           pre2_* = extended clean-tree baseline; candidate_* = patched tree
reproduce.sh               correctness -> portable -> native -> portable_c -> native_c -> perf;
                           pre2 (clean-tree extended baseline); candidate (patched-tree pass)
PLAN.md                    phase-2 edit plan (design options, accept criteria)
proposed.patch             phase-2 diff vs 386948be (captured; tree restored after)
review-260914.md           owner review: verdict CORRECT, tall-skinny caveat, dispositions
results/review260914/      review test patch + snippets + re-run instructions
```

## Phase-2 result (blocked 2-D strided kernel) — D3 PASS

Patch: [proposed.patch](proposed.patch) (+346 lines, 2 files:
`rstsr-native-impl/src/cpu_{serial,rayon}/op_with_func.rs`). A blocked
[TILE=64, TILE=64] fast path replaces the layout-iterator branch when, after
greedy translation, the layouts have no common f-contiguous prefix and are
2-D, size ≥ 4096, and free of broadcast (shape>1 & stride=0) axes in **all
participating layouts** (zero-stride/broadcast stays on the iterator path).
Applied to `op_mutc_refa_refb_func` and `op_muta_refb_func` (serial + rayon
twins). TILE offsets are accumulated in `isize` and cast to `usize` only at
the indexing site, with per-element `debug_assert` bounds checks.

Full before/after tables: [results/tables.md](results/tables.md)
(phase-2 section; baseline = `results/pre2_*`, candidate =
`results/candidate_*`). Headline, reuse variant B (primary judge):

| case (2048×2048 f64) | before | after | speedup |
|---|---|---|---|
| strided B serial native | 23.10 ms | 7.51 ms | **3.08×** |
| strided B serial portable | 22.96 ms | 7.62 ms | **3.01×** |
| strided B faer16 native | 2.14 ms | 0.96 ms | **2.23×** |
| strided B faer16 portable | 2.17 ms | 1.05 ms | **2.07×** |
| a.t()+b B serial native | 23.01 ms | 7.52 ms | 3.06× |
| c += bᵀ B serial native | 17.28 ms | 6.17 ms | 2.80× |
| strided A (idiomatic) serial native | 28.83 ms | 12.71 ms | 2.27× |
| strided odd 1000×777 B serial native | 3.69 ms | 0.43 ms | **8.7×** |
| strided small 64×64 B serial native | 20.7 µs | 2.9 µs | **7.1×** |
| strided small B faer16 native | 89.5 µs | 5.7 µs | **15.7×** |

Idiomatic composing (C variant): strided A under the T7 MALLOC tunables =
**7.06 ms = 4.0×** for allocating user code.

- **D3 (≥10% on large reuse-variant in BOTH configs): PASS** with ~30×
  margin (3.0× serial both configs; 2.1-2.2× faer16 both configs). The
  PLAN.md strided target "beat the 17.2 ms T7 emulated bound" is met:
  7.5 ms — blocking removes the 16 KiB hop pattern that the naive bound
  still pays.
- **perf stat** (strided B, serial native, before→after): ins/elem
  **161.6 → 19.0**, cyc/elem 31.2 → 10.4, GB/s 4.2 → 13.1
  (results/perf/perf_add_strided_b_serial{,_after}.txt).
- **Gates**: all serial gates within ±3% (contig small/medium/large, bcast,
  mul, scale; worst −2.7% contig odd). Two cells exceed the band in single
  runs and are documented as build-layout/run noise with multi-run samples
  in results/tables.md: portable contig-odd B (the CLEAN tree itself spans
  124.6–156.5 µs across rebuilds with A anti-correlated; A+B sum invariant;
  native stable) and portable f32-large B (+3-6% on one monomorphization,
  native −5.9%). faer16 sub-ms rows have ±4-6% run spread; medians within
  ±3%.
- **dispatch_simd: skipped** (PLAN.md §5): contig already auto-vectorizes
  (0.83 ins/elem), strided is a gather no fixed-lane SIMD helps; plain
  blocked indexing achieved 3× without any feature/dependency cost.

### PR-notes (visit order) — for the eventual rstsr PR text

The serial strided path now visits elements in tiled order instead of strict
layout order. This is **bit-identical for stateless per-element ops** (every
operator rstsr ships: each output element is written exactly once from one
input tuple, no cross-element dependency, no FP re-association). A
*stateful* user `FnMut` closure passed to the low-level `op_mutc_refa_refb_func`
driver would observe the reordered visit sequence (the rayon path was always
order-free). The same note lives in the patch's doc comments
(`blocked_2d_iter!` / `blocked_2d_3layouts_cpu_rayon`).

### Phase-2 deviations / notes

- `op_muta_refb_func` twin included: the clean-tree measurement
  (17.28 ms serial for `c += bᵀ`, same hop pattern, probe bound 6.8 ms)
  justified it up front; candidate confirms 2.80×. The numb/numa drivers
  were left untouched (no measured use case in the matrix; scalar-operand
  strided ops can follow the same recipe if a need appears).
- New benches vs the G1 list: `a.t() + b` (strided-first operand) and
  `c += bᵀ` rows, strided small/odd regression gates. New correctness
  fixtures: `flip(0)`, `flip(0)+flip(1)`, flip+transpose reuse (negative
  strides through the tile path), i32 and Complex<f64> add/mul spots (alloc
  + reuse, both devices). Perf spots for the new rows skipped (optional per
  G1).
- The layout-lottery investigation (clean-tree stash roundtrip) is committed
  as raw evidence in the tables; it resolves both >3% gate cells as
  build-layout effects, not kernel effects.

## Phase-1 findings (evidence base for phase 2)

1. **The contiguous elementwise kernel is already vectorized and
   memory-bound — the T0 "12.9 ins/elem, no autovectorization" reading was
   an artifact of profiling the allocating variant.** Reuse-variant add
   2048²: **0.83 ins/elem**, IPC 0.27, 40.3% L1d-miss, 43.7 GB/s (native,
   perf stat; `vaddpd %zmm` in the binary). The allocating variant A
   reproduces T0's 12.79 ins/elem — its instruction stream is page-fault +
   allocator work (8358 faults/iter), not kernel work. Kernel-shape probes
   (index loop vs zip vs chunks_exact, MaybeUninit closure vs direct) all tie
   at both configs, and rstsr B ties ndarray `zip_prealloc` (2.103 vs
   2.132 ms native large; +3% at medium). **No contig codegen change is
   justified** (hypotheses a/b of the plan are moot for contig).
2. **The strided branch (`a + b.t()`) is the real elementwise target: 23.2 ms
   reuse (4.3 GB/s), 161.6 ins/elem at IPC 5.17.** Two stacked causes: a
   16 KiB-hop access pattern after the greedy translation (raw-loop bound
   16.6 ms) and ~6.2 ms of `IterLayoutColMajor` machinery. A 64×64 blocked
   probe kernel runs **6.8 ms = 3.4×** (probe only; phase 2 ports this into
   `op_with_func.rs` serial + rayon twins). Beats the 17.2 ms T7 bound
   because blocking removes the hop pattern itself.
3. **Broadcast reuse is already fast** (1.49 ms serial native; row L1-hot) —
   it routes through the contig branch (chunks of 2048) and needs nothing.
4. **dispatch_simd: not needed for elementwise** (PLAN.md §5). Plain loops
   auto-vectorize under `native` (medium add: 49.5 → 40.8 µs = 1.21× ISA
   effect with identical code); fixed-lane SIMD would add cost for no gain.
5. **Variant C reconfirms T7**: `MALLOC_MMAP_THRESHOLD_=67108864
   MALLOC_TRIM_THRESHOLD_=134217728` recovers the large-class rider (add A
   6.23 → 2.14 ms native, tying B 2.10); small/medium/odd unchanged.
6. **f32 has no large-class rider** (16 MiB outputs < the 32 MiB glibc cap;
   A ≈ B at 740-782 µs serial), so f32 A is already kernel-faithful.

## Baseline tables

Full tables: [results/tables.md](results/tables.md). Extract (native, reuse
variant B, large 2048×2048 f64):

| case | serial B | nominal GB/s | faer16 B | ndarray zip_prealloc |
|---|---|---|---|---|
| add contig | 2.103 ms | 47.9 | 562 µs (179) | 2.132 ms |
| add broadcast row | 1.485 ms | 67.8 (b L1-hot) | 268 µs | — |
| add strided (bᵀ) | **23.20 ms** | **4.3** | 2.136 ms (47.1) | — |
| mul contig | 2.132 ms | 47.2 | 597 µs | — |
| scale (a·2) | 1.528 ms | 43.9 (16 B/elem) | 278 µs | — |

Gates (to protect in phase 2, reuse-variant, native): small 1.29 µs, medium
40.8 µs, odd 119.1 µs, bcast 1.485 ms — none touched by the planned
strided-branch edit (different code path).

## perf evidence (native, serial; results/perf/)

Derived: [results/perf/derived_summary.txt](results/perf/derived_summary.txt);
raw `perf stat -d` outputs alongside.

| op | ms/iter | GB/s | ins/elem | cyc/elem | IPC | L1d-miss% | faults/iter |
|---|---|---|---|---|---|---|---|
| add contig B | 2.21 | 43.7 | **0.83** | 3.09 | 0.27 | 40.3 | 0 (82 startup) |
| add strided B | 23.95 | 4.2 | **161.6** | 31.2 | 5.17 | 2.1 | 0 (247 startup) |
| add contig A | 6.47 | 15.1 | 12.79 | 9.04 | 1.42 | 8.1 | 8358 |

Contig B is memory-stalled (low IPC, huge L1d miss rate, minimal
instructions — vectorized streaming). Strided B is an instruction mountain
at a terrible access pattern. A is T7's rider reproduced exactly.

## Call-chain anatomy (`&a + &b`, at 386948be)

```
&TensorAny + &TensorAny
  -> TensorOpAPI::op_f              rstsr-core/src/tensor/operators/op_binary_arithmetic.rs:283
     broadcast_layout, get_layout_for_binary_op, device.uninit_impl (alloc),
     device.op_mutc_refa_refb
  -> OpAPI<TA,TB,TC,D> for Device   device_cpu_serial/operators/op_ternary_arithmetic.rs:16
     (DeviceFaer: feature_rayon/auto_impl/op_ternary_arithmetic.rs:16 -> rayon kernel)
     closure  |c, a, b| c.write(a.clone() + b.clone())   [concrete, static dispatch]
  -> Op_MutC_RefA_RefB_API::op_mutc_refa_refb_func
     device_cpu_serial/operators/op_with_func.rs:17 -> native kernel
  -> op_mutc_refa_refb_func_cpu_serial
     rstsr-native-impl/src/cpu_serial/op_with_func.rs:10
     translate_to_col_major(K) + translate_to_col_major_with_contig
     (rstsr-common/src/layout/rearrangement.rs:233,298; greedy_layout :36)
     size_contig >= 16: layout_col_major_dim_dispatch_3 + index loop  (contig/bcast)
     else:             izip of 3 IterLayoutColMajor per element       (strided)
```

Per-element cost (contig, after inlining — confirmed by measurement): one
vectorized load-add-store chain; the index arithmetic, bounds checks,
MaybeUninit read/write and `T::clone()` all compile away. Per-element cost
(strided): three `IterLayoutColMajor::next` (end-check + multi-axis index
update + `try_into().unwrap()`, `rstsr-common/src/layout/iterator.rs:290`) +
closure call + idx math = 161.6 ins/elem.

## Reuse-idiom / env note (fold-in for the eventual rstsr-book diff, per T7)

Works today at 386948be, no code change; cited by
`../2026-09-09-alloc-pagefault-study/EDIT-GUIDE.md` Option 1:

- Single-pass reuse into a pre-allocated output:
  `op_mutc_refa_refb_func(&mut c, &a, &b, &mut |c, a, b| c.write(a + b))`
  (`rstsr_core::tensor::operators::op_with_func`) — 66-77% faster than
  `&a + &b` at 2048² (this study: 6.23 -> 2.10 ms serial native, 2.64 ->
  0.56 ms faer16). Warm the output once (`c.fill(0.)`) outside timing;
  `rt::zeros` is calloc-lazy by design — never "fix" it into a touching fill.
- For ≥32 MiB transient outputs on glibc:
  `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`
  recovers ~96-97% of the rider for *allocating* code (this study's C
  variant reproduces T7 exactly). The tempting 32 MiB value does not work
  (chunk > threshold still mmaps); RSS grows; per-job setting, not a library
  default.

## Deviations from the brief

- **Strided operand**: benched as `a + bt.t()` with independent `bt`
  (square 2048², so identical layout/cost to `a + a.t()`); correctness uses
  `bt` shaped `[n, m]` so the odd-size case is expressible (T0's note).
- **Scale variant B**: no public tensor-level reuse wrapper exists for the
  numa-refb kernel (`a * 2.0`) at 386948be, so B is expressed as a `[1,n]`
  row of exactly 2.0 through the refa-refb driver (same loop structure as
  the broadcast add). Documented in the bench file and results/tables.md.
- **Variant C is an env re-run of the A filter** (`reproduce.sh` stages
  `portable_c`/`native_c`), not a separate Rust variant — same binary, same
  ids, glibc tunables in the environment.
- **f32 secondary** chosen (over Complex) because f32 isolates the
  rider/kernel split at the 16 MiB boundary (no rider) — justifying D5
  choice; complex arithmetic adds no new iteration-path coverage for this
  driver family (generic path), and phase-2 correctness will still add
  i32/Complex spots.
- **Probe benches** (`benches/kernels_probe.rs`) go beyond the required
  matrix; they are the load-bearing evidence for the codegen claims and the
  strided design choice. blocked128 kept for the TILE sweep reference.
- No `results/make_tables.py` (T0/T7 had one); tables are hand-maintained in
  `results/tables.md` from the committed raw logs for this smaller matrix.
- Phase-1 criterion runs use `--filter`-free full suites for
  portable/native and an `"A"` filter only for the C stages.

## Status & next step

Phase 1 (baseline + investigation + PLAN.md) and phase 2 (Option-A blocked
strided kernel, correctness + benches + perf + proposed.patch) are complete.
`git -C ../rstsr diff > proposed.patch` captured after all runs; the rstsr
working tree was then restored (`git checkout -- .`) and verified clean at
HEAD 386948b. `target/` left in place for reviewer re-runs (D14 deviation,
same note as before); the committed-baseline stages of reproduce.sh
(`portable`/`native`) and the `candidate` stage (to be run with the patch
applied) are both reproducible.

## Integration into rstsr (2026-09-14) — patch 2 of the campaign queue

Applied to `../rstsr` branch `260914-elementwise` (base = `origin/master`
`e835173`, i.e. post-PR#100 argmax+nanarg merge; base content is identical
to the 386948be tree in the touched files — patch applies clean, +346).
Status: gates + benches green, **awaiting owner review** (patch-1 cycle).

- Patch file: [proposed-v2-post-fmt.patch](proposed-v2-post-fmt.patch) —
  the applied diff after local `rustfmt` reflowed the added lines
  (`debug_assert!` split to multi-line form, guard chain joined; zero
  content-line changes, verified by a `-`-line-free diff against master and
  `rustfmt --check` on both files). [proposed.patch](proposed.patch) is the
  original campaign capture, kept for the record; CI's pinned rustfmt is the
  authority for comment wrapping (patch-1 lesson), and the workspace
  `fmt --check` noise on *untouched* files (build.rs, rearrangement.rs,
  tensordot_to_einsum.rs, reduction.rs ×2) is the known local-vs-CI rustfmt
  divergence, not part of this patch.
- Gates on the patched tree: rstsr-core lib 110/110 + entry_row_cpu 302/302,
  portable AND native (plus the faer-free usual-situation config: lib 94+1
  ign, entry 302); `cargo clippy -p rstsr-native-impl --all-targets
  -D warnings` clean both configs; elementwise correctness example 94/94
  PASS both configs (flip/negative-stride views, i32, complex64, f32,
  ndarray cross-check).
- G1 caller enumeration (the compose-smoke lesson): callers of
  `op_mutc_refa_refb_func` / `op_muta_refb_func` are the elementwise-op
  families (`op_binary_*`, `op_ternary_*` serial+rayon twins, `map_elementwise`,
  public `op_with_func` wrappers) and — outside elementwise-land — the
  reduction order-fixup sites in `cpu_{serial,rayon}/reduction.rs`. Those are
  dead under the default iter order (`default() == K`), and even if alive
  they copy between disjoint buffers (visit order harmless) with a
  contiguous destination layout the guard admits only for 2-D ≥4096
  no-broadcast problems. Broadcast and 1-D inputs fall through to the
  iterator path everywhere; the T8 union smoke (332 checks, entry 3-fix
  regression) already covered this patch in combination.
- Paired benches (3 alternating refA=master/cand=patched passes × portable +
  native, full 54-bench suite, same session, quiet machine):
  [results/integration260914/](results/integration260914/README.md).
  **Verdict: no stable regression in any cell; all headline wins reproduce**
  (strided B serial 23.4→7.4 ms = 0.32×; faer16 2.06→0.96 ms; odd 8–9×;
  small 64² 16×; `c += bᵀ` 0.34×; stridedfirst 0.33×; strided A 0.42–0.47×).
  Known lottery cells (portable contig 1000×777 serial, native contig
  2048² f64 B, faer large-A) all inside their documented bands.

