# PLAN — 03-elementwise: full elementwise surface + performance campaign

Branch: `261023/cuda-review-fixes` @ `b44cbef` (working tree only, no
commits). Baselines: cupy 14.2.0 (`ruaa-dev`), torch 2.13.0+cu132
(`rest-torch`), numpy 2.4.6 (`pyscf-pip`). Device: RTX 5070 Laptop,
36 SMs, 8 GB, sm_120. R1 answers in `GRILL-R1-ANSWERS.md`.

Goal: (1) every unary/binary/ternary elementwise op the CPU devices
support, running on CUDA for all supported real dtypes + c32/c64 incl.
complex transcendentals, arbitrary layouts (strided / transposed /
offset / negative-stride / broadcast, operands with differing layouts);
(2) assign (incl. reshape-like `assign_arbitary`) fast for all layout
classes; (3) correctness vs CPU reference; (4) f64 benchmark campaign
with gates vs cupy: per-cell ≤1.2× target, ≤2× hard (exceeding ⇒
written investigation: what was tried, why it stays), small sizes
informational; (5) report here, library comparisons only here.

## Workstream A — kernel infrastructure (the 2.3× fix lives here)

A1. **D9 ladder T2 (contiguous)**: dedicated kernel bodies with no
    shape/stride/offset args (pointer + n only). Kills the per-launch
    scratch H2D upload on the hottest path and shrinks the flat-launch
    arg marshaling; offset folded into the pointer at launch.
A2. **T1 (inner contiguous run)**: host-side detection of the trailing
    contiguous run (len ≥ ~8 elements); outer grid-stride over the
    non-contiguous remainder decomposed once per run, inner loop over
    the run with stride-1 loads (ILP-friendly, one index add per
    element); broadcast stride-0 stays valid (run of the *output*
    layout; operand run strides passed per operand).
A3. **T0 (generic strided) hardening**: precomputed fast-division
    (libdivide-style mulhi+shift magic) per axis, computed on host at
    launch, replacing the per-element `u64 div/mod` chain; keep exact
    semantics for negative strides (two's-complement wrap).
A4. **Dim-squash** (cupy `_reduce_dims` lesson): host-side axis merge
    when strides are composable (adjacent axes with stride ratio ==
    size, broadcast axes droppable when the other operand differs) —
    reduces rank before marshaling; benefits T0/T1 arg size and the
    division chain. Correctness-neutral by construction (proven by the
    same test matrix).
A5. **Launch-path host cost**: intern kernel identity as a cheap
    discriminator (family enum × dtype CNAMEs × variant bits) resolved
    to `Arc<CudaFunction>` without re-assembling or re-hashing source
    on hot paths (REVIEW-ee70fe4 #7, proper interning this time);
    source-hash only on the compile/disk-cache path.
A6. **Program batching amendment (amends 02-kernel D10)**: one NVRTC
    program per (family, dtype-combo) containing its variant entry
    points (T0/T1/T2, scalar forms) instead of one program per
    specialization — cuts cold-compile count ~4× as the op surface
    grows; laziness preserved (compile at first use of any variant).
    Record in DECISIONS.md at round close.
A7. Validation micro-bench first: A+B.T 1024² must drop from 69 µs to
    ≤36 µs (1.2× of cupy 30 µs) before building the rest on top;
    contig add must not regress (>5% regression = stop and rethink).

## Workstream B — op surface completion

Kernel-source families (generated from small Rust tables, split into
`src/kernels/` modules; per-file size cap, no total cap):

- B1 binary arithmetic remainder: `rem` (+scalar), incl. the CPU
  float-`ext_rem` semantics (fmod-class) vs Rust `%` for ints.
- B2 bitwise: `bitor/bitand/bitxor/shl/shr` (+bool, +scalar, +assign).
- B3 compound assign ×10 (`add_assign` … `shr_assign`): out-of-place
  kernel writing into the destination layout (order_input/output
  discipline as in `b44cbef`).
- B4 promoting binary: `atan2, copysign, floor_divide, hypot,
  log_add_exp, pow, maximum, minimum, nextafter` + `ext_{add,sub,mul,
  div,bitand,bitor,bitxor,shl,shr}` — mixed (TA,TB,TC) combos via
  C-level casts in the generated source; dtype set = CPU's per-family
  bound (ExtReal / ExtNum / PrimInt / promotion lattice).
- B5 non-transcendental unary: `abs, sign, square, floor, ceil, round,
  trunc, neg, not, positive, ext_neg, conj, real, imag, is_nan,
  is_inf, is_finite, signbit` — heterogeneous outputs (abs c→r,
  real/imag, bool-out predicates) use the TA≠TC unary machinery.
- B6 complex transcendentals: own prelude (`rstsr_cmath` section in
  the shared prelude) built from real libdevice calls, formulas
  cross-checked against cupy `complex.cuh` / pytorch `c10::complex`
  (brief code comment where a formula is taken from them); sin, cos,
  tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp,
  expm1(?), log, log1p, log2(?), log10, sqrt, inv, square — exactly
  the set CPU supports for complex (verify during B6; anything CPU
  lacks stays absent).
- B7 `where` (+scalar x/y forms), `clip` (scalar/tensor/None bounds),
  `*_with_output` for the 10 arithmetic families, `isclose`.
- B8 f16/bf16: arithmetic + transcendentals via cuda_fp16 native
  functions where present, else promote→f32→demote (match CPU's
  semantics, tolerance-tested).
- B9 type promotion: mixed-dtype binaries/where resolve at the tensor
  tier (std-op Output / DTypePromoteAPI); CUDA instantiates the
  required (TA,TB,TC) kernels. Spot-tests: f64⊕i32→f64, i32⊕i32→i32
  (wrapping semantics match CPU), bool⊕numeric where, int `sin`→f64.
  Not benchmarked (R1).

## Workstream C — assign / reshape-class copy

Already present (arbitrary rank/shape, cast, review-fixed). Extend:
- C1 T2/T1 fast paths for same-layout copy and the
  contiguous-run assign cases (row/col slices, padded rows).
- C2 bench coverage for `to_contig(T)` (existing, 1.3× today) and
  reshape-of-strided-view; both must reach ≤1.2×.

## Workstream D — correctness tests (extend `tests/test_kernels.rs`)

- D1 layout matrix generator: for each family, run operand layouts
  drawn from {contig, transpose, slice w/ offset, stepped slice,
  negative stride, broadcast-per-axis, mixed different layouts for
  a/b with same shape} on 1–4-D shapes incl. odd sizes; compare vs
  `DeviceCpuSerial` element-exact for ints/bool, tolerance for
  floats (1e-12-class like existing sin tests), f16/bf16 looser.
- D2 NaN/edge semantics: maximum/minimum NaN propagation, is_nan on
  injected NaN, log of 0/negative, div by zero for ints (match CPU
  panic policy — verify what CPU does before deciding CUDA behavior;
  report if inconsistent), where with NaN, complex formulas at branch
  cuts (compare vs num-complex CPU).
- D3 promotion spot-tests (B9), `with_output` forms, compound assign
  into strided/broadcast-invalid destinations (error paths).
- D4 keep every existing test green; new regressions cited by cause.

## Workstream E — benchmark campaign (f64)

E1 **rstsr bench** (`rstsr-cuda/benches/bench_elementwise.rs`,
rewrite; library-free): scenario table, hand-rolled timing
(sync-bracketed, warmup ≥30, **5 repeats (7 near gates), median**,
report min as diagnostic). Scenarios (each S ≈ 8 MB/operand, L ≈ 48 MB
unless noted):
  1. contig add size ladder: 1e3, 1e4, 1e5, 1e6, 4e6, 1.6e7 (small
     three = informational appendix);
  2. `A + B.T` square 1024² / 2500²;
  3. `A + A.T` same sizes;
  4. `A + B.swapaxes(0,1)` 3-D (64,64,256) / (160,160,256);
  5. outer `a[None,:] + b[:,None]` n=1024 / 2438;
  6. `A[:,:,:] + B[:,None,:]` (16,256,512) / (32,512,1024);
  7. `A[:,:,:] + B[:,:,None]` same shapes;
  8. sliced `A[:,10:1010] + B[:,50:1050]`, parents (1024,1280) /
     (4896,1280);
  9. `A += B.T` (compound, in-place) 1024² / 2500²;
  10. `to_contig(T)` 1024² / 2500²; reshape-of-strided-view same;
  11. `sin` 1e6 / 4e6; `fill` 2e6 / 1e7.
E2 **baselines here in trajectory**: `bench_cupy.py`, `bench_torch.py`
  (same scenario table, same protocol: stream-sync brackets, 5/7
  repeats, median; cuda graphs OFF, warmup until steady); a runner
  script producing the merged table.
E3 **protocol**: development runs unlocked (shared-GPU caveat
  documented); final tables under owner-run
  `sudo nvidia-smi -pm 1` + `-lgc <fixed>` (agent verifies supported
  clock range first; `-rgc` after). Median of 5 (7 near gates);
  any cell within ±10% of a gate boundary re-measured once.
E4 **gates**: per-cell rstsr/cupy ≤1.2× (target) / ≤2× (hard);
  >2× cells require written investigation (attempts + why rejected);
  ladder small cells informational; geomean over gated cells reported.
  torch column reported for context (no gate).
E5 **report**: `README.md` here — methodology, merged tables, gate
  verdicts, analysis per scenario class, CPU-side improvement
  suggestions (e.g. blocked/T1-style handling in native-impl),
  library comparisons allowed only here.

## Sequence and checkpoints

1. A1–A7 (perf core) → checkpoint: A+B.T ≤1.2×, no contig regression.
2. B1–B5+B7+B9 (real-dtype surface) with D1–D3 growing alongside.
3. B6 complex + B8 half; C1–C2.
4. E1–E2 harnesses; full D suite; fix-forward.
5. Final measured pass (E3), report (E5), DECISIONS.md + status log.

## Risks

- T1/T2 selection bugs (wrong fast path) — mitigated by running the
  whole D matrix through a debug env knob forcing each tier.
- Complex formula divergence vs num-complex at branch cuts — tolerance
  chosen per-op from CPU/CUDA libm differences (existing 1-ULP sin
  precedent).
- Compile-count blowup with the full surface — A6 batching; disk cache
  already in place.
- Shared-GPU noise near gates — 7 repeats + clock locking for finals.

## Status log

### 2026-10-23 — Workstream A implemented, A7 checkpoint PASS

Branch `261023/cuda-review-fixes` (working tree, no commits). New files:
`src/kernels/{mod,plan,elementwise}.rs`.

- **plan.rs**: host-side layout analysis — shape-1 drop, divmod-level
  merge (outer stride == d_inner * s_inner; both-zero broadcast pairs),
  magic division (Granlund-Montgomery round-up, pow2 shift sentinel),
  rank≤4 pack else `Generic` fallback. Unit tests include exhaustive
  magic-div verification for all d ≤ 2^16 against boundary/random
  numerators.
- **elementwise.rs**: planned launchers (binary/unary/scalar/fill/copy)
  emitting `_p` (parameterized) + `_f` (flat) entry points per program;
  plans travel by value as 136-B kernel parameters (no scratch H2D, no
  per-launch alloc); flat operands marshal as rank-0 plans (mixed
  selections always take `_p`; only all-flat takes `_f`); offsets folded
  into base pointers (wrapping math matches negative strides). Compiled
  functions interned behind a small discriminator string (per-ordinal)
  — source assembly + full-source hash off the hot path.
- **kernels_ops.rs**: rank-generic launchers kept as the Generic floor
  (renamed `*_generic`, incl. `copy_dispatch_generic` +
  `fill_dispatch_same` bridges); copy/fill now route through the
  planned kernels when shapes match. Dead legacy wrappers removed.
- **operators.rs**: binary arith + comparisons + unary float (+scalar
  forms, + in-place unary) all route through planned launchers.

A7 gate (locked 210 MHz, cupy same conditions): **add-strided
1.06–1.13x cupy (was 2.3x unlocked = past gate; now inside 1.2x)**,
contig add ~1.0–1.2x, sin 0.97x, fill ~0.97–1.27x, to_contig(T)
~1.0–1.3x, arange ~1.0–1.6x (small size, noisy). All 42 tests green.
Single-run spread at these sizes is ±10–15%; verdicts deferred to the
E-phase median protocol.

Design notes:
- Plans-by-value replaces the A1 "no layout args" sketch: the 136-B
  param struct avoids the scratch upload for P too, and T2-flat is the
  rank-0 encoding of the same mechanism (one code path).
- In-place unary got its own single-entry program (the flat/p split
  there hid a parameter-count mismatch — found by SIGSEGV in the
  in-place test, fixed before any test was green).
- Compile count per (dtype-combo, expr) doubled (two entries/program)
  but NVRTC cost is dominated by source length, and the disk cache
  dedups; interning makes the second entry free at lookup.

Next: B (op surface: rem, bitwise, compound assigns, promoting
binaries, non-transcendental unaries, where/clip, complex
transcendentals, half).

## Status log — 2026-10-24 (C/D/E progress + perf push)

Workstream B finished: all 5 wave tests + full suite green (51 tests).
Two test bugs fixed (mismatched operand shapes in the layout matrix;
assert_eq on NaN results). owner directive mid-session: push perf vs
torch below 1.2x where possible.

E-phase harness landed:
- bench_elementwise.rs: 19 cells, median-of-R, ITER/REPEAT env knobs,
  sustained warm-up (idle-clock mitigation).
- bench_cupy.py / bench_torch.py in trajectory: mirror cells,
  flush-per-line, 3-D arena freed before 2-D cells (silent-OOM CSV
  truncation found by misaligned paste comparisons).

C/D-phase perf work (three structural changes):
1. Narrow (u32) plan tier: ParamsN/_pn entries, 32-bit magic division,
   i32 wrapping offsets (negative strides exact), host-side eligibility
   (n < 2^31, strides and accumulated offsets fit i32). Marshaling bug
   found by tests: host struct was padded to 72 B while the device
   struct is 68 B — every parameter after the first plan shifted;
   removing the pad fixed wrong-indexing failures across test_kernels.
2. Grid-stride ILP4 on all flat entries (g + I*w; consecutive-elem
   per-thread pattern regressed 3x — uncoalesced).
3. Lean both-scalar clip: the fused min stage had textually embedded
   the max stage 4x (compiler cannot CSE isnan select chains); dedup
   via the sequential val local + host-checked non-NaN-bounds fast
   path (scalar_bytes_nan) + PTX max.NaN.f32 helpers: clip
   2.35 -> 1.03 ms (0.93x cupy, was 1.97x). max.NaN.f64 is
   NVRTC-OK but driver-JIT-rejected (INVALID_PTX) on this driver;
   f64 keeps the select fallback.
- Soundness fix found while chasing add_outer: single stride-1 level
  planned Flat even when its extent < n (broadcast operands) -> OOB
  reads; Flat now requires extent == n.
- Dead code removed (in-place clip2s superseded); clippy 0 warnings,
  fmt clean.

Table (median-of-5x200, same-day, see REPORT.md): 17/19 cells < 1.2x
vs max(cupy, torch). Flags: fill 2.25x torch (kernel exonerated —
in-place fill identical; clock caveat), rem 1.27x torch, add_outer
1.22x. All measurements today ran with the clock lock LOST (GPU idles
at P5/210 MHz; short kernels never spin it up — observed 3-4% util
during runs). Ratios need a re-measure after `sudo nvidia-smi -pm 1
-lgc 210,210`.

Next: re-lock clocks + re-run 3-way; f16/bf16 coverage check (B8);
promotion spot-tests (B9); cleanup task: remove dead clip2s entries.

## Status log — 2026-10-24 (boosted-clock re-measurement, Option D)

Owner chose Option D (software warm-up, no sudo). -lgc confirmed deprecated
on driver 595.91.07; persistence mode was the only setting that survived,
left enabled (harmless, recommended). No revert needed.

5-second sin/pow spin before timing proved sufficient: GPU runs ~2780 MHz
@ 55-80 W during measurement (was P5/210 MHz). All three harnesses carry it.

Re-measured 3-way table: every throttle-era flag collapsed. rem/add_outer/
add_step/flip/slice all <= 1.15x; swap3d/bcast at 1.00-1.01; where_tt 0.68;
where_ts 0.06 (our scalar-arm planned path beats library where by 16x).
Two flags remain:
- fill vs torch 6.7x: measured artifact, not kernel. Probe: our memset
  path (zeros) 0.0033 ms, our fill kernel 0.0117 ms, torch cell 0.0139 ms
  -- kernel parity; the cell gap is per-iteration allocation churn in our
  timed lambda + L2 residency (32 MiB < Blackwell L2). No code action.
- to_contig 1.34x: transposed-read gather uncoalesced; cupy/torch likely
  smem-tiled transpose. Follow-up wave: tile 2-D transpose in smem inside
  rstsr_copy_p/_pn.

Raw: rstsr_boost.txt / cupy_boost.csv / torch_boost.csv in this directory.
