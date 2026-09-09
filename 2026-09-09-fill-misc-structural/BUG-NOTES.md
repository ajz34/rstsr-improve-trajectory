# BUG-NOTES — upstream issues found during the campaign (for the human)

Consolidated for the rstsr maintainers. All locations are file:line at base
`386948be819baa334b8da02232f3a1944e5447d5`. None of these were fixed in the
campaign (out of scope / semantics decisions); the patches that touch the
same files deliberately preserve current behavior and say so.

---

## (a) Reductions: stride-0 summed axis multiplies by the wrong count

- **Where**: `rstsr-native-impl/src/cpu_serial/reduction.rs:226`
  ```rust
  let size_s0 = as0.iter().map(|&i| lm.shape()[i]).product::<usize>();
  ```
  `as0` indexes axes of the **summed** layout `ls`, but the shape is read
  from the **remaining** layout `lm`. For a broadcast (stride-0) summed axis
  the two shapes differ, so the reduce-multiplier is wrong. Same construct
  at `:252` and `:300` use `size_s0` for the loop counts.
- **Evidence** (T2'): `sum_axes(0)` of a `[1,n] → [m,n]` `broadcast_to`
  view yields **n×v** where numpy yields **m×v**; `mean_axes` yields
  n/m×v. `min`/`max` are insensitive (idempotent fold). The T2' phase-2
  rewrite preserves this behavior bit-for-bit and its gate locks it as
  CURRENT-BEHAVIOR (42-check suite includes the fixture).
- **Suggested fix owner**: rstsr maintainers. The fix itself is the obvious
  one-line `ls.shape()[i]`, but it **changes user-visible semantics** —
  should be its own PR with a numpy cross-check added to the functional
  suite. Coordinate with T2's proposed.patch (disjoint hunk, same file).

## (b) `inner_dot` reads uninitialized `c` when `beta != 0` — BOTH shapes

- **Where (serial)**: `rstsr-native-impl/src/cpu_serial/matmul_naive.rs` ≈
  :231–233
  ```rust
  let mut sum = beta * c[idx_c].clone();
  ```
  inside `inner_dot_naive_cpu_serial` — reads the output element **before**
  the loop writes it. The `%` surface allocates `c` fresh via `empty`
  (uninit).
- **Where (rayon)**: `rstsr-native-impl/src/cpu_rayon/matmul_naive.rs:91`
  ```rust
  *c = c_innerdot * alpha + c.clone() * beta;
  ```
  same read-after-nothing, post-fold.
- **Evidence** (T3', pre-existing at 386948be): with `beta = 0` (the only
  value the `%` surface ever passes) this is `0 × uninit-bits` — benign
  when the bits are finite/zero, **NaN-poisoning when the allocator hands
  back NaN-pattern bits** (fresh mmap pages are zero, which is why every
  test passes). T3's B1 fast path (`alpha==1 && beta==0`) incidentally
  removed the hazard on that guarded path; the B2 small-n fallback and the
  strided/scaled fallbacks PRESERVE the old behavior (no semantic fix
  smuggled into a perf patch).
- **Suggested fix owner**: rstsr maintainers, as a correctness PR: either
  skip the `c` read when `beta == 0` in all fallbacks (same pattern as B1),
  or document that `beta != 0` requires initialized output and zero-init in
  the drivers that allocate. Note the serial/rayon asymmetry (pre-loop vs
  post-fold read) when writing the fix.

## (c) `dispatch_dim_layout_iter`: f32 1-D vecdot regression (0.57×)

- **Where**: the feature itself (`rstsr-common/src/layout/iterator.rs`
  monomorphized dispatch) interacting with the f32 vecdot fold; observed in
  T0's D8 column (`2026-09-09-bench-harness-baseline/README.md` §D8,
  `results/tables.md:178`).
- **Evidence**: native build, vecdot 1e7 f32: serial 1.22 → 2.15 ms
  (**0.57×**), faer16 1.25 → 2.12 ms (0.59×) with the feature on; every
  other benched cell neutral or better (odd transpose up to 1.98×, strided
  add 1.36×, f64 vecdot unaffected).
- **Suggested owner**: whoever next touches the feature or the vecdot
  kernels; keep it **default-off and per-caller**, re-validate f32 1-D dot
  before enabling anywhere. Note the feature's useful wins for
  irregular-strided cases are largely superseded by the T1'/T4' blocked
  kernels now. Not a default-config issue.

## (d) numpy `.T.copy()` anomaly on this machine (context — NOT rstsr's bug)

- **Where/evidence**: T0 anchor table
  (`2026-09-09-bench-harness-baseline/README.md`): numpy 2.5.1 `.T.copy()`
  of 2048² f64 measured **99.8 ms** in-run (reproduced across repeats within
  the run) while the same op lands 2.6–4.6× above rstsr-serial across other
  runs — an environment/allocator anomaly on this box, not a numpy version
  behavior to anchor against.
- **Action**: do not cite the numpy transpose number as a reference; the T1
  verdict rests on the ndarray anchor. Listed so future benchmarking on this
  machine doesn't chase it.

## (e) Smaller flags from the five READMEs (not bugs, but decision records)

1. **API breaks carried by accepted patches** (flag in the integration PRs):
   T6 `ArgCmp` replaces `(Fcomp, Feq)` closures in `pub` kernels of the
   published `rstsr-native-impl`; T3' widens `TC: Zero + One + PartialEq` on
   `inner_dot_naive_cpu_serial`/`matmul_naive_cpu_serial` + the
   `DeviceMatMulAPI for DeviceCpuSerial` impl. Tensor-level API unchanged.
2. **T4' visit-order note**: the serial strided path now visits elements in
   tiled order — bit-identical for stateless ops; *stateful* `FnMut`
   closures on the low-level drivers no longer see layout order on the
   serial path (the rayon path was always order-free). Documented in the
   patch's doc comments.
3. **T2' ±0.0 sign-bit note**: min/max on ±0.0 ties now keep the incumbent
   (earlier) zero instead of the later operand (value-equal; sign bit may
   differ); locked by gate spots.
4. **T2' uncovered fallback**: the >128k-summed-positions cap fallback in
   `reduce_axes_cpu_serial` has no benchmark coverage (byte-identical
   pre-rewrite code); noted for reviewers.
5. **`zeros` doc nuance** (T5): "calloc-lazy" only holds for ≥ ~32 MiB
   outputs; below the glibc cap `vec![0.; n]` memsets (and at multi-MiB
   sizes does so non-temporally at DRAM speed — 2.4× slower than `full` at
   6.2 MiB). Docs should not claim unconditional laziness. No code change
   recommended.
6. **Bench-infra lessons that looked like bugs** (recorded so nobody
   re-chases them): build-layout/session lottery ±5–11 % on ~0.5 ms
   streaming cells (T2'/T4'/T6 — use interleaved back-to-back A/B);
   criterion `change:` lines inside saved-baseline reruns are NOT
   baseline-vs-candidate comparisons (T6); ndarray `.t().to_owned()`
   preserves f-order (T1 anchor caveat); the campaign's toolchain identity
   is unpinned — the rstsr checkout's undated `rust-toolchain.toml`
   (nightly) does not propagate along path deps, so the crates built with
   the rustup default (stable 1.97.1 recorded at T5 gate time;
   `results/toolchain_identity.txt` in the T5 dir) — treat
   cross-experiment absolute numbers as drift-prone.
