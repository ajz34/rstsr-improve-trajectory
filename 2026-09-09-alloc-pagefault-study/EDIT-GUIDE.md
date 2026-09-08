# EDIT-GUIDE — allocation/page-fault rider (T7, base 386948be)

Purpose: ranked, concrete options for the rstsr maintainers to act on the T7
findings. **Nothing here has been applied**; this directory is
measurement-only. Numbers: [README.md](README.md) (criterion + getrusage +
perf evidence, `results/`).

The problem in one line: every allocating op whose fresh output is ≳32 MiB
pays ~4 ms of mmap/zero-page-fault churn (8193 faults for 2048×2048 f64,
sys-time-dominated) on top of the kernel — 66% of `a + b`, 88% of `full`,
25–57% of transpose copy, both devices. Below ~32 MiB glibc already adapts
(0–5% rider), so this is a large-class/streaming-only phenomenon.

Options, ranked by (evidence-supported win) / (risk & effort):

---

## Option 1 — Document the reuse APIs and the env tunables (docs-only; do first)

**What.** Two documentation additions, both "user-side, works today":

1. **Reuse idiom page** (rstsr-book; candidate diff material for the final
   bookkeeping phase — do not merge from this repo):
   - `c` allocated once outside a loop, refilled per iteration with
     existing API:
     - elementwise: the public low-level driver
       `rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func(&mut c, &a, &b, &mut |c, a, b| { c.write(a + b); })`
       — single pass, broadcasting, any layout, both devices
       (66–77% faster than `&a + &b` at 2048²);
     - transpose: `c.assign(&a.t())` (25% serial / 57% faer16 faster than
       `a.t().to_contig(RowMajor)`);
     - fill: `c.fill(v)` vs `rt::full` (88% faster);
     - reductions: no reduce-into API exists today (see Option 2c) — say so.
   - Explicit warning: `rt::zeros` is calloc-lazy by design; never "fix" it
     into a touching fill (T0 + T7 evidence).
2. **Large-array performance note** (rstsr-book installation/performance
   page): for workloads with ≥32 MiB transient outputs on glibc,
   ```text
   MALLOC_MMAP_THRESHOLD_=67108864   # must EXCEED the largest recurring output
   MALLOC_TRIM_THRESHOLD_=134217728  # or free() shrinks the heap and refaults
   ```
   recovers ~96–97% of the rider with zero code change (6.18 → 2.27 ms
   serial, 2.70 → 0.72 ms faer16 for add 2048²). Caveats to document: the
   tempting `33554432` value does *not* work (chunk > threshold still
   mmaps); RSS grows; it is a per-job setting, not a library default.

**Expected win:** large: full 66–88% for users who adopt the idioms; medium:
96% via env. **Risk:** ~zero. **Evidence:** README RQ2a/RQ2b tables —
strongest-supported option.

---

## Option 2 — Small API additions (small diff; high value)

Sketch-level proposals, all consistent with existing trait structure:

a. **Safe tensor-method sugar over `op_mutc_refa_refb_func`**, e.g.
   `c.assign_with(&a, &b, |x, y| x + y)` or `rt::add_into(&mut c, &a, &b)`
   (fallible `_f` twin per house style). Hides the `MaybeUninit` closure and
   `impl TensorViewMutAPI` plumbing that currently makes the single-pass
   reuse path undiscoverable. The mechanism is already public and correct at
   386948be; this is ergonomic wrapping only. Expected doc examples stay
   valid under broadcasting (zero-stride cases verified in T7's gate).

b. **`into_contig`-style reuse for `to_contig`:** `c.assign(&a.t())` already
   covers it; optionally document a `to_contig_into(&mut c)` convenience.
   Low priority — Option 1's docs may suffice.

c. **`reduce_into` / `sum_axes_into(&mut c, &a, axes)`** — the only op
   family in the matrix with **no** reuse path (device `sum_axes` allocates
   unconditionally). The negative control shows small reduction outputs
   don't need it for fault reasons; its value is consistency + enabling
   kernel benches with reuse policy. Lowest priority of the three.

**Expected win:** makes Option 1's numbers attainable without low-level
APIs. **Risk:** small API surface additions; naming/order conventions need
maintainer care. **Evidence:** the underlying functions are proven in-crate
(T7 correctness gate + benches).

---

## Option 3 — Device-level allocation pool / cached large buffers (bigger; structural)

**What.** In the CPU storage allocation path (`uninit_impl` →
`uninitialized_vec`, `rstsr-common/src/alloc_vec.rs`), add size-class free
lists so that a freed ≥~4 MiB buffer is returned to the pool and the next
allocation of the same size class reuses it (no mmap/munmap, no faults).
Thread-local or sharded lists for the faer pool's 16 allocating threads.

**Evidence for the ceiling:** the THP probe's `reuse` mode (0.50 ms/iter for
a 32 MiB fill — the floor) and the env-tunable runs (2.27 ms for the full
allocating add, tying the reuse API): a pool delivers exactly this to
*idiomatic* code with no user changes.

**Trade-offs (why it's ranked third):** memory high-water policy (pool
retention vs RSS), peak-memory accounting (users tracking allocation may see
higher RSS), interaction with `aligned_alloc` feature and its 64-B layout,
faer pool sharding, and the fact that Option 1 already delivers ~the same
number for informed users. Consider only if rstsr wants the win *by
default* for all users.

**Alternative in the same tier (smaller, but blocked):**
`madvise(MADV_HUGEPAGE)` after large allocations — measured to hide the
fault cost (0.68 vs 4.20 ms/iter fresh-alloc fill) — **but it requires
page-aligned (ideally 2 MiB-aligned) buffers, and glibc's memalign trim path
returns non-page-aligned pointers for 64-B alignment** (expected per POSIX —
madvise requires a page-aligned addr — observed during development; no raw
log committed). So THP integration = change allocation alignment in
`alloc_vec.rs` + madvise + fallback handling. More invasive than it looks;
system THP is `[madvise]` here and often `never` elsewhere (HPC), so it is
not portable guidance.

---

## What the evidence supports

1. **Do Option 1 now** — zero risk, works at 386948be, recovers 66–97%
   depending on user control (env vs idiom), and immediately gives the
   campaign's later tasks the right denominator for judging kernel wins
   (kernel-only, not alloc-inclusive).
2. **Option 2a** is the best small code change if maintainers want the
   reuse idiom to be first-class; everything under it is already proven.
3. **Option 3** only if default-on behavior for all users is a goal; its
   ceiling is already reachable through Options 1–2.
4. **Do not** change `rt::zeros` (calloc-lazy is optimal), and do not add
   madvise without an alignment change (measured `EINVAL`).

## Cross-references

- Seed finding: T0 README, add_contig profile (844k faults, sys > user).
- Consuming tasks: T1 (transpose) and T4 (elementwise) should quote the
  reuse variants as kernel-only denominators; T5 may fold the
  `zeros`-must-stay-lazy warning into its fill work.
- Bookkeeping phase: the Option 1 docs belong to the separate rstsr-book
  diff directory, when the owner authorizes it.
