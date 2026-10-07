# Convergence checklist — rstsr_faer.api × array-api-tests (2026-10-07)

Ordered by ROI and by blocking. Counts are "suite nodes moved" (fails flipped,
plus currently-skipped tests that get *activated* where noted). Register ids
(G-nnn) refer to `../2026-10-04-rstsr-faer-py/GAP-REGISTER.md`.

Working baseline to beat: **1059 / 241 / 82 of 1382** (stamp `20261007-212921`).

## A. Shim-side quick wins — do first (~18 fails, no rust changes)

Wrapper-only edits in `crates-interop/rstsr-faer-py`; each is a register entry,
no algorithms. Lowest risk, immediate payoff.

- [ ] **A1. `finfo` accepts complex dtypes** (G-032) — 4
- [ ] **A2. namespace-info** — `devices()` → tuple, add `default_device()`,
      `dtypes(kind=)`, fix `default_dtypes()` keys (G-048/49/50) — 4
- [ ] **A3. 0-d result shape** in `isnan/isfinite/isinf` int/bool fallback (G-040) — 3
- [ ] **A4. creation `empty/full/ones/zeros` `shape` pos-or-kw** (G-047) — 4
- [ ] **A5. `astype(..., device=)` acceptance** (G-046) — 1
- [ ] **A6. u64 PyScalar carrier** for `bitwise_invert` (G-013) — 2

Exit check: full chunked run, expect **1059 + ~18 = ~1077 passed / ~223 failed
/ 82 skipped**, 0 regressions (test-for-test diff).

## B. Decisions needed before more binding work (each is a fork, not code)

- [ ] **B1. `linalg` namespace** — implement `xp.linalg` vs. scope-out via
      task-side `SKIPS_FILE`. Implement: **flips 37 fails + activates 49 skips
      = 86 tests** — the single biggest lever. Most kernels exist rust-side;
      QR/slogdet/solve_symmetric do not (G-004/G-005). (G-023)
- [ ] **B2. `fft`** — build an FFT subsystem vs. declare out of scope (W7 open).
      **42 tests** hinge (14 fails + 28 skips).
- [ ] **B3. dtype functions `can_cast/isdtype/result_type`** (G-027/G-008) —
      rstsr promotion exists only as associated types, no token-level query;
      needs a rust-side design decision. 12 fails.
- [ ] **B4. `clip`** (no rstsr primitive) and **`log1p`** (kernel is a
      `// TODO`) — rust-side; 3 + 2 name fails, `log1p` also 18 rust fails.

## C. Rust-side queue — ordered by failures per work item

Register + request; never fix agent-side.

- [ ] **C1. G-055 complex transcendental values/accuracy — 53**
      (one work item: special-value propagation + complex64 accuracy for
      acos…tanh, sqrt; `expm1` complex is a further 12)
- [ ] **C2. G-009 mixed-dtype arithmetic & joins — 29**
      (`add/sub/mul/div`, `bitwise_*`/shifts, `concat`/`stack`: Rust bounds are
      same-type only)
- [ ] **C3. G-058 `log1p` kernel — 18** (declared but every device impl is a TODO)
- [ ] **C4. G-057 `remainder` signed-zero / infinite-divisor — 12**
- [ ] **C5. G-054 `signbit` inverted semantics — 9**
- [ ] **C6. G-038/G-039 mask & fancy indexing — 6**
- [ ] **C7. G-053 `pow` int/bool/complex bases — 5**
- [ ] **C8. G-052 int `ceil/floor/trunc/round` dtype preservation — 4**
- [ ] **C9. G-056 NaN propagation (`max/min`, `maximum/minimum`) — 4**
- [ ] **C10. G-044 `negative` on unsigned — 2; G-045 empty `setitem` — 1**

## D. Process / measurement hygiene

- [ ] Build with `--release` at any opt-level, **never** the dev profile
      (debug-assertions flip exactly 10 integer-overflow nodes).
- [ ] `CARGO_PROFILE_RELEASE_OPT_LEVEL=0` for dev iterations; opt-3 only for a
      wheel whose numbers are recorded.
- [ ] Red runs always `CHUNKED=1`; canonical counts with `FRESH=1`;
      `NO_EXPLAIN=1` for speed.
- [ ] After every chunked run: confirm chunks == 19 `test_*.py` files.
- [ ] Before recording a flip: confirm the run's node set is test-for-test
      stable vs. the previous stamp.
- [ ] Fix the wheel-provenance gap (README caveat): point the local workspace at
      the state that builds the recorded wheel.
- [ ] Every divergence gets a `GAP-REGISTER` entry (rust-side / shim-side /
      suite); no rstsr-core edits without owner permission.

## Suggested next session

Take **group A** (≈18 fails, low risk, all shim-side) and, in parallel, put the
**B1 (linalg)** decision to the owner — it moves the most tests of any single
item.
