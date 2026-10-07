# array-api conformance: current status + convergence checklist (2026-10-07)

Snapshot of where `rstsr_faer.api` stands against the official array-api-tests
suite, and a concrete, owner-classified checklist to drive the remaining red
map down. Companion to the running task directory
`../2026-10-04-rstsr-faer-py/` (whose `GAP-REGISTER.md` is the source of the
G-numbers below); this directory is a **frozen status read-out + action list**,
not a new wave.

Files:
- `README.md` — this status summary.
- `CHECKLIST.md` — the actionable convergence checklist (grouped by owner).
- `failures-241.csv` — machine-readable classification of all 241 current
  failures (class / bucket / register id / nodeid / detail).

## Measurement

| item | value |
|---|---|
| suite | `array-api-tests@6c0b59f` (spec `5f847a3`), API version `2025.12` |
| module | `rstsr_faer.api` 0.9.0 |
| wheel | `/tmp/wheels/rstsr_faer_py-0.9.0-cp310-abi3-manylinux_2_34_x86_64.whl`, sha256 `c7477089af38d4a8e12c86a991cdcda86278efa85bdf0d7e592ced6a4750dc3e`, built 2026-10-07 11:58 |
| build | release profile, `CARGO_PROFILE_RELEASE_OPT_LEVEL=0 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_BUILD_JOBS=2` (skill §1; **not** the dev profile) |
| source state | manip/sort/set + tensor-tier API revision round; committed baseline `2aacd5e` (branch `261006/manip-sort-set`). `rstsr` main `af78036` contains #118 (`3bcc3a2`), so main is equivalent-or-newer. |
| stamps | baseline `20261007-092159`; re-verified `20261007-115926` (node-identical); latest full chunked report `harness/reports/rstsr_faer_api-20261007-212921.json` (19/19 chunks) — the classification source here |
| run | `FRESH=1 NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1` |

**Wheel-provenance caveat:** the common local workspace
`~/rstsr_pack/rstsr-local-workspace` is currently checked out at `e7cdc6a`
(#117), which *predates* the W6-8 bindings and does **not** build the wheel
whose numbers are recorded here. The recorded wheel corresponds to the
manip/sort/set state (`2aacd5e` / main `af78036`). Before the next run,
re-confirm which checkout builds the wheel (sync the local workspace, or build
from `rstsr/` main) so the stamp's provenance is unambiguous.

## Totals

| | passed | failed | skipped | total | pass-rate* |
|---|---|---|---|---|---|
| **current** | **1059** | **241** | **82** | 1382 | 76.6 % (81.5 % of graded) |
| NumPy 2.5.1 baseline | ~1335 | ~42 | 5 | 1382 | 96.6 % |

\* `passed/total`; parenthetical is `passed/(total − skipped)`.

**Trajectory** (per GAP-REGISTER waves): 255 → 302 → 902 (W2 elementwise) →
934 (W3 stats) → 984 (W4 creation/manip) → 996 (W5 searching) → 1014 (`where`)
→ **1059** (W6-8 manip/sort/set). The last +45 flips came almost entirely from
binding `repeat/roll/tile/sort/argsort/nonzero/isin/unique_*/take_along_axis/diff`.

## The 241 failures — partition

Grouped by owner class. Full per-test rows in `failures-241.csv`.

| class | n | buckets (register) |
|---|---|---|
| **rust-side** (rstsr fix needed; register + request only) | **155** | complex special values/accuracy **53** (G-055) · mixed-dtype arithmetic & joins **29** (G-009) · `log1p` **18** (G-058) · `remainder`/`__mod__` **12** (G-057) · `expm1` complex/int **12** (G-055/G-052) · `signbit` **9** (G-054) · mask/fancy indexing **6** (G-038/039) · `pow` **5** (G-053) · NaN propagation **4** (G-056) · int `ceil/floor/trunc/round` **4** (G-052) · `negative` unsigned **2** (G-044) · empty `setitem` **1** (G-045) |
| **missing surface (names)** | **70** | `linalg` namespace + top-level `matmul/matrix_transpose/tensordot/vecdot` **37** (G-023) · `fft` **14** (scope decision, W7) · dtype fns `can_cast/isdtype/result_type` **12** (G-027/G-008) · `clip` **3** · `log1p` name **2** |
| **shim-side** (agent-doable, wrapper-only) | **18** | `finfo(complex)` **4** (G-032) · namespace-info **4** (G-048/49/50) · `isnan/isfinite/isinf` 0-d shape **3** (G-040) · creation `shape` pos-only **4** (G-047) · `bitwise_invert` u64 carrier **2** (G-013) · `astype device=` **1** (G-046) |

Roll-up: **rust-side 155 · missing surface 70 · shim-side 18**.

Two levers dominate the surface half: `linalg` (37 fails **and** it activates 49
currently-skipped tests → 86 tests hinge on it) and `fft` (14 fails + 28 skips
→ 42 tests).

## The 82 skips

| # | cause | nodes |
|---|---|---|
| 49 | `linalg` namespace absent → suite self-skips (`test_linalg` 26 + `test_signatures` 23) | the extension's *names* still **fail** separately in the 37 above |
| 28 | `fft` namespace absent (`test_fft` 14 + `test_signatures` 14) | names fail separately in the 14 above |
| 5 | upstream `@pytest.mark.skip("flaky")` on `test_remainder` (suite `6c0b59f`) | backend-independent — same 5 NumPy skips |

None of the 82 is a hidden rstsr failure.

## Method

1. `harness/compliance_table.py reports/rstsr_faer_api-20261007-212921.json -o current.csv --census`
   → root-cause census: missing-surface 88 · wrong-value 79 · unexpected-exception
   62 · indexing-gap 6 · dtype-fn-bug 4 · unclassified 2.
2. Theme bucketing of the 241 rows by nodeid param / test function / detail
   string into the owner classes above, emitted to `failures-241.csv`
   (`class,bucket,register,suite_file,test_function,nodeid,detail`).

## Conclusion

The red map is **two thirds rust-side (155)** and one third surface (70), with a
small, fully enumerated shim-side remainder (18) that is the cheapest next win.
Convergence to the NumPy band therefore depends mostly on (a) two namespace
decisions (`linalg`, `fft`) and (b) a handful of rust-side semantics items, with
G-055 (complex, 53) and G-009 (mixed-dtype, 29) the two biggest single levers.
See `CHECKLIST.md` for the ordered plan.
