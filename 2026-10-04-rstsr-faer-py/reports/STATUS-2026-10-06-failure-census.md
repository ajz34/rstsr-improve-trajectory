# Failure census — rstsr_faer.api × array-api-tests (2026-10-06)

State: post-`where` merged main (`05cf6ce`), re-measured after the local
workspace was relocated to `~/rstsr_pack/rstsr-local-workspace` (the common
local workspace; the wheel below is rebuilt there from scratch, so the
relocation is also a build sanity check). All 286 failures are grouped below;
nothing is left in "unknown".

## Measurement

| item | value |
|---|---|
| wheel | `~/rstsr_pack/rstsr-local-workspace` @ branch `261006/faer-py-where` (`8d4cb28`; tree-identical to main `05cf6ce`), `CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build` (dev opt; suite verdicts are opt-independent, skill §1) |
| suite | array-api-tests `6c0b59f`, spec `5f847a3`, `FRESH=1 NO_EXPLAIN=1 CHUNKED=1`, 19/19 chunks, 62 s |
| stamp | `20261006-183350` (`harness/reports/…-MERGED-20261006-183350.json`, `COMPLIANCE-FULL-20261006-183350.csv`) |
| totals | **1014 passed / 286 failed / 82 skipped / 1382** |
| stability | test-for-test 0-flip and identical nodeid set vs the canonical `where` run `20261006-180651` |

Class census (`compliance_table.py --census`): missing-surface 133,
wrong-value 79, unexpected-exception 62, indexing-gap 6, dtype-fn-bug 4,
unclassified 2 (both are the complex64 `tan`/`tanh` accuracy cases already
inside G-055).

## Where the 286 sit (exact partition)

| bucket | n | register |
|---|---|---|
| missing names — `test_has_names` | 61 | names list below |
| missing names — `test_signatures` / `test_info_func_signature` | 24 | names list below |
| missing names — their runtime tests | 29 | linalg 4, set 6, manip 3, searching 3, sort 2, dtype 6, diff 2, elementwise(log1p/clip) 2, take_along_axis 1 |
| complex transcendendental special values / accuracy | 65 | G-055 |
| mixed-dtype arithmetic & joins (add/sub/mul/div, bitwise_*, shifts, concat, stack) | 29 | G-009 |
| `log1p` (17 special-value + the 3 name failures above) | 17 (20 total) | G-058 — kernel is a `// TODO` in rstsr |
| `signbit` inverted semantics | 9 | G-054 |
| `remainder`/`mod`/`__imod__` signed-zero + infinite-divisor | 12 | G-057 |
| boolean-mask / integer-array indexing declines | 6 | G-038 / G-039 |
| `pow` integer/bool/complex bases | 5 | G-053 |
| `ceil`/`floor`/`trunc`/`round` on integers | 4 | G-052 |
| NaN propagation (`max`/`min` reductions, `maximum`/`minimum`) | 4 | G-056 |
| `negative` on unsigned dtypes | 2 | G-044 |
| `bitwise_invert` on uint64 (Python carrier wraps via i64) | 2 | **G-013 — newly graded** |
| empty-tensor `__setitem__` (`InvalidLayout`) | 1 | G-045 |
| `finfo(complex64/128)` | 4 | G-032 (shim) |
| `empty`/`full`/`ones`/`zeros` `shape` pos-only | 4 | G-047 (shim) |
| namespace-info: `devices()` type, `default_device()`, `dtypes(kind=)`, `default_dtypes()` keys | 4 | G-048/049/050 (shim) |
| `isnan`/`isfinite`/`isinf` 0-d shape `(1,)` | 3 | G-040 (shim) |
| `astype` signature missing `device=` | 1 | G-046 (shim) |

Roll-up: **missing surface 114 · rust-side semantics/kernels 156 · shim-side
fixes 16**.

## The missing surface — 61 names

| block | names | cost | note |
|---|---|---|---|
| `xp.linalg` + top-level | linalg namespace 23 (cholesky cross det diagonal eigh eigvalsh inv matmul matrix_norm matrix_power matrix_rank matrix_transpose outer pinv qr slogdet solve svd svdvals tensordot trace vecdot vector_norm) + `matmul` `matrix_transpose` `tensordot` `vecdot` + `Array.__matmul__` | 37 failures | the single largest block; most kernels exist rust-side, QR/slogdet/solve_symmetric do not (G-004/G-005); landing `xp.linalg` also *activates* the currently self-skipped linalg extension tests |
| `xp.fft` | 14 | 14 failures | names only (test file self-skips); scoping decision still pending (W7) |
| searching/set/indexing | `nonzero` `searchsorted` `isin` `unique_all` `unique_counts` `unique_inverse` `unique_values` `take_along_axis` | 26 failures | G-029 remainder; all rust-side algorithms |
| manipulation | `repeat` `roll` `tile` | 9 failures | G-002/G-003, no rstsr primitive |
| sorting | `sort` `argsort` | 6 failures | G-001; `test_searchsorted` also dies on `sort` first |
| dtype functions | `can_cast` `isdtype` `result_type` | 12 failures | G-027/G-008 — needs a decision: rstsr promotion exists only as associated types, no token-level query |
| elementwise | `clip` (also rust-side) `log1p` (kernel TODO, G-058) | 3 failures (+17 log1p specials in the rust bucket) | |
| utility | `diff` | 4 failures | rust-side |

## Read-out

- Two thirds of the red map is **rust-side** (156): 65 = one work item
  (G-055 complex transcendental values/accuracy), then G-009 (29), G-058 (20),
  G-057 (12), G-054 (9). The absent-surface block is the other 114, of which
  28 names are the linalg decision and 14 the fft decision.
- Shim-side debt is now small and fully enumerated: **16 failures** across six
  registered items (G-032, G-040, G-046, G-047, G-048/049/050) — the cheapest
  next win.
- Latent items that currently cost no test: G-051 (`capabilities()["boolean
  indexing"] == True` while masks decline), G-035 (slice+array mixed
  indexing), G-016…G-020 (workarounds in place), G-070 (declined cases).
- New evidence this round: **G-013 upgraded from "ungraded edge" to graded** —
  `bitwise_invert` on uint64 is value-correct in the tensor, but `tolist` /
  scalar extraction wraps through i64 (`~[1] → -2`, `0xFFFFFFFFFFFFFFFE →
  -1`), which the suite grades (2 tests). Shim-side carrier fix.

## Reproduce

```bash
cd /home/a/rstsr_pack/rstsr-local-workspace/crates-interop/rstsr-faer-py
CARGO_PROFILE_RELEASE_OPT_LEVEL=0 maturin build --release -i "$TEST_PY" -o /tmp/wheels
"$TEST_PY" -m pip install --force-reinstall --no-deps /tmp/wheels/rstsr_faer_py-*.whl
cd /home/a/rstsr_pack/rstsr-improve-trajectory/2026-10-04-rstsr-faer-py/harness
FRESH=1 NO_EXPLAIN=1 MODULE=rstsr_faer.api CHUNKED=1 ./run.sh   # 1014/286/82, 19/19 chunks
```
