# Candidate faer16 / volatile-cell repeated trials (phase 2)

Criterion medians per trial (µs). "full-suite" = the first candidate run
(after the long bench session — hottest state); later trials are focused
re-runs of the suspect cells on the same binaries. Clean-tree reference
bands from phase 1 (three independent clean builds).

## native (RUSTFLAGS=-C target-cpu=native)

| cell | baseline clean builds | cand full-suite | trial 2 | trial 3 | trial 4 | trial 5 | trial 6 | verdict |
|---|---|---|---|---|---|---|---|---|
| dot1d_1000-faer16_f64 | 4.47–4.55 | 5.21 | 4.59 | 4.42 | 5.24 | — | — | band overlap, no regression |
| dot1d_1e5-serial_c64 | 34.74 / 34.80 | 52.21 | 34.74 | — | — | 34.73 | 34.72 | trial-1 transient; settled = baseline |
| dot1d_1e5-faer16_c64 | 37.64 / 37.70 | 55.30* | 37.44 | — | — | 37.57 | 37.73 | (*full-suite misparse guard) settled = baseline |
| batched_am1 c64 serial | 1060.6 / 1248.3 | 1230.9 | 999.3 | — | — | 989.5 | 989.5 | improved 7–20% |
| batched_am1 c64 faer16 | 315.5 / 248.0 | 283.7 | 253.9 | — | — | 239.0 | 242.8 | improved |
| batched_axis0 faer16 | 113.7 / 120.8 (band 113–178) | 129.98 | 110.1 | 122.4 | 114.0 | — | — | band overlap, no regression |
| batched_strided faer16 | 301.3 (band 301–344) | 312.2 | — | 305.8 | 313.0 | — | — | +1.5–4%, untouched path, within band |
| batched_am1 faer16 (from full suite both cfg) | 141.4 / 148.9–150.8 | 139.7 | — | — | — | — | — | improved |

## portable (no RUSTFLAGS)

| cell | baseline clean builds | cand full-suite | ptrial 2 | ptrial 3 | verdict |
|---|---|---|---|---|---|
| dot1d_1000-faer16_f64 | 4.69–4.70 | 5.21 | 4.56 | 4.65 | band overlap |
| batched_am1 serial | 443.8–466.0 | 352.2 | 363.9 | 363.4 | **−21…−23%** |
| batched_am1 c64 serial | 1383.2 / 1585.8 | 1230.9 | 1222.6 | 1243.4 | −5…−11% vs clean run-1 (baseline-stage cell was lottery-hit +14.6%) |
| batched_axis0 serial | 1495.3–1613.9 | 538.4 | 526.9 | 546.0 | **−64…−65%** |
| batched_axis0 faer16 | 162.5–178.2 | 130.0 | 130.6 | 134.1 | −20% improved |
| batched_strided serial | 3053.8–3141.2 | 3045.8 | 3112.5 | 3112.4 | band overlap (untouched path) |
| batched_strided faer16 | 321.8–344.2 | 312.2 | 312.0 | 314.0 | flat/improved |
| batched_am1 faer16 | 148.9–150.8 | 139.7 | 143.3 | 138.7 | −5% improved |

Reading: single full-suite passes on this box carry ±5–8% (faer16) and up to
+67% single-cell codegen/thermal transients (f32 1e7 and c64 cells between
independent clean builds; see tables.md note). Gate decisions above use
medians of repeated trials, never the single full-suite pass.
