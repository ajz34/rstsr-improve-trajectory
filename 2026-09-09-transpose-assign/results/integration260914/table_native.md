## native: cand/refA per-bench ratio (3 paired passes, 1.00× = parity)

| benchmark | refA (master) | cand (patched) | ratio range | verdict |
|---|---|---|---|---|
| `assign_gates/contig_large_2048x2048-faer16-f64/assign` | 1.31 ms … 1.33 ms | 1.30 ms … 1.34 ms | 0.99–1.02× |
| `assign_gates/contig_large_2048x2048-serial-f64/assign` | 1.31 ms … 1.32 ms | 1.31 ms … 1.33 ms | 1.00–1.01× |
| `assign_gates/contig_small_64x64-faer16-f64/assign` | 1.04 µs … 1.06 µs | 1.05 µs … 1.08 µs | 0.99–1.04× |
| `assign_gates/contig_small_64x64-serial-f64/assign` | 1.01 µs … 1.02 µs | 1.02 µs … 1.03 µs | 1.01–1.02× |
| `assign_gates/sliced_large_2048x2048-faer16-f64/assign` | 700 µs … 707 µs | 720 µs … 727 µs | 1.02–1.04× |
| `assign_gates/sliced_large_2048x2048-serial-f64/assign` | 6.31 ms … 6.34 ms | 6.35 ms … 6.38 ms | 1.00–1.01× |
| `assign_gates/sliced_odd_1000x777-faer16-f64/assign` | 243 µs … 244 µs | 248 µs … 251 µs | 1.02–1.03× |
| `assign_gates/sliced_odd_1000x777-serial-f64/assign` | 1.16 ms … 1.18 ms | 1.17 ms … 1.18 ms | 1.00–1.01× |
| `assign_gates/sliced_t_large_2048x2048-faer16-f64/assign` | 819 µs … 826 µs | 866 µs … 890 µs | 1.05–1.08× |
| `assign_gates/sliced_t_large_2048x2048-serial-f64/assign` | 8.27 ms … 8.48 ms | 8.34 ms … 8.51 ms | 1.00–1.01× |
| `assign_gates/sliced_t_odd_1000x777-faer16-f64/assign` | 246 µs … 248 µs | 250 µs … 252 µs | 1.01–1.02× |
| `assign_gates/sliced_t_odd_1000x777-serial-f64/assign` | 1.19 ms … 1.20 ms | 1.20 ms … 1.20 ms | 1.00–1.01× |
| `orderchange_extra/bcast_t_large_2048x2048-faer16-f64/A` | 2.45 ms … 2.74 ms | 2.29 ms … 2.59 ms | 0.85–0.96× |
| `orderchange_extra/bcast_t_large_2048x2048-faer16-f64/B` | 1.17 ms … 1.18 ms | 362 µs … 364 µs | 0.31–0.31× | **WIN**
| `orderchange_extra/bcast_t_large_2048x2048-serial-f64/A` | 16.74 ms … 16.81 ms | 5.31 ms … 5.37 ms | 0.32–0.32× | **WIN**
| `orderchange_extra/bcast_t_large_2048x2048-serial-f64/B` | 12.47 ms … 12.58 ms | 1.47 ms … 1.54 ms | 0.12–0.12× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-faer16-f64/A` | 353 µs … 355 µs | 87 µs … 89 µs | 0.24–0.25× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-faer16-f64/B` | 353 µs … 356 µs | 100 µs … 118 µs | 0.28–0.33× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-serial-f64/A` | 2.33 ms … 2.35 ms | 217 µs … 218 µs | 0.09–0.09× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-serial-f64/B` | 2.32 ms … 2.33 ms | 225 µs … 225 µs | 0.10–0.10× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-faer16-f64/A` | 10.83 ms … 10.99 ms | 2.72 ms … 2.74 ms | 0.25–0.25× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-faer16-f64/B` | 1.45 ms … 1.46 ms | 537 µs … 552 µs | 0.37–0.38× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-serial-f64/A` | 19.65 ms … 19.93 ms | 10.45 ms … 10.64 ms | 0.53–0.54× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-serial-f64/B` | 16.99 ms … 17.37 ms | 5.77 ms … 5.86 ms | 0.33–0.35× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-faer16-f64/A` | 374 µs … 378 µs | 93 µs … 96 µs | 0.25–0.25× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-faer16-f64/B` | 356 µs … 359 µs | 95 µs … 100 µs | 0.27–0.28× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-serial-f64/A` | 2.39 ms … 2.42 ms | 243 µs … 244 µs | 0.10–0.10× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-serial-f64/B` | 2.39 ms … 2.39 ms | 244 µs … 244 µs | 0.10–0.10× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f32/A` | 1.34 ms … 1.34 ms | 390 µs … 402 µs | 0.29–0.30× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f32/B` | 1.33 ms … 1.35 ms | 394 µs … 406 µs | 0.29–0.30× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f64/A` | 3.03 ms … 3.36 ms | 2.39 ms … 2.73 ms | 0.71–0.90× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f64/B` | 1.45 ms … 1.46 ms | 537 µs … 547 µs | 0.37–0.38× | **WIN**
| `transpose_copy/large_2048x2048-serial_f32/A` | 15.10 ms … 15.57 ms | 5.16 ms … 5.37 ms | 0.33–0.36× | **WIN**
| `transpose_copy/large_2048x2048-serial_f32/B` | 15.11 ms … 15.55 ms | 5.09 ms … 5.38 ms | 0.33–0.36× | **WIN**
| `transpose_copy/large_2048x2048-serial_f64/A` | 22.09 ms … 22.84 ms | 10.58 ms … 10.81 ms | 0.47–0.48× | **WIN**
| `transpose_copy/large_2048x2048-serial_f64/B` | 16.98 ms … 17.43 ms | 5.63 ms … 5.93 ms | 0.33–0.35× | **WIN**
| `transpose_copy/medium_512x512-faer16_f64/A` | 202 µs … 204 µs | 51 µs … 54 µs | 0.25–0.26× | **WIN**
| `transpose_copy/medium_512x512-faer16_f64/B` | 204 µs … 204 µs | 53 µs … 60 µs | 0.26–0.29× | **WIN**
| `transpose_copy/medium_512x512-serial_f64/A` | 801 µs … 819 µs | 299 µs … 311 µs | 0.37–0.38× | **WIN**
| `transpose_copy/medium_512x512-serial_f64/B` | 804 µs … 808 µs | 300 µs … 313 µs | 0.37–0.39× | **WIN**
| `transpose_copy/oddT_777x1000-faer16_f64/A` | 358 µs … 360 µs | 94 µs … 96 µs | 0.26–0.27× | **WIN**
| `transpose_copy/oddT_777x1000-faer16_f64/B` | 360 µs … 361 µs | 99 µs … 104 µs | 0.27–0.29× | **WIN**
| `transpose_copy/oddT_777x1000-serial_f64/A` | 2.41 ms … 2.44 ms | 239 µs … 300 µs | 0.10–0.12× | **WIN**
| `transpose_copy/oddT_777x1000-serial_f64/B` | 2.39 ms … 2.40 ms | 295 µs … 327 µs | 0.12–0.14× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f32/A` | 355 µs … 357 µs | 61 µs … 64 µs | 0.17–0.18× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f32/B` | 354 µs … 358 µs | 64 µs … 68 µs | 0.18–0.19× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f64/A` | 359 µs … 360 µs | 91 µs … 98 µs | 0.25–0.27× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f64/B` | 358 µs … 362 µs | 95 µs … 105 µs | 0.26–0.29× | **WIN**
| `transpose_copy/odd_1000x777-serial_f32/A` | 2.37 ms … 2.39 ms | 281 µs … 281 µs | 0.12–0.12× | **WIN**
| `transpose_copy/odd_1000x777-serial_f32/B` | 2.36 ms … 2.38 ms | 281 µs … 281 µs | 0.12–0.12× | **WIN**
| `transpose_copy/odd_1000x777-serial_f64/A` | 2.38 ms … 2.45 ms | 243 µs … 300 µs | 0.10–0.13× | **WIN**
| `transpose_copy/odd_1000x777-serial_f64/B` | 2.39 ms … 2.41 ms | 246 µs … 253 µs | 0.10–0.11× | **WIN**
| `transpose_copy/small_64x64-faer16_f64/A` | 13 µs … 13 µs | 1.55 µs … 1.85 µs | 0.12–0.14× | **WIN**
| `transpose_copy/small_64x64-faer16_f64/B` | 14 µs … 14 µs | 2.21 µs … 2.26 µs | 0.16–0.16× | **WIN**
| `transpose_copy/small_64x64-serial_f64/A` | 13 µs … 14 µs | 1.51 µs … 1.58 µs | 0.11–0.12× | **WIN**
| `transpose_copy/small_64x64-serial_f64/B` | 14 µs … 14 µs | 2.19 µs … 2.33 µs | 0.16–0.17× | **WIN**
