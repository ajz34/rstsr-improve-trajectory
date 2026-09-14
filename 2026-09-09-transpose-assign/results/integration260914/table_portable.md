## portable: cand/refA per-bench ratio (3 paired passes, 1.00× = parity)

| benchmark | refA (master) | cand (patched) | ratio range | verdict |
|---|---|---|---|---|
| `assign_gates/contig_large_2048x2048-faer16-f64/assign` | 1.32 ms … 1.35 ms | 1.36 ms … 1.38 ms | 1.02–1.03× |
| `assign_gates/contig_large_2048x2048-serial-f64/assign` | 1.34 ms … 1.36 ms | 1.24 ms … 1.35 ms | 0.93–1.00× |
| `assign_gates/contig_small_64x64-faer16-f64/assign` | 1.07 µs … 1.09 µs | 1.07 µs … 1.10 µs | 0.99–1.02× |
| `assign_gates/contig_small_64x64-serial-f64/assign` | 1.03 µs … 1.04 µs | 1.03 µs … 1.04 µs | 0.99–1.01× |
| `assign_gates/sliced_large_2048x2048-faer16-f64/assign` | 711 µs … 727 µs | 701 µs … 711 µs | 0.98–0.99× |
| `assign_gates/sliced_large_2048x2048-serial-f64/assign` | 6.34 ms … 6.44 ms | 6.33 ms … 6.35 ms | 0.98–1.00× |
| `assign_gates/sliced_odd_1000x777-faer16-f64/assign` | 248 µs … 253 µs | 247 µs … 248 µs | 0.98–0.99× |
| `assign_gates/sliced_odd_1000x777-serial-f64/assign` | 1.18 ms … 1.19 ms | 1.17 ms … 1.17 ms | 0.98–1.00× |
| `assign_gates/sliced_t_large_2048x2048-faer16-f64/assign` | 827 µs … 950 µs | 827 µs … 864 µs | 0.87–1.00× |
| `assign_gates/sliced_t_large_2048x2048-serial-f64/assign` | 8.34 ms … 8.37 ms | 8.25 ms … 8.62 ms | 0.99–1.03× |
| `assign_gates/sliced_t_odd_1000x777-faer16-f64/assign` | 252 µs … 257 µs | 250 µs … 251 µs | 0.97–0.99× |
| `assign_gates/sliced_t_odd_1000x777-serial-f64/assign` | 1.19 ms … 1.23 ms | 1.19 ms … 1.21 ms | 0.98–1.00× |
| `orderchange_extra/bcast_t_large_2048x2048-faer16-f64/A` | 2.60 ms … 2.67 ms | 2.17 ms … 2.47 ms | 0.81–0.95× | **WIN**
| `orderchange_extra/bcast_t_large_2048x2048-faer16-f64/B` | 1.18 ms … 1.19 ms | 322 µs … 332 µs | 0.27–0.28× | **WIN**
| `orderchange_extra/bcast_t_large_2048x2048-serial-f64/A` | 16.76 ms … 16.77 ms | 5.22 ms … 5.35 ms | 0.31–0.32× | **WIN**
| `orderchange_extra/bcast_t_large_2048x2048-serial-f64/B` | 12.54 ms … 12.59 ms | 1.48 ms … 1.49 ms | 0.12–0.12× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-faer16-f64/A` | 359 µs … 361 µs | 90 µs … 91 µs | 0.25–0.25× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-faer16-f64/B` | 361 µs … 367 µs | 97 µs … 98 µs | 0.27–0.27× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-serial-f64/A` | 2.31 ms … 2.33 ms | 218 µs … 220 µs | 0.09–0.10× | **WIN**
| `orderchange_extra/bcast_t_odd_1000x777-serial-f64/B` | 2.31 ms … 2.34 ms | 224 µs … 226 µs | 0.10–0.10× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-faer16-f64/A` | 10.43 ms … 11.60 ms | 2.64 ms … 2.69 ms | 0.23–0.25× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-faer16-f64/B` | 1.47 ms … 1.49 ms | 546 µs … 560 µs | 0.37–0.38× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-serial-f64/A` | 19.77 ms … 20.35 ms | 10.33 ms … 10.45 ms | 0.51–0.53× | **WIN**
| `orderchange_extra/to_fcontig_large_2048x2048-serial-f64/B` | 17.23 ms … 17.40 ms | 5.66 ms … 5.83 ms | 0.33–0.33× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-faer16-f64/A` | 378 µs … 381 µs | 91 µs … 95 µs | 0.24–0.25× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-faer16-f64/B` | 364 µs … 366 µs | 91 µs … 99 µs | 0.25–0.27× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-serial-f64/A` | 2.41 ms … 2.42 ms | 247 µs … 253 µs | 0.10–0.10× | **WIN**
| `orderchange_extra/to_fcontig_odd_1000x777-serial-f64/B` | 2.40 ms … 2.41 ms | 250 µs … 255 µs | 0.10–0.11× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f32/A` | 1.33 ms … 1.36 ms | 393 µs … 415 µs | 0.29–0.31× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f32/B` | 1.34 ms … 1.37 ms | 397 µs … 412 µs | 0.29–0.31× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f64/A` | 3.42 ms … 3.63 ms | 2.81 ms … 2.92 ms | 0.78–0.85× | **WIN**
| `transpose_copy/large_2048x2048-faer16_f64/B` | 1.45 ms … 1.47 ms | 545 µs … 552 µs | 0.37–0.38× | **WIN**
| `transpose_copy/large_2048x2048-serial_f32/A` | 15.14 ms … 15.38 ms | 5.24 ms … 5.65 ms | 0.35–0.37× | **WIN**
| `transpose_copy/large_2048x2048-serial_f32/B` | 15.15 ms … 15.35 ms | 5.26 ms … 5.64 ms | 0.35–0.37× | **WIN**
| `transpose_copy/large_2048x2048-serial_f64/A` | 21.93 ms … 22.64 ms | 10.42 ms … 10.84 ms | 0.46–0.48× | **WIN**
| `transpose_copy/large_2048x2048-serial_f64/B` | 16.92 ms … 17.24 ms | 5.88 ms … 5.90 ms | 0.34–0.35× | **WIN**
| `transpose_copy/medium_512x512-faer16_f64/A` | 205 µs … 207 µs | 50 µs … 52 µs | 0.24–0.25× | **WIN**
| `transpose_copy/medium_512x512-faer16_f64/B` | 205 µs … 210 µs | 52 µs … 64 µs | 0.25–0.31× | **WIN**
| `transpose_copy/medium_512x512-serial_f64/A` | 800 µs … 809 µs | 297 µs … 311 µs | 0.37–0.39× | **WIN**
| `transpose_copy/medium_512x512-serial_f64/B` | 802 µs … 815 µs | 299 µs … 311 µs | 0.37–0.38× | **WIN**
| `transpose_copy/oddT_777x1000-faer16_f64/A` | 363 µs … 365 µs | 93 µs … 100 µs | 0.26–0.28× | **WIN**
| `transpose_copy/oddT_777x1000-faer16_f64/B` | 366 µs … 371 µs | 94 µs … 97 µs | 0.25–0.27× | **WIN**
| `transpose_copy/oddT_777x1000-serial_f64/A` | 2.38 ms … 2.41 ms | 299 µs … 301 µs | 0.12–0.13× | **WIN**
| `transpose_copy/oddT_777x1000-serial_f64/B` | 2.38 ms … 2.41 ms | 241 µs … 300 µs | 0.10–0.12× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f32/A` | 360 µs … 361 µs | 63 µs … 65 µs | 0.18–0.18× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f32/B` | 364 µs … 367 µs | 63 µs … 69 µs | 0.17–0.19× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f64/A` | 364 µs … 368 µs | 91 µs … 98 µs | 0.25–0.27× | **WIN**
| `transpose_copy/odd_1000x777-faer16_f64/B` | 369 µs … 370 µs | 100 µs … 104 µs | 0.27–0.28× | **WIN**
| `transpose_copy/odd_1000x777-serial_f32/A` | 2.37 ms … 2.38 ms | 226 µs … 229 µs | 0.10–0.10× | **WIN**
| `transpose_copy/odd_1000x777-serial_f32/B` | 2.37 ms … 2.39 ms | 227 µs … 231 µs | 0.10–0.10× | **WIN**
| `transpose_copy/odd_1000x777-serial_f64/A` | 2.38 ms … 2.41 ms | 249 µs … 253 µs | 0.10–0.11× | **WIN**
| `transpose_copy/odd_1000x777-serial_f64/B` | 2.39 ms … 2.59 ms | 249 µs … 256 µs | 0.10–0.11× | **WIN**
| `transpose_copy/small_64x64-faer16_f64/A` | 13 µs … 14 µs | 1.62 µs … 1.65 µs | 0.12–0.12× | **WIN**
| `transpose_copy/small_64x64-faer16_f64/B` | 14 µs … 14 µs | 2.19 µs … 2.25 µs | 0.16–0.16× | **WIN**
| `transpose_copy/small_64x64-serial_f64/A` | 13 µs … 14 µs | 1.53 µs … 1.55 µs | 0.11–0.11× | **WIN**
| `transpose_copy/small_64x64-serial_f64/B` | 14 µs … 14 µs | 2.16 µs … 2.25 µs | 0.16–0.16× | **WIN**
