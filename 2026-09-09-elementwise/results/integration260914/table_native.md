## native: cand/refA per-bench ratio (3 paired passes, 1.00× = parity)

| benchmark | refA (master) | cand (patched) | ratio range | verdict |
|---|---|---|---|---|
| `add/add_bcast_2048x2048-faer16_f64/A` | 2.11 ms … 2.20 ms | 2.16 ms … 2.52 ms | 0.98–1.19× |
| `add/add_bcast_2048x2048-faer16_f64/B` | 273 µs … 279 µs | 264 µs … 273 µs | 0.97–1.00× |
| `add/add_bcast_2048x2048-serial_f64/A` | 5.55 ms … 5.64 ms | 5.55 ms … 5.60 ms | 0.99–1.01× |
| `add/add_bcast_2048x2048-serial_f64/B` | 1.42 ms … 1.51 ms | 1.40 ms … 1.49 ms | 0.93–1.01× |
| `add/add_contig_1000x777-faer16_f64/A` | 102 µs … 102 µs | 99 µs … 100 µs | 0.97–0.98× |
| `add/add_contig_1000x777-faer16_f64/B` | 102 µs … 102 µs | 99 µs … 100 µs | 0.97–0.98× |
| `add/add_contig_1000x777-serial_f64/A` | 114 µs … 121 µs | 114 µs … 116 µs | 0.96–1.00× |
| `add/add_contig_1000x777-serial_f64/B` | 114 µs … 121 µs | 114 µs … 119 µs | 0.95–1.00× |
| `add/add_contig_2048x2048-faer16_f32/A` | 251 µs … 253 µs | 229 µs … 232 µs | 0.91–0.93× | **WIN**
| `add/add_contig_2048x2048-faer16_f32/B` | 252 µs … 253 µs | 231 µs … 232 µs | 0.91–0.92× | **WIN**
| `add/add_contig_2048x2048-faer16_f64/A` | 2.48 ms … 2.66 ms | 2.48 ms … 2.66 ms | 0.99–1.00× |
| `add/add_contig_2048x2048-faer16_f64/B` | 538 µs … 584 µs | 517 µs … 575 µs | 0.88–1.07× |
| `add/add_contig_2048x2048-serial_f32/A` | 728 µs … 761 µs | 717 µs … 762 µs | 0.96–1.00× |
| `add/add_contig_2048x2048-serial_f32/B` | 734 µs … 744 µs | 716 µs … 757 µs | 0.96–1.02× |
| `add/add_contig_2048x2048-serial_f64/A` | 6.21 ms … 6.21 ms | 6.18 ms … 6.24 ms | 1.00–1.00× |
| `add/add_contig_2048x2048-serial_f64/B` | 2.04 ms … 2.13 ms | 2.07 ms … 2.17 ms | 0.97–1.06× |
| `add/add_contig_512x512-faer16_f32/A` | 57 µs … 57 µs | 55 µs … 55 µs | 0.96–0.97× |
| `add/add_contig_512x512-faer16_f32/B` | 58 µs … 58 µs | 56 µs … 57 µs | 0.97–0.97× |
| `add/add_contig_512x512-faer16_f64/A` | 62 µs … 63 µs | 61 µs … 62 µs | 0.98–0.99× |
| `add/add_contig_512x512-faer16_f64/B` | 63 µs … 63 µs | 62 µs … 62 µs | 0.98–0.98× |
| `add/add_contig_512x512-serial_f32/A` | 23 µs … 24 µs | 23 µs … 23 µs | 0.98–1.00× |
| `add/add_contig_512x512-serial_f32/B` | 23 µs … 23 µs | 23 µs … 23 µs | 0.98–0.98× |
| `add/add_contig_512x512-serial_f64/A` | 41 µs … 43 µs | 40 µs … 43 µs | 0.99–1.02× |
| `add/add_contig_512x512-serial_f64/B` | 41 µs … 44 µs | 40 µs … 43 µs | 0.95–1.02× |
| `add/add_contig_64x64-faer16_f64/A` | 27 µs … 27 µs | 26 µs … 27 µs | 0.99–1.00× |
| `add/add_contig_64x64-faer16_f64/B` | 27 µs … 27 µs | 26 µs … 27 µs | 0.98–1.00× |
| `add/add_contig_64x64-serial_f64/A` | 1.36 µs … 1.37 µs | 1.33 µs … 1.35 µs | 0.98–0.99× |
| `add/add_contig_64x64-serial_f64/B` | 1.32 µs … 1.32 µs | 1.30 µs … 1.32 µs | 0.99–1.00× |
| `add/add_strided_1000x777-faer16_f64/A` | 502 µs … 505 µs | 66 µs … 78 µs | 0.13–0.15× | **WIN**
| `add/add_strided_1000x777-faer16_f64/B` | 504 µs … 504 µs | 63 µs … 72 µs | 0.12–0.14× | **WIN**
| `add/add_strided_1000x777-serial_f64/A` | 3.67 ms … 3.68 ms | 407 µs … 420 µs | 0.11–0.11× | **WIN**
| `add/add_strided_1000x777-serial_f64/B` | 3.66 ms … 3.67 ms | 404 µs … 436 µs | 0.11–0.12× | **WIN**
| `add/add_strided_2048x2048-faer16_f64/A` | 3.94 ms … 4.34 ms | 3.13 ms … 3.29 ms | 0.75–0.83× | **WIN**
| `add/add_strided_2048x2048-faer16_f64/B` | 2.06 ms … 2.08 ms | 956 µs … 1.15 ms | 0.46–0.56× | **WIN**
| `add/add_strided_2048x2048-serial_f64/A` | 28.15 ms … 29.25 ms | 12.30 ms … 13.12 ms | 0.42–0.47× | **WIN**
| `add/add_strided_2048x2048-serial_f64/B` | 23.40 ms … 24.01 ms | 7.42 ms … 7.79 ms | 0.31–0.33× | **WIN**
| `add/add_strided_64x64-faer16_f64/A` | 89 µs … 89 µs | 5.32 µs … 5.89 µs | 0.06–0.07× | **WIN**
| `add/add_strided_64x64-faer16_f64/B` | 89 µs … 89 µs | 5.49 µs … 5.68 µs | 0.06–0.06× | **WIN**
| `add/add_strided_64x64-serial_f64/A` | 21 µs … 21 µs | 2.92 µs … 2.97 µs | 0.14–0.14× | **WIN**
| `add/add_strided_64x64-serial_f64/B` | 21 µs … 21 µs | 2.89 µs … 2.95 µs | 0.14–0.14× | **WIN**
| `add/add_stridedfirst_2048x2048-faer16_f64/A` | 3.85 ms … 4.44 ms | 3.21 ms … 3.26 ms | 0.73–0.83× | **WIN**
| `add/add_stridedfirst_2048x2048-faer16_f64/B` | 2.06 ms … 2.09 ms | 961 µs … 1.09 ms | 0.46–0.53× | **WIN**
| `add/add_stridedfirst_2048x2048-serial_f64/A` | 28.21 ms … 28.76 ms | 12.47 ms … 12.51 ms | 0.43–0.44× | **WIN**
| `add/add_stridedfirst_2048x2048-serial_f64/B` | 22.80 ms … 23.19 ms | 7.34 ms … 7.49 ms | 0.32–0.33× | **WIN**
| `add/addasgn_strided_2048x2048-faer16_f64/B` | 1.48 ms … 1.52 ms | 699 µs … 728 µs | 0.47–0.48× | **WIN**
| `add/addasgn_strided_2048x2048-serial_f64/B` | 17.41 ms … 17.89 ms | 5.94 ms … 6.10 ms | 0.34–0.35× | **WIN**
| `mul/mul_contig_2048x2048-faer16_f64/A` | 2.42 ms … 2.53 ms | 2.25 ms … 2.53 ms | 0.93–1.00× |
| `mul/mul_contig_2048x2048-faer16_f64/B` | 545 µs … 576 µs | 535 µs … 585 µs | 0.93–1.07× |
| `mul/mul_contig_2048x2048-serial_f64/A` | 6.09 ms … 6.25 ms | 6.15 ms … 6.21 ms | 0.98–1.01× |
| `mul/mul_contig_2048x2048-serial_f64/B` | 2.06 ms … 2.08 ms | 2.06 ms … 2.09 ms | 0.99–1.02× |
| `scale/scale_contig_2048x2048-faer16_f64/A` | 2.07 ms … 2.36 ms | 2.00 ms … 2.07 ms | 0.85–0.97× |
| `scale/scale_contig_2048x2048-faer16_f64/B` | 266 µs … 277 µs | 269 µs … 273 µs | 0.97–1.03× |
| `scale/scale_contig_2048x2048-serial_f64/A` | 5.29 ms … 5.37 ms | 5.32 ms … 5.40 ms | 0.99–1.02× |
| `scale/scale_contig_2048x2048-serial_f64/B` | 1.42 ms … 1.46 ms | 1.45 ms … 1.50 ms | 1.01–1.02× |
