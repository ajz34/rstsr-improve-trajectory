## portable: cand/refA per-bench ratio (3 paired passes, 1.00× = parity)

| benchmark | refA (master) | cand (patched) | ratio range | verdict |
|---|---|---|---|---|
| `add/add_bcast_2048x2048-faer16_f64/A` | 1.94 ms … 2.23 ms | 1.92 ms … 2.30 ms | 0.94–1.05× |
| `add/add_bcast_2048x2048-faer16_f64/B` | 284 µs … 286 µs | 278 µs … 285 µs | 0.98–1.00× |
| `add/add_bcast_2048x2048-serial_f64/A` | 5.74 ms … 5.85 ms | 5.83 ms … 5.87 ms | 1.00–1.02× |
| `add/add_bcast_2048x2048-serial_f64/B` | 1.52 ms … 1.63 ms | 1.53 ms … 1.54 ms | 0.94–1.01× |
| `add/add_contig_1000x777-faer16_f64/A` | 99 µs … 100 µs | 99 µs … 100 µs | 1.00–1.01× |
| `add/add_contig_1000x777-faer16_f64/B` | 99 µs … 100 µs | 100 µs … 100 µs | 1.00–1.01× |
| `add/add_contig_1000x777-serial_f64/A` | 123 µs … 147 µs | 118 µs … 127 µs | 0.80–1.04× |
| `add/add_contig_1000x777-serial_f64/B` | 129 µs … 152 µs | 117 µs … 126 µs | 0.77–0.98× |
| `add/add_contig_2048x2048-faer16_f32/A` | 231 µs … 233 µs | 233 µs … 233 µs | 1.00–1.01× |
| `add/add_contig_2048x2048-faer16_f32/B` | 231 µs … 233 µs | 233 µs … 233 µs | 1.00–1.01× |
| `add/add_contig_2048x2048-faer16_f64/A` | 2.35 ms … 2.86 ms | 2.28 ms … 2.82 ms | 0.84–1.15× |
| `add/add_contig_2048x2048-faer16_f64/B` | 545 µs … 579 µs | 520 µs … 555 µs | 0.90–1.02× |
| `add/add_contig_2048x2048-serial_f32/A` | 761 µs … 780 µs | 749 µs … 782 µs | 0.97–1.03× |
| `add/add_contig_2048x2048-serial_f32/B` | 758 µs … 771 µs | 768 µs … 793 µs | 1.00–1.04× |
| `add/add_contig_2048x2048-serial_f64/A` | 6.44 ms … 6.86 ms | 6.47 ms … 6.51 ms | 0.95–1.01× |
| `add/add_contig_2048x2048-serial_f64/B` | 2.04 ms … 2.06 ms | 2.05 ms … 2.06 ms | 0.99–1.00× |
| `add/add_contig_512x512-faer16_f32/A` | 55 µs … 55 µs | 55 µs … 56 µs | 1.00–1.01× |
| `add/add_contig_512x512-faer16_f32/B` | 55 µs … 55 µs | 55 µs … 56 µs | 1.00–1.01× |
| `add/add_contig_512x512-faer16_f64/A` | 61 µs … 61 µs | 61 µs … 61 µs | 1.00–1.00× |
| `add/add_contig_512x512-faer16_f64/B` | 61 µs … 61 µs | 61 µs … 61 µs | 1.00–1.01× |
| `add/add_contig_512x512-serial_f32/A` | 23 µs … 24 µs | 23 µs … 24 µs | 1.00–1.00× |
| `add/add_contig_512x512-serial_f32/B` | 24 µs … 24 µs | 23 µs … 24 µs | 0.98–1.00× |
| `add/add_contig_512x512-serial_f64/A` | 48 µs … 50 µs | 45 µs … 50 µs | 0.91–1.03× |
| `add/add_contig_512x512-serial_f64/B` | 46 µs … 48 µs | 48 µs … 48 µs | 0.99–1.04× |
| `add/add_contig_64x64-faer16_f64/A` | 26 µs … 26 µs | 26 µs … 26 µs | 1.00–1.00× |
| `add/add_contig_64x64-faer16_f64/B` | 26 µs … 26 µs | 26 µs … 26 µs | 1.00–1.01× |
| `add/add_contig_64x64-serial_f64/A` | 1.46 µs … 1.47 µs | 1.45 µs … 1.47 µs | 1.00–1.00× |
| `add/add_contig_64x64-serial_f64/B` | 1.38 µs … 1.39 µs | 1.38 µs … 1.40 µs | 1.00–1.00× |
| `add/add_strided_1000x777-faer16_f64/A` | 515 µs … 521 µs | 60 µs … 68 µs | 0.12–0.13× | **WIN**
| `add/add_strided_1000x777-faer16_f64/B` | 515 µs … 518 µs | 58 µs … 65 µs | 0.11–0.13× | **WIN**
| `add/add_strided_1000x777-serial_f64/A` | 3.64 ms … 3.66 ms | 465 µs … 469 µs | 0.13–0.13× | **WIN**
| `add/add_strided_1000x777-serial_f64/B` | 3.63 ms … 3.64 ms | 421 µs … 459 µs | 0.12–0.13× | **WIN**
| `add/add_strided_2048x2048-faer16_f64/A` | 3.75 ms … 4.16 ms | 3.04 ms … 3.77 ms | 0.75–0.91× | **WIN**
| `add/add_strided_2048x2048-faer16_f64/B` | 2.12 ms … 2.14 ms | 979 µs … 1.13 ms | 0.46–0.53× | **WIN**
| `add/add_strided_2048x2048-serial_f64/A` | 28.09 ms … 29.64 ms | 12.51 ms … 12.55 ms | 0.42–0.45× | **WIN**
| `add/add_strided_2048x2048-serial_f64/B` | 22.79 ms … 23.43 ms | 7.54 ms … 7.59 ms | 0.32–0.33× | **WIN**
| `add/add_strided_64x64-faer16_f64/A` | 91 µs … 92 µs | 5.04 µs … 5.39 µs | 0.06–0.06× | **WIN**
| `add/add_strided_64x64-faer16_f64/B` | 91 µs … 92 µs | 5.08 µs … 5.33 µs | 0.06–0.06× | **WIN**
| `add/add_strided_64x64-serial_f64/A` | 21 µs … 21 µs | 3.17 µs … 3.18 µs | 0.15–0.15× | **WIN**
| `add/add_strided_64x64-serial_f64/B` | 21 µs … 21 µs | 3.11 µs … 3.16 µs | 0.15–0.15× | **WIN**
| `add/add_stridedfirst_2048x2048-faer16_f64/A` | 3.94 ms … 4.21 ms | 3.09 ms … 3.21 ms | 0.73–0.81× | **WIN**
| `add/add_stridedfirst_2048x2048-faer16_f64/B` | 2.10 ms … 2.16 ms | 957 µs … 1.06 ms | 0.44–0.50× | **WIN**
| `add/add_stridedfirst_2048x2048-serial_f64/A` | 28.13 ms … 28.87 ms | 12.47 ms … 12.57 ms | 0.43–0.45× | **WIN**
| `add/add_stridedfirst_2048x2048-serial_f64/B` | 22.45 ms … 22.93 ms | 7.51 ms … 7.54 ms | 0.33–0.34× | **WIN**
| `add/addasgn_strided_2048x2048-faer16_f64/B` | 1.49 ms … 1.53 ms | 673 µs … 874 µs | 0.45–0.57× | **WIN**
| `add/addasgn_strided_2048x2048-serial_f64/B` | 17.84 ms … 18.12 ms | 5.84 ms … 5.86 ms | 0.32–0.33× | **WIN**
| `mul/mul_contig_2048x2048-faer16_f64/A` | 2.30 ms … 2.71 ms | 2.31 ms … 2.95 ms | 0.85–1.28× |
| `mul/mul_contig_2048x2048-faer16_f64/B` | 547 µs … 568 µs | 537 µs … 567 µs | 0.98–1.00× |
| `mul/mul_contig_2048x2048-serial_f64/A` | 6.52 ms … 6.89 ms | 6.40 ms … 6.52 ms | 0.95–0.98× |
| `mul/mul_contig_2048x2048-serial_f64/B` | 2.04 ms … 2.07 ms | 2.04 ms … 2.05 ms | 0.99–1.00× |
| `scale/scale_contig_2048x2048-faer16_f64/A` | 2.03 ms … 2.37 ms | 1.97 ms … 2.56 ms | 0.95–1.08× |
| `scale/scale_contig_2048x2048-faer16_f64/B` | 274 µs … 281 µs | 280 µs … 284 µs | 1.00–1.04× |
| `scale/scale_contig_2048x2048-serial_f64/A` | 5.38 ms … 5.46 ms | 5.33 ms … 5.52 ms | 0.99–1.01× |
| `scale/scale_contig_2048x2048-serial_f64/B` | 1.55 ms … 1.67 ms | 1.42 ms … 1.55 ms | 0.85–1.00× |
