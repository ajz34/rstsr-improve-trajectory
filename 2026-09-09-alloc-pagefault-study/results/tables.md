# T7 tables (regenerated from criterion raw estimates)

## native

| group | case | A_alloc | B_reuse | B_bound | best B | rider (A-bestB)/A |
|---|---|---|---|---|---|---|
| add_contig | large_2048x2048-faer16_f64 | 2.741 ms | bound=718.82 µs / 2pass=2.120 ms / mutc=629.25 µs | - | 629.25 µs | 77% |
| add_contig | large_2048x2048-serial_f64 | 6.318 ms | bound=2.173 ms / 2pass=2.518 ms / mutc=2.154 ms | - | 2.154 ms | 66% |
| add_contig | medium_512x512-faer16_f64 | 70.33 µs | bound=163.61 µs / 2pass=119.93 µs / mutc=69.65 µs | - | 69.65 µs | 1% |
| add_contig | medium_512x512-serial_f64 | 41.16 µs | bound=41.16 µs / 2pass=64.66 µs / mutc=42.17 µs | - | 41.16 µs | 0% |
| add_contig | odd_1000x777-faer16_f64 | 117.39 µs | bound=200.07 µs / 2pass=238.35 µs / mutc=116.51 µs | - | 116.51 µs | 1% |
| add_contig | odd_1000x777-serial_f64 | 116.72 µs | bound=111.04 µs / 2pass=157.93 µs / mutc=115.01 µs | - | 111.04 µs | 5% |
| add_strided | large_2048x2048-faer16_f64 | 4.144 ms | bound=1.306 ms / 2pass=3.533 ms / mutc=2.122 ms | - | 1.306 ms | 68% |
| add_strided | large_2048x2048-serial_f64 | 28.298 ms | bound=17.249 ms / 2pass=19.878 ms / mutc=23.041 ms | - | 17.249 ms | 39% |
| fill_full | large_2048x2048-faer16_f64 | 4.576 ms | fill=534.75 µs | - | 534.75 µs | 88% |
| fill_full | large_2048x2048-serial_f64 | 4.482 ms | fill=542.00 µs | - | 542.00 µs | 88% |
| sum_axis0 | large_2048x2048-faer16_f64 | 385.84 µs | bound=725.00 µs | - | 725.00 µs | -88% |
| sum_axis0 | large_2048x2048-serial_f64 | 1.844 ms | bound=613.72 µs | - | 613.72 µs | 67% |
| transpose_copy | large_2048x2048-faer16_f64 | 3.544 ms | bound=1.512 ms / assign=1.507 ms | - | 1.507 ms | 57% |
| transpose_copy | large_2048x2048-serial_f64 | 22.809 ms | bound=17.061 ms / assign=17.309 ms | - | 17.061 ms | 25% |
| transpose_copy | odd_1000x777-faer16_f64 | 357.63 µs | bound=206.91 µs / assign=357.68 µs | - | 206.91 µs | 42% |
| transpose_copy | odd_1000x777-serial_f64 | 2.430 ms | bound=329.78 µs / assign=2.443 ms | - | 329.78 µs | 86% |
| vecdot_batched | batched_4096x512-faer16_f64 | 138.88 µs | bound=220.57 µs | - | 220.57 µs | -59% |
| vecdot_batched | batched_4096x512-serial_f64 | 455.41 µs | bound=736.43 µs | - | 736.43 µs | -62% |
| zeros | large_2048x2048-faer16_f64 | 2.38 µs | fill=529.05 µs | - | 529.05 µs | -22110% |
| zeros | large_2048x2048-serial_f64 | 2.37 µs | fill=521.79 µs | - | 521.79 µs | -21874% |

## portable

| group | case | A_alloc | B_reuse | B_bound | best B | rider (A-bestB)/A |
|---|---|---|---|---|---|---|
| add_contig | large_2048x2048-faer16_f64 | 3.056 ms | bound=700.13 µs / 2pass=2.236 ms / mutc=654.82 µs | - | 654.82 µs | 79% |
| add_contig | large_2048x2048-serial_f64 | 6.381 ms | bound=2.386 ms / 2pass=2.505 ms / mutc=2.103 ms | - | 2.103 ms | 67% |
| add_contig | medium_512x512-faer16_f64 | 68.24 µs | bound=171.59 µs / 2pass=131.82 µs / mutc=69.00 µs | - | 69.00 µs | -1% |
| add_contig | medium_512x512-serial_f64 | 49.74 µs | bound=47.86 µs / 2pass=65.01 µs / mutc=49.44 µs | - | 47.86 µs | 4% |
| add_contig | odd_1000x777-faer16_f64 | 113.86 µs | bound=206.40 µs / 2pass=275.49 µs / mutc=113.44 µs | - | 113.44 µs | 0% |
| add_contig | odd_1000x777-serial_f64 | 154.10 µs | bound=120.84 µs / 2pass=204.96 µs / mutc=153.71 µs | - | 120.84 µs | 22% |
| add_strided | large_2048x2048-faer16_f64 | 4.342 ms | bound=1.221 ms / 2pass=3.146 ms / mutc=2.104 ms | - | 1.221 ms | 72% |
| add_strided | large_2048x2048-serial_f64 | 28.327 ms | bound=17.670 ms / 2pass=20.008 ms / mutc=22.824 ms | - | 17.670 ms | 38% |
| fill_full | large_2048x2048-faer16_f64 | 4.646 ms | fill=595.18 µs | - | 595.18 µs | 87% |
| fill_full | large_2048x2048-serial_f64 | 4.571 ms | fill=578.99 µs | - | 578.99 µs | 87% |
| sum_axis0 | large_2048x2048-faer16_f64 | 475.23 µs | bound=647.55 µs | - | 647.55 µs | -36% |
| sum_axis0 | large_2048x2048-serial_f64 | 1.403 ms | bound=607.53 µs | - | 607.53 µs | 57% |
| transpose_copy | large_2048x2048-faer16_f64 | 3.364 ms | bound=1.195 ms / assign=1.482 ms | - | 1.195 ms | 64% |
| transpose_copy | large_2048x2048-serial_f64 | 22.804 ms | bound=17.068 ms / assign=17.299 ms | - | 17.068 ms | 25% |
| transpose_copy | odd_1000x777-faer16_f64 | 362.48 µs | bound=203.29 µs / assign=364.33 µs | - | 203.29 µs | 44% |
| transpose_copy | odd_1000x777-serial_f64 | 2.433 ms | bound=330.20 µs / assign=2.435 ms | - | 330.20 µs | 86% |
| vecdot_batched | batched_4096x512-faer16_f64 | 146.42 µs | bound=220.93 µs | - | 220.93 µs | -51% |
| vecdot_batched | batched_4096x512-serial_f64 | 468.74 µs | bound=736.76 µs | - | 736.76 µs | -57% |
| zeros | large_2048x2048-faer16_f64 | 2.54 µs | fill=594.92 µs | - | 594.92 µs | -23278% |
| zeros | large_2048x2048-serial_f64 | 2.51 µs | fill=572.08 µs | - | 572.08 µs | -22666% |
