# T2' phase-2 candidate vs phase-1 baseline

criterion `change`% vs the saved phase-1 `portable`/`native` baselines;
**candidate = mean of two full-suite runs** (r1/r2 columns show each run's
mid estimate; the build-to-build layout lottery on ~0.5 ms streaming cells
is ±5% per the T4' precedent, so single-run cells in the ±2-5% band are
not conclusive). Negative = faster than phase 1.


## config: portable

| case | baseline ms | cand r1 | cand r2 | change r1 | change r2 | mean change |
|---|---|---|---|---|---|---|
| reduce1d/large_1e7-faer16_f64/max_all | 0.2537 | 0.1598 | 0.1552 | -37.02% | -38.04% | -37.53% <-- IMPROVED |
| reduce1d/large_1e7-faer16_f64/mean_all | 0.1571 | 0.1516 | 0.1512 | -3.52% | -6.04% | -4.78% |
| reduce1d/large_1e7-faer16_f64/min_all | 0.2517 | 0.1616 | 0.1565 | -35.79% | -37.94% | -36.86% <-- IMPROVED |
| reduce1d/large_1e7-faer16_f64/norm_all | 0.1627 | 0.1577 | 0.1581 | -3.10% | -2.80% | -2.95% |
| reduce1d/large_1e7-faer16_f64/sum_all | 0.1710 | 0.1586 | 0.1503 | -7.27% | -9.93% | -8.60% |
| reduce1d/large_1e7-faer16_f64/var_all | 0.2536 | 0.2406 | 0.2417 | -5.14% | -4.45% | -4.79% |
| reduce1d/large_1e7-serial_f64/max_all | 2.1535 | 1.2828 | 1.2701 | -40.43% | -41.02% | -40.73% <-- IMPROVED |
| reduce1d/large_1e7-serial_f64/mean_all | 1.2945 | 1.2560 | 1.2466 | -2.97% | -3.70% | -3.34% |
| reduce1d/large_1e7-serial_f64/min_all | 2.1482 | 1.2391 | 1.2625 | -42.32% | -41.23% | -41.78% <-- IMPROVED |
| reduce1d/large_1e7-serial_f64/norm_all | 1.3311 | 1.2927 | 1.2789 | -2.89% | -3.93% | -3.41% |
| reduce1d/large_1e7-serial_f64/sum_all | 1.2793 | 1.2567 | 1.2396 | -1.77% | -3.10% | -2.43% |
| reduce1d/large_1e7-serial_f64/var_all | 1.9069 | 1.9269 | 1.9161 | +1.05% | +0.48% | +0.77% |
| reduce2d/large_2048x2048-faer16_f64/max_axis0 | 0.4209 | 0.3893 | 0.3960 | -7.50% | -8.09% | -7.80% |
| reduce2d/large_2048x2048-faer16_f64/mean_axis1 | 0.1004 | 0.1005 | 0.0976 | +0.05% | -2.09% | -1.02% |
| reduce2d/large_2048x2048-faer16_f64/min_axis0 | 0.4305 | 0.3942 | 0.4233 | -8.44% | -4.87% | -6.66% |
| reduce2d/large_2048x2048-faer16_f64/norm_all | 0.0567 | 0.0563 | 0.0564 | -0.78% | -0.83% | -0.80% |
| reduce2d/large_2048x2048-faer16_f64/sum_all | 0.0554 | 0.0548 | 0.0545 | -0.95% | -2.18% | -1.57% |
| reduce2d/large_2048x2048-faer16_f64/sum_axis0 | 0.3701 | 0.4137 | 0.4437 | +11.78% | +9.49% | +10.63% <-- REGRESSION? |
| reduce2d/large_2048x2048-faer16_f64/sum_axis1 | 0.0994 | 0.1002 | 0.0974 | +0.81% | -0.58% | +0.12% |
| reduce2d/large_2048x2048-faer16_f64/var_axis1 | 0.1286 | 0.1290 | 0.1292 | +0.30% | -0.63% | -0.16% |
| reduce2d/large_2048x2048-serial_f64/max_axis0 | 1.7449 | 1.2096 | 1.1955 | -30.68% | -31.48% | -31.08% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/mean_axis1 | 0.4891 | 0.4826 | 0.4871 | -1.32% | +0.65% | -0.34% |
| reduce2d/large_2048x2048-serial_f64/min_axis0 | 2.9347 | 1.2002 | 1.1979 | -59.10% | -59.18% | -59.14% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/norm_all | 0.4759 | 0.4717 | 0.4881 | -0.87% | -0.49% | -0.68% |
| reduce2d/large_2048x2048-serial_f64/sum_all | 0.4463 | 0.4646 | 0.4626 | +4.10% | +4.32% | +4.21% <-- REGRESSION? |
| reduce2d/large_2048x2048-serial_f64/sum_axis0 | 1.4349 | 1.2077 | 1.1971 | -15.83% | -16.57% | -16.20% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/sum_axis1 | 0.4657 | 0.4886 | 0.4859 | +4.92% | +4.64% | +4.78% <-- REGRESSION? |
| reduce2d/large_2048x2048-serial_f64/var_axis1 | 0.8695 | 0.8659 | 0.8809 | -0.41% | +1.31% | +0.45% |
| reduce2d/medium_512x512-faer16_f64/max_axis0 | 0.0669 | 0.0981 | 0.0790 | +46.68% | +27.40% | +37.04% <-- REGRESSION? |
| reduce2d/medium_512x512-faer16_f64/mean_axis1 | 0.0460 | 0.0460 | 0.0462 | +0.07% | +0.25% | +0.16% |
| reduce2d/medium_512x512-faer16_f64/min_axis0 | 0.0528 | 0.0914 | 0.0885 | +73.31% | +73.12% | +73.22% <-- REGRESSION? |
| reduce2d/medium_512x512-faer16_f64/norm_all | 0.0164 | 0.0162 | 0.0163 | -1.10% | -0.68% | -0.89% |
| reduce2d/medium_512x512-faer16_f64/sum_all | 0.0163 | 0.0160 | 0.0161 | -1.59% | -0.79% | -1.19% |
| reduce2d/medium_512x512-faer16_f64/sum_axis0 | 0.0816 | 0.0811 | 0.0562 | -0.65% | -23.86% | -12.26% <-- IMPROVED |
| reduce2d/medium_512x512-faer16_f64/sum_axis1 | 0.0465 | 0.0456 | 0.0457 | -2.05% | -1.64% | -1.85% |
| reduce2d/medium_512x512-faer16_f64/var_axis1 | 0.0479 | 0.0479 | 0.0477 | +0.03% | -0.12% | -0.05% |
| reduce2d/medium_512x512-serial_f64/max_axis0 | 0.0611 | 0.0397 | 0.0401 | -35.02% | -34.39% | -34.70% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/mean_axis1 | 0.0222 | 0.0218 | 0.0227 | -1.76% | +1.96% | +0.10% |
| reduce2d/medium_512x512-serial_f64/min_axis0 | 0.0604 | 0.0396 | 0.0400 | -34.43% | -34.43% | -34.43% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/norm_all | 0.0170 | 0.0171 | 0.0171 | +0.76% | +0.75% | +0.75% |
| reduce2d/medium_512x512-serial_f64/sum_all | 0.0158 | 0.0162 | 0.0160 | +2.32% | +1.13% | +1.73% |
| reduce2d/medium_512x512-serial_f64/sum_axis0 | 0.0520 | 0.0405 | 0.0401 | -22.10% | -22.21% | -22.16% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/sum_axis1 | 0.0215 | 0.0215 | 0.0221 | +0.20% | +2.87% | +1.53% |
| reduce2d/medium_512x512-serial_f64/var_axis1 | 0.0502 | 0.0502 | 0.0507 | +0.16% | +1.12% | +0.64% |
| reduce2d/odd_1000x777-faer16_f64/sum_all | 0.0247 | 0.0243 | 0.0245 | -1.51% | -0.95% | -1.23% |
| reduce2d/odd_1000x777-faer16_f64/sum_axis0 | 0.1090 | 0.1463 | 0.1171 | +34.23% | -7.19% | +13.52% <-- REGRESSION? |
| reduce2d/odd_1000x777-faer16_f64/sum_axis1 | 0.0605 | 0.0603 | 0.0604 | -0.34% | -0.11% | -0.23% |
| reduce2d/odd_1000x777-serial_f64/sum_all | 0.0496 | 0.0486 | 0.0486 | -1.90% | -1.84% | -1.87% |
| reduce2d/odd_1000x777-serial_f64/sum_axis0 | 0.1631 | 0.1417 | 0.1325 | -13.10% | -18.88% | -15.99% <-- IMPROVED |
| reduce2d/odd_1000x777-serial_f64/sum_axis1 | 0.0590 | 0.0596 | 0.0587 | +0.87% | -0.68% | +0.10% |
| reduce2d/small_64x64-faer16_f64/sum_all | 0.0067 | 0.0068 | 0.0068 | +2.51% | +3.29% | +2.90% <-- REGRESSION? |
| reduce2d/small_64x64-faer16_f64/sum_axis0 | 0.0132 | 0.0129 | 0.0131 | -2.09% | -0.70% | -1.39% |
| reduce2d/small_64x64-faer16_f64/sum_axis1 | 0.0215 | 0.0211 | 0.0213 | -1.51% | -0.78% | -1.15% |
| reduce2d/small_64x64-serial_f64/sum_all | 0.0004 | 0.0004 | 0.0004 | +2.65% | +5.72% | +4.18% <-- REGRESSION? |
| reduce2d/small_64x64-serial_f64/sum_axis0 | 0.0019 | 0.0019 | 0.0019 | -3.41% | -0.94% | -2.17% |
| reduce2d/small_64x64-serial_f64/sum_axis1 | 0.0020 | 0.0020 | 0.0020 | +0.35% | +1.47% | +0.91% |
| reduce2d_f32/large_1e7-serial_f32/sum_all | 0.7116 | 0.6936 | 0.6757 | -2.53% | -5.09% | -3.81% |
| reduce2d_f32/large_2048x2048-faer16_f32/sum_axis0 | 0.1662 | 0.1625 | 0.1336 | -2.26% | -20.13% | -11.19% <-- IMPROVED |
| reduce2d_f32/large_2048x2048-faer16_f32/sum_axis1 | 0.0876 | 0.0862 | 0.0867 | -1.66% | -1.18% | -1.42% |
| reduce2d_f32/large_2048x2048-serial_f32/min_axis0 | 0.4653 | 0.2881 | 0.3231 | -38.08% | -30.07% | -34.07% <-- IMPROVED |
| reduce2d_f32/large_2048x2048-serial_f32/sum_all | 0.2252 | 0.2234 | 0.2268 | -0.79% | +1.07% | +0.14% |
| reduce2d_f32/large_2048x2048-serial_f32/sum_axis0 | 0.4352 | 0.2925 | 0.3306 | -32.79% | -24.93% | -28.86% <-- IMPROVED |
| reduce2d_f32/large_2048x2048-serial_f32/sum_axis1 | 0.2476 | 0.2429 | 0.2482 | -1.89% | +0.16% | -0.86% |

## config: native

| case | baseline ms | cand r1 | cand r2 | change r1 | change r2 | mean change |
|---|---|---|---|---|---|---|
| reduce1d/large_1e7-faer16_f64/max_all | 0.3758 | 0.1541 | 0.1529 | -58.98% | -58.90% | -58.94% <-- IMPROVED |
| reduce1d/large_1e7-faer16_f64/mean_all | 0.1550 | 0.1533 | 0.1527 | -1.05% | -0.45% | -0.75% |
| reduce1d/large_1e7-faer16_f64/min_all | 0.3756 | 0.1511 | 0.1556 | -59.76% | -58.38% | -59.07% <-- IMPROVED |
| reduce1d/large_1e7-faer16_f64/norm_all | 0.1638 | 0.1595 | 0.1610 | -2.59% | -0.54% | -1.56% |
| reduce1d/large_1e7-faer16_f64/sum_all | 0.1776 | 0.1555 | 0.1545 | -12.41% | -13.73% | -13.07% <-- IMPROVED |
| reduce1d/large_1e7-faer16_f64/var_all | 0.2286 | 0.2301 | 0.2243 | +0.68% | +1.72% | +1.20% |
| reduce1d/large_1e7-serial_f64/max_all | 10.4236 | 1.2204 | 1.2195 | -88.29% | -88.30% | -88.30% <-- IMPROVED |
| reduce1d/large_1e7-serial_f64/mean_all | 1.2729 | 1.2722 | 1.2406 | -0.05% | -2.53% | -1.29% |
| reduce1d/large_1e7-serial_f64/min_all | 10.4215 | 1.2214 | 1.2273 | -88.28% | -88.22% | -88.25% <-- IMPROVED |
| reduce1d/large_1e7-serial_f64/norm_all | 1.3251 | 1.2991 | 1.2877 | -1.96% | -2.82% | -2.39% |
| reduce1d/large_1e7-serial_f64/sum_all | 1.2419 | 1.2398 | 1.2598 | -0.17% | +1.44% | +0.63% |
| reduce1d/large_1e7-serial_f64/var_all | 1.6941 | 1.6769 | 1.6897 | -1.01% | -0.26% | -0.64% |
| reduce2d/large_2048x2048-faer16_f64/max_axis0 | 0.2666 | 0.3200 | 0.3164 | +20.05% | +19.25% | +19.65% <-- REGRESSION? |
| reduce2d/large_2048x2048-faer16_f64/mean_axis1 | 0.0941 | 0.0936 | 0.0941 | -0.52% | -0.07% | -0.29% |
| reduce2d/large_2048x2048-faer16_f64/min_axis0 | 0.2698 | 0.3164 | 0.3197 | +17.28% | +20.17% | +18.72% <-- REGRESSION? |
| reduce2d/large_2048x2048-faer16_f64/norm_all | 0.0551 | 0.0557 | 0.0560 | +1.03% | +1.54% | +1.28% |
| reduce2d/large_2048x2048-faer16_f64/sum_all | 0.0535 | 0.0541 | 0.0544 | +1.17% | +1.50% | +1.33% |
| reduce2d/large_2048x2048-faer16_f64/sum_axis0 | 0.2869 | 0.3188 | 0.3204 | +11.11% | +10.34% | +10.73% <-- REGRESSION? |
| reduce2d/large_2048x2048-faer16_f64/sum_axis1 | 0.0930 | 0.0931 | 0.0933 | +0.10% | +0.05% | +0.07% |
| reduce2d/large_2048x2048-faer16_f64/var_axis1 | 0.1194 | 0.1187 | 0.1187 | -0.58% | -0.66% | -0.62% |
| reduce2d/large_2048x2048-serial_f64/max_axis0 | 2.1408 | 1.4963 | 1.4908 | -30.11% | -30.36% | -30.23% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/mean_axis1 | 0.4721 | 0.4762 | 0.4695 | +0.87% | +0.12% | +0.50% |
| reduce2d/large_2048x2048-serial_f64/min_axis0 | 2.2437 | 1.4793 | 1.4828 | -34.07% | -33.91% | -33.99% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/norm_all | 0.4873 | 0.4825 | 0.4760 | -1.00% | -2.45% | -1.73% |
| reduce2d/large_2048x2048-serial_f64/sum_all | 0.4323 | 0.4308 | 0.4300 | -0.34% | -2.17% | -1.25% |
| reduce2d/large_2048x2048-serial_f64/sum_axis0 | 1.9942 | 1.3648 | 1.2280 | -31.56% | -38.43% | -34.99% <-- IMPROVED |
| reduce2d/large_2048x2048-serial_f64/sum_axis1 | 0.4870 | 0.4792 | 0.4813 | -1.59% | -2.31% | -1.95% |
| reduce2d/large_2048x2048-serial_f64/var_axis1 | 0.7667 | 0.7744 | 0.7713 | +1.01% | -0.19% | +0.41% |
| reduce2d/medium_512x512-faer16_f64/max_axis0 | 0.0396 | 0.0494 | 0.0470 | +24.81% | +19.36% | +22.08% <-- REGRESSION? |
| reduce2d/medium_512x512-faer16_f64/mean_axis1 | 0.0449 | 0.0448 | 0.0446 | -0.13% | -0.37% | -0.25% |
| reduce2d/medium_512x512-faer16_f64/min_axis0 | 0.0519 | 0.0559 | 0.0456 | +7.80% | -11.63% | -1.91% |
| reduce2d/medium_512x512-faer16_f64/norm_all | 0.0164 | 0.0164 | 0.0164 | -0.38% | -0.56% | -0.47% |
| reduce2d/medium_512x512-faer16_f64/sum_all | 0.0161 | 0.0161 | 0.0161 | -0.06% | -0.09% | -0.08% |
| reduce2d/medium_512x512-faer16_f64/sum_axis0 | 0.0626 | 0.0553 | 0.0482 | -11.65% | -31.00% | -21.33% <-- IMPROVED |
| reduce2d/medium_512x512-faer16_f64/sum_axis1 | 0.0441 | 0.0445 | 0.0441 | +1.01% | +0.23% | +0.62% |
| reduce2d/medium_512x512-faer16_f64/var_axis1 | 0.0462 | 0.0461 | 0.0461 | -0.11% | -0.34% | -0.22% |
| reduce2d/medium_512x512-serial_f64/max_axis0 | 0.0458 | 0.0332 | 0.0356 | -27.55% | -22.27% | -24.91% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/mean_axis1 | 0.0217 | 0.0215 | 0.0218 | -1.11% | -0.05% | -0.58% |
| reduce2d/medium_512x512-serial_f64/min_axis0 | 0.0484 | 0.0350 | 0.0357 | -27.70% | -26.40% | -27.05% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/norm_all | 0.0167 | 0.0161 | 0.0163 | -3.23% | -2.08% | -2.65% |
| reduce2d/medium_512x512-serial_f64/sum_all | 0.0155 | 0.0154 | 0.0157 | -0.96% | +0.67% | -0.14% |
| reduce2d/medium_512x512-serial_f64/sum_axis0 | 0.0421 | 0.0253 | 0.0325 | -39.77% | -22.80% | -31.28% <-- IMPROVED |
| reduce2d/medium_512x512-serial_f64/sum_axis1 | 0.0213 | 0.0210 | 0.0214 | -1.31% | +0.06% | -0.62% |
| reduce2d/medium_512x512-serial_f64/var_axis1 | 0.0461 | 0.0462 | 0.0463 | +0.20% | +0.33% | +0.26% |
| reduce2d/odd_1000x777-faer16_f64/sum_all | 0.0241 | 0.0242 | 0.0243 | +0.35% | +1.07% | +0.71% |
| reduce2d/odd_1000x777-faer16_f64/sum_axis0 | 0.0916 | 0.0922 | 0.1049 | +0.72% | -11.69% | -5.48% |
| reduce2d/odd_1000x777-faer16_f64/sum_axis1 | 0.0576 | 0.0577 | 0.0577 | +0.08% | +0.17% | +0.13% |
| reduce2d/odd_1000x777-serial_f64/sum_all | 0.0481 | 0.0478 | 0.0494 | -0.66% | +2.75% | +1.04% |
| reduce2d/odd_1000x777-serial_f64/sum_axis0 | 0.1196 | 0.0813 | 0.0850 | -32.00% | -28.98% | -30.49% <-- IMPROVED |
| reduce2d/odd_1000x777-serial_f64/sum_axis1 | 0.0593 | 0.0584 | 0.0580 | -1.63% | -2.37% | -2.00% |
| reduce2d/small_64x64-faer16_f64/sum_all | 0.0066 | 0.0065 | 0.0066 | -0.73% | -2.94% | -1.83% |
| reduce2d/small_64x64-faer16_f64/sum_axis0 | 0.0129 | 0.0129 | 0.0127 | -0.13% | -1.64% | -0.88% |
| reduce2d/small_64x64-faer16_f64/sum_axis1 | 0.0210 | 0.0211 | 0.0209 | +0.30% | -0.43% | -0.06% |
| reduce2d/small_64x64-serial_f64/sum_all | 0.0004 | 0.0004 | 0.0004 | +0.22% | +0.43% | +0.32% |
| reduce2d/small_64x64-serial_f64/sum_axis0 | 0.0018 | 0.0018 | 0.0018 | -0.50% | +1.67% | +0.58% |
| reduce2d/small_64x64-serial_f64/sum_axis1 | 0.0019 | 0.0019 | 0.0020 | -0.15% | +2.15% | +1.00% |
| reduce2d_f32/large_1e7-serial_f32/sum_all | 0.7386 | 0.7352 | 0.7195 | -0.47% | -1.76% | -1.11% |
| reduce2d_f32/large_2048x2048-faer16_f32/sum_axis0 | 0.1015 | 0.0832 | 0.0843 | -18.01% | -19.33% | -18.67% <-- IMPROVED |
| reduce2d_f32/large_2048x2048-faer16_f32/sum_axis1 | 0.0844 | 0.0834 | 0.0832 | -1.16% | -1.60% | -1.38% |
| reduce2d_f32/large_2048x2048-serial_f32/min_axis0 | 0.4807 | 0.4428 | 0.4626 | -7.90% | -3.84% | -5.87% |
| reduce2d_f32/large_2048x2048-serial_f32/sum_all | 0.2451 | 0.2417 | 0.2432 | -1.39% | -0.55% | -0.97% |
| reduce2d_f32/large_2048x2048-serial_f32/sum_axis0 | 0.4879 | 0.4155 | 0.4350 | -14.84% | -10.68% | -12.76% <-- IMPROVED |
| reduce2d_f32/large_2048x2048-serial_f32/sum_axis1 | 0.2580 | 0.2549 | 0.2560 | -1.22% | -0.75% | -0.99% |
