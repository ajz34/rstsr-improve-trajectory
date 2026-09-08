### Bandwidth ceiling (pure Rust, serial, 240 MB / 160 MB working sets)

| op | portable | GB/s | native | GB/s |
|---|---|---|---|---|
| triad large (3x1e7 f64) | 6.05 ms |   39.6 | 6.08 ms |   39.5 |
| memcpy large (2x1e7 f64) | 3.73 ms |   42.9 | 3.91 ms |   40.9 |

### Large class 2048x2048 f64 (32 MiB) — time and derived GB/s (alloc included)

| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| transpose copy | 21.73 ms |    3.1 | 22.13 ms |    3.0 | 3.51 ms |   19.1 | 3.54 ms |   19.0 |
| add contiguous | 6.33 ms |   15.9 | 6.30 ms |   16.0 | 2.87 ms |   35.1 | 2.72 ms |   37.0 |
| add broadcast row | 5.91 ms |   17.0 | 5.67 ms |   17.7 | 2.55 ms |   39.5 | 2.47 ms |   40.7 |
| add strided (b.t) | 28.07 ms |    3.6 | 28.25 ms |    3.6 | 4.29 ms |   23.5 | 4.39 ms |   22.9 |
| sum axis0 | 1.38 ms |   24.3 | 1.90 ms |   17.7 | 444.66 µs |   75.5 | 341.06 µs |   98.4 |
| sum axislast | 468.03 µs |   71.7 | 483.55 µs |   69.4 | 98.23 µs |  341.6 | 94.81 µs |  353.9 |
| sum all | 461.56 µs |   72.7 | 433.46 µs |   77.4 | 56.66 µs |  592.2 | 55.65 µs |  603.0 |
| fill: full | 4.49 ms |    7.5 | 4.63 ms |    7.2 | 4.67 ms |    7.2 | 4.60 ms |    7.3 |
| fill: zeros (calloc-lazy) | 2.27 µs | - | 2.33 µs | - | 2.31 µs | - | 2.34 µs | - |
| argmax 1e7 | 93.53 ms |    0.9 | 93.95 ms |    0.9 | 10.76 ms |    7.4 | 11.69 ms |    6.8 |
| vecdot 1e7 | 2.81 ms |   56.9 | 2.82 ms |   56.7 | 2.91 ms |   55.0 | 2.79 ms |   57.4 |
| vecdot batched 4096x512 | 465.78 µs |   72.0 | 461.78 µs |   72.7 | 147.85 µs |  226.9 | 144.41 µs |  232.4 |
| transpose copy f32 | 15.43 ms |    2.2 | 15.55 ms |    2.2 | 1.34 ms |   25.0 | 1.37 ms |   24.5 |
| add contiguous f32 | 789.44 µs |   63.8 | 772.47 µs |   65.2 | 238.28 µs |  211.2 | 255.77 µs |  196.8 |
| sum axis0 f32 | 426.49 µs |   39.3 | 454.12 µs |   36.9 | 177.59 µs |   94.5 | 95.19 µs |  176.3 |
| vecdot 1e7 f32 | 1.20 ms |   66.9 | 1.22 ms |   65.8 | 1.20 ms |   66.8 | 1.25 ms |   64.1 |

### Class medium 512x512 f64 — time and derived GB/s

| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| transpose copy | 811.81 µs |    5.2 | 812.90 µs |    5.2 | 205.11 µs |   20.4 | 205.34 µs |   20.4 |
| add contiguous | 49.40 µs |  127.3 | 39.23 µs |  160.4 | 62.62 µs |  100.5 | 63.38 µs |   99.3 |
| sum axis0 | 51.55 µs |   40.7 | 41.29 µs |   50.8 | 71.03 µs |   29.5 | 59.74 µs |   35.1 |
| sum axislast | 21.58 µs |   97.2 | 21.26 µs |   98.7 | 45.51 µs |   46.1 | 43.95 µs |   47.7 |
| sum all | 15.93 µs |  131.7 | 15.37 µs |  136.4 | 16.34 µs |  128.4 | 16.30 µs |  128.6 |
| argmax | 9.33 ms | - | 9.34 ms | - | 1.21 ms | - | 1.32 ms | - |

### Class odd 1000x777 f64 — time and derived GB/s

| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| transpose copy | 2.43 ms |    5.1 | 2.44 ms |    5.1 | 362.52 µs |   34.3 | 366.58 µs |   33.9 |
| sum axis0 | 163.13 µs |   38.1 | 125.05 µs |   49.7 | 172.73 µs |   36.0 | 84.83 µs |   73.3 |
| sum axislast | 59.30 µs |  104.8 | 59.06 µs |  105.3 | 60.44 µs |  102.9 | 57.91 µs |  107.3 |
| sum all | 49.66 µs |  125.2 | 48.50 µs |  128.2 | 25.02 µs |  248.4 | 24.70 µs |  251.7 |
| argmax | - | - | - | - | - | - | - | - |

### Class small 64x64 f64 — time and derived GB/s

| op | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s |
|---|---|---|---|---|---|---|---|---|
| transpose copy | 13.61 µs |    4.8 | 13.55 µs |    4.8 | 13.53 µs |    4.8 | 13.60 µs |    4.8 |
| sum axis0 | 1.91 µs |   17.2 | 1.77 µs |   18.5 | 13.26 µs |    2.5 | 12.90 µs |    2.5 |
| sum axislast | 1.97 µs |   16.7 | 1.94 µs |   16.9 | 21.30 µs |    1.5 | 20.99 µs |    1.6 |
| sum all | 408 ns |   80.3 | 372 ns |   88.2 | 7.21 µs |    4.5 | 7.01 µs |    4.7 |
| argmax | - | - | - | - | - | - | - | - |

### vecdot 1-D f64 across sizes — time and GB/s

| size | serial portable | GB/s | serial native | GB/s | faer16 portable | GB/s | faer16 native | GB/s | ndarray dot |
|---|---|---|---|---|---|---|---|---|---|
| 1e3 | 1.60 µs |   10.0 | 1.60 µs |   10.0 | 4.59 µs |    3.5 | 4.61 µs |    3.5 | 90 ns |
| 1e5 | 13.09 µs |  122.2 | 12.76 µs |  125.4 | 16.14 µs |   99.1 | 15.65 µs |  102.2 | 10.86 µs |
| 1e7 | 2.81 ms |   56.9 | 2.82 ms |   56.7 | 2.91 ms |   55.0 | 2.79 ms |   57.4 | 2.75 ms |

### numpy anchors (numpy 2.5.1, conda torch env, single-threaded unless BLAS)

```
numpy 2.5.1 (conda env: torch); single-threaded anchors
repeat=5, best/median wall time

transpose_copy small_64x64 (a.T.copy())                 best=     0.001 ms  median=     0.001 ms  best_GBs=    55.5
sum_axis0     small_64x64 (a.sum(0))                    best=     0.001 ms  median=     0.002 ms  best_GBs=    22.4
sum_axislast  small_64x64 (a.sum(1))                    best=     0.002 ms  median=     0.002 ms  best_GBs=    19.3
sum_all       small_64x64 (a.sum())                     best=     0.001 ms  median=     0.001 ms  best_GBs=    40.5
transpose_copy medium_512x512 (a.T.copy())              best=     0.391 ms  median=     0.393 ms  best_GBs=    10.7
sum_axis0     medium_512x512 (a.sum(0))                 best=     0.029 ms  median=     0.029 ms  best_GBs=    73.5
sum_axislast  medium_512x512 (a.sum(1))                 best=     0.034 ms  median=     0.034 ms  best_GBs=    62.3
sum_all       medium_512x512 (a.sum())                  best=     0.028 ms  median=     0.029 ms  best_GBs=    74.0
transpose_copy large_2048x2048 (a.T.copy())             best=    99.825 ms  median=   100.023 ms  best_GBs=     0.7
sum_axis0     large_2048x2048 (a.sum(0))                best=     0.460 ms  median=     0.478 ms  best_GBs=    72.9
sum_axislast  large_2048x2048 (a.sum(1))                best=     0.655 ms  median=     0.715 ms  best_GBs=    51.2
sum_all       large_2048x2048 (a.sum())                 best=     0.791 ms  median=     0.870 ms  best_GBs=    42.4
transpose_copy odd_1000x777 (a.T.copy())                best=     0.346 ms  median=     0.347 ms  best_GBs=    35.9
sum_axis0     odd_1000x777 (a.sum(0))                   best=     0.062 ms  median=     0.062 ms  best_GBs=   100.2
sum_axislast  odd_1000x777 (a.sum(1))                   best=     0.089 ms  median=     0.089 ms  best_GBs=    70.1
sum_all       odd_1000x777 (a.sum())                    best=     0.079 ms  median=     0.079 ms  best_GBs=    78.6

vecdot np.dot       small_1000                          best=     0.000 ms  median=     0.000 ms  best_GBs=    40.0
vecdot np.einsum    small_1000                          best=     0.001 ms  median=     0.001 ms  best_GBs=    15.0
vecdot np.dot       medium_100000                       best=     0.003 ms  median=     0.003 ms  best_GBs=   613.0
vecdot np.einsum    medium_100000                       best=     0.019 ms  median=     0.020 ms  best_GBs=    82.9
vecdot np.dot       large_10000000                      best=     1.187 ms  median=     1.247 ms  best_GBs=   134.8
vecdot np.einsum    large_10000000                      best=     2.692 ms  median=     2.753 ms  best_GBs=    59.4

add contig     large_2048x2048 (a+b)                    best=     2.772 ms  median=     2.882 ms  best_GBs=    36.3
add broadcast  large_2048x2048 (a+brow)                 best=     2.485 ms  median=     2.568 ms  best_GBs=    40.5
add strided    large_2048x2048 (a+bt.T)                 best=    87.385 ms  median=    99.844 ms  best_GBs=     1.2

zeros          large_2048x2048 (np.zeros)               best=     0.003 ms  median=     0.004 ms  best_GBs= 10551.7
full           large_2048x2048 (np.full)                best=     1.190 ms  median=     1.332 ms  best_GBs=    28.2

argmax         medium_1000000                           best=     0.072 ms  median=     0.072 ms  best_GBs=   111.5
argmax         large_10000000                           best=     1.321 ms  median=     1.334 ms  best_GBs=    60.6

triad (a+3b->c) large_10000000                          best=    11.676 ms  median=    11.927 ms  best_GBs=    20.6
memcpy (c[:] = a) large_10000000                        best=     3.567 ms  median=     3.650 ms  best_GBs=    44.9
triad (a+3b->c) medium_1000000                          best=     0.317 ms  median=     0.326 ms  best_GBs=    75.7
memcpy (c[:] = a) medium_1000000                        best=     0.090 ms  median=     0.091 ms  best_GBs=   177.0
```

### vecdot 1e7 f64 vs analytic peak (32 DP FLOP/cycle/core @ ~5.7 GHz => ~182 GFLOP/s/core)

| variant | portable t | GFLOP/s | %peak | native t | GFLOP/s | %peak |
|---|---|---|---|---|---|---|
| rstsr serial | 2.81 ms | 7.1 | 3.9% | 2.82 ms | 7.1 | 3.9% |
| rstsr faer16 | 2.91 ms | 6.9 | 3.8% | 2.79 ms | 7.2 | 3.9% |
| ndarray dot (serial) | 2.80 ms | 7.1 | 3.9% | 2.75 ms | 7.3 | 4.0% |

### D8 secondary column: native + dispatch_dim_layout_iter (large|odd subset)

| bench | native | native+D8 | speedup |
|---|---|---|---|
| `anchor_ndarray/dot1d_large_10000000-f64/dot` | 2.75 ms | 2.82 ms | 0.97x |
| `anchor_ndarray/large_10000000-f64/argmax` | 5.46 ms | 5.52 ms | 0.99x |
| `anchor_ndarray/large_2048x2048-contig-f64/add_alloc` | 6.56 ms | 6.35 ms | 1.03x |
| `anchor_ndarray/large_2048x2048-contig-f64/add_zip_prealloc` | 2.17 ms | 2.16 ms | 1.00x |
| `anchor_ndarray/large_2048x2048-f64/full` | 4.70 ms | 4.44 ms | 1.06x |
| `anchor_ndarray/large_2048x2048-f64/sum_all` | 432.80 µs | 451.43 µs | 0.96x |
| `anchor_ndarray/large_2048x2048-f64/sum_axis0` | 581.80 µs | 587.70 µs | 0.99x |
| `anchor_ndarray/large_2048x2048-f64/sum_axislast` | 451.59 µs | 443.92 µs | 1.02x |
| `anchor_ndarray/large_2048x2048-f64/transpose_copy` | 5.36 ms | 5.36 ms | 1.00x |
| `anchor_ndarray/large_2048x2048-f64/zeros` | 2.42 µs | 2.23 µs | 1.08x |
| `anchor_ndarray/odd_1000x777-f64/sum_all` | 47.18 µs | 47.46 µs | 0.99x |
| `anchor_ndarray/odd_1000x777-f64/sum_axis0` | 48.24 µs | 53.78 µs | 0.90x |
| `anchor_ndarray/odd_1000x777-f64/sum_axislast` | 49.41 µs | 49.75 µs | 0.99x |
| `anchor_ndarray/odd_1000x777-f64/transpose_copy` | 72.71 µs | 73.80 µs | 0.99x |
| `argmax/large_10000000-faer16_f64/argmax` | 11.69 ms | 11.70 ms | 1.00x |
| `argmax/large_10000000-serial_f64/argmax` | 93.95 ms | 93.28 ms | 1.01x |
| `elementwise_add/large_2048x2048-broadcast/faer16_f64` | 2.47 ms | 2.55 ms | 0.97x |
| `elementwise_add/large_2048x2048-broadcast/serial_f64` | 5.67 ms | 5.62 ms | 1.01x |
| `elementwise_add/large_2048x2048-contig/faer16_f32` | 255.77 µs | 234.96 µs | 1.09x |
| `elementwise_add/large_2048x2048-contig/faer16_f64` | 2.72 ms | 2.83 ms | 0.96x |
| `elementwise_add/large_2048x2048-contig/serial_f32` | 772.47 µs | 759.82 µs | 1.02x |
| `elementwise_add/large_2048x2048-contig/serial_f64` | 6.30 ms | 6.28 ms | 1.00x |
| `elementwise_add/large_2048x2048-strided/faer16_f64` | 4.39 ms | 3.57 ms | 1.23x |
| `elementwise_add/large_2048x2048-strided/serial_f64` | 28.25 ms | 20.79 ms | 1.36x |
| `fill/large_2048x2048-full/faer16_f64` | 4.60 ms | 4.57 ms | 1.01x |
| `fill/large_2048x2048-full/serial_f64` | 4.63 ms | 4.53 ms | 1.02x |
| `fill/large_2048x2048-zeros/faer16_f64` | 2.34 µs | 2.45 µs | 0.96x |
| `fill/large_2048x2048-zeros/serial_f64` | 2.33 µs | 2.36 µs | 0.99x |
| `memcpy/copy_f64/large` | 3.91 ms | 3.93 ms | 1.00x |
| `reduce/large_2048x2048-faer16_f32/sum_axis0` | 95.19 µs | 119.73 µs | 0.80x |
| `reduce/large_2048x2048-faer16_f64/sum_all` | 55.65 µs | 55.47 µs | 1.00x |
| `reduce/large_2048x2048-faer16_f64/sum_axis0` | 341.06 µs | 249.09 µs | 1.37x |
| `reduce/large_2048x2048-faer16_f64/sum_axislast` | 94.81 µs | 92.32 µs | 1.03x |
| `reduce/large_2048x2048-serial_f32/sum_axis0` | 454.12 µs | 489.63 µs | 0.93x |
| `reduce/large_2048x2048-serial_f64/sum_all` | 433.46 µs | 422.71 µs | 1.03x |
| `reduce/large_2048x2048-serial_f64/sum_axis0` | 1.90 ms | 1.86 ms | 1.02x |
| `reduce/large_2048x2048-serial_f64/sum_axislast` | 483.55 µs | 450.82 µs | 1.07x |
| `reduce/odd_1000x777-faer16_f64/sum_all` | 24.70 µs | 24.68 µs | 1.00x |
| `reduce/odd_1000x777-faer16_f64/sum_axis0` | 84.83 µs | 100.32 µs | 0.85x |
| `reduce/odd_1000x777-faer16_f64/sum_axislast` | 57.91 µs | 56.78 µs | 1.02x |
| `reduce/odd_1000x777-serial_f64/sum_all` | 48.50 µs | 48.63 µs | 1.00x |
| `reduce/odd_1000x777-serial_f64/sum_axis0` | 125.05 µs | 111.60 µs | 1.12x |
| `reduce/odd_1000x777-serial_f64/sum_axislast` | 59.06 µs | 57.83 µs | 1.02x |
| `transpose_copy/large_2048x2048-faer16_f32/to_contig` | 1.37 ms | 856.09 µs | 1.60x |
| `transpose_copy/large_2048x2048-faer16_f64/to_contig` | 3.54 ms | 3.14 ms | 1.13x |
| `transpose_copy/large_2048x2048-serial_f32/to_contig` | 15.55 ms | 10.66 ms | 1.46x |
| `transpose_copy/large_2048x2048-serial_f64/to_contig` | 22.13 ms | 19.51 ms | 1.13x |
| `transpose_copy/odd_1000x777-faer16_f64/to_contig` | 366.58 µs | 197.32 µs | 1.86x |
| `transpose_copy/odd_1000x777-serial_f64/to_contig` | 2.44 ms | 1.23 ms | 1.98x |
| `triad/triad_f64/large` | 6.08 ms | 6.05 ms | 1.01x |
| `vecdot/dot1d_large_10000000-faer16_f32/vecdot` | 1.25 ms | 2.12 ms | 0.59x |
| `vecdot/dot1d_large_10000000-faer16_f64/vecdot` | 2.79 ms | 2.86 ms | 0.98x |
| `vecdot/dot1d_large_10000000-serial_f32/vecdot` | 1.22 ms | 2.15 ms | 0.57x |
| `vecdot/dot1d_large_10000000-serial_f64/vecdot` | 2.82 ms | 2.78 ms | 1.01x |
