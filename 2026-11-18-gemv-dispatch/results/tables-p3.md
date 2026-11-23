<!-- generated from results/bench-run5-old.txt (before) and results/bench-run6.txt (after) -->

| device/dtype | cell | orientation | before (ms) | before/anchor | after (ms) | after/anchor | speedup |
|---|---|---|---|---|---|---|---|
| openblas/f64 | 1024x1024 | matvec | 0.0690 | 9.42× | 0.0078 | 1.07× | 8.8× |
| openblas/f64 | 1024x1024 | vecmat | 0.1162 | 1.78× | 0.0259 | 0.40× | 4.5× |
| openblas/f64 | 4096x4096 | matvec | 1.0902 | 1.50× | 0.7277 | 1.00× | 1.5× |
| openblas/f64 | 4096x4096 | vecmat | 1.7993 | 1.81× | 1.0510 | 0.96× | 1.7× |
| openblas/f64 | 8192x8192 | matvec | 10.3290 | 1.26× | 8.1764 | 1.00× | 1.3× |
| openblas/f64 | 8192x8192 | vecmat | 12.6730 | 1.46× | 8.5807 | 1.00× | 1.5× |
| openblas/f64 | 65536x64 | matvec | 0.0571 | 1.24× | 0.0463 | 1.01× | 1.2× |
| openblas/f64 | 65536x64 | vecmat | 0.8946 | 2.96× | 0.7225 | 1.00× | 1.2× |
| openblas/f64 | 256x452676 | matvec | 17.5190 | 1.18× | 14.8250 | 1.00× | 1.2× |
| openblas/f64 | 256x452676 | vecmat | 55.1000 | 3.45× | 15.7620 | 1.00× | 3.5× |
| openblas/f64 | 452676x256 | matvec | 17.8800 | 1.04× | 17.1240 | 1.00× | 1.0× |
| openblas/f64 | 452676x256 | vecmat | 26.7650 | 1.22× | 21.4860 | 0.92× | 1.2× |
| faer/f64 | 1024x1024 | matvec | 0.0166 | 2.49× | 0.0159 | 2.34× | 1.0× |
| faer/f64 | 1024x1024 | vecmat | 0.0305 | 0.59× | 0.0297 | 0.42× | 1.0× |
| faer/f64 | 4096x4096 | matvec | 1.1131 | 1.52× | 1.0928 | 1.49× | 1.0× |
| faer/f64 | 4096x4096 | vecmat | 0.9427 | 0.85× | 1.0708 | 0.94× | 0.9× |
| faer/f64 | 8192x8192 | matvec | 8.0669 | 0.99× | 8.0650 | 0.99× | 1.0× |
| faer/f64 | 8192x8192 | vecmat | 8.1071 | 0.91× | 8.1159 | 0.94× | 1.0× |
| faer/f64 | 65536x64 | matvec | 0.0636 | 1.44× | 0.0640 | 1.41× | 1.0× |
| faer/f64 | 65536x64 | vecmat | 0.0953 | 0.13× | 0.1003 | 0.32× | 1.0× |
| faer/f64 | 256x452676 | matvec | 14.6350 | 0.99× | 14.5320 | 0.98× | 1.0× |
| faer/f64 | 256x452676 | vecmat | 27.5460 | 1.72× | 27.6650 | 1.76× | 1.0× |
| faer/f64 | 452676x256 | matvec | 15.5320 | 0.91× | 15.3970 | 0.90× | 1.0× |
| faer/f64 | 452676x256 | vecmat | 15.2420 | 0.67× | 15.2180 | 0.70× | 1.0× |

| extra | before (ms) | after (ms) | speedup |
|---|---|---|---|
| openblas/f64/extra/matmul_from-beta/1024x1024 | 0.068 | 0.007 | 9.50× |
| openblas/f64/extra/batched-gemv/64x1024x1024 | 11.084 | 8.356 | 1.33× |
| openblas/f64/extra/batched-gemm/64x512x512 | 22.323 | 22.319 | 1.00× |
| faer/f64/extra/matmul_from-beta/1024x1024 | 0.017 | 0.016 | 1.03× |
| faer/f64/extra/batched-gemv/64x1024x1024 | 9.161 | 8.903 | 1.03× |
| faer/f64/extra/batched-gemm/64x512x512 | 88.512 | 90.768 | 0.98× |
