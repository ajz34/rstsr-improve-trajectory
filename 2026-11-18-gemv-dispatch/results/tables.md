<!-- generated from results/bench-run4.txt by results/make_tables.py -->

| device/dtype | cell | orientation | rstsr (ms) | cblas gemv (ms) | ratio |
|---|---|---|---|---|---|
| openblas/f64 | 1024x1024 | matvec | 0.0679 | 0.0076 | 8.96× |
| openblas/f64 | 1024x1024 | vecmat | 0.1133 | 0.0728 | 1.56× |
| openblas/f64 | 4096x4096 | matvec | 1.0565 | 0.7238 | 1.46× |
| openblas/f64 | 4096x4096 | vecmat | 1.7979 | 1.2469 | 1.44× |
| openblas/f64 | 8192x8192 | matvec | 10.3270 | 8.1679 | 1.26× |
| openblas/f64 | 8192x8192 | vecmat | 12.5160 | 8.5967 | 1.46× |
| openblas/f64 | 65536x64 | matvec | 0.0585 | 0.0459 | 1.27× |
| openblas/f64 | 65536x64 | vecmat | 0.7997 | 0.3308 | 2.42× |
| openblas/f64 | 256x452676 | matvec | 17.0460 | 14.8870 | 1.15× |
| openblas/f64 | 256x452676 | vecmat | 55.4870 | 16.0100 | 3.47× |
| openblas/f64 | 452676x256 | matvec | 17.8330 | 17.1960 | 1.04× |
| openblas/f64 | 452676x256 | vecmat | 27.4200 | 22.8400 | 1.20× |
| faer/f64 | 1024x1024 | matvec | 0.0167 | 0.0078 | 2.15× |
| faer/f64 | 1024x1024 | vecmat | 0.0313 | 0.0618 | 0.51× |
| faer/f64 | 4096x4096 | matvec | 1.1818 | 0.7241 | 1.63× |
| faer/f64 | 4096x4096 | vecmat | 0.9833 | 1.1064 | 0.89× |
| faer/f64 | 8192x8192 | matvec | 7.8957 | 8.1648 | 0.97× |
| faer/f64 | 8192x8192 | vecmat | 8.0993 | 8.6090 | 0.94× |
| faer/f64 | 65536x64 | matvec | 0.0631 | 0.0451 | 1.40× |
| faer/f64 | 65536x64 | vecmat | 0.1028 | 0.6794 | 0.15× |
| faer/f64 | 256x452676 | matvec | 14.7400 | 14.9530 | 0.99× |
| faer/f64 | 256x452676 | vecmat | 30.6070 | 17.4010 | 1.76× |
| faer/f64 | 452676x256 | matvec | 16.5890 | 17.1880 | 0.97× |
| faer/f64 | 452676x256 | vecmat | 15.3470 | 20.9450 | 0.73× |
