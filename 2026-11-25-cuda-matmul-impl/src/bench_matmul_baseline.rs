//! Baseline matmul timing (rstsr-cuda before the driver chain, `9c7d49a`): `rt::matmul` for `f32`
//! and `f64` on the shapes of `bench_matmul`, under the driver the environment picks
//! (`RSTSR_CUDA_MATMUL` unset = cuBLAS, `tiled` = the tiled kernel).
//!
//! Copied into `crates-device/rstsr-cuda/benches/` of a worktree of `9c7d49a` (with a
//! `[[bench]]` entry), run there, then removed:
//!
//! ```text
//! git worktree add /tmp/wt-base 9c7d49a
//! cargo bench -p rstsr-cuda --bench bench_matmul_baseline                  # cuBLAS
//! RSTSR_CUDA_MATMUL=tiled cargo bench -p rstsr-cuda --bench bench_matmul_baseline
//! ```

use std::time::Instant;

use rstsr::prelude::*;
use rstsr_cuda::funcs;
use rstsr_cuda::prelude_dev::*;

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn shapes() -> Vec<(&'static str, usize, usize, usize, usize)> {
    vec![
        ("square 2048", 2048, 2048, 2048, 1),
        ("square 512", 512, 512, 512, 1),
        ("tall-skinny 65536x64x64", 65536, 64, 64, 1),
        ("batch 256 x (256,256,256)", 256, 256, 256, 256),
    ]
}

fn bench_cell<T>(d: &DeviceCuda, mut f: impl FnMut() -> T) -> f64 {
    let budget = std::env::var("RSTSR_BENCH_TIME").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.2);
    let repeat = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
    for _ in 0..3 {
        let _ = f();
    }
    funcs::synchronize(d).unwrap();
    let t0 = Instant::now();
    let _ = f();
    funcs::synchronize(d).unwrap();
    let one = t0.elapsed().as_secs_f64().max(1e-7);
    let n_iter = ((budget / one) as usize).clamp(1, 5000);
    let mut samples = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            let _ = f();
        }
        funcs::synchronize(d).unwrap();
        samples.push(t0.elapsed().as_secs_f64() / n_iter as f64);
    }
    samples.sort_by(|x, y| x.total_cmp(y));
    samples[samples.len() / 2]
}

fn main() {
    if !funcs::is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let driver = std::env::var("RSTSR_CUDA_MATMUL").unwrap_or_else(|_| "cublas".to_string());
    println!("triple,shape,driver,median_ms,tflops");
    for (label, m, k, n, p) in shapes() {
        let flops = 2.0 * m as f64 * k as f64 * n as f64 * p as f64;
        let d = DeviceCuda::new(0);
        let conv: Vec<f32> = data(p * m * k, 1).into_iter().map(|x| x as f32).collect();
        let a: Tensor<f32, DeviceCuda, _> = rt::asarray((conv, &d)).into_shape(if p > 1 { vec![p, m, k] } else { vec![m, k] });
        let conv: Vec<f32> = data(p * k * n, 2).into_iter().map(|x| x as f32).collect();
        let b: Tensor<f32, DeviceCuda, _> = rt::asarray((conv, &d)).into_shape(if p > 1 { vec![p, k, n] } else { vec![k, n] });
        let s = bench_cell(&d, || rt::matmul(&a, &b));
        println!("f32 -> f32,{label},{driver},{:.4},{:.3}", s * 1e3, flops / s / 1e12);
        let d = DeviceCuda::new(0);
        let a: Tensor<f64, DeviceCuda, _> = rt::asarray((data(p * m * k, 1), &d)).into_shape(if p > 1 { vec![p, m, k] } else { vec![m, k] });
        let b: Tensor<f64, DeviceCuda, _> = rt::asarray((data(p * k * n, 2), &d)).into_shape(if p > 1 { vec![p, k, n] } else { vec![k, n] });
        let s = bench_cell(&d, || rt::matmul(&a, &b));
        println!("f64 -> f64,{label},{driver},{:.4},{:.3}", s * 1e3, flops / s / 1e12);
    }
}
