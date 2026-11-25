//! Timing of `rt::matmul` per driver and shape (`Default` compute mode): the first call of a
//! shape on a fresh device (cuBLASLt pays the heuristic there) and the steady state.
//!
//! ```text
//! cargo run --release -p rstsr-cuda --example bench_matmul_driver_check -- <reps>
//! ```

use rstsr::prelude::*;
use rstsr_cuda::prelude_dev::*;

fn time<T>(d: &DeviceCuda, reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let _ = f();
    }
    d.synchronize().unwrap();
    t.elapsed().as_secs_f64() / reps as f64
}

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

/// `[(label, m, k, n, batch)]`: square, tall-skinny, small, strided batch.
fn shapes() -> Vec<(&'static str, usize, usize, usize, usize)> {
    vec![
        ("square 2048", 2048, 2048, 2048, 1),
        ("square 512", 512, 512, 512, 1),
        ("tall-skinny 65536x64x64", 65536, 64, 64, 1),
        ("small 64", 64, 64, 64, 1),
        ("batch 256 x (256,256,256)", 256, 256, 256, 256),
    ]
}

fn run_f64(driver: &str, reps: usize) {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = match driver {
        "cublaslt" => MatMulDrivers::CUBLAS_LT,
        "cublasex" => MatMulDrivers::CUBLAS_EX,
        "cublas" => MatMulDrivers::CUBLAS,
        _ => panic!(),
    };
    println!("driver = {driver} (f64, Default mode)\n");
    println!("| shape | flops | first call ms | steady ms | TFLOP/s |");
    println!("|---|---|---|---|---|");
    for (label, m, k, n, p) in shapes() {
        // a fresh device per shape: its first call pays the heuristic (the cache is per ordinal)
        let d = DeviceCuda::new(0).with_flags(flags);
        let a: Tensor<f64, DeviceCuda, _> = rt::asarray((data(m * k * p, 1), &d)).into_shape(if p > 1 { vec![p, m, k] } else { vec![m, k] });
        let b: Tensor<f64, DeviceCuda, _> = rt::asarray((data(k * n * p, 2), &d)).into_shape(if p > 1 { vec![p, k, n] } else { vec![k, n] });
        d.synchronize().unwrap();
        let first = {
            let t = std::time::Instant::now();
            let _ = rt::matmul(&a, &b);
            d.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        };
        // steady state includes the allocation of the output, as every call does
        let secs = time(&d, reps, || rt::matmul(&a, &b));
        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64) * (p as f64);
        println!("| {label} | {:.2e} | {first:.3} | {:.3} | {:.2} |", flops, secs * 1e3, flops / secs / 1e12);
    }
    println!();
}

fn run_f32(driver: &str, reps: usize) {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = match driver {
        "cublaslt" => MatMulDrivers::CUBLAS_LT,
        "cublasex" => MatMulDrivers::CUBLAS_EX,
        "cublas" => MatMulDrivers::CUBLAS,
        _ => panic!(),
    };
    println!("driver = {driver} (f32, Default mode)\n");
    println!("| shape | flops | first call ms | steady ms | TFLOP/s |");
    println!("|---|---|---|---|---|");
    for (label, m, k, n, p) in shapes() {
        let d = DeviceCuda::new(0).with_flags(flags);
        let av: Vec<f32> = data(m * k * p, 1).into_iter().map(|x| x as f32).collect();
        let bv: Vec<f32> = data(k * n * p, 2).into_iter().map(|x| x as f32).collect();
        let a: Tensor<f32, DeviceCuda, _> = rt::asarray((av, &d)).into_shape(if p > 1 { vec![p, m, k] } else { vec![m, k] });
        let b: Tensor<f32, DeviceCuda, _> = rt::asarray((bv, &d)).into_shape(if p > 1 { vec![p, k, n] } else { vec![k, n] });
        d.synchronize().unwrap();
        let first = {
            let t = std::time::Instant::now();
            let _ = rt::matmul(&a, &b);
            d.synchronize().unwrap();
            t.elapsed().as_secs_f64() * 1e3
        };
        let secs = time(&d, reps, || rt::matmul(&a, &b));
        let flops = 2.0 * (m as f64) * (k as f64) * (n as f64) * (p as f64);
        println!("| {label} | {:.2e} | {first:.3} | {:.3} | {:.2} |", flops, secs * 1e3, flops / secs / 1e12);
    }
    println!();
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    println!("reps = {reps}");
    for driver in ["cublaslt", "cublasex", "cublas"] {
        run_f32(driver, reps);
    }
    for driver in ["cublaslt", "cublasex", "cublas"] {
        run_f64(driver, reps);
    }
}
