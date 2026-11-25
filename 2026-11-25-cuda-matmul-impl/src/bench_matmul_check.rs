//! Matmul driver timing for `rt::matmul`. The driver is forced through the device flags
//! (`CudaRuntimeFlags::matmul_drivers`), by name: `cublas`, `tiled`, `naive`, `all` (the default
//! chain; later batches add `cublaslt` and `cublasex`). Wall time around a synchronized loop, so it
//! includes the per-call launch overhead, not just the kernel.
//!
//! ```text
//! cargo run --release -p rstsr-cuda --example bench_matmul_check -- <reps> [driver]
//! ```

use rstsr::prelude::*;
use rstsr_cuda::prelude_dev::*;

fn time<T>(d: &DeviceCuda, reps: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..2 {
        let _ = f();
    }
    d.synchronize().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        let _ = f();
    }
    d.synchronize().unwrap();
    t.elapsed().as_secs_f64() / reps as f64
}

macro_rules! bench {
    ($d:expr, $reps:expr, $t:ty, $label:expr, $m:expr, $k:expr, $n:expr, $batch:expr) => {{
        let (m, k, n, p) = ($m, $k, $n, $batch);
        let d = &$d;
        let a: Tensor<$t, _, _> = rt::full(([p * m * k], 0.5 as $t, d)).into_shape(if p > 1 { vec![p, m, k] } else { vec![m, k] });
        let b: Tensor<$t, _, _> = rt::full(([p * k * n], 0.25 as $t, d)).into_shape(if p > 1 { vec![p, k, n] } else { vec![k, n] });
        let _ = rt::matmul(&a, &b);
        d.synchronize().unwrap();
        let ms = time(d, $reps, || rt::matmul(&a, &b));
        let gflops = 2.0 * p as f64 * m as f64 * k as f64 * n as f64 / ms / 1e9;
        println!("| {} | {}x{}x{}{} | {:.4} | {:.1} |", $label, m, k, n, if p > 1 { format!(" x{p}") } else { String::new() }, ms * 1e3, gflops);
    }};
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(20);
    let driver = std::env::args().nth(2).unwrap_or_else(|| "all".into());
    let d = {
        let mut flags = CudaRuntimeFlags::default();
        flags.matmul_drivers = match driver.as_str() {
            "all" => MatMulDrivers::all(),
            "cublas" => MatMulDrivers::CUBLAS,
            "tiled" => MatMulDrivers::TILED,
            "naive" => MatMulDrivers::NAIVE,
            other => panic!("unknown driver {other:?}: expected cublas, tiled, naive or all"),
        };
        DeviceCuda::new(0).with_flags(flags)
    };
    println!("driver = {driver} ({:?})", d.flags().matmul_drivers);
    println!("| dtype | m x k x n | ms | GFLOP/s |");
    println!("|---|---|---|---|");
    bench!(d, reps, f32, "f32", 512, 512, 512, 1);
    bench!(d, reps, f32, "f32", 2048, 2048, 2048, 1);
    bench!(d, reps, f32, "f32", 65536, 64, 64, 1);
    bench!(d, reps, f32, "f32", 64, 64, 65536, 1);
    bench!(d, reps, f32, "f32", 65536, 16, 16, 1);
    bench!(d, reps, f32, "f32", 256, 256, 256, 128);
    bench!(d, reps, f32, "f32", 32, 32, 32, 4096);
    bench!(d, reps, f32, "f32", 17, 33, 5, 1);
    bench!(d, reps, f64, "f64", 512, 512, 512, 1);
    bench!(d, reps, f64, "f64", 2048, 2048, 2048, 1);
    bench!(d, reps, f64, "f64", 65536, 64, 64, 1);
    bench!(d, reps, f64, "f64", 256, 256, 256, 128);
    bench!(d, reps, f64, "f64", 17, 33, 5, 1);
}
