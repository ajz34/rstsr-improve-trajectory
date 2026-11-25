//! Timing and accuracy of `rt::matmul` per compute mode. The mode is set through the device
//! flags (`compute_type_32f`, `compute_type_64f`), the driver is forced by name. Accuracy is the
//! max relative error against the **device** `Default` result of the same driver (so no CPU
//! product of a 2048³ problem is needed); bit-identity with `Default` is reported as `same bits`.
//!
//! ```text
//! cargo run --release -p rstsr-cuda --example bench_matmul_mode_check -- <reps> [driver]
//! ```
//!
//! `driver` = `cublasex` (default), `cublaslt`, `cublas` or `all`.

use rstsr::prelude::*;
use rstsr_cuda::dtype::CudaDType;
use rstsr_cuda::matmul::{ComputeType32F, ComputeType64F};
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

fn device(driver: &str) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = match driver {
        "all" => MatMulDrivers::all(),
        "cublasex" => MatMulDrivers::CUBLAS_EX,
        "cublaslt" => MatMulDrivers::CUBLAS_LT,
        "cublas" => MatMulDrivers::CUBLAS,
        other => panic!("unknown driver {other:?}"),
    };
    DeviceCuda::new(0).with_flags(flags)
}

/// Random data in `[-1, 1]`, the same for every run.
fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn host<T, D>(t: &Tensor<T, DeviceCuda, D>) -> Vec<T>
where
    T: CudaDType + Copy + Clone + Send + Sync + 'static,
    D: DimAPI,
{
    let contig = t.to_contig(RowMajor);
    DeviceToCpuAPI::<T>::to_cpu_vec(contig.storage()).rstsr_unwrap()
}

/// `(ms, GFLOP/s, max rel err vs the reference, same bits as the reference)`.
fn run_f32(d: &DeviceCuda, mode: ComputeType32F, a: &Tensor<f32, DeviceCuda, IxD>, b: &Tensor<f32, DeviceCuda, IxD>, reference: &[f32], reps: usize) -> (f64, f64, f64, bool) {
    let d = &d.clone().with_flags({
        let mut f = d.flags();
        f.compute_type_32f = mode;
        f
    });
    // a tensor carries its device: the operands are moved onto the flagged device value
    let (a, b) = (a.to_device(d), b.to_device(d));
    let secs = time(d, reps, || rt::matmul(&a, &b));
    let got = host(&rt::matmul(&a, &b));
    let n = a.shape()[0] as f64;
    let gflops = 2.0 * n * a.shape()[1] as f64 * b.shape()[1] as f64 / secs / 1e9;
    let scale = reference.iter().fold(0.0f32, |s, &x| s.max(x.abs())) as f64;
    let err = got.iter().zip(reference).map(|(g, w)| ((g - w) as f64).abs()).fold(0.0, |e: f64, x| e.max(x)) / scale;
    (secs * 1e3, gflops, err, got == reference)
}

fn run_f64(d: &DeviceCuda, mode: ComputeType64F, a: &Tensor<f64, DeviceCuda, IxD>, b: &Tensor<f64, DeviceCuda, IxD>, reference: &[f64], reps: usize) -> (f64, f64, f64, bool) {
    let d = &d.clone().with_flags({
        let mut f = d.flags();
        f.compute_type_64f = mode;
        f
    });
    // a tensor carries its device: the operands are moved onto the flagged device value
    let (a, b) = (a.to_device(d), b.to_device(d));
    let secs = time(d, reps, || rt::matmul(&a, &b));
    let got = host(&rt::matmul(&a, &b));
    let n = a.shape()[0] as f64;
    let gflops = 2.0 * n * a.shape()[1] as f64 * b.shape()[1] as f64 / secs / 1e9;
    let scale = reference.iter().fold(0.0f64, |s, &x| s.max(x.abs()));
    let err = got.iter().zip(reference).map(|(g, w)| (g - w).abs()).fold(0.0, |e: f64, x| e.max(x)) / scale;
    (secs * 1e3, gflops, err, got == reference)
}

fn main() {
    let reps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let driver = std::env::args().nth(2).unwrap_or_else(|| "cublasex".into());
    let d = device(&driver);
    println!("driver = {driver} ({:?}); math-mode bindings: {}, fixed-point bindings: {}", d.flags().matmul_drivers, cfg!(cublas_math_mode), cfg!(cublas_fixedpoint_math));
    let n = 2048usize;

    println!("\n## f32 {n}^3 (reference: Default on the same driver)\n");
    println!("| mode | ms | TFLOP/s | max rel err vs default | same bits |");
    println!("|---|---|---|---|---|");
    let a: Tensor<f32, _, _> = rt::asarray((data(n * n, 1).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &d)).into_shape([n, n]);
    let b: Tensor<f32, _, _> = rt::asarray((data(n * n, 2).into_iter().map(|x| x as f32).collect::<Vec<_>>(), &d)).into_shape([n, n]);
    let reference = host(&rt::matmul(&a, &b));
    for (mode, name) in [(ComputeType32F::Default, "Default"), (ComputeType32F::Pedantic, "Pedantic"), (ComputeType32F::FastTF32, "FastTF32"), (ComputeType32F::Fast16F, "Fast16F"), (ComputeType32F::Fast16BF, "Fast16BF")] {
        // the classic entries have no math mode converting to bf16
        if driver == "cublas" && mode == ComputeType32F::Fast16BF {
            println!("| {name} | - | - | not served (no classic math mode) | - |");
            continue;
        }
        let (ms, gf, err, same) = run_f32(&d, mode, &a, &b, &reference, reps);
        println!("| {name} | {ms:.3} | {:.2} | {err:.3e} | {same} |", gf / 1e3);
    }

    println!("\n## f64 {n}^3 (reference: Default on the same driver)\n");
    println!("| mode | ms | TFLOP/s | max rel err vs default | same bits |");
    println!("|---|---|---|---|---|");
    let a: Tensor<f64, _, _> = rt::asarray((data(n * n, 3), &d)).into_shape([n, n]);
    let b: Tensor<f64, _, _> = rt::asarray((data(n * n, 4), &d)).into_shape([n, n]);
    let reference = host(&rt::matmul(&a, &b));
    for (mode, name) in [(ComputeType64F::Default, "Default"), (ComputeType64F::Pedantic, "Pedantic"), (ComputeType64F::EmulatedFixedPoint, "EmulatedFixedPoint")] {
        // fixed-point emulation is served by the classic handle only
        if driver != "cublas" && mode == ComputeType64F::EmulatedFixedPoint {
            println!("| {name} | - | - | not served (classic handle only) | - |");
            continue;
        }
        let (ms, gf, err, same) = run_f64(&d, mode, &a, &b, &reference, reps);
        println!("| {name} | {ms:.3} | {:.2} | {err:.3e} | {same} |", gf / 1e3);
    }
}
