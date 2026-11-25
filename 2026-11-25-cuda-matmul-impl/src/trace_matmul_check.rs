//! Prints the `trace_matmul` lines of a few launches: the default chain on a stepped operand
//! (cuBLAS and tiled decline, naive runs), a plain contiguous product, and a forced tiled driver
//! on the stepped operand (an error naming the driver and the triple).
//!
//! ```text
//! cargo run -p rstsr-cuda --example trace_matmul_check
//! ```

use rstsr::prelude::*;
use rstsr_cuda::prelude_dev::*;

fn dev(drivers: MatMulDrivers) -> DeviceCuda {
    let mut flags = CudaRuntimeFlags::default();
    flags.matmul_drivers = drivers;
    flags.trace_matmul = true;
    DeviceCuda::new(0).with_flags(flags)
}

fn main() {
    let dcpu = DeviceCpuSerial::default();
    let vals: Vec<f64> = (0..2 * 20 * 2 * 16).map(|i| ((i * 7) % 9) as f64 - 4.0).collect();
    let base: Tensor<f64, DeviceCpuSerial, _> = rt::asarray((vals, &dcpu)).into_shape([2 * 20, 2 * 16]);
    let stepped = base.i(s![slice!(None, None, 2), slice!(None, None, 2)]);
    let b: Tensor<f64, DeviceCpuSerial, _> = rt::asarray(((0..16 * 12).map(|i| ((i * 5) % 9) as f64 - 4.0).collect::<Vec<_>>(), &dcpu)).into_shape([16, 12]);

    println!("--- default chain, stepped operand");
    let d = dev(MatMulDrivers::all());
    let _ = rt::matmul(&stepped.to_device(&d), &b.to_device(&d));
    println!("--- default chain, contiguous (cublas only enabled)");
    let d = dev(MatMulDrivers::CUBLAS);
    let contig = stepped.to_contig(RowMajor).into_owned();
    let _ = rt::matmul(&contig.to_device(&d), &b.to_device(&d));
    println!("--- tiled only, stepped operand (error)");
    let d = dev(MatMulDrivers::TILED);
    match rt::matmul_f(&stepped.to_device(&d), &b.to_device(&d)) {
        Ok(_) => println!("unexpectedly ran"),
        Err(e) => println!("{e:?}"),
    }
}
