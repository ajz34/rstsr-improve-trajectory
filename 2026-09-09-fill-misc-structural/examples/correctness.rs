//! T5 fill/creation correctness gate.
//!
//! Checks, against naive scalar references:
//! - `rt::full` / `rt::ones` / `rt::zeros` element values (incl. odd and
//!   degenerate shapes, f64 + f32, both devices);
//! - `c.fill(v)` into existing storage (fresh + re-fill, `fill(0.)` ==
//!   zeros content);
//! - device-level `fill` on non-contiguous layouts: diagonal layout (the
//!   `eye` path), stride-2-both-axes layout, and a zero-stride broadcast
//!   layout (fill must write through the stride-0 axis consistently).
//!
//! Run under BOTH RUSTFLAGS configs (portable + native); exit non-zero on
//! any failure.

use fill_misc_structural::{assert_faer_threads, faer_device, serial_device, FILL_VALUE_F32, FILL_VALUE_F64};
use rstsr_core::operators::assignment::OpAssignAPI;
use rstsr_core::prelude::*;

static mut PASS: usize = 0;
static mut FAIL: usize = 0;

fn check(cond: bool, what: &str) {
    unsafe {
        if cond {
            PASS += 1;
        } else {
            FAIL += 1;
            eprintln!("FAIL: {what}");
        }
    }
}

fn naive_fill(v: &mut [f64], val: f64) {
    for x in v.iter_mut() {
        *x = val;
    }
}

fn test_device_full_ones_zeros_serial(dev: &DeviceCpuSerial) {
    for &(m, n) in &[(64usize, 64usize), (1000, 777), (1, 7), (7, 1), (7, 1)] {
        let t: Tensor<f64, _> = rt::full(([m, n], FILL_VALUE_F64, dev));
        let ok = t.iter().all(|&x| x == FILL_VALUE_F64);
        check(ok && t.size() == m * n, &format!("rt::full f64 [{m},{n}] serial"));

        let t: Tensor<f64, _> = rt::ones(([m, n], dev));
        check(t.iter().all(|&x| x == 1.0), &format!("rt::ones f64 [{m},{n}] serial"));

        let t: Tensor<f64, _> = rt::zeros(([m, n], dev));
        check(t.iter().all(|&x| x == 0.0), &format!("rt::zeros f64 [{m},{n}] serial"));
    }
    // 1-D and f32 spots + negative fill value
    let t: Tensor<f64, _> = rt::full(([997], -0.5, dev));
    check(t.iter().all(|&x| x == -0.5), "rt::full f64 [997] serial, negative value");
    let t: Tensor<f32, _> = rt::full(([64, 64], FILL_VALUE_F32, dev));
    check(t.iter().all(|&x| x == FILL_VALUE_F32), "rt::full f32 [64,64] serial");
    let t: Tensor<f32, _> = rt::ones(([1000, 777], dev));
    check(t.iter().all(|&x| x == 1.0f32), "rt::ones f32 [1000,777] serial");
}

fn test_device_full_ones_zeros_faer(dev: &DeviceFaer) {
    for &(m, n) in &[(64usize, 64usize), (1000, 777), (1, 7), (7, 1)] {
        let t: Tensor<f64, _> = rt::full(([m, n], FILL_VALUE_F64, dev));
        check(t.iter().all(|&x| x == FILL_VALUE_F64), &format!("rt::full f64 [{m},{n}] faer"));
        let t: Tensor<f64, _> = rt::ones(([m, n], dev));
        check(t.iter().all(|&x| x == 1.0), &format!("rt::ones f64 [{m},{n}] faer"));
        let t: Tensor<f64, _> = rt::zeros(([m, n], dev));
        check(t.iter().all(|&x| x == 0.0), &format!("rt::zeros f64 [{m},{n}] faer"));
    }
    let t: Tensor<f32, _> = rt::full(([2048, 2048], FILL_VALUE_F32, dev));
    check(t.iter().all(|&x| x == FILL_VALUE_F32), "rt::full f32 [2048,2048] faer");
}

fn test_fill_method(dev_order: usize) {
    // serial device, odd size, re-fill + fill(0.)
    let mut c: Tensor<f64, _> = rt::zeros(([1000, 777], &DeviceCpuSerial::default()));
    c.fill(FILL_VALUE_F64);
    check(c.iter().all(|&x| x == FILL_VALUE_F64), "c.fill f64 [1000,777] first fill");
    c.fill(-1.5);
    check(c.iter().all(|&x| x == -1.5), "c.fill f64 [1000,777] re-fill");
    c.fill(0.0);
    check(c.iter().all(|&x| x == 0.0), "c.fill(0.) == zeros content");
    let _ = dev_order;

    // faer device
    let mut c: Tensor<f64, _> = rt::zeros(([512, 512], &faer_device()));
    c.fill(FILL_VALUE_F64);
    check(c.iter().all(|&x| x == FILL_VALUE_F64), "c.fill f64 [512,512] faer");

    // f32
    let mut c: Tensor<f32, _> = rt::zeros(([64, 97], &DeviceCpuSerial::default()));
    c.fill(FILL_VALUE_F32);
    check(c.iter().all(|&x| x == FILL_VALUE_F32), "c.fill f32 [64,97]");
}

fn test_device_level_strided_fill() {
    // (a) diagonal layout (the `eye` path): shape [n], stride [n+1]
    let dev = serial_device();
    let n = 257usize;
    let mut v = vec![0.0f64; n * n];
    let layout = Layout::new([n], [(n + 1) as isize], 0).unwrap();
    dev.fill(&mut v, &layout, FILL_VALUE_F64).unwrap();
    let mut want = vec![0.0f64; n * n];
    for i in 0..n {
        want[i * (n + 1)] = FILL_VALUE_F64;
    }
    check(v == want, "device fill diag layout serial");

    // (b) stride-2 both axes: shape [33, 97] over stride [2*200, 2] buffer of 400*100
    let (m, k) = (33usize, 97usize);
    let mut v = vec![9.0f64; 400 * 100];
    let layout = Layout::new([m, k], [200isize, 2isize], 0).unwrap();
    dev.fill(&mut v, &layout, -2.5).unwrap();
    let mut want = vec![9.0f64; 400 * 100];
    for i in 0..m {
        for j in 0..k {
            want[i * 200 + j * 2] = -2.5;
        }
    }
    check(v == want, "device fill stride-2-both-axes serial");

    // (c) zero-stride broadcast layout: shape [3, 8], stride [0, 1]
    //     (one physical row written through three logical rows)
    let mut v = vec![0.0f64; 8];
    let layout = Layout::new([3, 8], [0isize, 1isize], 0).unwrap();
    dev.fill(&mut v, &layout, 4.75).unwrap();
    check(v.iter().all(|&x| x == 4.75) && v.len() == 8, "device fill broadcast layout serial");

    // (d) faer twin of (a)
    let dev = faer_device();
    let n = 257usize;
    let mut v = vec![0.0f64; n * n];
    let layout = Layout::new([n], [(n + 1) as isize], 0).unwrap();
    dev.fill(&mut v, &layout, FILL_VALUE_F64).unwrap();
    check(v.iter().step_by(n + 1).all(|&x| x == FILL_VALUE_F64) && v.iter().enumerate().all(|(i, &x)| (i % (n + 1) == 0) || x == 0.0), "device fill diag layout faer");
}

fn main() {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    test_device_full_ones_zeros_serial(&dev_serial);
    test_device_full_ones_zeros_faer(&dev_faer);
    test_fill_method(0);
    test_device_level_strided_fill();

    // explicit naive cross-check of the reuse fill path over a fresh buffer
    let mut v = vec![0.0f64; 1000 * 777];
    naive_fill(&mut v, FILL_VALUE_F64);
    let mut c: Tensor<f64, _> = rt::zeros(([1000, 777], &DeviceCpuSerial::default()));
    c.fill(FILL_VALUE_F64);
    check(c.raw().as_slice() == v.as_slice(), "c.fill == naive fill [1000,777]");

    let (p, f) = unsafe { (PASS, FAIL) };
    println!("correctness: {p} passed, {f} failed");
    if f > 0 {
        std::process::exit(1);
    }
}
