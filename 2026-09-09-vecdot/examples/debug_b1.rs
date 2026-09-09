//! Debug: is the B1 serial inner_dot fast path firing at 1e7?

use exp_vecdot::gen_vec_f64;
use rstsr_core::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let dev = DeviceCpuSerial::default();
    let n = 10_000_000usize;
    let a = rt::asarray((gen_vec_f64(n, 1), [n], &dev));
    let b = rt::asarray((gen_vec_f64(n, 2), [n], &dev));

    // route check: layouts
    println!("a layout: {:?}, ndim {}", a.layout().stride(), a.layout().ndim());

    // 1: vecdot (uses unrolled_binary_reduce — expected ~2.9 ms)
    let _ = rt::vecdot(&a, &b, None);
    let t0 = Instant::now();
    for _ in 0..50 {
        black_box(rt::vecdot(black_box(&a), black_box(&b), None));
    }
    println!("vecdot  : {:8.3} ms/iter", t0.elapsed().as_secs_f64() * 20.0);

    // 2: % (expected ~2.9 ms if B1 fast path fires; ~4.1 ms if not)
    let _ = black_box(&a) % black_box(&b);
    let t0 = Instant::now();
    for _ in 0..50 {
        black_box(black_box(&a) % black_box(&b));
    }
    println!("percent : {:8.3} ms/iter", t0.elapsed().as_secs_f64() * 20.0);

    // 3: lower-level entry with explicit alpha/beta via rt::matmul_from on
    // owned tensors is the only public alpha/beta surface; skip here.
    println!("done");
}
