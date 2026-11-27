// Hand-kernel baseline for the cuTENSOR elementwise comparison.
//
// Cells mirror the names emitted by probe/bench.cpp. Binary cells time the device op with a
// preallocated output (`*_with_output`, kernel only, tier checks included); unary cells use the
// allocating tensor-tier call because no `_with_output` variant exists; `muladd*` is the two-op
// sequence rstsr needs today (there is no fused multiply-add op).
use std::time::Instant;

use num::Complex;
use rstsr::prelude::*;
use rstsr_cuda::prelude_dev::*;
use rstsr_cuda::registry::is_available;

type C64 = Complex<f64>;
type C32 = Complex<f32>;

fn bench_cell<F: FnMut()>(
    d: &DeviceCuda,
    name: &str,
    n: usize,
    accesses: usize,
    elem_bytes: usize,
    n_iter: usize,
    repeat: usize,
    mut f: F,
) {
    for _ in 0..10 {
        f();
    }
    d.synchronize();
    let mut samples: Vec<f64> = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let t0 = Instant::now();
        for _ in 0..n_iter {
            f();
        }
        d.synchronize();
        samples.push(t0.elapsed().as_secs_f64() / n_iter as f64);
    }
    samples.sort_by(|x, y| x.total_cmp(y));
    let med = samples[samples.len() / 2];
    let gbps = accesses as f64 * n as f64 * elem_bytes as f64 / med / 1e9;
    println!("{name},{:.6},{:.1}", med * 1e3, gbps);
}

macro_rules! add_cell {
    ($d:expr, $name:expr, $ty:ty, $eb:expr, $a:expr, $b:expr, $shape:expr, $n:expr, $it:expr, $rep:expr) => {{
        let mut c: Tensor<$ty, _, _> = rt::zeros(($shape, $d));
        bench_cell($d, $name, $n, 3, $eb, $it, $rep, || {
            rt::add_with_output(&$a, &$b, c.view_mut());
        });
    }};
}

macro_rules! copy_cell {
    ($d:expr, $name:expr, $ty:ty, $eb:expr, $a:expr, $shape:expr, $n:expr, $it:expr, $rep:expr) => {{
        let mut c: Tensor<$ty, _, _> = rt::zeros(($shape, $d));
        bench_cell($d, $name, $n, 2, $eb, $it, $rep, || {
            c.view_mut().assign(&$a);
        });
    }};
}

macro_rules! exp_cell {
    ($d:expr, $name:expr, $a:expr, $eb:expr, $n:expr, $it:expr, $rep:expr) => {{
        bench_cell($d, $name, $n, 2, $eb, $it, $rep, || {
            let e = rt::exp(&$a);
            std::hint::black_box(&e);
        });
    }};
}

fn main() {
    if !is_available() {
        eprintln!("no CUDA device");
        return;
    }
    let d = DeviceCuda::new(0);
    let n_iter: usize = std::env::var("RSTSR_BENCH_ITER").ok().and_then(|v| v.parse().ok()).unwrap_or(50).max(1);
    let repeat: usize = std::env::var("RSTSR_BENCH_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(7).max(1);

    println!("cell,med_ms,gbps");

    // power-heavy warm-up, same as the rstsr-cuda bench
    {
        let n = 2048usize;
        let w: Tensor<f64, _, _> = rt::arange((0., (n * n) as f64, 1., &d)).into_shape([n, n]);
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < 5.0 {
            let m = rt::pow(&rt::sin(&w), 2.5);
            std::hint::black_box(&m);
        }
        d.synchronize();
    }

    let sizes: [(&str, usize, usize); 6] =
        [("S", 100, 100), ("M", 2048, 2048), ("L", 4096, 4096), ("O1", 1000, 1001), ("O2", 2047, 2053), ("O3", 4093, 4099)];

    // contiguous add/copy/exp over sizes
    for (tag, r, c) in sizes {
        let n = r * c;
        let a: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let b: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let name = format!("add_f64_contig_{tag}");
        add_cell!(&d, &name, f64, 8, a, b, [r, c], n, n_iter, repeat);
        let name = format!("copy_f64_contig_{tag}");
        copy_cell!(&d, &name, f64, 8, a, [r, c], n, n_iter, repeat);
        if tag == "M" || tag == "S" || tag == "O1" {
            let name = format!("exp_f64_contig_{tag}");
            exp_cell!(&d, &name, a, 8, n, n_iter, repeat);
            let name = format!("add_f64_contig_{tag}_full");
            bench_cell(&d, &name, n, 3, 8, n_iter, repeat, || {
                let s = rt::add(&a, &b);
                std::hint::black_box(&s);
            });
        }
    }

    // layouts at M
    {
        let (r, c) = (2048usize, 2048usize);
        let n = r * c;
        let a: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let b: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        // A transposed: view of a [C,R] arena
        let xa: Tensor<f64, _, _> = rt::zeros(([c, r], &d));
        let v_at = xa.view();
        let at = v_at.swapaxes(0, 1);
        add_cell!(&d, "add_f64_At_M", f64, 8, at, b, [r, c], n, n_iter, repeat);
        let xb: Tensor<f64, _, _> = rt::zeros(([c, r], &d));
        let v_bt = xb.view();
        let bt = v_bt.swapaxes(0, 1);
        add_cell!(&d, "add_f64_Bt_M", f64, 8, a, bt, [r, c], n, n_iter, repeat);
        add_cell!(&d, "add_f64_bothT_M", f64, 8, at, bt, [r, c], n, n_iter, repeat);
        // windowed (offset, padded row stride)
        let xw: Tensor<f64, _, _> = rt::zeros(([r + 20, c + 20], &d));
        let v_aw = xw.view();
        let aw = v_aw.i((10..10 + r, 10..10 + c));
        let xw2: Tensor<f64, _, _> = rt::zeros(([r + 20, c + 20], &d));
        let v_bw = xw2.view();
        let bw = v_bw.i((10..10 + r, 10..10 + c));
        add_cell!(&d, "add_f64_win_M", f64, 8, aw, bw, [r, c], n, n_iter, repeat);
        // step 2
        let xs: Tensor<f64, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_asv = xs.view();
        let asv = v_asv.i((slice!(None, None, 2), slice!(None, None, 2)));
        let xs2: Tensor<f64, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_bsv = xs2.view();
        let bsv = v_bsv.i((slice!(None, None, 2), slice!(None, None, 2)));
        add_cell!(&d, "add_f64_step_M", f64, 8, asv, bsv, [r, c], n, n_iter, repeat);
        // broadcast B
        let b1: Tensor<f64, _, _> = rt::zeros(([r, 1], &d));
        add_cell!(&d, "add_f64_bcast_M", f64, 8, a, b1, [r, c], n, n_iter, repeat);
        // transposed output
        {
            let mut cb: Tensor<f64, _, _> = rt::zeros(([c, r], &d));
            bench_cell(&d, "add_f64_outT_M", n, 3, 8, n_iter, repeat, || {
                rt::add_with_output(&a, &b, cb.view_mut().into_permute_dims((1, 0)));
            });
        }
    }

    // dtypes at M
    {
        let (r, c) = (2048usize, 2048usize);
        let n = r * c;
        let a32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        let b32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        add_cell!(&d, "add_f32_contig_M", f32, 4, a32, b32, [r, c], n, n_iter, repeat);
        let a64: Tensor<C64, _, _> = rt::zeros(([r, c], &d));
        let b64: Tensor<C64, _, _> = rt::zeros(([r, c], &d));
        add_cell!(&d, "add_C64F_contig_M", C64, 16, a64, b64, [r, c], n, n_iter, repeat);
        let a32c: Tensor<C32, _, _> = rt::zeros(([r, c], &d));
        let b32c: Tensor<C32, _, _> = rt::zeros(([r, c], &d));
        add_cell!(&d, "add_C32F_contig_M", C32, 8, a32c, b32c, [r, c], n, n_iter, repeat);
        // f32 transposed + step
        let xa: Tensor<f32, _, _> = rt::zeros(([c, r], &d));
        let v_at = xa.view();
        let at = v_at.swapaxes(0, 1);
        add_cell!(&d, "add_f32_At_M", f32, 4, at, b32, [r, c], n, n_iter, repeat);
        let xs: Tensor<f32, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_asv = xs.view();
        let asv = v_asv.i((slice!(None, None, 2), slice!(None, None, 2)));
        let xs2: Tensor<f32, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_bsv = xs2.view();
        let bsv = v_bsv.i((slice!(None, None, 2), slice!(None, None, 2)));
        add_cell!(&d, "add_f32_step_M", f32, 4, asv, bsv, [r, c], n, n_iter, repeat);
        // complex transposed
        let xc: Tensor<C64, _, _> = rt::zeros(([c, r], &d));
        let v_atc = xc.view();
        let atc = v_atc.swapaxes(0, 1);
        add_cell!(&d, "add_C64F_At_M", C64, 16, atc, b64, [r, c], n, n_iter, repeat);
    }

    // two-op muladd sequences at M (the only way rstsr can express a*b+c today)
    {
        let (r, c) = (2048usize, 2048usize);
        let n = r * c;
        let a: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let b: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let mut tmp: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let mut c0: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let mut out: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        bench_cell(&d, "muladd_f64_contig_M", n, 4, 8, n_iter, repeat, || {
            rt::mul_with_output(&a, &b, tmp.view_mut());
            rt::add_with_output(&tmp, &c0, out.view_mut());
        });
        let mut out2: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        bench_cell(&d, "muladdoop_f64_contig_M", n, 4, 8, n_iter, repeat, || {
            rt::mul_with_output(&a, &b, out2.view_mut());
            rt::add_with_output(&out2, &c0, out.view_mut());
        });
        bench_cell(&d, "muladd_f64_contig_M_full", n, 4, 8, n_iter, repeat, || {
            let s = rt::add(&rt::mul(&a, &b), &c0);
            std::hint::black_box(&s);
        });
        let a32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        let b32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        let mut tmp32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        let mut c32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        let mut out32: Tensor<f32, _, _> = rt::zeros(([r, c], &d));
        bench_cell(&d, "muladd_f32_contig_M", n, 4, 4, n_iter, repeat, || {
            rt::mul_with_output(&a32, &b32, tmp32.view_mut());
            rt::add_with_output(&tmp32, &c32, out32.view_mut());
        });
    }

    // mul at M
    {
        let (r, c) = (2048usize, 2048usize);
        let n = r * c;
        let a: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let b: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let mut o: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        bench_cell(&d, "mul_f64_contig_M", n, 3, 8, n_iter, repeat, || {
            rt::mul_with_output(&a, &b, o.view_mut());
        });
    }

    // small-size layout matrix
    {
        let (r, c) = (100usize, 100usize);
        let n = r * c;
        let a: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let b: Tensor<f64, _, _> = rt::zeros(([r, c], &d));
        let xa: Tensor<f64, _, _> = rt::zeros(([c, r], &d));
        let v_at = xa.view();
        let at = v_at.swapaxes(0, 1);
        add_cell!(&d, "add_f64_At_S", f64, 8, at, b, [r, c], n, n_iter, repeat);
        let xw: Tensor<f64, _, _> = rt::zeros(([r + 20, c + 20], &d));
        let v_aw = xw.view();
        let aw = v_aw.i((10..10 + r, 10..10 + c));
        let xw2: Tensor<f64, _, _> = rt::zeros(([r + 20, c + 20], &d));
        let v_bw = xw2.view();
        let bw = v_bw.i((10..10 + r, 10..10 + c));
        add_cell!(&d, "add_f64_win_S", f64, 8, aw, bw, [r, c], n, n_iter, repeat);
        let xs: Tensor<f64, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_asv = xs.view();
        let asv = v_asv.i((slice!(None, None, 2), slice!(None, None, 2)));
        let xs2: Tensor<f64, _, _> = rt::zeros(([2 * r, 2 * c], &d));
        let v_bsv = xs2.view();
        let bsv = v_bsv.i((slice!(None, None, 2), slice!(None, None, 2)));
        add_cell!(&d, "add_f64_step_S", f64, 8, asv, bsv, [r, c], n, n_iter, repeat);
        let b1: Tensor<f64, _, _> = rt::zeros(([r, 1], &d));
        add_cell!(&d, "add_f64_bcast_S", f64, 8, a, b1, [r, c], n, n_iter, repeat);
        {
            let mut cb: Tensor<f64, _, _> = rt::zeros(([c, r], &d));
            bench_cell(&d, "add_f64_outT_S", n, 3, 8, n_iter, repeat, || {
                rt::add_with_output(&a, &b, cb.view_mut().into_permute_dims((1, 0)));
            });
        }
    }

    d.synchronize();
}
