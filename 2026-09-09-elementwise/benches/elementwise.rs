//! T4' elementwise benchmarks (baseline stage, rstsr 386948be, clean tree).
//!
//! Cases: add contig (small/medium/large/odd), add broadcast ([1,2048] row
//! auto-broadcast, zero-stride), add strided (`a + bt.t()` view), mul contig
//! (large spot), scale contig (large spot, `a * 2.0`).
//!
//! Variants per case:
//! - `A`: idiomatic allocating op (`&a + &b` etc.) — allocation included.
//! - `B`: reuse via public `op_mutc_refa_refb_func` into a pre-allocated,
//!   pre-warmed output `c` — single kernel pass; PRIMARY judge for the
//!   large class (T7 carry-forward: the ~4 ms glibc page-fault rider masks
//!   kernel deltas in A at 32 MiB outputs).
//! - `C`: (not a separate id) re-run of this binary under
//!   `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`;
//!   see reproduce.sh stage `*_c`.
//!
//! Devices: `serial` = DeviceCpuSerial; `faer16` = DeviceFaer with
//! RAYON_NUM_THREADS=16 (asserted). f64 primary; f32 secondary spots for
//! add contig medium/large.
//!
//! Scale variant B deviation (documented): the numa-refb kernel that backs
//! `a * 2.0` has no public tensor-level reuse wrapper at 386948be, so B for
//! scale is expressed as a `[1,n]` row of exactly 2.0 auto-broadcast (same
//! refa-refb driver as the broadcast add; identical loop structure).

use std::hint::black_box;
use std::mem::MaybeUninit;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use elementwise::{assert_faer_threads, configure_group, faer_device, gen_mat_f32, gen_mat_f64, serial_device};
use elementwise::{warm_output_mat, warm_output_mat_f32};
use rstsr_core::prelude::*;
use rstsr_core::tensor::operators::op_with_func::op_mutc_refa_refb_func;

/// id helper: `<case>_<label>_<m>x<n>-<devtag>_<dtype>` / `<variant>`
fn bid(case: &str, label: &str, m: usize, n: usize, devtag: &str, dtype: &str, variant: &str) -> BenchmarkId {
    BenchmarkId::new(format!("{case}_{label}_{m}x{n}-{devtag}_{dtype}"), variant)
}

/// Deterministic row vector [1, n] with every element exactly 2.0.
fn row_of_2(n: usize, dev: &DeviceCpuSerial) -> Tensor<f64, DeviceCpuSerial> {
    let v: Vec<f64> = vec![2.0; n];
    rt::asarray((v, [1, n], dev))
}

fn row_of_2_faer(n: usize, dev: &DeviceFaer) -> Tensor<f64, DeviceFaer> {
    let v: Vec<f64> = vec![2.0; n];
    rt::asarray((v, [1, n], dev))
}

fn bench_elementwise(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    macro_rules! bench_add_pair {
        // (group, dev, devtag, dtype-suffix, m, n, gen-macro, warm-macro)
        ($group:expr, $dev:expr, $devtag:expr, $dtype:expr, $m:expr, $n:expr, $gen:ident, $warm:ident,
         a = $sa:expr, b = $sb:expr) => {{
            let (m, n) = ($m, $n);
            let (a, b) = ($gen!(m, n, $sa, $dev), $gen!(m, n, $sb, $dev));
            $group.bench_function(bid("add", "contig", m, n, $devtag, $dtype, "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) + black_box(&b)))
            });
            let mut c = $warm!(m, n, $dev);
            $group.bench_function(bid("add", "contig", m, n, $devtag, $dtype, "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<_>, x: &_, y: &_| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    macro_rules! bench_add_bcast_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $salt_a:expr, $salt_row:expr) => {{
            let (m, n) = ($m, $n);
            let a = gen_mat_f64!(m, n, $salt_a, $dev);
            let brow = gen_mat_f64!(1, n, $salt_row, $dev);
            $group.bench_function(bid("add", "bcast", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) + black_box(&brow)))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("add", "bcast", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&brow), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    macro_rules! bench_add_strided_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $salt_a:expr, $salt_bt:expr) => {{
            let (m, n) = ($m, $n);
            let a = gen_mat_f64!(m, n, $salt_a, $dev);
            let bt = gen_mat_f64!(m, n, $salt_bt, $dev);
            $group.bench_function(bid("add", "strided", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) + black_box(&bt.t())))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("add", "strided", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&bt.t()), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    // strided with explicit bt dims (bt is [bm, bn]; bt.t() is [m, n]) — used
    // for the small/odd regression gates where bt must be rectangular.
    macro_rules! bench_add_strided_rect_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $bm:expr, $bn:expr, $salt_a:expr, $salt_bt:expr) => {{
            let (m, n) = ($m, $n);
            let a = gen_mat_f64!(m, n, $salt_a, $dev);
            let bt = gen_mat_f64!($bm, $bn, $salt_bt, $dev);
            $group.bench_function(bid("add", "strided", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) + black_box(&bt.t())))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("add", "strided", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&bt.t()), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    // strided FIRST operand (a.t() + b): same kernel, roles swapped
    macro_rules! bench_add_stridedfirst_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $salt_at:expr, $salt_b:expr) => {{
            let (m, n) = ($m, $n);
            let at = gen_mat_f64!(m, n, $salt_at, $dev); // square: at.t() is [m, n]
            let b = gen_mat_f64!(m, n, $salt_b, $dev);
            $group.bench_function(bid("add", "stridedfirst", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&at.t()) + black_box(&b)))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("add", "stridedfirst", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x + y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&at.t()), black_box(&b), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    // in-place strided add `c += bt.t()` (exercises the op_muta_refb twin path)
    macro_rules! bench_addasgn_strided_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $salt_a:expr, $salt_bt:expr) => {{
            let (m, n) = ($m, $n);
            let a = gen_mat_f64!(m, n, $salt_a, $dev);
            let bt = gen_mat_f64!(m, n, $salt_bt, $dev);
            let mut c = warm_output_mat!(m, n, $dev);
            c.assign(&a); // deterministic start; values evolve during benching (perf-only row)
            $group.bench_function(bid("addasgn", "strided", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    c += black_box(&bt.t());
                    black_box(&c);
                })
            });
        }};
    }

    macro_rules! bench_mul_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr) => {{
            let (m, n) = ($m, $n);
            let (a, b) = (gen_mat_f64!(m, n, 1, $dev), gen_mat_f64!(m, n, 2, $dev));
            $group.bench_function(bid("mul", "contig", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) * black_box(&b)))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("mul", "contig", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x * y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&b), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    macro_rules! bench_scale_pair {
        ($group:expr, $dev:expr, $devtag:expr, $m:expr, $n:expr, $row2:expr) => {{
            let (m, n) = ($m, $n);
            let a = gen_mat_f64!(m, n, 1, $dev);
            let row2 = $row2(n, $dev);
            $group.bench_function(bid("scale", "contig", m, n, $devtag, "f64", "A"), |bench| {
                bench.iter(|| black_box(black_box(&a) * 2.0))
            });
            let mut c = warm_output_mat!(m, n, $dev);
            $group.bench_function(bid("scale", "contig", m, n, $devtag, "f64", "B"), |bench| {
                bench.iter(|| {
                    let mut f = |cv: &mut MaybeUninit<f64>, x: &f64, y: &f64| { cv.write(x * y); };
                    op_mutc_refa_refb_func(&mut c, black_box(&a), black_box(&row2), &mut f).unwrap();
                    black_box(&c);
                })
            });
        }};
    }

    // =====================================================================
    // add — the deep case (contig on all sizes, broadcast, strided)
    // =====================================================================
    let mut group = c.benchmark_group("add");
    configure_group(&mut group);

    // small 64x64 (regression gate)
    bench_add_pair!(group, &dev_serial, "serial", "f64", 64, 64, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f64", 64, 64, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    // medium / large / odd
    bench_add_pair!(group, &dev_serial, "serial", "f64", 512, 512, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f64", 512, 512, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_serial, "serial", "f64", 2048, 2048, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f64", 2048, 2048, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_serial, "serial", "f64", 1000, 777, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f64", 1000, 777, gen_mat_f64, warm_output_mat, a = 1, b = 2);
    // f32 secondary spots (medium + large)
    bench_add_pair!(group, &dev_serial, "serial", "f32", 512, 512, gen_mat_f32, warm_output_mat_f32, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f32", 512, 512, gen_mat_f32, warm_output_mat_f32, a = 1, b = 2);
    bench_add_pair!(group, &dev_serial, "serial", "f32", 2048, 2048, gen_mat_f32, warm_output_mat_f32, a = 1, b = 2);
    bench_add_pair!(group, &dev_faer, "faer16", "f32", 2048, 2048, gen_mat_f32, warm_output_mat_f32, a = 1, b = 2);
    // broadcast + strided (large, f64)
    bench_add_bcast_pair!(group, &dev_serial, "serial", 2048, 2048, 1, 3);
    bench_add_bcast_pair!(group, &dev_faer, "faer16", 2048, 2048, 1, 3);
    bench_add_strided_pair!(group, &dev_serial, "serial", 2048, 2048, 1, 4);
    bench_add_strided_pair!(group, &dev_faer, "faer16", 2048, 2048, 1, 4);
    // strided regression gates: small (degenerate tiles) and odd (tail tiles;
    // bt is [777, 1000] so bt.t() is [1000, 777])
    bench_add_strided_rect_pair!(group, &dev_serial, "serial", 64, 64, 64, 64, 1, 4);
    bench_add_strided_rect_pair!(group, &dev_faer, "faer16", 64, 64, 64, 64, 1, 4);
    bench_add_strided_rect_pair!(group, &dev_serial, "serial", 1000, 777, 777, 1000, 1, 4);
    bench_add_strided_rect_pair!(group, &dev_faer, "faer16", 1000, 777, 777, 1000, 1, 4);
    // strided-first operand (a.t() + b) and in-place strided add (c += bt.t())
    bench_add_stridedfirst_pair!(group, &dev_serial, "serial", 2048, 2048, 6, 2);
    bench_add_stridedfirst_pair!(group, &dev_faer, "faer16", 2048, 2048, 6, 2);
    bench_addasgn_strided_pair!(group, &dev_serial, "serial", 2048, 2048, 1, 4);
    bench_addasgn_strided_pair!(group, &dev_faer, "faer16", 2048, 2048, 1, 4);
    group.finish();

    // =====================================================================
    // mul — spot check (large contig)
    // =====================================================================
    let mut group = c.benchmark_group("mul");
    configure_group(&mut group);
    bench_mul_pair!(group, &dev_serial, "serial", 2048, 2048);
    bench_mul_pair!(group, &dev_faer, "faer16", 2048, 2048);
    group.finish();

    // =====================================================================
    // scale — spot check (large contig)
    // =====================================================================
    let mut group = c.benchmark_group("scale");
    configure_group(&mut group);
    bench_scale_pair!(group, &dev_serial, "serial", 2048, 2048, row_of_2);
    bench_scale_pair!(group, &dev_faer, "faer16", 2048, 2048, row_of_2_faer);
    group.finish();
}

criterion_group!(benches, bench_elementwise);
criterion_main!(benches);
