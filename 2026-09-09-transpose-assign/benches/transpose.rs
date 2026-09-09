//! T1' transpose-copy benchmarks (baseline stage, rstsr 386948be, clean tree).
//!
//! The op: `a.t().to_contig(RowMajor).into_owned()` — the cheapest user
//! idiom for materializing a transpose (`.t()` is a VIEW; `to_contig` on a
//! non-contiguous view must copy via `change_contig_f` -> `change_layout_f`
//! -> `assign_arbitary_uninit_*`). This is the exact T0 idiom, so numbers
//! are comparable across experiments.
//!
//! Variants per case:
//! - `A`: the allocating idiom above — allocation included (comparable to
//!   T0/T7 tables; glibc fault rider included at 32 MiB outputs).
//! - `B`: reuse via `c.assign(&a.t())` into a pre-allocated, pre-warmed
//!   output `c` — kernel-only; PRIMARY judge for the large class (T7).
//! - `C`: (not a separate id) re-run of this binary under
//!   `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`;
//!   see reproduce.sh stages `portable_c` / `native_c` (filter "A").
//!
//! Devices: `serial` = DeviceCpuSerial; `faer16` = DeviceFaer with
//! RAYON_NUM_THREADS=16 (asserted). f64 primary (all sizes); f32 secondary
//! (large + odd). Sizes: 64x64, 512x512, 2048x2048, 1000x777, 777x1000.
//!
//! Second group `assign_gates`: contiguous assign + sliced-view assign
//! (fast-axis stride 2) — phase-2 regression gates that must stay on the
//! untouched generic/slice-copy paths (see PLAN.md).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude::*;
use transpose_assign::{
    assert_faer_threads, configure_group, faer_device, serial_device, MAT_SIZES,
};
use transpose_assign::{gen_mat_f32, gen_mat_f64, warm_output_mat, warm_output_mat_f32};

/// id helper: `<label>_<m>x<n>-<devtag>_<dtype>` / `<variant>`
fn bid(label: &str, m: usize, n: usize, devtag: &str, dtype: &str, variant: &str) -> BenchmarkId {
    BenchmarkId::new(format!("{label}_{m}x{n}-{devtag}_{dtype}"), variant)
}

/// A/B pair for one (device, dtype, size). Macro so both device types and
/// both dtype fixtures inherit rstsr's trait bounds at the call site.
macro_rules! bench_pair {
    ($group:expr, $dev:expr, $devtag:expr, $dtype:expr, $label:expr, $m:expr, $n:expr, $gen:ident, $warm:ident) => {{
        let (m, n) = ($m, $n);
        let a = $gen!(m, n, 1, $dev);

        // A: allocating idiom (allocation included; T0-comparable)
        $group.bench_function(bid($label, m, n, $devtag, $dtype, "A"), |bench| {
            bench.iter(|| {
                let at = black_box(&a).t();
                let out = at.to_contig(RowMajor).into_owned();
                black_box(out)
            })
        });

        // B: reuse into pre-warmed c (kernel-only; primary judge)
        let mut c = $warm!(n, m, $dev); // c.shape == a.t().shape == [n, m]
        $group.bench_function(bid($label, m, n, $devtag, $dtype, "B"), |bench| {
            bench.iter(|| {
                c.assign(&black_box(&a).t());
                black_box(&c);
            })
        });
    }};
}

fn bench_transpose(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("transpose_copy");
    configure_group(&mut group);

    // f64 primary: all five sizes
    for &(label, m, n) in MAT_SIZES {
        bench_pair!(&mut group, &dev_serial, "serial", "f64", label, m, n, gen_mat_f64, warm_output_mat);
        bench_pair!(&mut group, &dev_faer, "faer16", "f64", label, m, n, gen_mat_f64, warm_output_mat);
    }

    // f32 secondary: large + odd (both orientations)
    for &(label, m, n) in MAT_SIZES.iter().skip(2).take(2) {
        bench_pair!(&mut group, &dev_serial, "serial", "f32", label, m, n, gen_mat_f32, warm_output_mat_f32);
        bench_pair!(&mut group, &dev_faer, "faer16", "f32", label, m, n, gen_mat_f32, warm_output_mat_f32);
    }

    group.finish();
}

/// Phase-2 regression gates: assign-kernel cases that must NOT be affected
/// by the planned 2-D order-change routing (see PLAN.md):
/// - `contig_assign`: `c.assign(&a)` same-shape contiguous -> contiguous
///   (slice-copy branch of the assign kernels; never enters the strided
///   branch, but is the cheap-path canary).
/// - `sliced_assign`: `c.assign(&a_sliced.t())` where the source's fast axis
///   has stride 2 -> the guard (fast-axis stride == 1) MUST reject the
///   blocked kernel and fall through to the existing iterator path.
///   This is the canary for "fall-through stays untouched".
/// - `sliced_contig_assign`: `c.assign(&a_sliced)` — strided assign without
///   any transpose (no common contig prefix, generic path).
fn bench_assign_gates(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("assign_gates");
    configure_group(&mut group);

    macro_rules! bench_gates {
        ($dev:expr, $devtag:expr) => {{
            // contig assign: small + large (canary)
            for &(label, m, n) in [("small", 64usize, 64usize), ("large", 2048usize, 2048usize)].iter() {
                let a = gen_mat_f64!(m, n, 1, $dev);
                let mut c = warm_output_mat!(m, n, $dev);
                group.bench_function(
                    BenchmarkId::new(format!("contig_{label}_{m}x{n}-{}-f64", $devtag), "assign"),
                    |bench| {
                        bench.iter(|| {
                            c.assign(black_box(&a));
                            black_box(&c);
                        })
                    },
                );
            }
            // sliced assign: large + odd (fall-through canary)
            for &(label, m, n) in [("large", 2048usize, 2048usize), ("odd", 1000usize, 777usize)].iter() {
                let n2 = n / 2;
                let v = transpose_assign::gen_vec_f64(m * n, 2);
                let a = rt::asarray((v, [m, n], $dev));
                let sliced = a.i((.., slice!(0, n2 * 2, 2))); // [m, n2] stride [n, 2]
                let mut ct = warm_output_mat!(n2, m, $dev);
                group.bench_function(
                    BenchmarkId::new(format!("sliced_t_{label}_{m}x{n}-{}-f64", $devtag), "assign"),
                    |bench| {
                        bench.iter(|| {
                            ct.assign(&black_box(&sliced).t()); // view [n2, m] stride [2, n]
                            black_box(&ct);
                        })
                    },
                );
                let mut cs = warm_output_mat!(m, n2, $dev);
                group.bench_function(
                    BenchmarkId::new(format!("sliced_{label}_{m}x{n}-{}-f64", $devtag), "assign"),
                    |bench| {
                        bench.iter(|| {
                            cs.assign(black_box(&sliced)); // [m, n2] stride [n, 2]
                            black_box(&cs);
                        })
                    },
                );
            }
        }};
    }
    bench_gates!(&dev_serial, "serial");
    bench_gates!(&dev_faer, "faer16");

    group.finish();
}

/// Phase-2 review additions (G1 item 3): orientation coverage.
/// - `to_fcontig`: `a.to_contig(ColMajor)` on a c-contig `a` — the R2C
///   orientation (input fast axis 1, output fast axis 0). All other
///   transpose benches are c2r-oriented (input stride [1, n]).
/// - `bcast_t`: `c.assign(&brow.t())` where brow is a [1, n] row broadcast
///   to [m, n] — the zero-stride SLOW-axis case the guard admits
///   (input stride [1, 0]); reads one row repeatedly, L1-hot.
/// Variants A (allocating) and B (reuse) as usual.
fn bench_orderchange_extra(c: &mut Criterion) {
    let dev_serial = serial_device();
    let dev_faer = faer_device();
    assert_faer_threads(&dev_faer, 16);

    let mut group = c.benchmark_group("orderchange_extra");
    configure_group(&mut group);

    macro_rules! bench_extra {
        ($dev:expr, $devtag:expr) => {{
            for &(label, m, n) in [("large", 2048usize, 2048usize), ("odd", 1000usize, 777usize)].iter() {
                // r2c orientation: c-contig -> f-contig
                let a = gen_mat_f64!(m, n, 1, $dev);
                group.bench_function(
                    BenchmarkId::new(format!("to_fcontig_{label}_{m}x{n}-{}-f64", $devtag), "A"),
                    |bench| {
                        bench.iter(|| {
                            let out = black_box(&a).to_contig(ColMajor).into_owned();
                            black_box(out)
                        })
                    },
                );
                let mut cf = warm_output_mat!(m, n, $dev).to_contig(ColMajor).into_owned();
                group.bench_function(
                    BenchmarkId::new(format!("to_fcontig_{label}_{m}x{n}-{}-f64", $devtag), "B"),
                    |bench| {
                        bench.iter(|| {
                            cf.assign(black_box(&a));
                            black_box(&cf);
                        })
                    },
                );
                // zero-stride slow axis: broadcast row transposed
                let brow = gen_mat_f64!(1, n, 3, $dev);
                let brow_bc = brow.broadcast_to(vec![m, n]);
                group.bench_function(
                    BenchmarkId::new(format!("bcast_t_{label}_{m}x{n}-{}-f64", $devtag), "A"),
                    |bench| {
                        bench.iter(|| {
                            let out = black_box(&brow_bc).t().to_contig(RowMajor).into_owned();
                            black_box(out)
                        })
                    },
                );
                let mut c = warm_output_mat!(n, m, $dev);
                group.bench_function(
                    BenchmarkId::new(format!("bcast_t_{label}_{m}x{n}-{}-f64", $devtag), "B"),
                    |bench| {
                        bench.iter(|| {
                            c.assign(&black_box(&brow_bc).t());
                            black_box(&c);
                        })
                    },
                );
            }
        }};
    }
    bench_extra!(&dev_serial, "serial");
    bench_extra!(&dev_faer, "faer16");

    group.finish();
}

criterion_group!(benches, bench_transpose, bench_assign_gates, bench_orderchange_extra);
criterion_main!(benches);
