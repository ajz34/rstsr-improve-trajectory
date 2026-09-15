//! A-B timing harness: T1 §4.1 AtomicPtr hoist (removal of write-through `as_ptr()`).
//!
//! Dependency-free: warmup runs then N timed runs per case; prints CSV
//! `case,median_ns,min_ns` to stdout. Identical source is used for the
//! `before/` and `after/` crates; only the dependency paths differ.
//!
//! All cases run on `DeviceFaer` — the tensor-facing rayon device whose op
//! dispatch is `feature_rayon/auto_impl` (symlinked as `device_faer/
//! rayon_auto_impl`), which routes to the edited `rstsr-native-impl/src/
//! cpu_rayon` kernels.
//!
//! Notes:
//! - Fresh-output cases pay the glibc 32 MiB page-fault rider identically in
//!   both variants, so A/B deltas stay valid; owned-reuse cases
//!   (`add_owned_reuse_*`) isolate the kernel cost (T7 lesson).
//! - THREADS pins the device pool explicitly (no RAYON_NUM_THREADS reliance).

use std::hint::black_box;
use std::time::Instant;

use rstsr_core::prelude_dev::*;
use rstsr_sci_traits::distance::metric::MetricEuclidean;
use rstsr_sci_traits::distance::traits::cdist;

const THREADS: usize = 16;
const REPS: usize = 30;
const WARMUP: usize = 3;

fn time_case<F: FnMut()>(name: &str, mut f: F) {
    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as u64);
    }
    samples.sort();
    println!("{name},{},{}", samples[REPS / 2], samples[0]);
}

fn main() {
    println!("case,median_ns,min_ns");

    let dev = DeviceFaer::new(THREADS);

    // add, contiguous, fresh output (op_with_func: op_mutc_refa_refb_func)
    for n in [1usize << 20, 1usize << 24] {
        let a = full((vec![n], 1.0, &dev));
        let b = full((vec![n], 2.0, &dev));
        time_case(&format!("add_contig_{n}"), || {
            let c = &a + &b;
            black_box(&c);
        });
    }

    // add, contiguous, owned-reuse output (op_with_func: op_muta_refb_func)
    for n in [1usize << 20, 1usize << 24] {
        let mut a = Some(full((vec![n], 1.0, &dev)));
        let b = full((vec![n], 2.0, &dev));
        time_case(&format!("add_owned_reuse_{n}"), || {
            a = Some(add(a.take().unwrap(), &b));
            black_box(a.as_ref().unwrap());
        });
    }

    // blocked 2-D path, fresh output (blocked_2d_3layouts)
    {
        let m = 2048usize;
        let a = full((vec![m, m], 1.0, &dev));
        let b = full((vec![m, m], 2.0, &dev));
        let bt = b.t();
        time_case("add_blocked2d_3layout_2048sq", || {
            let c = &a + &bt;
            black_box(&c);
        });
    }

    // blocked 2-D path, owned-reuse (blocked_2d_2layouts, in-place)
    {
        let m = 2048usize;
        let mut a = Some(full((vec![m, m], 1.0, &dev)));
        let b = full((vec![m, m], 2.0, &dev));
        let bt = b.t();
        time_case("add_blocked2d_2layout_inplace_2048sq", || {
            a = Some(add(a.take().unwrap(), &bt));
            black_box(a.as_ref().unwrap());
        });
    }

    // tall-skinny, generic outer-parallel path (documented blocked-path caveat)
    {
        let a = full((vec![70, 10000], 1.0, &dev));
        let b = full((vec![10000, 70], 2.0, &dev));
        let bt = b.t();
        time_case("add_tallskinny_70x10000", || {
            let c = &a + &bt;
            black_box(&c);
        });
    }

    // fill (assignment.rs fill_promote)
    {
        let n = 1usize << 24;
        let mut c = full((vec![n], 0.0, &dev));
        time_case("fill_2p24", || {
            fill(&mut c, 1.5);
            black_box(&c);
        });
    }

    // generic non-contiguous assign (assignment.rs generic branch): 3-D permuted
    {
        let (p, q, r) = (192usize, 256usize, 224usize);
        let a = full((vec![p, q, r], 1.0, &dev));
        let v = permute_dims(&a, (2, 0, 1));
        let mut c = full((vec![r, p, q], 0.0, &dev));
        time_case("assign_perm3d", || {
            c.assign(&v);
            black_box(&c);
        });
    }

    // 2-D order-change (transpose.rs orderchange_out_r2c family)
    {
        let m = 4096usize;
        let a = full((vec![m, m], 1.0, &dev));
        let at = a.t();
        let mut c = full((vec![m, m], 0.0, &dev));
        time_case("transpose_assign_4096sq", || {
            c.assign(&at);
            black_box(&c);
        });
    }

    // reduction (reduction.rs reduce_axes): contiguous + strided-axis
    {
        let n = 1usize << 24;
        let a = full((vec![n], 1.0, &dev));
        time_case("sum_contig_2p24", || {
            black_box(a.sum());
        });
    }
    {
        let m = 2048usize;
        let a = full((vec![m, m], 1.0, &dev));
        let at = a.t();
        time_case("sum_strided_axis0_2048sq", || {
            black_box(at.sum_axes(&[0]));
        });
    }

    // vecdot (vecdot.rs vecdot_naive_cpu_rayon)
    {
        let n = 1usize << 22;
        let a = full((vec![n], 1.0, &dev));
        let b = full((vec![n], 2.0, &dev));
        time_case("vecdot_2p22", || {
            black_box(vecdot(&a, &b, (-1, -1)));
        });
    }

    // index_select (adv_indexing.rs)
    {
        let (rows, cols) = (8192usize, 512usize);
        let t = full((vec![rows, cols], 1.0, &dev));
        let idx: Vec<usize> = (0..4096).collect();
        time_case("index_select_4096_of_8192x512", || {
            black_box(index_select(&t, 0, &idx));
        });
    }

    // gemm_ix2_naive_cpu_rayon via faer-device fallback (i64: no faer gemm)
    {
        let m = 256usize;
        let a = full((vec![m, m], 1i64, &dev));
        let b = full((vec![m, m], 2i64, &dev));
        time_case("matmul_naive_i64_256sq", || {
            black_box(matmul(&a, &b));
        });
    }

    // cdist_rayon / cdist_weighted_rayon (sci-traits distance, rayon device)
    {
        let (m, k) = (512usize, 3usize);
        let xa = full((vec![m, k], 0.5, &dev));
        let xb = full((vec![m, k], 1.5, &dev));
        time_case("cdist_euclidean_512x3", || {
            black_box(cdist((xa.view(), xb.view(), MetricEuclidean)));
        });
        let w = full((vec![k], 1.0, &dev));
        time_case("cdist_weighted_euclidean_512x3", || {
            black_box(cdist((xa.view(), xb.view(), MetricEuclidean, w.view())));
        });
    }
}
