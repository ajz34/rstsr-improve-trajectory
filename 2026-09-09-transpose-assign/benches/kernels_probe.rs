//! T1' raw-kernel probes: the assign kernels called DIRECTLY on slices +
//! layouts (they are `pub` in rstsr-native-impl), bypassing all tensor
//! wiring. This isolates kernel cost from wiring/alloc cost:
//!
//! - `P_generic_serial` / `P_generic_rayon16`:
//!   `assign_arbitary_uninit_cpu_{serial,rayon}` — the kernel the tensor
//!   path actually runs today (layout-iterator zip). P_generic_serial should
//!   reproduce the transpose kernel floor seen through the API.
//! - `P_blocked_c2r_serial` / `P_blocked_r2c_serial` /
//!   `P_blocked_c2r_rayon16`:
//!   the DORMANT blocked kernels `orderchange_out_c2r/r2c_ix2_cpu_*`
//!   (cpu_{serial,rayon}/transpose.rs, BLOCK_SIZE=64) — the phase-2
//!   candidate. c2r runs on the exact `a.t().to_contig(RowMajor)` layout
//!   pair (input [n,m] stride [1,n] -> output [n,m] stride [m,1]); r2c on
//!   the role-swapped pair (same physical transpose; both "orientations" of
//!   the kernel's stride guard).
//! - `P_blocked_swap_serial`: local probe, same blocked loop shape but with
//!   the inner loop walking the INPUT-fast axis (contiguous reads, strided
//!   writes) instead of the OUTPUT-fast axis (strided reads, contiguous
//!   writes). Design input: which stream should be the contiguous one.
//! - `P_blocked_rawptr_serial`: exact dormant loop but with raw-pointer
//!   writes + debug_assert bounds contracts (the in-tree serial kernel
//!   bounds-checks every write). Quantifies the check cost.
//!
//! All probes write into pre-allocated buffers (B policy; allocation outside
//! the timed region). f64 only (the kernels are dtype-generic; f64 is the
//! campaign primary). Sizes: large 2048x2048 + odd 1000x777. The explicit
//! 16-thread pool mirrors the RAYON_NUM_THREADS=16 convention without
//! depending on the env var.
//!
//! Layout math (verified in examples/correctness.rs): for `a` = [m, n]
//! row-major (stride [n, 1]), the transpose-copy call passes
//! la = a.t() = shape [n, m], stride [1, n]; lc = output c-contig =
//! shape [n, m], stride [m, 1].

use std::hint::black_box;
use std::mem::MaybeUninit;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rstsr_core::prelude_dev::*;
use transpose_assign::{configure_group, gen_vec_f64};

fn bid(label: &str, m: usize, n: usize) -> BenchmarkId {
    BenchmarkId::new(format!("probe_{m}x{n}_f64"), label)
}

/// Layout pair the tensor path passes for `a.t().to_contig(RowMajor)` with
/// `a` = [m, n] row-major: la = a.t() = [n, m] stride [1, n] (axis-0 fast);
/// lc = c-contig output = [n, m] stride [m, 1] (axis-1 fast).
/// This is the `c2r` orientation of the dormant kernel.
fn layouts_c2r(m: usize, n: usize) -> (Layout<Ix2>, Layout<Ix2>) {
    let lc: Layout<Ix2> = Layout::new([n, m], [m as isize, 1], 0).unwrap();
    let la: Layout<Ix2> = Layout::new([n, m], [1, n as isize], 0).unwrap();
    (lc, la)
}

/// Role-swapped pair expressing the same transpose as `r2c`:
/// input a = [m, n] stride [n, 1] (axis-1 fast), output col-major [m, n]
/// stride [1, m] (axis-0 fast; ldc = row count).
fn layouts_r2c(m: usize, n: usize) -> (Layout<Ix2>, Layout<Ix2>) {
    let lc: Layout<Ix2> = Layout::new([m, n], [1, m as isize], 0).unwrap();
    let la: Layout<Ix2> = Layout::new([m, n], [n as isize, 1], 0).unwrap();
    (lc, la)
}

fn bench_kernels(c: &mut Criterion) {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();

    let mut group = c.benchmark_group("kernels_probe");
    configure_group(&mut group);

    for &(m, n) in [(2048usize, 2048usize), (1000, 777)].iter() {
        let av = gen_vec_f64(m * n, 1); // a: [m, n] row-major fixture
        let size = m * n;

        // (lc, la) for the c2r orientation (the tensor-path orientation)
        let (lc, la) = layouts_c2r(m, n);
        // role-swapped pair for the r2c orientation
        let (lc2, la2) = layouts_r2c(m, n);

        // -- current generic kernel, serial (tensor-path kernel floor) -----
        let mut c_mu: Vec<MaybeUninit<f64>> = vec![MaybeUninit::uninit(); size];
        group.bench_function(bid("P_generic_serial", m, n), |bench| {
            bench.iter(|| {
                assign_arbitary_uninit_cpu_serial(
                    black_box(&mut c_mu),
                    black_box(&lc),
                    black_box(&av),
                    black_box(&la),
                    RowMajor,
                )
                .unwrap();
                black_box(&c_mu);
            })
        });

        // -- current generic kernel, rayon twin -----------------------------
        group.bench_function(bid("P_generic_rayon16", m, n), |bench| {
            bench.iter(|| {
                assign_arbitary_uninit_cpu_rayon(
                    black_box(&mut c_mu),
                    black_box(&lc),
                    black_box(&av),
                    black_box(&la),
                    RowMajor,
                    Some(&pool),
                )
                .unwrap();
                black_box(&c_mu);
            })
        });

        // -- blocked kernels + probes write an initialized f64 buffer --------
        let mut c: Vec<f64> = vec![0.0; size];

        // -- dormant blocked kernel, c2r orientation, serial ----------------
        group.bench_function(bid("P_blocked_c2r_serial", m, n), |bench| {
            bench.iter(|| {
                orderchange_out_c2r_ix2_cpu_serial(black_box(&mut c), black_box(&lc), black_box(&av), black_box(&la))
                    .unwrap();
                black_box(&c);
            })
        });

        // -- dormant blocked kernel, r2c orientation (roles swapped) --------
        group.bench_function(bid("P_blocked_r2c_serial", m, n), |bench| {
            bench.iter(|| {
                orderchange_out_r2c_ix2_cpu_serial(black_box(&mut c), black_box(&lc2), black_box(&av), black_box(&la2))
                    .unwrap();
                black_box(&c);
            })
        });

        // -- dormant blocked kernel, c2r, rayon twin ------------------------
        group.bench_function(bid("P_blocked_c2r_rayon16", m, n), |bench| {
            bench.iter(|| {
                orderchange_out_c2r_ix2_cpu_rayon(
                    black_box(&mut c),
                    black_box(&lc),
                    black_box(&av),
                    black_box(&la),
                    Some(&pool),
                )
                .unwrap();
                black_box(&c);
            })
        });

        // -- probe: inner loop walks the INPUT-fast axis --------------------
        //    (contiguous reads, strided writes) instead of the output-fast
        //    axis of the dormant body.
        group.bench_function(bid("P_blocked_swap_serial", m, n), |bench| {
            bench.iter(|| {
                blocked_transpose_swap(black_box(&mut c), &lc, black_box(&av), &la);
                black_box(&c);
            })
        });

        // -- probe: exact dormant loop with raw-pointer writes --------------
        group.bench_function(bid("P_blocked_rawptr_serial", m, n), |bench| {
            bench.iter(|| {
                blocked_transpose_rawptr(black_box(&mut c), &lc, black_box(&av), &la);
                black_box(&c);
            })
        });
    }

    group.finish();
}

/// Local probe: blocked transpose where the INNER loop steps along the
/// input's fast axis (axis 0 here: la.stride()[0] == 1) — contiguous reads,
/// strided writes (hop m). The dormant kernel does the opposite pairing.
/// c2r orientation fixtures only (la.stride()[0] == 1, lc.stride()[1] == 1).
fn blocked_transpose_swap(c: &mut [f64], lc: &Layout<Ix2>, a: &[f64], la: &Layout<Ix2>) {
    const BLOCK_SIZE: usize = 64;
    let [nrow, ncol] = *la.shape(); // [n, m]
    let (sa0, sa1) = (la.stride()[0], la.stride()[1]);
    let (sc0, sc1) = (lc.stride()[0], lc.stride()[1]);
    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    (0..ncol).step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        (0..nrow).step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i as isize * sa0 + j as isize * sa1) as usize;
                    let dst_idx = (offset_c + i as isize * sc0 + j as isize * sc1) as usize;
                    c[dst_idx] = a[src_idx];
                }
            }
        });
    });
}

/// Local probe: loop-for-loop the dormant c2r body (internally reverses the
/// axis pair, so in original space: shape [m, n], reads hop n, writes walk
/// the output contiguously), but with raw-pointer writes + debug_assert
/// bounds contracts instead of the in-tree per-element slice bounds checks.
fn blocked_transpose_rawptr(c: &mut [f64], lc: &Layout<Ix2>, a: &[f64], la: &Layout<Ix2>) {
    const BLOCK_SIZE: usize = 64;
    let [n, m] = *la.shape(); // la: [n, m] stride [1, n]
    // reversed (r2c) space: shape [m, n], lda = la.stride()[1] = n,
    // ldc = lc.stride()[0] = m
    let (nrow, ncol) = (m, n);
    let lda = la.stride()[1];
    let ldc = lc.stride()[0];
    let offset_a = la.offset() as isize;
    let offset_c = lc.offset() as isize;
    let c_ptr = c.as_ptr() as *mut f64;
    (0..ncol).step_by(BLOCK_SIZE).for_each(|j_start| {
        let j_end = (j_start + BLOCK_SIZE).min(ncol);
        (0..nrow).step_by(BLOCK_SIZE).for_each(|i_start| {
            let i_end = (i_start + BLOCK_SIZE).min(nrow);
            for j in j_start..j_end {
                for i in i_start..i_end {
                    let src_idx = (offset_a + i as isize * lda + j as isize) as usize;
                    let dst_idx = (offset_c + j as isize * ldc + i as isize) as usize;
                    debug_assert!(src_idx < a.len() && dst_idx < c.len());
                    unsafe {
                        *c_ptr.add(dst_idx) = *a.as_ptr().add(src_idx);
                    }
                }
            }
        });
    });
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
