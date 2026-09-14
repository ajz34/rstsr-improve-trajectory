// Review-verification: end-to-end tensor-API test for the blocked 2-D
// elementwise path. Applied TEMPORARILY during the 2026-09-14 owner review as
// rstsr-core/tests/tmp_review_blocked2d.rs, then deleted.
// Captured in review-tests.patch (this file is that test file; see
// review-260914.md for run instructions).

// TEMPORARY review test — to be deleted.
#![cfg(feature = "row_major")]

use rstsr::prelude::*;

#[test]
fn test_add_transposed_end_to_end() {
    let mut device = DeviceCpuSerial::default();
    device.set_default_order(RowMajor);

    // sizes: 70*130 >= TILE_SWITCH(4096), non-multiples of TILE
    let (m, n) = (70usize, 130usize);
    let vec_a: Vec<f64> = (0..m * n).map(|i| i as f64 * 0.25 - 100.0).collect();
    let vec_b: Vec<f64> = (0..n * m).map(|i| -(i as f64) * 0.5 + 7.0).collect();
    let a: Tensor<f64, _> = rt::asarray((vec_a.clone(), &device)).into_shape([m, n]);
    let b: Tensor<f64, _> = rt::asarray((vec_b.clone(), &device)).into_shape([n, m]);

    // c = a + b.t()  (b.t() is a col-major view over a row-major buffer)
    let c = rt::add(&a, &b.t());
    for i in 0..m {
        for j in 0..n {
            let expect = vec_a[i * n + j] + vec_b[j * m + i];
            let got = c.i((i, j)).to_scalar();
            assert!((got - expect).abs() < 1e-9, "({i},{j}): {got} vs {expect}");
        }
    }

    // in-place: c += b.t()
    let mut d = a.clone();
    rt::add_assign(&mut d, &b.t());
    for i in 0..m {
        for j in 0..n {
            let expect = vec_a[i * n + j] + vec_b[j * m + i];
            let got = d.i((i, j)).to_scalar();
            assert!((got - expect).abs() < 1e-9, "inplace ({i},{j}): {got} vs {expect}");
        }
    }

    // flipped (negative-stride) operands
    let e = rt::add(&a.flip([0, 1]), &b.t());
    for i in 0..m {
        for j in 0..n {
            let expect = vec_a[(m - 1 - i) * n + (n - 1 - j)] + vec_b[j * m + i];
            let got = e.i((i, j)).to_scalar();
            assert!((got - expect).abs() < 1e-9, "flip ({i},{j}): {got} vs {expect}");
        }
    }

    // broadcast operand (row broadcast): guard must keep generic path
    let vec_row: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let row: Tensor<f64, _> = rt::asarray((vec_row.clone(), &device));
    let f = rt::add(&a, &row);
    for i in 0..m {
        for j in 0..n {
            let expect = vec_a[i * n + j] + vec_row[j];
            let got = f.i((i, j)).to_scalar();
            assert!((got - expect).abs() < 1e-9, "broadcast ({i},{j}): {got} vs {expect}");
        }
    }

    // sliced (interleaved) operands: a view of every 2nd element
    let big: Vec<f64> = (0..2 * m * n).map(|i| i as f64 * 0.05).collect();
    let bigt: Tensor<f64, _> = rt::asarray((big.clone(), &device)).into_shape([m, 2 * n]);
    let a2 = bigt.i((.., slice!(None, None, 2)));
    let g = rt::add(&a2, &b.t());
    for i in 0..m {
        for j in 0..n {
            let expect = big[i * 2 * n + 2 * j] + vec_b[j * m + i];
            let got = g.i((i, j)).to_scalar();
            assert!((got - expect).abs() < 1e-9, "sliced ({i},{j}): {got} vs {expect}");
        }
    }

    // 3-D problem stays on the generic path (ndim != 2), still correct
    let (m3, k3) = (16usize, 512usize); // 8192*2 elements, 3-D
    let va: Vec<f64> = (0..m3 * 2 * k3).map(|i| i as f64).collect();
    let vb: Vec<f64> = (0..m3 * 2 * k3).map(|i| -(i as f64)).collect();
    let ta: Tensor<f64, _> = rt::asarray((va, &device)).into_shape([m3, 2, k3]);
    let tb: Tensor<f64, _> = rt::asarray((vb, &device)).into_shape([2, m3, k3]);
    let tc = rt::add(&ta, &tb.permute_dims([1, 0, 2]));
    for i in 0..m3 {
        for j in 0..2 {
            for k in 0..k3 {
                let ia = (i * 2 + j) * k3 + k;
                let ib = (j * m3 + i) * k3 + k;
                let expect = ia as f64 - ib as f64;
                let got = tc.i((i, j, k)).to_scalar();
                assert!((got - expect).abs() < 1e-9, "3d ({i},{j},{k}): {got} vs {expect}");
            }
        }
    }
}

// Spot-check perf probe (debug build; relative patched-vs-stashed ratios only —
// release numbers live in the campaign tables).
#[test]
fn test_perf_transposed_add() {
    let mut device = DeviceCpuSerial::default();
    device.set_default_order(RowMajor);
    let (m, n) = (2000usize, 2000usize);
    let a: Tensor<f64, _> = rt::asarray(((0..m * n).map(|i| i as f64).collect::<Vec<f64>>(), &device)).into_shape([m, n]);
    let b: Tensor<f64, _> = rt::asarray(((0..m * n).map(|i| i as f64).collect::<Vec<f64>>(), &device)).into_shape([n, m]);
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let c = rt::add(&a, &b.t());
        let s = c.i((0, 0)).to_scalar();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(s);
    }
    eprintln!("PERF: add(a, b.t()) 2000x2000: {best:.2} ms");
    let mut best2 = f64::INFINITY;
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let c2 = rt::add(&a, &a);
        let s2 = c2.i((0, 0)).to_scalar();
        best2 = best2.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(s2);
    }
    eprintln!("PERF: add(a, a) contig 2000x2000: {best2:.2} ms");
}
