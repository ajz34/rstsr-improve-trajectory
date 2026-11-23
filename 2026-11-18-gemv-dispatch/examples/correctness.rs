//! Correctness gate for the P0 numbers: every timed rstsr call must compute the
//! same product as the raw cblas gemv anchor on the same buffer.
//!
//! Run before the benchmarks (`cargo run --release --example correctness`).

use gemv_dispatch_p0::*;

macro_rules! check_device {
    ($dev:expr, $label:literal, $ty:ty, $tol:expr) => {{
        let dev = &$dev;
        let label: &str = $label;
        assert_eq!(dev.get_num_threads(), NTHREADS);
        for &(p, q, name) in CELLS {
            let a = linspace((0 as $ty, 1 as $ty, p * q, dev)).into_shape([p, q]);
            let xp = linspace((0 as $ty, 1 as $ty, p, dev));
            let xq = linspace((0 as $ty, 1 as $ty, q, dev));
            let ap = a.as_ptr(); // col-major view (q, p), lda = q

            // matvec: y(p) = a(p,q) xq(q)
            let y = &a % &xq;
            assert_eq!(y.layout().offset(), 0, "{name} matvec: non-zero offset");
            assert_eq!(y.layout().stride()[0], 1, "{name} matvec: non-unit output stride");
            let got = unsafe { std::slice::from_raw_parts(y.as_ptr(), p) };
            let mut want = vec![0 as $ty; p];
            unsafe { <$ty as CblasGemv>::cblas_gemv(true, q, p, ap, q, xq.as_ptr(), want.as_mut_ptr()) };
            let rel = max_rel_diff(got, &want);
            println!("{label} {name:>12} matvec rel={rel:.3e}");
            assert!(rel < $tol, "{label} {name} matvec rel={rel:.3e}");

            // vecmat: y(q) = xp(p)^T a(p,q)
            let y = &xp % &a;
            assert_eq!(y.layout().offset(), 0, "{name} vecmat: non-zero offset");
            assert_eq!(y.layout().stride()[0], 1, "{name} vecmat: non-unit output stride");
            let got = unsafe { std::slice::from_raw_parts(y.as_ptr(), q) };
            let mut want = vec![0 as $ty; q];
            unsafe { <$ty as CblasGemv>::cblas_gemv(false, q, p, ap, q, xp.as_ptr(), want.as_mut_ptr()) };
            let rel = max_rel_diff(got, &want);
            println!("{label} {name:>12} vecmat rel={rel:.3e}");
            assert!(rel < $tol, "{label} {name} vecmat rel={rel:.3e}");
        }
    }};
}

fn main() {
    println!("blas: {}", blas_config());
    pin_blas_threads(NTHREADS);
    check_device!(DeviceOpenBLAS::new(NTHREADS), "openblas", f64, 1e-10);
    check_device!(DeviceFaer::new(NTHREADS), "faer", f64, 1e-10);
    check_device!(DeviceOpenBLAS::new(NTHREADS), "openblas", f32, 1e-4);
    println!("correctness gate: OK");
}
