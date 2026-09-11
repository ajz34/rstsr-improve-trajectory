// Spot check: final arg-family semantics on the rayon/faer device path,
// cross-checked against the serial device.
use rstsr_core::prelude::*;


fn check(name: &str, cond: bool) {
    println!("[{}] {}", if cond { "ok" } else { "FAIL" }, name);
    if !cond {
        std::process::exit(1);
    }
}

fn main() {
    let dev_rayon = DeviceFaer::default();
    let dev_serial = DeviceCpuSerial::default();

    // ---- NaN-free large: agreement serial vs rayon; nanarg == plain ----
    let n = 4_000_000;
    let v: Vec<f64> = (0..n).map(|i| ((i * 2654435761u64) % 1000003) as f64 - 500000.0).collect();
    let a_rayon = rt::asarray((v.as_slice(), &dev_rayon));
    let a_serial = rt::asarray((v.as_slice(), &dev_serial));
    check("argmax NaN-free serial==rayon", rt::argmax(&a_rayon) == rt::argmax(&a_serial));
    check("argmin NaN-free serial==rayon", rt::argmin(&a_rayon) == rt::argmin(&a_serial));
    check("nanargmax NaN-free == argmax", rt::nanargmax(&a_rayon) == rt::argmax(&a_rayon));
    check("nanargmin NaN-free == argmin", rt::nanargmin(&a_rayon) == rt::argmin(&a_rayon));

    // ---- mid NaN: plain skips it, nanarg skips it ----
    let k = 2_500_001;
    let mut w = v.clone();
    w[k] = f64::NAN;
    let b_rayon = rt::asarray((w.as_slice(), &dev_rayon));
    let b_serial = rt::asarray((w.as_slice(), &dev_serial));
    check("argmax skips mid NaN (rayon)", rt::argmax(&b_rayon) == rt::argmax(&a_rayon));
    check("argmax serial==rayon with mid NaN", rt::argmax(&b_rayon) == rt::argmax(&b_serial));
    check("nanargmax skips mid NaN (rayon)", rt::nanargmax(&b_rayon) == rt::argmax(&a_rayon));

    // ---- front NaN: plain poisons to 0, nanarg skips ----
    let mut u = v.clone();
    u[0] = f64::NAN;
    let c_rayon = rt::asarray((u.as_slice(), &dev_rayon));
    check("argmax front-NaN -> 0 (rayon)", rt::argmax(&c_rayon) == 0);
    check("nanargmax skips front NaN (rayon)", rt::nanargmax(&c_rayon) == rt::argmax(&a_rayon));

    // ---- all-NaN: plain -> 0, nanarg -> error ----
    let z = vec![f64::NAN; 1_000_000];
    let d_rayon = rt::asarray((z.as_slice(), &dev_rayon));
    check("argmax all-NaN -> 0 (rayon)", rt::argmax(&d_rayon) == 0);
    check("nanargmax all-NaN errors (rayon)", rt::nanargmax_f(&d_rayon).is_err());
    check("nanargmin all-NaN errors (rayon)", rt::nanargmin_f(&d_rayon).is_err());

    // ---- strided 2-D, rayon path ----
    let t = rt::asarray((vec![1.0, 5.0, 3.0, f64::NAN, 2.0, 6.0], &dev_rayon)).into_shape([2, 3]);
    let tt = t.t();
    // t.t() = [[1,nan],[5,2],[3,6]] row-major scan [1, nan, 5, 2, 3, 6]:
    // plain argmax skips the NaN -> max 6 at flat 5; nanargmax same.
    check("argmax strided skips NaN (rayon)", rt::argmax(&tt) == 5);
    check(
        "argmax strided serial==rayon",
        rt::argmax(&tt) == {
            let ts = rt::asarray((vec![1.0, 5.0, 3.0, f64::NAN, 2.0, 6.0], &dev_serial)).into_shape([2, 3]);
            rt::argmax(&ts.t())
        },
    );
    check("nanargmax strided (rayon)", rt::nanargmax(&tt) == 5);

    // ---- axes variants, rayon path ----
    let m = rt::asarray((vec![f64::NAN, 1.0, 2.0, f64::NAN], &dev_rayon)).into_shape([2, 2]);
    check("nanargmax_axes(0) (rayon)", m.nanargmax_axes(0).to_vec() == vec![1, 0]);
    check("nanargmin_axes(0) (rayon)", m.nanargmin_axes(0).to_vec() == vec![1, 0]);
    let m_allnan_col = rt::asarray((vec![f64::NAN, 2.0, f64::NAN, 1.0], &dev_rayon)).into_shape([2, 2]);
    check(
        "nanargmax_axes(0) all-NaN column errors (rayon)",
        m_allnan_col.nanargmax_axes_f(0).is_err(),
    );

    println!("ALL RAYON NAN CHECKS PASSED");
}
