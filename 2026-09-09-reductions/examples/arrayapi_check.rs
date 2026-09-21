//! Array-API 2024.12 compliance probe for reduction NaN semantics.
//! Spec requirements (statistical_functions, `max`/`min` Special Cases):
//!   - If any x_i is NaN, the min/max value is NaN (propagation REQUIRED).
//!   - Zero-size: implementation-defined (error permitted).
//!   - Signed-zero order: implementation-defined.
//! argmax/argmin: no NaN special case in the spec (unspecified).
//! Prints rstsr's actual behavior; NumPy comparisons recorded in the report.

use rstsr_core::prelude::*;

fn main() {
    let dev = DeviceCpuSerial::default();

    println!("== min_all / max_all (spec: NaN propagates) ==");
    let a = rt::asarray((vec![1.0f64, f64::NAN, 3.0], &dev));
    println!("  min_all([1, NaN, 3])       = {}   (spec: NaN; NumPy: NaN)", a.min_all());
    println!("  max_all([1, NaN, 3])       = {}   (spec: NaN; NumPy: NaN)", a.max_all());
    let a = rt::asarray((vec![f64::NAN, 5.0, 3.0], &dev));
    println!("  min_all([NaN, 5, 3])       = {}   (spec: NaN)", a.min_all());
    println!("  max_all([NaN, 5, 3])       = {}   (spec: NaN)", a.max_all());
    let a = rt::asarray((vec![f64::NAN, f64::NAN], &dev));
    println!("  min_all([NaN, NaN])        = {}   (spec: NaN)", a.min_all());
    println!("  max_all([NaN, NaN])        = {}   (spec: NaN)", a.max_all());

    println!("== min_axes / max_axes ==");
    let a = rt::asarray((vec![1.0f64, f64::NAN, 3.0, 2.0], [2usize, 2], &dev));
    let mn = a.min_axes(0);
    let mx = a.max_axes(0);
    println!("  min_axes(0, [[1, NaN], [3, 2]]) = [{}, {}]   (spec: [NaN, 2])", mn[[0usize]], mn[[1usize]]);
    println!("  max_axes(0, [[1, NaN], [3, 2]]) = [{}, {}]   (spec: [3, NaN])", mx[[0usize]], mx[[1usize]]);

    println!("== sum-family (spec: NaN propagates, natural) ==");
    let a = rt::asarray((vec![1.0f64, f64::NAN, 3.0], &dev));
    println!("  sum_all([1, NaN, 3])       = {}   (spec: NaN)", a.sum_all());
    println!("  mean_all([1, NaN, 3])      = {}   (spec: NaN)", a.mean_all());

    println!("== zero-size (spec: implementation-defined) ==");
    let a = rt::asarray((Vec::<f64>::new(), [0usize], &dev));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.min_all()));
    println!("  min_all([])                = {}   (spec: impl-defined)", r.map(|v| v.to_string()).unwrap_or_else(|_| "panic".into()));
    let s = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.sum_all()));
    println!("  sum_all([])                = {}   (spec: impl-defined)", s.map(|v| v.to_string()).unwrap_or_else(|_| "panic".into()));

    println!("== signed zeros (spec: implementation-defined) ==");
    let a = rt::asarray((vec![-0.0f64, 0.0], &dev));
    let m = a.min_all();
    println!("  min_all([-0.0, +0.0])      = {m:+}  sign-bit: {}", m.is_sign_negative());
    let a = rt::asarray((vec![0.0f64, -0.0], &dev));
    let m = a.min_all();
    println!("  min_all([+0.0, -0.0])      = {m:+}  sign-bit: {}", m.is_sign_negative());

    println!("== arg family (spec: NaN unspecified) ==");
    let a = rt::asarray((vec![1.0f64, f64::NAN, 3.0], &dev));
    println!("  argmax([1, NaN, 3])        = {}   (spec: unspecified; NumPy: 1)", rt::argmax(&a));
}
