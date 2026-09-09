//! T3' benchmark harness for the rstsr CPU-efficiency campaign (vecdot +
//! inner_dot). Copied from the T0 harness pattern
//! (`2026-09-09-bench-harness-baseline/src/lib.rs`) with additions for this
//! experiment: complex fixtures, batched-vecdot sizes, matmul-`%` helpers.
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (path dep;
//! PHASE 1 is read-only for ../rstsr).
//!
//! Conventions (see T0 README + campaign plan §3):
//! - Two devices per op: [`serial_device`] (`DeviceCpuSerial`) and
//!   [`faer_device`] (`DeviceFaer`, the default device: rayon pool sized by
//!   `RAYON_NUM_THREADS`, convention 16 on this machine).
//! - Anti-cheat: inputs AND outputs go through `std::hint::black_box`.
//! - Allocation policy: ops allocate their outputs internally (`rt::vecdot`,
//!   `%`); that allocation is *part of the measured op* and identical across
//!   devices/configs. Outputs are 4096 f64 = 32 KiB for the primary batched
//!   case — no page-fault rider (T7 study: rider only matters ≥ MiB-scale
//!   outputs).
//! - Deterministic fixtures (no PRNG): same data on both devices.

use std::time::Duration;

use rstsr_core::prelude::*;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// Explicit serial device (`DeviceCpuSerial`).
pub fn serial_device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// The "default device" path: `DeviceFaer::new(0)` — exactly what
/// `Device::default()` constructs under feature `faer_as_default`.
///
/// With `num_threads = 0` the pool size is `rayon::current_num_threads()` at
/// construction time, which honors `RAYON_NUM_THREADS`. Campaign convention:
/// `RAYON_NUM_THREADS=16`.
pub fn faer_device() -> DeviceFaer {
    DeviceFaer::new(0)
}

/// Panic unless the faer device really owns `expected` threads.
pub fn assert_faer_threads(device: &DeviceFaer, expected: usize) {
    let n = device.get_num_threads();
    assert_eq!(
        n, expected,
        "DeviceFaer pool has {n} threads, expected {expected}; \
         set RAYON_NUM_THREADS={expected} before running"
    );
    eprintln!("[harness] DeviceFaer thread pool: {n} threads (RAYON_NUM_THREADS convention)");
}

// ---------------------------------------------------------------------------
// Criterion configuration
// ---------------------------------------------------------------------------

/// Uniform criterion settings (2 s measurement + 0.7 s warm-up, as T0).
pub const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
pub const WARM_UP_TIME: Duration = Duration::from_millis(700);

/// Apply the uniform criterion configuration to a benchmark group.
pub fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);
}

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/// Deterministic pseudo-data in `[-0.5, 0.5)`; no PRNG dependency, identical
/// across devices. `salt` decorrelates operand buffers (a/b).
#[inline]
pub fn gen_value(i: usize, salt: usize) -> f64 {
    let mut x =
        (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    ((x % 2003) as f64) / 2003.0 - 0.5
}

/// Fresh `Vec<f64>` of deterministic values in `[-0.5, 0.5)`.
pub fn gen_vec_f64(n: usize, salt: usize) -> Vec<f64> {
    (0..n).map(|i| gen_value(i, salt)).collect()
}

/// Fresh `Vec<Complex<f64>>` (re/im from decorrelated f64 patterns).
pub fn gen_vec_c64(n: usize, salt: usize) -> Vec<num::Complex<f64>> {
    (0..n)
        .map(|i| num::Complex::new(gen_value(i, salt), gen_value(i, salt.wrapping_add(77))))
        .collect()
}

/// Fresh `Vec<f32>` cast from the f64 pattern.
pub fn gen_vec_f32(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| gen_value(i, salt) as f32).collect()
}

/// Contiguous row-major `[m, n]` f64 tensor on any device.
#[macro_export]
macro_rules! gen_mat_f64 {
    ($m:expr, $n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<f64> = $crate::gen_vec_f64(($m) * ($n), $salt);
        rstsr_core::prelude::rt::asarray((v, [$m, $n], $device))
    }};
}

/// Contiguous row-major `[m, n]` Complex<f64> tensor on any device.
#[macro_export]
macro_rules! gen_mat_c64 {
    ($m:expr, $n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<num::Complex<f64>> = $crate::gen_vec_c64(($m) * ($n), $salt);
        rstsr_core::prelude::rt::asarray((v, [$m, $n], $device))
    }};
}

/// Contiguous f64 vector of length `n` on any device.
#[macro_export]
macro_rules! gen_vec_dev_f64 {
    ($n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<f64> = $crate::gen_vec_f64($n, $salt);
        rstsr_core::prelude::rt::asarray((v, [$n], $device))
    }};
}

/// Contiguous f32 vector of length `n` on any device.
#[macro_export]
macro_rules! gen_vec_dev_f32 {
    ($n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<f32> = $crate::gen_vec_f32($n, $salt);
        rstsr_core::prelude::rt::asarray((v, [$n], $device))
    }};
}

/// Contiguous Complex<f64> vector of length `n` on any device.
#[macro_export]
macro_rules! gen_vec_dev_c64 {
    ($n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<num::Complex<f64>> = $crate::gen_vec_c64($n, $salt);
        rstsr_core::prelude::rt::asarray((v, [$n], $device))
    }};
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

/// 1-D dot sizes (gates): (label, n). Same as T0.
pub const DOT1D_SIZES: &[(&str, usize)] =
    &[("small", 1_000), ("medium", 100_000), ("large", 10_000_000)];

/// inner_dot (`%` on 1-D) sizes: (label, n).
pub const INNERDOT_SIZES: &[(&str, usize)] =
    &[("small", 64), ("medium", 10_000), ("large", 1_000_000), ("xlarge", 10_000_000)];

/// Batched vecdot cases: (label, rows, cols, contract_axis).
///
/// - `am1`: contract LAST axis — row-dot shape (numpy `einsum('ij,ij->i')`).
///   (4096,512) is T0's primary; (8192,256) is the same-data second point.
/// - `axis0`: contract FIRST axis — column-accumulation shape
///   (`einsum('ij,ij->j')`); this is the geometry that hits the
///   contiguous-REMAINING (MaybeUninit read-modify-write) branch of
///   `vecdot_naive_cpu_serial`.
/// - `strided`: b stored (cols, rows) contiguous and viewed transposed, so
///   the contracted axis of b has stride `rows` (general branch).
/// - `odd`/`small`: correctness-adjacent gates.
pub const BATCHED_CASES: &[(&str, usize, usize, i32)] = &[
    ("small", 64, 64, -1),
    ("odd", 1000, 777, -1),
    ("batched_am1", 4096, 512, -1),
    ("batched_am1", 8192, 256, -1),
    ("batched_axis0", 512, 4096, 0),
    ("batched_strided", 4096, 512, -2), // special: b viewed as (cols, rows).t()
];

/// GB/s helper: bytes moved / seconds / 1e9.
pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    bytes / seconds / 1e9
}
