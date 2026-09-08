//! T0 benchmark harness for the rstsr CPU-efficiency campaign.
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (read-only;
//! T0 is a no-patch task). This crate is the *reusable harness pattern*: later
//! experiment dirs copy this lib + the bench skeletons and swap in candidate
//! kernels.
//!
//! Conventions (see README.md):
//! - Two devices per op: [`serial_device`] (`DeviceCpuSerial`) and
//!   [`faer_device`] (`DeviceFaer`, i.e. the default device: rayon pool sized
//!   by `RAYON_NUM_THREADS`, convention 16 on this machine).
//! - Anti-cheat: inputs AND outputs go through `std::hint::black_box`.
//! - Allocation policy: ops allocate their outputs internally (`to_contig`,
//!   `sum_axes`, `+`, `zeros`, `full`, `vecdot`); that allocation is *part of
//!   the measured op* and identical across devices/configs. No pre-allocated
//!   output is reused between iterations anywhere in this crate.
//! - Deterministic fixtures (no PRNG): same data on both devices, so
//!   correctness comparisons are exact modulo summation order.
//! - Fixture constructors are **macros**, not generic fns: they inherit
//!   rstsr's trait bounds at the call site instead of re-declaring them.

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
/// construction time, i.e. the global pool default, which honors
/// `RAYON_NUM_THREADS`. Campaign convention: `RAYON_NUM_THREADS=16`.
pub fn faer_device() -> DeviceFaer {
    DeviceFaer::new(0)
}

/// Panic unless the faer device really owns `expected` threads.
///
/// Every bench/example binary calls this once in `main` so a mis-set
/// `RAYON_NUM_THREADS` can never silently falsify parallel numbers.
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

/// Uniform criterion settings for every suite (keeps total run time at
/// minutes scale while staying stable for the ~ms-large benches).
pub const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
pub const WARM_UP_TIME: Duration = Duration::from_millis(700);

/// Apply the uniform criterion configuration to a benchmark group.
pub fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);
    // sampling mode left at criterion's Auto default
}

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/// Deterministic pseudo-data in `[-0.5, 0.5)`; no PRNG dependency, identical
/// across devices. `salt` decorrelates operand buffers (a/b/c).
#[inline]
pub fn gen_value(i: usize, salt: usize) -> f64 {
    // integer hash (SplitMix-ish); deterministic across platforms
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

/// Fresh `Vec<f32>` cast from the f64 pattern (same values, f32 storage).
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

/// Contiguous row-major `[m, n]` f32 tensor on any device.
#[macro_export]
macro_rules! gen_mat_f32 {
    ($m:expr, $n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<f32> = $crate::gen_vec_f32(($m) * ($n), $salt);
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

// ---------------------------------------------------------------------------
// Size classes (plan D9)
// ---------------------------------------------------------------------------

/// The four 2-D size classes: (label, rows, cols).
pub const MAT_SIZES: &[(&str, usize, usize)] = &[
    ("small", 64, 64),     // ~32 KiB f64: L1-class
    ("medium", 512, 512),  // 2 MiB f64: L2-class
    ("large", 2048, 2048), // 32 MiB f64: streams, >> L2
    ("odd", 1000, 777),    // non-power-of-2 edges
];

/// 1-D sizes for vecdot: (label, n).
pub const VECDOT_SIZES: &[(&str, usize)] =
    &[("small", 1_000), ("medium", 100_000), ("large", 10_000_000)];

/// argmax 1-D sizes: (label, n).
pub const ARGMAX_SIZES: &[(&str, usize)] = &[("medium", 1_000_000), ("large", 10_000_000)];

/// GB/s helper: bytes moved / seconds / 1e9.
pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    bytes / seconds / 1e9
}
