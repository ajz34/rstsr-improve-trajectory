//! T6 benchmark harness (argmax/argmin) for the rstsr CPU-efficiency campaign.
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (path dep;
//! PHASE 1 is no-patch — ../rstsr is read-only).
//!
//! Copied from the T0 harness pattern (`../2026-09-09-bench-harness-baseline`),
//! trimmed to what the arg* benches need:
//! - Two devices per op: [`serial_device`] (`DeviceCpuSerial`) and
//!   [`faer_device`] (`DeviceFaer`, i.e. the default device: rayon pool sized
//!   by `RAYON_NUM_THREADS`, convention 16 on this machine).
//! - Anti-cheat: inputs AND outputs go through `std::hint::black_box`.
//! - Allocation policy: arg* ops return a scalar index (no output allocation);
//!   identical across devices/configs. Fixture construction is outside the
//!   timed region.
//! - Deterministic fixtures (no PRNG): same data on both devices so
//!   correctness comparisons are exact.
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

/// Uniform criterion settings for every suite (minutes-scale totals).
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
/// across devices. `salt` decorrelates buffers.
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
// Size classes for this task
// ---------------------------------------------------------------------------

/// 1-D arg* sizes: (label, n). `small` is the regression gate (L1-class +
/// below-CONTIG_SWITCH fold path), `medium`/`large` are the headline sizes.
pub const ARG_SIZES: &[(&str, usize)] = &[
    ("small", 64),          // L1-class, below every parallel/contig switch
    ("small_odd", 1000),    // small but above PARALLEL_SWITCH/CONTIG_SWITCH edges
    ("medium", 1_000_000),  // 8 MB f64: L3-class on this X3D part
    ("large", 10_000_000),  // 80 MB f64: streams (L3-assisted on 128 MiB X3D)
];

/// 2-D whole-tensor arg* sizes: (label, rows, cols).
pub const ARG_MAT_SIZES: &[(&str, usize, usize)] = &[
    ("small", 64, 64),     // 32 KiB f64: L1-class
    ("medium", 512, 512),  // 2 MiB f64: L2-class
    ("large", 2048, 2048), // 32 MiB f64: streams
];

/// GB/s helper: bytes moved / seconds / 1e9.
pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    bytes / seconds / 1e9
}
