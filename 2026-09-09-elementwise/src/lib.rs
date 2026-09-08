//! T4' benchmark harness for the rstsr CPU-efficiency campaign (elementwise).
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (phase 1:
//! read-only). Copy of the T0 harness pattern
//! (`../2026-09-09-bench-harness-baseline`) with the T7 carry-forwards baked
//! in:
//! - variant **A** = idiomatic allocating op (allocation included);
//! - variant **B** = reuse via the public single-pass driver
//!   `op_mutc_refa_refb_func` into a pre-allocated, pre-warmed output
//!   (this is the PRIMARY judge for >=32 MiB-output cases per T7);
//! - variant **C** is a re-run of the A binary under
//!   `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`
//!   (secondary column; see reproduce.sh stage `native_c`/`portable_c`).
//!
//! Conventions: `black_box` on inputs AND outputs; deterministic fixtures
//! (no PRNG); `RAYON_NUM_THREADS=16` asserted for every faer device; stock
//! cargo release profile; criterion 2 s measurement + 0.7 s warm-up.

use std::time::Duration;

use rstsr_core::prelude::*;

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

/// Explicit serial device (`DeviceCpuSerial`).
pub fn serial_device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// The "default device" path: `DeviceFaer::new(0)` (pool sized from
/// `RAYON_NUM_THREADS` at process start; campaign convention 16).
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

pub const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
pub const WARM_UP_TIME: Duration = Duration::from_millis(700);

pub fn configure_group<M: criterion::measurement::Measurement>(group: &mut criterion::BenchmarkGroup<'_, M>) {
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);
}

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/// Deterministic pseudo-data in `[-0.5, 0.5)`; identical across devices.
#[inline]
pub fn gen_value(i: usize, salt: usize) -> f64 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    ((x % 2003) as f64) / 2003.0 - 0.5
}

pub fn gen_vec_f64(n: usize, salt: usize) -> Vec<f64> {
    (0..n).map(|i| gen_value(i, salt)).collect()
}

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

/// Pre-allocated output tensor with warmed pages (T7 reuse-variant protocol:
/// `rt::zeros` is calloc-lazy, so fill once OUTSIDE timing to normalize
/// first-touch faults, then judge kernel-only).
#[macro_export]
macro_rules! warm_output_mat {
    ($m:expr, $n:expr, $device:expr) => {{
        let mut c: Tensor<f64, _> = rstsr_core::prelude::rt::zeros(([$m, $n], $device));
        c.fill(0.0);
        c
    }};
}

#[macro_export]
macro_rules! warm_output_mat_f32 {
    ($m:expr, $n:expr, $device:expr) => {{
        let mut c: Tensor<f32, _> = rstsr_core::prelude::rt::zeros(([$m, $n], $device));
        c.fill(0.0);
        c
    }};
}

// ---------------------------------------------------------------------------
// Size classes (plan D9; T4' brief)
// ---------------------------------------------------------------------------

/// add contig runs on all four classes; the other cases are large-class.
pub const MAT_SIZES: &[(&str, usize, usize)] = &[
    ("small", 64, 64),     // ~32 KiB f64: L1-class (regression gate)
    ("medium", 512, 512),  // 2 MiB f64: L2-class (regression gate)
    ("large", 2048, 2048), // 32 MiB f64: streaming (primary)
    ("odd", 1000, 777),    // non-power-of-2 edges (regression gate)
];

/// GB/s helper: bytes moved / seconds / 1e9.
///
/// For `c = a + b` the traffic convention here is 24 B/element
/// (2 reads + 1 write of f64) — same convention as the T0 tables.
pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    bytes / seconds / 1e9
}
