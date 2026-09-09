//! T5 (half 1) benchmark harness — fill/creation kernel study.
//!
//! Copied from the T0 harness pattern
//! (`../2026-09-09-bench-harness-baseline/src/lib.rs`), same conventions:
//! - Two devices per op: [`serial_device`] and [`faer_device`] (pool asserted
//!   16 threads via `RAYON_NUM_THREADS`).
//! - Anti-cheat: inputs AND outputs through `std::hint::black_box`.
//! - Allocation policy is per-bench-variant and identical across
//!   devices/configs: A = creation allocates inside the op (it IS the op);
//!   B = output allocated ONCE outside the loop and pre-warmed (kernel-only);
//!   BOUND = raw slice loop into a pre-warmed `Vec` (emulated floor, not an
//!   rstsr API).
//! - Deterministic fixtures; constructor macros inherit rstsr's bounds.

use std::time::Duration;

use rstsr_core::prelude::*;

// ---------------------------------------------------------------------------
// Devices (verbatim from the T0 pattern)
// ---------------------------------------------------------------------------

/// Explicit serial device (`DeviceCpuSerial`).
pub fn serial_device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// The "default device" path: `DeviceFaer::new(0)` (= `Device::default()`
/// under `faer_as_default`); pool size honors `RAYON_NUM_THREADS`.
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
// Criterion configuration (verbatim from the T0 pattern)
// ---------------------------------------------------------------------------

pub const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
pub const WARM_UP_TIME: Duration = Duration::from_millis(700);

pub fn configure_group<M: criterion::measurement::Measurement>(group: &mut criterion::BenchmarkGroup<'_, M>) {
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);
}

// ---------------------------------------------------------------------------
// Fixtures / size classes
// ---------------------------------------------------------------------------

pub const FILL_VALUE_F64: f64 = 3.25;
pub const FILL_VALUE_F32: f32 = 1.5;

/// 2-D fill sizes: (label, rows, cols). Same classes as the campaign matrix.
pub const FILL_SIZES: &[(&str, usize, usize)] = &[
    ("small", 64, 64),     // 32 KiB f64: L1-class
    ("medium", 512, 512),  // 2 MiB f64: L2-class
    ("large", 2048, 2048), // 32 MiB f64: the glibc mmap class (T7 rider)
    ("odd", 1000, 777),    // 6.2 MiB, non-power-of-2
];
