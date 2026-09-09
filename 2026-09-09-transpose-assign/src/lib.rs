//! T1' benchmark harness for the rstsr CPU-efficiency campaign
//! (transpose copy + generic strided assign).
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (read-only;
//! phase 1 of T1' is baseline + anatomy + plan, NO edits to rstsr).
//! Harness pattern copied from `2026-09-09-bench-harness-baseline` (T0) with
//! the elementwise crate's A/B/C protocol baked in.
//!
//! Conventions (see README.md):
//! - Two devices per op: [`serial_device`] (`DeviceCpuSerial`) and
//!   [`faer_device`] (`DeviceFaer`, i.e. the default device: rayon pool sized
//!   by `RAYON_NUM_THREADS`, convention 16 on this machine).
//! - Allocation variants per case:
//!   - `A`: idiomatic allocating call (`.t().to_contig(RowMajor).into_owned()`)
//!     — allocation included (T0's policy; comparable to T0/T7 tables).
//!   - `B`: reuse into a pre-allocated, pre-warmed output `c` via
//!     `c.assign(&a.t())` — kernel-only; PRIMARY judge for the large class
//!     (T7 carry-forward: the ~4 ms glibc page-fault rider masks kernel
//!     deltas in A at 32 MiB outputs; transpose rider = 25% serial / 57%
//!     faer16).
//!   - `C`: (not a separate id) re-run of this binary under
//!     `MALLOC_MMAP_THRESHOLD_=67108864 MALLOC_TRIM_THRESHOLD_=134217728`;
//!     see reproduce.sh stages `portable_c` / `native_c`.
//! - Anti-cheat: inputs AND outputs go through `std::hint::black_box`.
//! - Deterministic fixtures (no PRNG): same data on both devices, so
//!   correctness comparisons are exact. Fixture constructors are macros,
//!   not generic fns (they inherit rstsr's trait bounds at the call site).

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

/// Uniform criterion settings for every suite (2 s measurement + 0.7 s
/// warm-up per bench; minutes-scale totals per config).
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
// Deterministic fixtures (identical generator to T0/elementwise crates)
// ---------------------------------------------------------------------------

/// Deterministic pseudo-data in `[-0.5, 0.5)`; no PRNG dependency, identical
/// across devices. `salt` decorrelates operand buffers.
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

/// Pre-allocated, page-warmed `[m, n]` f64 output for the reuse variant B.
///
/// `rt::zeros` is calloc-lazy by design (~2 µs, no page touching); `fill`
/// touches every page ONCE outside the timed region so variant B measures
/// kernel work, not first-touch faults (T7 protocol).
#[macro_export]
macro_rules! warm_output_mat {
    ($m:expr, $n:expr, $device:expr) => {{
        let mut c: rstsr_core::prelude::Tensor<f64, _> = rstsr_core::prelude::rt::zeros(([$m, $n], $device));
        c.fill(0.0);
        c
    }};
}

/// Page-warmed `[m, n]` f32 output (f32 variant of [`warm_output_mat!`]).
#[macro_export]
macro_rules! warm_output_mat_f32 {
    ($m:expr, $n:expr, $device:expr) => {{
        let mut c: rstsr_core::prelude::Tensor<f32, _> = rstsr_core::prelude::rt::zeros(([$m, $n], $device));
        c.fill(0.0);
        c
    }};
}

// ---------------------------------------------------------------------------
// Size classes (plan D9 + T1' brief: both odd orientations)
// ---------------------------------------------------------------------------

/// The five 2-D size classes for transpose copy: (label, rows, cols).
///
/// `odd` and `oddT` are the two orientations of the same odd footprint
/// (6.2 MiB f64): one has the source fast axis on columns (`1000x777`),
/// the other on rows (`777x1000`) — the dormant blocked kernel's guard
/// cares about which axis carries stride 1.
pub const MAT_SIZES: &[(&str, usize, usize)] = &[
    ("small", 64, 64),     // ~32 KiB f64: L1-class (regression gate)
    ("medium", 512, 512),  // 2 MiB f64: L2-class
    ("large", 2048, 2048), // 32 MiB f64: streaming (primary)
    ("odd", 1000, 777),    // non-power-of-2 edges
    ("oddT", 777, 1000),   // same footprint, transposed orientation
];

/// GB/s helper: bytes moved / seconds / 1e9.
///
/// For a transpose copy the honest traffic is 2x the tensor size
/// (read a + write c).
pub fn gbps(bytes: f64, seconds: f64) -> f64 {
    bytes / seconds / 1e9
}
