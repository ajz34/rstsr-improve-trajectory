//! T8 compose-smoke harness. Devices + deterministic fixture helpers,
//! copied from the campaign's T0/T2' harness conventions.

use rstsr_core::prelude::*;

/// Explicit serial device (`DeviceCpuSerial`).
pub fn serial_device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// The "default device" path: `DeviceFaer::new(0)` — exactly what
/// `Device::default()` constructs under feature `faer_as_default`.
/// Pool size honors `RAYON_NUM_THREADS` (campaign convention: 16).
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
}

/// Deterministic pseudo-data in `[-0.5, 0.5)`; no PRNG, identical across
/// devices. `salt` decorrelates operand buffers.
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

/// Fresh `Vec<f32>` cast from the f64 pattern.
pub fn gen_vec_f32(n: usize, salt: usize) -> Vec<f32> {
    (0..n).map(|i| gen_value(i, salt) as f32).collect()
}
