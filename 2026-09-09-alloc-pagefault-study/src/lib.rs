//! T7 benchmark harness: alloc/page-fault rider study.
//!
//! rstsr base commit: `386948be819baa334b8da02232f3a1944e5447d5` (read-only;
//! measurement-only task). This crate reuses the T0 harness pattern
//! (`2026-09-09-bench-harness-baseline`) unchanged where possible:
//!
//! - Two devices per op: [`serial_device`] (`DeviceCpuSerial`) and
//!   [`faer_device`] (`DeviceFaer`, rayon pool sized by `RAYON_NUM_THREADS`,
//!   convention 16 on this machine, asserted at every bench start).
//! - `std::hint::black_box` on inputs AND outputs.
//! - Deterministic fixtures via macros (no PRNG, identical across devices).
//!
//! Deviation from T0 (this is the *point* of T7): the allocation policy is
//! the variable under study, not a fixed convention. Two variants per op:
//!
//! - **Variant A ("alloc")**: idiomatic allocating call — output allocated
//!   inside the op on every iteration (exactly T0's policy).
//! - **Variant B ("reuse")**: output storage allocated once outside the
//!   timed loop; the op (or its closest existing-API equivalent) writes into
//!   it. Variant `B_bound` emulates a single-pass kernel writing into a
//!   preallocated `Vec` — labeled *emulated kernel bound, not an rstsr API*.

use std::time::Duration;

use rstsr_core::prelude::*;

// ---------------------------------------------------------------------------
// Devices (verbatim from the T0 harness)
// ---------------------------------------------------------------------------

/// Explicit serial device (`DeviceCpuSerial`).
pub fn serial_device() -> DeviceCpuSerial {
    DeviceCpuSerial::default()
}

/// The "default device" path: `DeviceFaer::new(0)` — exactly what
/// `Device::default()` constructs under feature `faer_as_default`.
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
// Criterion configuration (verbatim from the T0 harness)
// ---------------------------------------------------------------------------

pub const MEASUREMENT_TIME: Duration = Duration::from_secs(2);
pub const WARM_UP_TIME: Duration = Duration::from_millis(700);

pub fn configure_group<M: criterion::measurement::Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
) {
    group.measurement_time(MEASUREMENT_TIME);
    group.warm_up_time(WARM_UP_TIME);
}

// ---------------------------------------------------------------------------
// Deterministic fixtures (verbatim from the T0 harness)
// ---------------------------------------------------------------------------

#[inline]
pub fn gen_value(i: usize, salt: usize) -> f64 {
    let mut x =
        (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (salt as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    ((x % 2003) as f64) / 2003.0 - 0.5
}

pub fn gen_vec_f64(n: usize, salt: usize) -> Vec<f64> {
    (0..n).map(|i| gen_value(i, salt)).collect()
}

#[macro_export]
macro_rules! gen_mat_f64 {
    ($m:expr, $n:expr, $salt:expr, $device:expr) => {{
        let v: Vec<f64> = $crate::gen_vec_f64(($m) * ($n), $salt);
        rstsr_core::prelude::rt::asarray((v, [$m, $n], $device))
    }};
}

// ---------------------------------------------------------------------------
// getrusage accounting for the fixed-iteration profilers (examples/)
// ---------------------------------------------------------------------------

/// Snapshot of `getrusage(RUSAGE_SELF)` fields relevant to the fault study.
#[derive(Clone, Copy, Debug)]
pub struct Rusage {
    /// minor (soft) page faults — zero-page mapping churn lives here.
    pub min_flt: i64,
    /// major faults (should stay ~0 in this study).
    pub maj_flt: i64,
    /// user CPU time, seconds.
    pub utime: f64,
    /// system CPU time, seconds.
    pub stime: f64,
}

pub fn getrusage() -> Rusage {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    assert_eq!(ret, 0, "getrusage failed");
    Rusage {
        min_flt: ru.ru_minflt,
        maj_flt: ru.ru_majflt,
        utime: ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6,
        stime: ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 / 1e6,
    }
}

impl Rusage {
    pub fn delta(&self, before: &Rusage) -> Rusage {
        Rusage {
            min_flt: self.min_flt - before.min_flt,
            maj_flt: self.maj_flt - before.maj_flt,
            utime: self.utime - before.utime,
            stime: self.stime - before.stime,
        }
    }
}

/// Print a one-line accounting block for a timed section (machine-parsable).
pub fn report_accounting(tag: &str, iters: usize, wall: f64, d: &Rusage) {
    println!(
        "RESULT tag={tag} iters={iters} wall_s={wall:.4} ms_per_iter={:.4} \
         minflt={} majflt={} minflt_per_iter={:.1} user_s={:.4} sys_s={:.4}",
        wall * 1e3 / iters as f64,
        d.min_flt,
        d.maj_flt,
        d.min_flt as f64 / iters as f64,
        d.utime,
        d.stime,
    );
}
