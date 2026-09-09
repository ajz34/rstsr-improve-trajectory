//! Manual-loop probes for the T2' design options (phase-1 evidence).
//!
//! Reproduces the SHAPES of `reduce_axes_cpu_serial` branch 2 (axis0) and
//! branch 1 (axis1) against the design candidates, timing them directly.
//! Pure Rust on a plain Vec (device/kernel dispatch overhead excluded) —
//! these are DESIGN-ANCHOR numbers, not rstsr numbers.
//!
//! Run: cargo run --release --example probe_manual [-- features ignored]

use std::hint::black_box;
use std::time::Instant;

fn gen(n: usize, salt: u64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut x = i as u64 ^ (salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            ((x % 2003) as f64) / 2003.0 - 0.5
        })
        .collect()
}

fn time_it<F: FnMut()>(name: &str, iters: usize, bytes_per_iter: f64, mut f: F) {
    f(); // warmup
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let el = t0.elapsed().as_secs_f64() / iters as f64;
    println!("{:44} {:10.6} ms/iter  {:7.1} GB/s", name, el * 1e3, bytes_per_iter / el / 1e9);
}

fn main() {
    let (m, n) = (2048usize, 2048usize);
    let a = gen(m * n, 1);
    let bytes = (m * n) as f64 * 8.0;
    let iters = 400;

    // ------------------------------------------------------------------
    // axis0 (branch 2 shapes)
    // ------------------------------------------------------------------

    // current shape: fresh vacc per outer group, CHUNK=48 bands, per-chunk
    // slice + bounds check + clone-through-closure add (f64: plain add)
    time_it("axis0 current-shape: vacc+CHUNK=48 bands", iters, bytes, || {
        let mut out = vec![0.0_f64; n];
        const CHUNK: usize = 48;
        for vacc_chunk in out.chunks_mut(CHUNK) {
            let start = 0usize; // single outer group here
            let nchunk = vacc_chunk.len();
            for i in 0..m {
                let slc = &a[i * n + start..i * n + start + nchunk];
                vacc_chunk.iter_mut().zip(slc).for_each(|(acc, x)| {
                    *acc = *acc + *x;
                });
            }
        }
        black_box(&out);
    });

    // candidate A: plain 2048-wide vacc, contiguous row adds, no chunking,
    // no per-chunk bounds checks
    time_it("axis0 candidate: full vacc, row adds", iters, bytes, || {
        let mut out = vec![0.0_f64; n];
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            out.iter_mut().zip(row).for_each(|(acc, x)| *acc += *x);
        }
        black_box(&out);
    });

    // candidate A2: same but 8-lane unrolled by hand (chunks_exact 8)
    time_it("axis0 candidate: full vacc, 8-lane unroll", iters, bytes, || {
        let mut out = vec![0.0_f64; n];
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            for (acc, x) in out.chunks_exact_mut(8).zip(row.chunks_exact(8)) {
                for l in 0..8 {
                    acc[l] += x[l];
                }
            }
        }
        black_box(&out);
    });

    // candidate B: lane accumulators [f64; 8] over rows (register vacc),
    // for the case where output rows are narrow
    time_it("axis0 candidate: [f64;8] row-lane accum", iters, bytes, || {
        let mut out = vec![0.0_f64; n];
        let mut lanes = [0.0_f64; 8];
        for i in 0..m {
            let row = &a[i * n..i * n + n];
            for (l, x) in row.chunks_exact(8).enumerate() {
                let _ = l;
                for k in 0..8 {
                    lanes[k] += x[k];
                }
            }
            row[8 * (n / 8)..].iter().for_each(|x| out[0] += x); // tail sink
        }
        out[0] = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        black_box(&out);
    });

    // column-wise scalar walk (the "output-order" naive — BAD pattern)
    time_it("axis0 naive: column walk (stride-n reads)", iters, bytes, || {
        let mut out = vec![0.0_f64; n];
        for j in 0..n {
            let mut acc = 0.0;
            for i in 0..m {
                acc += a[i * n + j];
            }
            out[j] = acc;
        }
        black_box(&out);
    });

    println!();

    // ------------------------------------------------------------------
    // axis1 (branch 1 shapes)
    // ------------------------------------------------------------------
    time_it("axis1 current-shape: per-row unrolled fold", iters, bytes, || {
        let mut out = vec![0.0_f64; m];
        for i in 0..m {
            let slc = &a[i * n..i * n + n];
            // unrolled_reduce shape: 8 scalar lanes
            let mut p = [0.0_f64; 8];
            let mut chunks = slc.chunks_exact(8);
            while let Some(c) = chunks.next() {
                for l in 0..8 {
                    p[l] += c[l];
                }
            }
            let mut acc = 0.0;
            for (l, x) in chunks.remainder().iter().enumerate() {
                acc += if l < 7 { 0.0 } else { *x };
            }
            out[i] = acc + p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }
        black_box(&out);
    });

    time_it("axis1 naive: per-row scalar fold", iters, bytes, || {
        let mut out = vec![0.0_f64; m];
        for i in 0..m {
            let slc = &a[i * n..i * n + n];
            let mut acc = 0.0;
            for x in slc {
                acc += x;
            }
            out[i] = acc;
        }
        black_box(&out);
    });

    println!();

    // ------------------------------------------------------------------
    // sum_all shapes (already-good path — must not regress)
    // ------------------------------------------------------------------
    time_it("sum_all: 8-lane unrolled (current shape)", iters, bytes, || {
        let mut p = [0.0_f64; 8];
        let mut chunks = a.chunks_exact(8);
        while let Some(c) = chunks.next() {
            for l in 0..8 {
                p[l] += c[l];
            }
        }
        let mut acc = 0.0;
        for x in chunks.remainder() {
            acc += x;
        }
        black_box(acc + p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7]);
    });

    time_it("sum_all: plain scalar fold (reference)", iters, bytes, || {
        let mut acc = 0.0;
        for x in &a {
            acc += x;
        }
        black_box(acc);
    });

    println!();

    // ------------------------------------------------------------------
    // min_all shapes (reduce_all with ext_min — native pathology at 1e7)
    // ------------------------------------------------------------------
    // f64::min semantics (NaN-skipping, seeded) == `if x < acc { acc = x }`
    // for a MAX-seeded accumulator: NaN never passes the strict compare.
    time_it("min_all: 8-lane unrolled via f64::min (rstsr shape)", iters, bytes, || {
        let mut p = [f64::MAX; 8];
        let mut chunks = a.chunks_exact(8);
        while let Some(c) = chunks.next() {
            for l in 0..8 {
                p[l] = p[l].min(c[l]);
            }
        }
        let mut acc = f64::MAX;
        for x in chunks.remainder() {
            acc = acc.min(*x);
        }
        black_box(p.iter().fold(acc, |a, b| a.min(*b)));
    });

    time_it("min_all: 8-lane unrolled via strict compare", iters, bytes, || {
        let mut p = [f64::MAX; 8];
        let mut chunks = a.chunks_exact(8);
        while let Some(c) = chunks.next() {
            for l in 0..8 {
                if c[l] < p[l] {
                    p[l] = c[l];
                }
            }
        }
        let mut acc = f64::MAX;
        for x in chunks.remainder() {
            if *x < acc {
                acc = *x;
            }
        }
        black_box(p.iter().fold(acc, |a, b| a.min(*b)));
    });

    time_it("min_all: plain scalar fold (f64::min)", iters, bytes, || {
        let mut acc = f64::MAX;
        for x in &a {
            acc = acc.min(*x);
        }
        black_box(acc);
    });

    time_it("min_all: 16-lane compare unroll", iters, bytes, || {
        let mut p = [f64::MAX; 16];
        let mut chunks = a.chunks_exact(16);
        while let Some(c) = chunks.next() {
            for l in 0..16 {
                if c[l] < p[l] {
                    p[l] = c[l];
                }
            }
        }
        let mut acc = f64::MAX;
        for x in p {
            acc = acc.min(x);
        }
        black_box(acc);
    });
}
