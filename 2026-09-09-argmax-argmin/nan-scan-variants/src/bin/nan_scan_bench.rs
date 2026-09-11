// Scan-variant benchmark: why plain argmin/argmax keep NaN-skipping
// semantics instead of NumPy's first-NaN-wins, and why nanarg seeds are
// free. Run: cargo run --release --bin nan_scan_bench
//
// Measured on 9950X3D (native: `-C target-cpu=native`), n=1e7 f64:
//   plain (A)   ~1.45 ms  (auto-vectorizes to AVX-512 vmax/select)
//   fused (E)   ~1.5 ms large (+5..18%), but +73..100% at small sizes
//               (unordered-aware compare = scalar, no vectorization)
//   pre-pass    +100..340% everywhere (a second pass = a full DRAM pass;
//               block-tiling via per-block calls is even worse)
//   block8 (H)  +55..80% (the interleaved check de-vectorizes the loop)
//   no-earlyret: BROKEN semantics (poisoned lanes wash out; kept only to
//               document why the early return is required)
use std::time::Instant;

fn gen(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9u64);
            x ^= x >> 27;
            ((x % 2003) as f64) / 2003.0 - 0.5
        })
        .collect()
}

// A: plain 8-lane scan (committed kernel; NaN never wins an update)
fn scan_plain(xs: &[f64]) -> usize {
    let mut vs = [xs[0]; 8];
    let mut idx = [usize::MAX; 8];
    let mut chunks = xs.chunks_exact(8);
    let mut base = 0;
    for ch in chunks.by_ref() {
        for l in 0..8 {
            if ch[l] > vs[l] {
                vs[l] = ch[l];
                idx[l] = base + l;
            }
        }
        base += 8;
    }
    for (l, x) in chunks.remainder().iter().enumerate() {
        if *x > vs[l] {
            vs[l] = *x;
            idx[l] = base + l;
        }
    }
    let mut bi = 0;
    for l in 1..8 {
        if vs[l] > vs[bi] || (vs[l] == vs[bi] && idx[l] < idx[bi]) {
            bi = l;
        }
    }
    if vs[bi] == xs[0] && idx[bi] == usize::MAX {
        0
    } else {
        idx[bi]
    }
}

// D: NaN pre-scan only
fn prescan_only(xs: &[f64]) -> usize {
    for (i, x) in xs.iter().enumerate() {
        if *x != *x {
            return i;
        }
    }
    usize::MAX
}

// B: NaN pre-scan over the whole buffer, then one plain scan
fn scan_pre_whole(xs: &[f64]) -> usize {
    let i = prescan_only(xs);
    if i != usize::MAX {
        return i;
    }
    scan_plain(xs)
}

// C: block-tiled pre-scan + scan (per-block function calls)
fn scan_tiled(xs: &[f64]) -> usize {
    let mut offset = 0;
    for block in xs.chunks(2048) {
        for (i, x) in block.iter().enumerate() {
            if *x != *x {
                return offset + i;
            }
        }
        let _ = scan_plain(block);
        offset += block.len();
    }
    0
}

// E: fused unordered-compare update with early return (NumPy's scalar trick)
fn scan_fused(xs: &[f64]) -> usize {
    let mut vs = [xs[0]; 8];
    let mut idx = [usize::MAX; 8];
    let mut chunks = xs.chunks_exact(8);
    let mut base = 0;
    for ch in chunks.by_ref() {
        for l in 0..8 {
            if !(ch[l] <= vs[l]) {
                vs[l] = ch[l];
                idx[l] = base + l;
                if !(vs[l] == vs[l]) {
                    return base + l;
                }
            }
        }
        base += 8;
    }
    let mut bi = 0;
    for l in 1..8 {
        if vs[l] > vs[bi] || (vs[l] == vs[bi] && idx[l] < idx[bi]) {
            bi = l;
        }
    }
    if vs[bi] == xs[0] && idx[bi] == usize::MAX {
        0
    } else {
        idx[bi]
    }
}

// H: plain scan + per-8-block NaN or-reduce, scalar in-block fallback
fn scan_block8(xs: &[f64]) -> usize {
    let mut vs = [xs[0]; 8];
    let mut idx = [usize::MAX; 8];
    let mut chunks = xs.chunks_exact(8);
    let mut base = 0;
    for ch in chunks.by_ref() {
        let has_nan = ch[0] != ch[0]
            || ch[1] != ch[1]
            || ch[2] != ch[2]
            || ch[3] != ch[3]
            || ch[4] != ch[4]
            || ch[5] != ch[5]
            || ch[6] != ch[6]
            || ch[7] != ch[7];
        if has_nan {
            for (l, x) in ch.iter().enumerate() {
                if *x != *x {
                    return base + l;
                }
            }
        }
        for l in 0..8 {
            if ch[l] > vs[l] {
                vs[l] = ch[l];
                idx[l] = base + l;
            }
        }
        base += 8;
    }
    for (l, x) in chunks.remainder().iter().enumerate() {
        if *x != *x {
            return base + l;
        }
        if *x > vs[l] {
            vs[l] = *x;
            idx[l] = base + l;
        }
    }
    let mut bi = 0;
    for l in 1..8 {
        if vs[l] > vs[bi] || (vs[l] == vs[bi] && idx[l] < idx[bi]) {
            bi = l;
        }
    }
    if vs[bi] == xs[0] && idx[bi] == usize::MAX {
        0
    } else {
        idx[bi]
    }
}

// BROKEN on purpose: fused update without the early return. A NaN that wins
// a lane is immediately replaced by every later element (`!(x <= NaN)` is
// always true), so the NaN index is lost and the result is a plain max.
// Kept to document why the early return (and thus the vectorization loss)
// is unavoidable in the fused design.
fn scan_fused_no_earlyret_BROKEN(xs: &[f64]) -> usize {
    let mut vs = [xs[0]; 8];
    let mut idx = [usize::MAX; 8];
    let mut chunks = xs.chunks_exact(8);
    let mut base = 0;
    for ch in chunks.by_ref() {
        for l in 0..8 {
            if !(ch[l] <= vs[l]) {
                vs[l] = ch[l];
                idx[l] = base + l;
            }
        }
        base += 8;
    }
    for (l, x) in chunks.remainder().iter().enumerate() {
        if !(*x <= vs[l]) {
            vs[l] = *x;
            idx[l] = base + l;
        }
    }
    let any_nan = vs.iter().any(|v| *v != *v);
    if any_nan {
        for (i, x) in xs.iter().enumerate() {
            if *x != *x {
                return i;
            }
        }
    }
    let mut bi = 0;
    for l in 1..8 {
        if vs[l] > vs[bi] || (vs[l] == vs[bi] && idx[l] < idx[bi]) {
            bi = l;
        }
    }
    if vs[bi] == xs[0] && idx[bi] == usize::MAX {
        0
    } else {
        idx[bi]
    }
}

fn main() {
    // correctness spot checks (the no-earlyret variant is expected to fail
    // its check; that is the documented defect)
    let mut nan_mid: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    nan_mid[777] = f64::NAN;
    assert_eq!(scan_plain(&nan_mid), 999); // NaN skipped: plain max at 999
    assert_eq!(scan_fused(&nan_mid), 777); // first NaN wins
    assert_eq!(scan_block8(&nan_mid), 777);
    assert_eq!(scan_fused_no_earlyret_BROKEN(&nan_mid), 999); // defect!
    assert_eq!(scan_fused(&[f64::NAN; 8]), 0);
    println!("correctness spot checks ok");

    let iters = 20000;
    for &n in &[64usize, 1000, 4096, 65536, 262144, 1000000, 10000000] {
        let xs = gen(n);
        let it = iters.max(400_000_000 / n);
        let mut times = Vec::new();
        for (name, f) in [
            ("plain   ", scan_plain as fn(&[f64]) -> usize),
            ("fused   ", scan_fused as fn(&[f64]) -> usize),
            ("pre+all ", scan_pre_whole as fn(&[f64]) -> usize),
            ("tiled   ", scan_tiled as fn(&[f64]) -> usize),
            ("block8  ", scan_block8 as fn(&[f64]) -> usize),
        ] {
            let t = Instant::now();
            let mut acc = 0usize;
            for _ in 0..it {
                acc = acc.wrapping_add(f(&xs));
            }
            std::hint::black_box(acc);
            times.push((name, t.elapsed().as_secs_f64() * 1e9 / it as f64));
        }
        let base = times[0].1;
        let row: String = times
            .iter()
            .map(|(name, d)| format!("{}={:9.1}ns ({:+5.1}%)  ", name, d, (d / base - 1.0) * 100.0))
            .collect();
        println!("n={n:>9}: {row}");
    }
}
