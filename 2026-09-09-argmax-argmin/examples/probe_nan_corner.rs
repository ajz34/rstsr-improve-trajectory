//! Probe for NaN-at-fold-split instability: run against the CLEAN tree to demonstrate the pre-patch pairing-order dependence (results in results/probe_nan_clean_tree.txt); run against the PATCHED tree to verify the deterministic combine (expect "0 unstable cases").: find a case
//! where the PRE-PATCH rayon argmax returns a NaN index even though a real
//! global max exists — demonstrating the pairing-order dependence that the
//! phase-2 kernel fixes.

use bench_argmax_argmin::gen_vec_f64;
use rstsr_core::prelude::*;

fn main() {
    let dev = DeviceFaer::new(0);
    let mut found = 0;
    for n in (1025..40_000).step_by(977) {
        for nan_pos in (1..n).step_by(n.max(1) / 241 + 1) {
            if nan_pos == n - 1 {
                continue; // skip the degenerate case (NaN overwrites the max)
            }
            let mut d = gen_vec_f64(n, 7);
            let max_idx = n - 1; // real max at the very end (a "far" position)
            d[max_idx] = 1000.0;
            d[nan_pos] = f64::NAN;
            let a = rt::asarray((d, [n], &dev));
            let got = rt::argmax(&a);
            if got != max_idx {
                println!("INSTABILITY: n={n} nan@{nan_pos} max@{max_idx} -> argmax={got}");
                found += 1;
                if found >= 5 {
                    return;
                }
            }
        }
    }
    println!("probe done, {found} unstable cases found");
}
