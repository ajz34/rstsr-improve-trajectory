// Review-verification tests for the blocked 2-D elementwise kernels (serial).
// Applied TEMPORARILY during the 2026-09-14 owner review by appending to
// rstsr-native-impl/src/cpu_serial/op_with_func.rs, then reverted.
// Captured in review-tests.patch (this file is the appended block; see
// review-260914.md for run instructions).

/* TEMPORARY review tests — to be reverted */
#[cfg(test)]
mod review_tests {
    use super::*;

    fn layout2(shape: [usize; 2], stride: [isize; 2], offset: usize) -> Layout<Ix2> {
        unsafe { Layout::new_unchecked(shape, stride, offset) }
    }

    // Adversarial layout matrix for the 3-layout kernel: every scenario is
    // bit-compared against the layout-iterator reference. Exercises negative
    // strides on each operand (flip views), all-negative combos, interleaved
    // (sliced) strides, nonzero offsets, c-col-major, and shapes at/around
    // the TILE boundary that are not multiples of 64.
    #[test]
    fn test_blocked_3layout_matrix() {
        for &(m, n) in &[
            (70usize, 130usize),
            (130, 70),
            (4096, 2),
            (2, 4096),
            (64, 64),
            (65, 65),
            (1, 5000),
            (5000, 1),
            (63, 129),
            (4096, 1),
        ] {
            let size = m * n;
            if size < 4096 {
                continue;
            }
            // scenarios: (name, c=(strides, offset), a=(strides, offset, buffer len), b=same)
            let scenarios: Vec<(&str, ([isize; 2], usize), ([isize; 2], usize, usize), ([isize; 2], usize, usize))> = vec![
                ("c_row_a_row_b_col", ([n as isize, 1], 0), ([n as isize, 1], 0, size), ([1, m as isize], 0, size)),
                ("c_col_a_row_b_col", ([1, m as isize], 0), ([n as isize, 1], 0, size), ([1, m as isize], 0, size)),
                ("neg_a_flip0", ([n as isize, 1], 0), ([-(n as isize), 1], (m - 1) * n, size), ([1, m as isize], 0, size)),
                ("neg_c_flip1", ([n as isize, -1], n - 1), ([n as isize, 1], 0, size), ([1, m as isize], 0, size)),
                ("neg_b_flip0_col", ([n as isize, 1], 0), ([n as isize, 1], 0, size), ([-1, m as isize], m - 1, size)),
                (
                    "neg_all",
                    ([n as isize, -1], n - 1),
                    ([-(n as isize), 1], (m - 1) * n, size),
                    ([-1, -(m as isize)], (m - 1) + (n - 1) * m, size),
                ),
                ("sliced_a_and_b", ([n as isize, 1], 0), ([2 * n as isize, 2], 0, 2 * size), ([2, 2 * m as isize], 0, 2 * size)),
                ("sliced_c", ([2 * n as isize, 2], 0), ([n as isize, 1], 0, size), ([1, m as isize], 0, size)),
                ("offset_views", ([n as isize, 1], n), ([n as isize, 1], 2 * n, 3 * size), ([1, m as isize], m, 2 * size)),
            ];
            for (name, (sc, oc), (sa, oa, len_a), (sb, ob, len_b)) in scenarios {
                let buf_a: Vec<f64> = (0..len_a).map(|i| i as f64 * 0.25 - 100.0).collect();
                let buf_b: Vec<f64> = (0..len_b).map(|i| -(i as f64) * 0.5 + 7.0).collect();
                let len_c = match name {
                    "sliced_c" => 2 * size,
                    "offset_views" => 2 * size,
                    _ => size,
                };
                let mut buf_c: Vec<MaybeUninit<f64>> = (0..len_c).map(|_| MaybeUninit::new(1e300)).collect();
                let lc = layout2([m, n], sc, oc);
                let la = layout2([m, n], sa, oa);
                let lb = layout2([m, n], sb, ob);
                op_mutc_refa_refb_func_cpu_serial(&mut buf_c, &lc, &buf_a, &la, &buf_b, &lb, |c, a, b| {
                    c.write(a + b);
                })
                .unwrap();
                // reference: layout-iterator walk over identical layouts
                let lc2 = layout2([m, n], sc, oc);
                let la2 = layout2([m, n], sa, oa);
                let lb2 = layout2([m, n], sb, ob);
                let it = IterLayoutColMajor::new(&lc2).unwrap();
                let ita = IterLayoutColMajor::new(&la2).unwrap();
                let itb = IterLayoutColMajor::new(&lb2).unwrap();
                for ((ic, ia), ib) in it.zip(ita).zip(itb) {
                    let expect = buf_a[ia] + buf_b[ib];
                    let got = unsafe { buf_c[ic].assume_init() };
                    assert!((got - expect).abs() < 1e-9, "case {} shape ({},{}) idx {}: got {} expect {}", name, m, n, ic, got, expect);
                }
                // no unwritten sentinel remains — dense-c scenarios only;
                // strided-c scenarios legitimately leave interleaved slots alone
                if name != "sliced_c" && name != "offset_views" {
                    for (i, v) in buf_c.iter().enumerate() {
                        let x = unsafe { v.assume_init() };
                        assert!(x.abs() < 1e6, "case {} shape ({},{}): sentinel/unwritten at {}", name, m, n, i);
                    }
                }
            }
        }
    }

    // 2-layout (in-place) kernel: same matrix shape, mutc→muta.
    #[test]
    fn test_blocked_2layout_inplace_matrix() {
        for &(m, n) in &[(70usize, 130usize), (130, 70), (4096, 2), (64, 64), (65, 65), (1, 5000), (5000, 1)] {
            let size = m * n;
            if size < 4096 {
                continue;
            }
            let scenarios: Vec<(&str, ([isize; 2], usize, usize), ([isize; 2], usize, usize))> = vec![
                ("a_row_b_col", ([n as isize, 1], 0, size), ([1, m as isize], 0, size)),
                ("a_col_b_row", ([1, m as isize], 0, size), ([n as isize, 1], 0, size)),
                ("neg_a_flip0", ([-(n as isize), 1], (m - 1) * n, size), ([1, m as isize], 0, size)),
                ("neg_b_flip1", ([n as isize, 1], 0, size), ([n as isize, -1], n - 1, size)),
                ("sliced", ([2 * n as isize, 2], 0, 2 * size), ([2, 2 * m as isize], 0, 2 * size)),
            ];
            for (name, (sa, oa, len_a), (sb, ob, len_b)) in scenarios {
                let mut buf_a: Vec<MaybeUninit<f64>> = (0..len_a).map(|i| MaybeUninit::new(i as f64 * 0.25 - 100.0)).collect();
                let buf_a0: Vec<f64> = (0..len_a).map(|i| i as f64 * 0.25 - 100.0).collect();
                let buf_b: Vec<f64> = (0..len_b).map(|i| -(i as f64) * 0.5 + 7.0).collect();
                let la = layout2([m, n], sa, oa);
                let lb = layout2([m, n], sb, ob);
                op_muta_refb_func_cpu_serial(&mut buf_a, &la, &buf_b, &lb, |a, b| {
                    let x = unsafe { a.assume_init() } + b;
                    a.write(x);
                })
                .unwrap();
                let ita = IterLayoutColMajor::new(&la).unwrap();
                let itb = IterLayoutColMajor::new(&lb).unwrap();
                for (ia, ib) in ita.zip(itb) {
                    let expect = buf_a0[ia] + buf_b[ib];
                    let got = unsafe { buf_a[ia].assume_init() };
                    assert!((got - expect).abs() < 1e-9, "case {} shape ({},{}) idx {}: got {} expect {}", name, m, n, ia, got, expect);
                }
            }
        }
    }
}
