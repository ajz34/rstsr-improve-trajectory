use rstsr::prelude::*;

macro_rules! run_case {
    ($name:expr, $b:expr, $naux:expr, $dev:expr) => {{
        let b = $b;
        {
            let v = b.view();
            let b2 = v.to_dim::<Ix2>();
            println!("[{}] shape={:?} stride={:?} ld_col={:?} ld_row={:?}",
                $name, b2.shape(), b2.stride(), b2.ld_col(), b2.ld_row());
        }
        let _ = rt::linalg::solve_triangular((rt::eye(($naux, $dev)).t(), b, Lower));
        println!("[{}] returned", $name);
    }};
}

fn main() {
    let case: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    let dev = DeviceBLAS::default();
    let naux = 18usize;
    let mut buf: Vec<f64> = Vec::new();

    match case {
        1 => run_case!("issue: static [0,18].f() reversed",
                rt::asarray((&mut buf[..], [0, naux].f(), &dev)).into_reverse_axes(), naux, &dev),
        2 => run_case!("static [18,0].f() direct",
                rt::asarray((&mut buf[..], [18, 0].f(), &dev)), naux, &dev),
        3 => {
            let b: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((vec![18, 0], RowMajor, &dev));
            run_case!("dynamic [18,0] RowMajor", b, naux, &dev)
        },
        4 => {
            let base: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((vec![5, 18], ColMajor, &dev));
            run_case!("f-order [5,18] rows 0..0, transposed",
                base.i((0..0, ..)).into_dim::<IxD>().into_reverse_axes(), naux, &dev)
        },
        5 => {
            let base: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((vec![18, 5], RowMajor, &dev));
            run_case!("row-major [18,5] cols 0..0",
                base.i((.., 0..0)).into_dim::<IxD>(), naux, &dev)
        },
        6 => {
            let base: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((vec![5, 18], ColMajor, &dev));
            let v = base.i((0..0, ..)).into_dim::<IxD>().into_reverse_axes();
            run_case!("owned copy of case 4", v.to_owned(), naux, &dev)
        },
        7 => {
            let base: Tensor<f64, DeviceBLAS, IxD> = rt::zeros((vec![18, 5], RowMajor, &dev));
            let v = base.i((.., 0..0)).into_dim::<IxD>();
            run_case!("owned copy of case 5", v.to_owned(), naux, &dev)
        },
        8 => {
            let mut buf: Vec<f64> = Vec::new();
            let b = rt::asarray((&mut buf[..], [0, naux].f(), &dev)).into_reverse_axes();
            let a = rt::eye((naux, &dev));
            let r = rt::linalg::solve_general_f((a, b));
            match r {
                Ok(x) => println!("solve_general ok: shape={:?} stride={:?}", x.shape(), x.stride()),
                Err(e) => println!("solve_general err: {e:?}"),
            }
        },
        _ => unreachable!(),
    }
}
