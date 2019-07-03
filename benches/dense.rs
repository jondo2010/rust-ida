use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use ida::linear::*;
use ndarray::{s, Array};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

use core::ffi::c_void;
use std::slice;

use sundials_sys::*;

const NEQ: usize = 3;
const RTOL: f64 = 1.0e-4;
const ATOL: [f64; 3] = [1.0e-8, 1.0e-6, 1.0e-6];
const YY0: [f64; 3] = [1.0, 0.0, 0.0];
const YP0: [f64; 3] = [-0.04, 0.04, 0.0];

unsafe extern "C" fn resrob(
    _tres: f64,
    yy: N_Vector,
    yp: N_Vector,
    rr: N_Vector,
    _user_data: *mut c_void,
) -> i32 {
    let yval = slice::from_raw_parts(N_VGetArrayPointer(yy), NEQ);
    let ypval = slice::from_raw_parts(N_VGetArrayPointer(yp), NEQ);
    let mut rval = slice::from_raw_parts_mut(N_VGetArrayPointer(rr), NEQ);

    rval[0] = -0.04 * yval[0] + 1.0e4 * yval[1] * yval[2];
    rval[1] = -rval[0] - 3.0e7 * yval[1] * yval[1] - ypval[1];
    rval[0] -= ypval[0];
    rval[2] = yval[0] + yval[1] + yval[2] - 1.0;

    return 0;
}

unsafe extern "C" fn grob(
    t: f64,
    yy: N_Vector,
    yp: N_Vector,
    gout: *mut f64,
    _user_data: *mut c_void,
) -> i32 {
    let yval = slice::from_raw_parts(N_VGetArrayPointer(yy), NEQ);
    let y1 = yval[0];
    let y3 = yval[2];
    let mut gout = slice::from_raw_parts_mut(gout, 2);
    gout[0] = y1 - 0.0001;
    gout[1] = y3 - 0.01;
    return 0;
}

unsafe extern "C" fn jacrob(
    tt: f64,
    cj: f64,
    yy: N_Vector,
    yp: N_Vector,
    resvec: N_Vector,
    JJ: SUNMatrix,
    _user_data: *mut c_void,
    _tempv1: N_Vector,
    _tempv2: N_Vector,
    _tempv3: N_Vector,
) -> i32 {
    let yval = slice::from_raw_parts(N_VGetArrayPointer(yy), NEQ);
    let raw_ptr = (*((*JJ).content as SUNMatrixContent_Dense)).cols;
    let mut raw_vec: Vec<_> = slice::from_raw_parts_mut(*raw_ptr, NEQ * NEQ)
        .chunks_mut(NEQ)
        .collect();
    let J: &mut [&mut [_]] = raw_vec.as_mut_slice();

    J[0][0] = -0.04 - cj;
    J[0][1] = 0.04;
    J[0][2] = 1.0;
    J[1][0] = 1.0e4 * yval[2];
    J[1][1] = -1.0e4 * yval[2] - 6.0e7 * yval[1] - cj;
    J[1][2] = 1.0;
    J[2][0] = 1.0e4 * yval[1];
    J[2][1] = -1.0e4 * yval[1];
    J[2][2] = 1.0;

    return 0;
}

struct SundialsIdaTest {
    ida_mem: *mut std::ffi::c_void,
    yy: N_Vector,
    yp: N_Vector,
    avtol: N_Vector,
    A: *mut sundials_sys::_generic_SUNMatrix,
    LS: *mut sundials_sys::_generic_SUNLinearSolver,
    NLS: *mut sundials_sys::_generic_SUNNonlinearSolver,
}

impl SundialsIdaTest {
    pub fn new(t0: f64) -> Self {
        unsafe {
            /* Allocate N-vectors. */
            let yy = N_VNew_Serial(NEQ as sunindextype);
            //if(check_retval((void *)yy, "N_VNew_Serial", 0)) return(1);
            let yp = N_VNew_Serial(NEQ as sunindextype);
            //if(check_retval((void *)yp, "N_VNew_Serial", 0)) return(1);
            let avtol = N_VNew_Serial(NEQ as sunindextype);
            //if(check_retval((void *)avtol, "N_VNew_Serial", 0)) return(1);

            /* Create and initialize  y, y', and absolute tolerance vectors. */
            std::slice::from_raw_parts_mut(N_VGetArrayPointer(yy), NEQ).copy_from_slice(&YY0);
            std::slice::from_raw_parts_mut(N_VGetArrayPointer(yp), NEQ).copy_from_slice(&YP0);
            std::slice::from_raw_parts_mut(N_VGetArrayPointer(avtol), NEQ).copy_from_slice(&ATOL);

            let mut ida_mem = IDACreate();

            let retval = IDAInit(ida_mem, Some(resrob), t0, yy, yp);
            assert_eq!(retval, IDA_SUCCESS);
            let retval = IDASVtolerances(ida_mem, RTOL, avtol);
            assert_eq!(retval, IDA_SUCCESS);

            /* Call IDARootInit to specify the root function grob with 2 components */
            let retval = IDARootInit(ida_mem, 2, Some(grob));
            assert_eq!(retval, IDA_SUCCESS);

            let A = SUNDenseMatrix(NEQ as sunindextype, NEQ as sunindextype);
            let LS = SUNLinSol_Dense(yy, A);
            let retval = IDASetLinearSolver(ida_mem, LS, A);
            assert_eq!(retval, IDA_SUCCESS);
            let retval = IDASetJacFn(ida_mem, Some(jacrob));
            assert_eq!(retval, IDA_SUCCESS);
            let NLS = SUNNonlinSol_Newton(yy);
            let retval = IDASetNonlinearSolver(ida_mem, NLS);
            assert_eq!(retval, IDA_SUCCESS);

            Self {
                ida_mem,
                yy,
                yp,
                avtol,
                A,
                LS,
                NLS,
            }
        }
    }
}

impl Drop for SundialsIdaTest {
    fn drop(&mut self) {
        unsafe {
            /* Free memory */
            IDAFree(&mut self.ida_mem);
            SUNNonlinSolFree(self.NLS);
            SUNLinSolFree(self.LS);
            SUNMatDestroy(self.A);
            N_VDestroy(self.avtol);
            N_VDestroy(self.yy);
            N_VDestroy(self.yp);
        }
    }
}

fn bench_dense_sundials(bencher: &mut criterion::Bencher, nout: usize) {
    /* Integration limits */
    let t0 = 0.0;
    let mut iout = 0;
    let mut tout = 0.4;
    let mut tret: f64 = 0.0;
    unsafe {
        bencher.iter_with_setup(
            || SundialsIdaTest::new(t0),
            |ida| {
                loop {
                    let retval = IDASolve(ida.ida_mem, tout, &mut tret, ida.yy, ida.yp, IDA_NORMAL);
                    //assert_eq!(retval, IDA_SUCCESS);

                    if retval == IDA_ROOT_RETURN {
                        //retvalr = IDAGetRootInfo(mem, rootsfound);
                        //check_retval(&retvalr, "IDAGetRootInfo", 1);
                        //PrintRootInfo(rootsfound[0],rootsfound[1]);
                    }

                    if retval == IDA_SUCCESS {
                        iout += 1;
                        tout *= 10.0;
                    }

                    if iout == nout {
                        break;
                    }
                }
            },
        );

        //let mut nje = 0;
        //IDAGetNumJacEvals(ida.ida_mem, &mut nje);
        //println!("tret={}, nje={}", tret, nje);
    }
}

fn bench_dense_ida_rs(bencher: &mut criterion::Bencher, nout: usize) {
    let t0 = 0.0;
    let mut iout = 0;
    let mut tout = 0.4;
    let mut tret: f64 = 0.0;

    bencher.iter_with_setup(
        || {
            let problem = ida::sample_problems::Roberts {};
            let yy0 = ndarray::Array::from_iter(YY0.iter().cloned());
            let yp0 = ndarray::Array::from_iter(YP0.iter().cloned());
            let ec = ida::tol_control::TolControlSV::new(
                RTOL,
                ndarray::Array1::from_iter(ATOL.iter().cloned()),
            );
            let mut ida: ida::Ida<_, Dense<_>, ida::nonlinear::Newton<_>, _> =
                ida::Ida::new(problem, yy0, yp0, ec);
            ida
        },
        |mut ida| {
            loop {
                let retval = ida.solve(tout, &mut tret, ida::IdaTask::Normal).unwrap();
                //assert_eq!(retval.unwrap(), ida::IdaSolveStatus::Success);

                if let ida::IdaSolveStatus::Root = retval {}

                if let ida::IdaSolveStatus::Success = retval {
                    iout += 1;
                    tout *= 10.0;
                }

                if iout == nout {
                    break;
                }
            }
        },
    );

    /*
    // Fill A matrix with uniform random data in [0,1/cols]
    // Add anti-identity to ensure the solver needs to do row-swapping
    let mut mat_a = Array::random((cols, cols), Uniform::new(0., 1.0 / (cols as f64)))
        + Array::eye(cols).slice_move(s![.., ..;-1]);

    // Fill x vector with uniform random data in [0,1]
    let mut x = Array::random(cols, Uniform::new(0.0, 1.0));
    let b = x.clone();

    let mut dense = Dense::new(cols);

    bencher.iter(|| dense.setup(mat_a.clone()).unwrap());
    dense.setup(mat_a.view_mut()).unwrap();
    bencher.iter(|| {
        dense
            .solve(mat_a.view(), x.view_mut(), b.view(), 0.0)
            .unwrap()
    });
    */

    /*
    let b_comp = mat_a_original.dot(&x);
    let norm = (b - b_comp)
        .iter()
        .map(|x| x.powi(2))
        .fold(0.0, |acc, x| acc + x)
        / (cols as f64).sqrt();

    assert_nearly_eq!(norm, 0.0);
    */

    //assert_nearly_eq!(&b_comp, &b, 1e-14);
    //println!("b (original) = {:#?}", &b);
    //println!("b (computed) = {:#?}", &b_comp);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench(
        "Roberts",
        ParameterizedBenchmark::new(
            "sundials-sys",
            |b, &i| bench_dense_sundials(b, black_box(i)),
            vec![1, 2, 4, 8, 12],
        )
        .with_function("ida-rs", |b, &i| bench_dense_ida_rs(b, black_box(i))),
    );

    //c.bench_function("Dense solver 5", |b| bench_dense(b, black_box(5)));
    //c.bench_function("Dense solver 10", |b| bench_dense(b, black_box(10)));
    //c.bench_function("Dense solver 50", |b| bench_dense(b, black_box(50)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
