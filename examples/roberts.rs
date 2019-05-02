//! This simple example problem for IDA, due to Robertson, is from chemical kinetics, and consists
//! of the following three equations:
//!
//!      dy1/dt = -.04*y1 + 1.e4*y2*y3
//!      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
//!         0   = y1 + y2 + y3 - 1
//!
//! on the interval from t = 0.0 to t = 4.e10, with initial conditions: y1 = 1, y2 = y3 = 0.
//!
//! While integrating the system, we also use the rootfinding feature to find the points at which
//! y1 = 1e-4 or at which y3 = 0.01.
//!
//! The problem is solved with IDA using the DENSE linear solver, with a user-supplied Jacobian.
//! Output is printed at t = .4, 4, 40, ..., 4e10.

use ida::{linear::*, nonlinear::*, tol_control::*, traits::*, *};

use ndarray::{array, prelude::*};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Roberts {}

impl ModelSpec for Roberts {
    type Scalar = f64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        3
    }
}

impl Residual for Roberts {
    fn res<S1, S2, S3>(
        &self,
        _tres: Self::Scalar,
        yy: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        mut resval: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {
        resval[0] = -0.04 * yy[0] + 1.0e4 * yy[1] * yy[2];
        resval[1] = -resval[0] - 3.0e7 * yy[1] * yy[1] - yp[1];
        resval[0] -= yp[0];
        resval[2] = yy[0] + yy[1] + yy[2] - 1.0;
    }
}

impl Jacobian for Roberts {
    fn jac<S1, S2, S3, S4>(
        &self,
        _tt: Self::Scalar,
        cj: Self::Scalar,
        yy: ArrayBase<S1, Ix1>,
        _yp: ArrayBase<S2, Ix1>,
        _rr: ArrayBase<S3, Ix1>,
        mut jac: ArrayBase<S4, Ix2>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::Data<Elem = Self::Scalar>,
        S4: ndarray::DataMut<Elem = Self::Scalar>,
    {
        jac[[0, 0]] = -0.04 - cj;
        jac[[0, 1]] = 1.0e4 * yy[2];
        jac[[0, 2]] = 1.0e4 * yy[1];

        jac[[1, 0]] = 0.04;
        jac[[1, 1]] = -1.0e4 * yy[2] - 6.0e7 * yy[1] - cj;
        jac[[1, 2]] = -1.0e4 * yy[1];

        jac[[2, 0]] = 1.0;
        jac[[2, 1]] = 1.0;
        jac[[2, 2]] = 1.0;
    }
}

use prettytable::{cell, row, table, Table};

fn main() {
    pretty_env_logger::init();
    profiler::register_thread_with_profiler();

    const RTOL: f64 = 1.0e-4;
    const ATOL: [f64; 3] = [1.0e-8, 1.0e-6, 1.0e-6];

    let problem = Roberts {};

    let yy0 = array![1.0, 0.0, 0.0];
    let yp0 = array![-0.04, 0.04, 0.0];

    let ec = TolControlSV::new(RTOL, ndarray::Array1::from_iter(ATOL.iter().cloned()));
    //let t0 = 0.0;

    let header = &[
        "idaRoberts_dns: Robertson kinetics DAE serial example problem for IDA Three equation chemical kinetics problem.",
        "Linear solver: DENSE, with user-supplied Jacobian.",
        &format!("Tolerance parameters: rtol = {:e} atol = [{:e}, {:e}, {:e}]", RTOL, ATOL[0], ATOL[1], ATOL[2]),
        &format!("Initial conditions y0 = [{:e} {:e} {:e}]", yy0[0], yy0[1], yy0[2]),
        "Constraints and id not used.",
    ].join("\n");

    let th = table!([header]);
    th.printstd();

    let mut table_out = Table::new();
    table_out.set_titles(row!["t", "y1", "y2", "y3", "nst", "k", "h"]);
    table_out.set_format(*prettytable::format::consts::FORMAT_NO_LINESEP_WITH_TITLE);

    let mut ida: Ida<_, Dense<_>, Newton<_>, _> = Ida::new(problem, yy0, yp0, ec);

    // In loop, call IDASolve, print results, and test for error.
    // Break out of loop when NOUT preset output times have been reached.

    let mut iout = 0;
    let mut tout = 0.4;
    let retval = loop {
        let mut tret = 0.0;

        let retval = ida.solve(tout, &mut tret, IdaTask::Normal);

        let nst = ida.get_num_steps();
        let kused = ida.get_last_order();
        let hused = ida.get_last_step();

        let yy = ida.get_yy();

        //println!("\"yy\":{:.6e}, \"k\":{}, \"hh\":{:.6e}", yy, kused, hused);

        table_out.add_row(row![
            format!("{:.5e}", tret),
            format!("{:.5e}", yy[0]),
            format!("{:.5e}", yy[1]),
            format!("{:.5e}", yy[2]),
            nst,
            kused,
            format!("{:.5e}", hused),
        ]);

        match retval {
            Err(_) => {
                break retval.map(|_| ());
            }
            Ok(IdaSolveStatus::Root) => {}
            Ok(IdaSolveStatus::Success) => {
                iout += 1;
                tout *= 10.0;
            }
            _ => {}
        }

        if iout == 12 {
            break Ok(());
        }
    };

    table_out.printstd();
    dbg!(retval.unwrap());

    let mut stats = table!(
        ["Number of steps:", ida.get_num_steps()],
        [
            "Number of residual evaluations:",
            ida.get_num_res_evals() + ida.get_num_lin_res_evals(),
        ],
        ["Number of Jacobian evaluations:", ida.get_num_jac_evals()],
        [
            "Number of nonlinear iterations:",
            ida.get_num_nonlin_solv_iters(),
        ],
        [
            "Number of error test failures:",
            ida.get_num_err_test_fails(),
        ]
    );

    stats.set_format(*prettytable::format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    stats.set_titles(row![bFgH2->"Final Run Statistics:"]);
    //"Number of nonlinear conv. failures:", ncfn,
    //"Number of root fn. evaluations:", nge,

    stats.printstd();

    profiler::write_profile("profile.json");
}
