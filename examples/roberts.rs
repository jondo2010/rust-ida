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

use ida::{linear::*, nonlinear::*, tol_control::*, *};

use ndarray::{array, prelude::*};

/// compare the solution at the final time 4e10s to a reference solution computed using a relative
/// tolerance of 1e-8 and absoltue tolerance of 1e-14
fn check_ans<S1, S2>(y: ArrayBase<S1, Ix1>, _t: f64, rtol: f64, atol: ArrayBase<S2, Ix1>)
where
    S1: ndarray::Data<Elem = f64>,
    S2: ndarray::Data<Elem = f64>,
{
    //int      passfail=0;        /* answer pass (0) or fail (1) retval */
    //N_Vector ref;               /* reference solution vector        */
    //N_Vector ewt;               /* error weight vector              */
    //realtype err;               /* wrms error                       */
    // create reference solution and error weight vectors

    // set the reference solution data
    let reference = array![
        5.2083474251394888e-08,
        2.0833390772616859e-13,
        9.9999994791631752e-01
    ];

    // compute the error weight vector, loosen atol
    // ewt = rtol*ewt + 10.0*atol
    let mut ewt = rtol * &reference.mapv(f64::abs);
    ewt.scaled_add(10.0, &atol);

    dbg!(&ewt);

    //if (N_VMin(ewt) <= ZERO) {
    //  fprintf(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
    //  return(-1);
    //}
    //N_VInv(ewt, ewt);
    let ewt = ewt.mapv(f64::recip);

    // compute the solution error
    let diff = &y - &reference;
    let err = diff.norm_wrms(&ewt);

    // is the solution within the tolerances?
    let passfail = dbg!(err >= 1.0);

    if passfail {
        println!("SUNDIALS_WARNING: check_ans error={} \n\n", err);
    }
}

use prettytable::{cell, row, table, Table};

fn main() {
    pretty_env_logger::init();
    profiler::register_thread_with_profiler();

    const RTOL: f64 = 1.0e-4;
    const ATOL: [f64; 3] = [1.0e-8, 1.0e-6, 1.0e-6];

    let problem = ida::example_problems::Roberts {};

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
    let mut tret = 0.0;
    let retval = loop {
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
            Ok(IdaSolveStatus::Root) => {
                println!("Roots found: at t={:.6e}: {}", tret, ida.get_root_info());
            }
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
        ],
        [
            "Number of nonlinear conv. failures:",
            ida.get_num_nonlin_solv_conv_fails(),
        ],
        ["Number of root fn. evaluations:", ida.get_num_g_evals(),]
    );

    stats.set_format(*prettytable::format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
    stats.set_titles(row![bFgH2->"Final Run Statistics:"]);
    stats.printstd();

    check_ans(
        ida.get_yy(),
        tret,
        RTOL,
        ndarray::Array1::from_iter(ATOL.iter().cloned()),
    );

    profiler::write_profile("profile.json");
}
