use ida::{Ida, IdaSolveStatus, IdaTask, TolControl};
use nalgebra::{vector, Storage, Vector, Vector3, U3};
use nonlinear::norm_wrms::NormWRMS;
use prettytable::{row, table, Table};

/// compare the solution at the final time 4e10s to a reference solution computed using a relative
/// tolerance of 1e-8 and absoltue tolerance of 1e-14
fn check_ans<SA, SB>(y: &Vector<f64, U3, SA>, _t: f64, rtol: f64, atol: &Vector<f64, U3, SB>)
where
    SA: Storage<f64, U3>,
    SB: Storage<f64, U3>,
{
    //int      passfail=0;        /* answer pass (0) or fail (1) retval */
    //N_Vector ref;               /* reference solution vector        */
    //N_Vector ewt;               /* error weight vector              */
    //realtype err;               /* wrms error                       */
    // create reference solution and error weight vectors

    // set the reference solution data
    let reference = vector![
        5.2083474251394888e-08,
        2.0833390772616859e-13,
        9.9999994791631752e-01
    ];

    // compute the error weight vector, loosen atol; ewt = rtol*ewt + 10.0*atol
    let mut ewt = rtol * &reference;
    ewt.axpy(10.0, atol, 1.0);

    dbg!(&ewt);

    //if (N_VMin(ewt) <= ZERO) {
    //  fprintf(stderr, "\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
    //  return(-1);
    //}
    //N_VInv(ewt, ewt);
    ewt.iter_mut().for_each(|x| *x = 1.0 / *x);

    // compute the solution error
    let diff = y - &reference;
    let err = diff.norm_wrms(&ewt);

    // is the solution within the tolerances?
    let fail = dbg!(err) >= 1.0;

    if fail {
        println!("SUNDIALS_WARNING: check_ans error={} \n\n", err);
    }
}

fn main() {
    tracing_subscriber::fmt::init();

    #[cfg(feature = "profiler")]
    profiler::register_thread_with_profiler();

    const RTOL: f64 = 1.0e-4;
    const ATOL: [f64; 3] = [1.0e-8, 1.0e-6, 1.0e-6];

    let problem = sample_problems::Roberts {};

    let yy0 = vector![1.0, 0.0, 0.0];
    let yp0 = vector![-0.04, 0.04, 0.0];

    let atol = Vector3::from(ATOL);
    let ec = TolControl::new_sv(RTOL, atol);
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

    let ls = ida::linear::Dense::new();
    let nls = ida::nonlinear::newton::Newton::new(4);
    let mut ida = Ida::new(problem, ls, nls, &yy0, &yp0, ec);

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

    if let Err(e) = retval {
        println!("Error: {e}");
    }

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

    check_ans(ida.get_yy(), tret, RTOL, &Vector3::from(ATOL));

    #[cfg(feature = "profiler")]
    profiler::write_profile("profile.json");
}
