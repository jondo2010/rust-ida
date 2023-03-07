//! Roberts example from the SUNDIALS documentation implemented with `sundials-sys`.

use std::ffi::{c_int, c_void};

use super::adapter::{dense_matrix_view, vector_view};
use nalgebra::*;

unsafe extern "C" fn res(
    _t: f64,
    _yy: sundials_sys::N_Vector,
    _yp: sundials_sys::N_Vector,
    _resval: sundials_sys::N_Vector,
    _user_data: *mut c_void,
) -> c_int {
    0
}

unsafe extern "C" fn g(
    _t: f64,
    yy: sundials_sys::N_Vector,
    _yp: sundials_sys::N_Vector,
    gout: *mut f64,
    _user_data: *mut c_void,
) -> c_int {
    let yval = std::slice::from_raw_parts(
        sundials_sys::N_VGetArrayPointer(yy),
        sundials_sys::N_VGetLength(yy) as usize,
    );
    let gout = std::slice::from_raw_parts_mut(gout, 3);
    gout[0] = yval[0] - 0.0001;
    gout[1] = yval[2] - 0.01;
    0
}

unsafe extern "C" fn jac(
    _tt: f64,
    cj: f64,
    yy: sundials_sys::N_Vector,
    _yp: sundials_sys::N_Vector,
    _respect: sundials_sys::N_Vector,
    JJ: sundials_sys::SUNMatrix,
    _user_data: *mut c_void,
    _tempv1: sundials_sys::N_Vector,
    _tempv2: sundials_sys::N_Vector,
    _tempv3: sundials_sys::N_Vector,
) -> c_int {
    let yval = vector_view::<U3>(yy);
    let mut jj = dense_matrix_view::<U3, U3>(JJ);
    jj.copy_from(&matrix![
        -0.04 - cj, 0.04, 1.0;
        1.0e4 * yval[2], -1.0e4 * yval[2] - 6.0e7 * yval[1] - cj, 1.0;
        1.0e4 * yval[1], -1.0e4 * yval[1], 1.0;
    ]);
    0
}

/// Build an IDA solver for the Roberts problem.
///
/// Returns the initial state vector, the initial derivative vector, and the IDA solver.
pub unsafe fn build_ida() -> (
    sundials_sys::N_Vector,
    sundials_sys::N_Vector,
    sundials_sys::IDAMem,
) {
    let yy = sundials_sys::N_VNew_Serial(3);
    let mut yval = vector_view::<U3>(yy);
    yval.copy_from(&(vector![1.0, 0.0, 0.0]));

    let yp = sundials_sys::N_VNew_Serial(3);
    let mut ypval = vector_view::<U3>(yp);
    ypval.copy_from(&(vector![-0.04, 0.04, 0.0]));

    let rtol = 1.0e-4;

    let avtol = sundials_sys::N_VNew_Serial(3);
    let mut atval = vector_view::<U3>(avtol);
    atval.copy_from(&vector![1.0e-8, 1.0e-6, 1.0e-6]);

    let t0 = 0.0;

    let mem = sundials_sys::IDACreate();
    assert_eq!(sundials_sys::IDAInit(mem, Some(res), t0, yy, yp), 0);

    // Set tolerances
    assert_eq!(sundials_sys::IDASVtolerances(mem, rtol, avtol), 0);

    // Call IDARootInit to specify the root function g with 2 components
    assert_eq!(sundials_sys::IDARootInit(mem, 2, Some(g)), 0);

    // Create dense SUNMatrix for use in linear solves
    let A = sundials_sys::SUNDenseMatrix(3, 3);

    // Create dense SUNLinearSolver object
    let LS = sundials_sys::SUNDenseLinearSolver(yy, A);

    // Attach the matrix and linear solver
    assert_eq!(sundials_sys::IDASetLinearSolver(mem, LS, A), 0);

    // Set the user-supplied Jacobian routine
    assert_eq!(sundials_sys::IDASetJacFn(mem, Some(jac)), 0);

    // Create Newton SUNNonlinearSolver object. IDA uses a Newton SUNNonlinearSolver by default, so it is unecessary to create it and attach it. It is done in this example code solely for demonstration purposes.
    let NLS = sundials_sys::SUNNonlinSol_Newton(yy);

    // Attach the nonlinear solver
    assert_eq!(sundials_sys::IDASetNonlinearSolver(mem, NLS), 0);

    let mem = mem as sundials_sys::IDAMem;

    (yy, yp, mem)
}
