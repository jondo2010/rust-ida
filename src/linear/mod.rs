mod dense;

use ndarray::prelude::*;

use crate::traits::ModelSpec;

pub use dense::Dense;

pub trait LSolver_x<M: ModelSpec> {
    //IDA_mem->ida_linit  = idaLsInitialize;
    fn new() -> Self;

    /// idaLsSetup
    ///
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS 'setup' routine
    /// to prepare for subsequent calls to the LS 'solve' routine.
    fn ls_setup<S1, S2, S3>(
        &self,
        _y: &ArrayBase<S1, Ix1>,
        _yp: &ArrayBase<S2, Ix1>,
        _r: &ArrayBase<S3, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {
        Ok(())
    }

    /// idaLsSolve
    ///
    /// This routine interfaces between IDA and the generic
    /// SUNLinearSolver object LS, by setting the appropriate tolerance
    /// and scaling vectors, calling the solver, accumulating
    /// statistics from the solve for use/reporting by IDA, and scaling
    /// the result if using a non-NULL SUNMatrix and cjratio does not
    /// equal one.
    fn ls_solve<S1, S2, S3>(
        &self,
        _b: &mut ArrayBase<S1, Ix1>,
        _weight: &ArrayBase<S2, Ix1>,
        _ycur: &ArrayBase<S3, Ix1>,
        _ypcur: &ArrayBase<S3, Ix1>,
        _rescur: &ArrayBase<S3, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::DataMut<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {
        Ok(())
    }
}

pub trait LSolver<Scalar> {
    fn new(size: usize) -> Self;

    /// Performs any linear solver setup needed, based on an updated system sunmatrix A. This may
    /// be called frequently (e.g., with a full Newton method) or infrequently (for a modified
    /// Newton method), based on the type of integrator and/or nonlinear solver requesting the
    /// solves.
    fn setup<S1>(&mut self, mat_a: &mut ArrayBase<S1, Ix2>) -> Result<(), failure::Error>
    where
        S1: ndarray::DataMut<Elem = Scalar>;

    /// solves a linear system Ax = b.
    ///
    /// ## Arguments
    /// * `matA` the matrix A.
    /// * `x` the initial guess for the solution of the linear system, and the solution to the linear system upon return.
    /// * `b` the linear system right-hand side.
    /// * `tol` the desired linear solver tolerance.
    ///
    /// ## Notes
    /// Direct solvers: can ignore the tol argument.
    /// Matrix-free solvers: can ignore the input matA, and should instead rely on the
    ///     matrix-vector product function supplied through the routine SUNLinSolSetATimes.
    /// Iterative solvers: should attempt to solve to the specified tolerance tol in a weighted
    ///     2-norm. If the solver does not support scaling then it should just use a 2-norm.
    fn solve<S1, S2, S3>(
        &self,
        mat_a: &ArrayBase<S1, Ix2>,
        x: &mut ArrayBase<S2, Ix1>,
        b: &ArrayBase<S3, Ix1>,
        tol: Scalar,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>;

    /// provides left/right scaling vectors for the linear system solve. Here, s1 and s2 are
    /// vectors of positive scale factors containing the diagonal of the matrices S1 and S2 from
    /// equations (8.1)-(8.2), respectively. Neither of these vectors need to be tested for positivity, and a NULL argument for either indicates that the corresponding scaling matrix is the identity.
    /// 
    /// ## Arguments
    /// * `s1` diagonal of the matrix S1
    /// * `s2` diagonal of the matrix S2
    fn set_scaling_vectors<S1, S2>(&mut self, _s1: &ArrayBase<S1, Ix1>, _s2: &ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::Data<Elem = Scalar>
    {}
}
