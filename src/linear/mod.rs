mod dense;

use ndarray::prelude::*;

use crate::traits::ModelSpec;

pub use dense::Dense;

pub trait LProblem<M>
where
    M: ModelSpec,
{
    /// idaLsSetup
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS `setup`
    /// routine to prepare for subsequent calls to the LS `solve` routine.
    fn setup<S1, S2, S3>(
        &mut self,
        y: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        r: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>;

    /// idaLsSolve
    /// This routine interfaces between IDA and the generic LSovler object LS, by setting the
    /// appropriate tolerance and scaling vectors, calling the solver, accumulating statistics
    /// from the solve for use/reporting by IDA, and scaling the result if using a non-NULL Matrix
    /// and cjratio does not equal one.
    fn solve<S1, S2>(
        &mut self,
        b: ArrayBase<S1, Ix1>,
        weight: ArrayBase<S2, Ix1>,
        ycur: ArrayBase<S1, Ix1>,
        ypcur: ArrayBase<S1, Ix1>,
        rescur: ArrayBase<S1, Ix1>,
    ) where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>;
}

#[derive(Debug)]
pub enum LSolverType {
    Direct,
    Iterative,
    MatrixIterative,
}

pub trait LSolver<Scalar>
where
    Scalar: num_traits::Zero
{
    fn new(size: usize) -> Self;

    fn get_type(&self) -> LSolverType;

    /// provides left/right scaling vectors for the linear system solve. Here, s1 and s2 are
    /// vectors of positive scale factors containing the diagonal of the matrices S1 and S2 from
    /// equations (8.1)-(8.2), respectively.
    ///
    /// Neither of these vectors need to be tested for positivity, and a NULL argument for either
    /// indicates that the corresponding scaling matrix is the identity.
    ///
    /// ## Arguments
    /// * `s1` diagonal of the matrix S1
    /// * `s2` diagonal of the matrix S2
    fn set_scaling_vectors<S1, S2>(&mut self, _s1: ArrayBase<S1, Ix1>, _s2: ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::Data<Elem = Scalar>,
    {
    }

    /// Performs any linear solver setup needed, based on an updated system sunmatrix A. This may
    /// be called frequently (e.g., with a full Newton method) or infrequently (for a modified
    /// Newton method), based on the type of integrator and/or nonlinear solver requesting the
    /// solves.
    fn setup<S1>(&mut self, mat_a: ArrayBase<S1, Ix2>) -> Result<(), failure::Error>
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
        mat_a: ArrayBase<S1, Ix2>,
        mut x: ArrayBase<S2, Ix1>,
        b: ArrayBase<S3, Ix1>,
        tol: Scalar,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>;

    /// should return the number of linear iterations performed in the last ‘solve’ call.
    fn num_iters(&self) -> usize {
        0
    }

    /// should return the final residual norm from the last ‘solve’ call.
    fn res_norm(&self) -> Scalar {
        Scalar::zero()
    }
}
