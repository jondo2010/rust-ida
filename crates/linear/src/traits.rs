use nalgebra::{Dim, Matrix, Scalar, StorageMut, VectorView, U1};
use num_traits::Zero;

use crate::{Error, LSolverType};

pub trait LProblem<T: Scalar, D: Dim> {
    /// idaLsSetup
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS `setup`
    /// routine to prepare for subsequent calls to the LS `solve` routine.
    fn setup(&mut self, y: VectorView<T, D>, yp: VectorView<T, D>, r: VectorView<T, D>);

    /// idaLsSolve
    /// This routine interfaces between IDA and the generic LSovler object LS, by setting the
    /// appropriate tolerance and scaling vectors, calling the solver, accumulating statistics
    /// from the solve for use/reporting by IDA, and scaling the result if using a non-NULL Matrix
    /// and cjratio does not equal one.
    fn solve(
        &mut self,
        b: VectorView<T, D>,
        weight: VectorView<T, D>,
        ycur: VectorView<T, D>,
        ypcur: VectorView<T, D>,
        rescur: VectorView<T, D>,
    );
}

pub trait LSolver<T, D>
where
    T: Scalar + Zero,
    D: Dim,
{
    fn new() -> Self;

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
    fn set_scaling_vectors<S1, S2>(&mut self, _s1: VectorView<T, D>, _s2: VectorView<T, D>) {}

    /// Performs any linear solver setup needed, based on an updated system sunmatrix A. This may
    /// be called frequently (e.g., with a full Newton method) or infrequently (for a modified
    /// Newton method), based on the type of integrator and/or nonlinear solver requesting the
    /// solves.
    fn setup<S>(&mut self, mat_a: &mut Matrix<T, D, D, S>) -> Result<(), Error>
    where
        S: StorageMut<T, D, D>;

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
    fn solve<SA, SB, SC>(
        &self,
        mat_a: &Matrix<T, D, D, SA>,
        x: &mut Matrix<T, D, U1, SB>,
        b: &Matrix<T, D, U1, SC>,
        tol: T,
    ) -> Result<(), Error>
    where
        SA: StorageMut<T, D, D>,
        SB: StorageMut<T, D>,
        SC: StorageMut<T, D>;

    /// should return the number of linear iterations performed in the last ‘solve’ call.
    fn num_iters(&self) -> usize {
        0
    }

    /// should return the final residual norm from the last ‘solve’ call.
    fn res_norm(&self) -> T {
        T::zero()
    }
}
