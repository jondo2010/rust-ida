use nalgebra::{Dim, Matrix, Scalar, Storage, StorageMut, U1};

use crate::Error;

pub trait NLProblem<T, D>
where
    T: Scalar,
    D: Dim,
{
    /// `sys` evaluates the nonlinear system `F(y)` for ROOTFIND type modules or `G(y)` for
    /// FIXEDPOINT type modules.
    ///
    /// # Arguments
    ///
    /// * `y` is the state vector at which the nonlinear system should be evaluated.
    /// * `f` is the output vector containing `F(y)` or `G(y)`, depending on the solver type.
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` indicates whether the routine has updated the Jacobian A (`true`) or not (`false`).
    /// * `Err()` for a recoverable error,
    /// * `Err(_) for an unrecoverable error
    fn sys<SB1, SB2>(
        &mut self,
        y: &Matrix<T, D, U1, SB1>,
        f: &mut Matrix<T, D, U1, SB2>,
    ) -> Result<(), Error>
    where
        SB1: Storage<T, D, U1>,
        SB2: StorageMut<T, D, U1>;

    /// `lsetup` is called by integrators to provide the nonlinear solver with access to its linear
    /// solver setup function.
    ///
    /// # Arguments
    ///
    /// * `y` is the state vector at which the linear system should be setup.
    /// * `f` is the value of the nonlinear system function at y.
    /// * `jbad` is an input indicating whether the nonlinear solver believes that A has gone stale
    ///     (`true`) or not (`false`).
    ///
    /// # Returns
    ///
    /// * `Ok(bool)` indicates whether the routine has updated the Jacobian A (`true`) or not
    ///     (`false`).
    /// * `Err()` for a recoverable error,
    /// * `Err(_) for an unrecoverable error
    ///
    /// The `lsetup` function sets up the linear system `Ax = b` where `A = ∂F/∂y` is the
    /// linearization of the nonlinear residual function `F(y) = 0` (when using direct linear
    /// solvers) or calls the user-defined preconditioner setup function (when using iterative
    /// linear solvers). `lsetup` implementations that do not require solving this system, do not
    /// utilize linear solvers, or use linear solvers that do not require setup may ignore these
    /// functions.
    fn setup<SA, SB>(
        &mut self,
        y: &Matrix<T, D, U1, SA>,
        f: &Matrix<T, D, U1, SB>,
        jbad: bool,
    ) -> Result<bool, Error>
    where
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>;

    /// `lsolve` is called by integrators to provide the nonlinear solver with access to its linear
    /// solver solve function.
    ///
    /// # Arguments
    /// * `y` is the input vector containing the current nonlinear iteration.
    /// * `b` contains the right-hand side vector for the linear solve on input and the solution to
    ///     the linear system on output.
    ///
    /// # Returns
    ///
    /// Return value The return value retval (of type int) is zero for a successul solve, a positive value for a recoverable error, and a negative value for an unrecoverable error.
    ///
    /// # Notes
    ///
    /// The `lsove` function solves the linear system `Ax = b` where `A = ∂F/∂y` is the linearization
    /// of the nonlinear residual function F(y) = 0. Implementations that do not require solving
    /// this system or do not use sunlinsol linear solvers may ignore these functions.
    fn solve<SA, SB>(
        &mut self,
        y: &Matrix<T, D, U1, SA>,
        b: &mut Matrix<T, D, U1, SB>,
    ) -> Result<(), Error>
    where
        SA: Storage<T, D, U1>,
        SB: StorageMut<T, D, U1>;

    /// `ctest` is an integrator-specific convergence test for nonlinear solvers and are typically
    /// supplied by each integrator, but users may supply custom problem-specific versions as desired.
    ///
    /// # Arguments
    ///
    /// * `nls` is the nonlinear solver
    /// * `y` is the current nonlinear iterate.
    /// * `del` is the difference between the current and prior nonlinear iterates.
    /// * `tol` is the nonlinear solver tolerance.
    /// * `ewt` is the weight vector used in computing weighted norms.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - the iteration is converged.
    /// * `Ok(false)` - the iteration has not converged, keep iterating.
    /// * `Err(Error::ConvergenceRecover)` - the iteration appears to be diverging, try to recover.
    /// * `Err(_)` - an unrecoverable error occurred.
    ///
    /// # Notes
    ///
    /// The tolerance passed to this routine by integrators is the tolerance in a weighted
    /// root-mean-squared norm with error weight vector `ewt`. Modules utilizing their own
    /// convergence criteria may ignore these functions.
    fn ctest<NLS, SA, SB, SC>(
        &mut self,
        solver: &NLS,
        y: &Matrix<T, D, U1, SA>,
        del: &Matrix<T, D, U1, SB>,
        tol: T,
        ewt: &Matrix<T, D, U1, SC>,
    ) -> Result<bool, Error>
    where
        NLS: NLSolver<T, D>,
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
        SC: Storage<T, D, U1>;
}

pub trait NLSolver<T: Scalar, D: Dim> {
    /// Create a new NLSolver
    ///
    /// # Arguments
    ///
    /// * `size` - The problem size
    /// * `maxiters` - The maximum number of iterations per solve attempt
    fn new(maxiters: usize) -> Self;

    /// Description
    /// The optional function SUNNonlinSolSetup performs any solver setup needed for a nonlinear solve.
    ///
    /// # Arguments
    /// NLS (SUNNonlinearSolver) a sunnonlinsol object.
    /// y (N Vector) the initial iteration passed to the nonlinear solver.
    ///
    /// Return value
    ///
    /// The return value retval (of type int) is zero for a successful call and a negative value for a failure.
    /// Notes sundials integrators call SUNonlinSolSetup before each step attempt. sunnonlinsol implementations that do not require setup may set this operation to NULL.
    fn setup<S>(&self, _y: &mut Matrix<T, D, U1, S>) -> Result<(), Error>
    where
        S: StorageMut<T, D, U1>,
    {
        Ok(())
    }

    /// Solves the nonlinear system `F(y)=0` or `G(y)=y`.
    ///
    /// # Arguments
    ///
    /// * `problem` -
    /// * `y0` - the initial iterate for the nonlinear solve.
    /// * `y` - (output) the solution to the nonlinear system.
    /// * `w` - the solution error weight vector used for computing weighted error norms.
    /// * `tol` - the requested solution tolerance in the weighted root-mean-squared norm.
    /// * `call_lsetup` - a flag indicating that the integrator recommends for the linear solver
    ///     setup function to be called.
    ///
    /// Note: The `lsetup` function sets up the linear system `Ax = b` where `A = ∂F/∂y` is the
    /// linearization of the nonlinear residual function `F(y) = 0` (when using direct linear
    /// solvers) or calls the user-defined preconditioner setup function (when using iterative
    /// linear solvers). Implementations that do not require solving this system, do not utilize
    /// linear solvers, or use linear solvers that do not require setup may skip the implementation.
    ///
    /// # Returns
    ///
    /// * Ok(()) - Successfully converged on a solution
    ///
    /// # Errors
    ///
    /// * `Err(Error::ConvergenceRecover)` - the iteration appears to be diverging, try to recover.
    /// * `Err(_)` - an unrecoverable error occurred.
    fn solve<NLP, SA, SB, SC>(
        &mut self,
        problem: &mut NLP,
        y0: &Matrix<T, D, U1, SA>,
        y: &mut Matrix<T, D, U1, SB>,
        w: &Matrix<T, D, U1, SC>,
        tol: T,
        call_lsetup: bool,
    ) -> Result<(), Error>
    where
        NLP: NLProblem<T, D>,
        SA: Storage<T, D, U1>,
        SB: StorageMut<T, D, U1>,
        SC: Storage<T, D, U1>,
        Self: std::marker::Sized;

    /// get the total number on nonlinear iterations (optional)
    fn get_num_iters(&self) -> usize {
        0
    }

    /// SUNNonlinSolGetCurIter
    /// get the iteration count for the current nonlinear solve
    fn get_cur_iter(&self) -> usize;

    /// get the total number on nonlinear solve convergence failures (optional)
    fn get_num_conv_fails(&self) -> usize;
}
