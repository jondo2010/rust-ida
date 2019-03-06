use failure::Fail;
use ndarray::*;

use crate::traits::{ModelSpec, Residual};

#[derive(Debug, Fail)]
pub enum Error {
    // Recoverable
    /// SUN_NLS_CONTINUE
    /// not converged, keep iterating
    #[fail(display = "")]
    Continue {},

    /// convergece failure, try to recover
    /// SUN_NLS_CONV_RECVR
    #[fail(display = "")]
    ConvergenceRecover {},

    // Unrecoverable
    /// illegal function input
    ///SUN_NLS_ILL_INPUT
    #[fail(display = "")]
    IllegalInput {},
    // failed NVector operation
    //SUN_NLS_VECTOROP_ERR
}

pub trait NLProblem: ModelSpec {
    fn res<S1, S2>(
        &self,
        y: &ArrayBase<S1, Ix1>,
        f: &mut ArrayBase<S2, Ix1>,
    ) -> Result<(), Error>
    where
        S1: Data<Elem = <Self as ModelSpec>::Scalar>,
        S2: DataMut<Elem = <Self as ModelSpec>::Scalar>;

    // int LSetup(N_Vector y, N_Vector f, booleantype jbad, booleantype* jcur, void* mem)
    /// Purpose These functions are wrappers to the sundials integrator’s function for setting up linear solves with sunlinsol modules.
    ///
    /// # Arguments
    ///
    /// * `y` is the state vector at which the linear system should be setup.
    /// * `F is the value of the nonlinear system function at y.
    /// * `jbad is an input indicating whether the nonlinear solver believes that A has gone stale (SUNTRUE) or not (SUNFALSE).
    /// * `jcur is an output indicating whether the routine has updated the Jacobian A (SUNTRUE) or not (SUNFALSE).
    fn lsetup<S1>(
        &mut self,
        y: &ArrayBase<S1, Ix1>,
        F: &ArrayView<<Self as ModelSpec>::Scalar, Ix1>,
        jbad: bool,
    ) -> bool
    where
        S1: Data<Elem = <Self as ModelSpec>::Scalar>,
    {
        false
    }

    // int LSolve(N_Vector y, N_Vector b, void* mem)

    /// # Arguments
    /// * `y` is the input vector containing the current nonlinear iteration.
    /// * `b` contains the right-hand side vector for the linear solve on input and the solution to
    ///     the linear system on output.
    ///
    /// Return value The return value retval (of type int) is zero for a successul solve, a positive value for a recoverable error, and a negative value for an unrecoverable error.
    ///
    /// # Notes
    /// The lsove function solves the linear system `Ax = b` where `A = ∂F/∂y` is the linearization
    /// of the nonlinear residual function F(y) = 0. Implementations that do not require solving
    /// this system or do not use sunlinsol linear solvers may ignore these functions.
    fn lsolve<S1, S2>(&self, y: &ArrayBase<S1, Ix1>, b: &mut ArrayBase<S2, Ix1>)
    where
        S1: Data<Elem = <Self as ModelSpec>::Scalar>,
        S2: DataMut<Elem = <Self as ModelSpec>::Scalar>;

    /// Proxy for integrator convergence test function */
    ///
    /// # Arguments
    ///
    /// * `y` is the current nonlinear iterate.
    /// * `del` is the difference between the current and prior nonlinear iterates.
    /// * `tol` is the nonlinear solver tolerance.
    /// * `ewt` is the weight vector used in computing weighted norms.
    ///
    /// # Returns
    ///
    /// The return value of this routine will be a negative value if an unrecoverable error occurred or one of the following:
    /// Ok(`true`) - the iteration is converged.
    /// Ok(`false`) - the iteration has not converged, keep iterating.
    /// Err() - SUN NLS CONV RECVR the iteration appears to be diverging, try to recover.
    ///
    /// # Notes
    ///
    /// The tolerance passed to this routine by sundials integrators is the tolerance in a
    /// weighted root-mean-squared norm with error weight vector ewt. sunnonlinsol modules
    /// utilizing their own convergence criteria may ignore these functions.
    fn ctest<S1, S2, S3>(
        &self,
        y: &ArrayBase<S1, Ix1>,
        del: &ArrayBase<S2, Ix1>,
        tol: <Self as ModelSpec>::Scalar,
        ewt: &ArrayBase<S3, Ix1>,
    ) -> Result<bool, Error>
    where
        S1: Data<Elem = <Self as ModelSpec>::Scalar>,
        S2: Data<Elem = <Self as ModelSpec>::Scalar>,
        S3: Data<Elem = <Self as ModelSpec>::Scalar>;
}

pub trait NonlinearSolver<F: Residual> {
    // core functions
    //fn setup(&self, y: &Vector1<f64>);
    //fn solve(&self, N_Vector y0, N_Vector y, N_Vector w, realtype tol, booleantype callLSetup, void *mem);

    /// # Arguments
    ///
    /// * `y0`
    /// * `w`
    /// * `tol`
    /// * `lsetup` - the package-supplied function for setting up the linear solver,
    fn solve_with_setup<L>(
        &self,
        y0: Array1<F::Scalar>,
        w: Array1<F::Scalar>,
        tol: F::Scalar,
        lsetup: L,
    ) -> Result<(), failure::Error>
    where
        L: FnOnce();
    // set functions
    /*
    fn setSysFn(&self, SUNNonlinSolSysFn SysFn);
    fn setLSetupFn(&self, SUNNonlinSolLSetupFn SetupFn);
    fn setLSolveFn(SUNNonlinearSolver NLS, SUNNonlinSolLSolveFn SolveFn);
    fn setConvTestFn(SUNNonlinearSolver NLS, SUNNonlinSolConvTestFn CTestFn);
    fn setMaxIters(SUNNonlinearSolver NLS, int maxiters);
    // get functions
    fn getNumIters(SUNNonlinearSolver NLS, long int *niters);
    fn getCurIter(SUNNonlinearSolver NLS, int *iter);
    fn getNumConvFails(SUNNonlinearSolver NLS, long int *nconvfails);
    */
}
