use thiserror::Error;

#[derive(Debug)]
pub enum RecoverableKind {
    /// IDA_RES_RECVR
    Residual,
    /// IDA_LSETUP_RECVR
    LSetup,
    /// IDA_LSOLVE_RECVR
    LSolve,
    /// IDA_CONSTR_RECVR
    Constraint,
    /// IDA_NLS_SETUP_RECVR
    NLSSetup,
}

#[derive(Debug, Error)]
pub enum Error {
    // ERROR_TEST_FAIL
    #[error("Error Test Failed")]
    TestFail,

    // LSETUP_ERROR_NONRECVR
    // IDA_ERR_FAIL
    #[error("Error test failures occurred too many times during one internal time step or minimum step size was reached.")]
    ErrFail,

    /// IDA_REP_RES_ERR:
    #[error("The user's residual function repeatedly returned a recoverable error flag, but the solver was unable to recover")]
    RepeatedResidualError {},

    /// IDA_ILL_INPUT
    #[error("One of the input arguments was illegal. See printed message")]
    IllegalInput { msg: String },

    /// IDA_LINIT_FAIL
    #[error("The linear solver's init routine failed")]
    LinearInitFail {},

    /// IDA_BAD_EWT
    #[error( "Some component of the error weight vector is zero (illegal), either for the input value of y0 or a corrected value")]
    BadErrorWeightVector {},

    /// IDA_RES_FAIL
    #[error("The user's residual routine returned a non-recoverable error flag")]
    ResidualFail,

    /// IDA_FIRST_RES_FAIL
    #[error( "The user's residual routine returned a recoverable error flag on the first call, but IDACalcIC was unable to recover")]
    FirstResidualFail {},

    /// IDA_LSETUP_FAIL
    #[error("The linear solver's setup routine had a non-recoverable error")]
    LinearSetupFail {},

    /// IDA_LSOLVE_FAIL
    #[error("The linear solver's solve routine had a non-recoverable error")]
    LinearSolveFail {},

    /// IDA_NO_RECOVERY
    #[error(
        "The user's residual routine, or the linear solver's setup or solve routine \
                   had a recoverable error, but IDACalcIC was unable to recover"
    )]
    NoRecovery {},

    #[error("Recoverable failure")]
    RecoverableFail(RecoverableKind),

    /// IDA_CONSTR_FAIL
    /// The inequality constraints were violated, and the solver was unable to recover.
    #[error("IDACalcIC was unable to find a solution satisfying the inequality constraints")]
    ConstraintFail,

    /// IDA_LINESEARCH_FAIL
    #[error(
        "The Linesearch algorithm failed to find a solution with a step larger than \
                   steptol in weighted RMS norm"
    )]
    LinesearchFail {},

    /// IDA_CONV_FAIL
    #[error("IDACalcIC failed to get convergence of the Newton iterations")]
    ConvergenceFail,

    ///MSG_BAD_K
    #[error("Illegal value for k, should be 0 <= k <= {kused}.")]
    BadK { kused: usize },

    //MSG_NULL_DKY       "dky = NULL illegal."
    ///MSG_BAD_T
    #[error("Illegal value for t: t = {t} is not between tcur - hu = {tdiff} and tcur = {tcurr}.")]
    BadTimeValue { t: f64, tdiff: f64, tcurr: f64 },

    #[error("At t = {t}, the rootfinding routine failed in an unrecoverable manner.")]
    RootFunctionFail { t: f64 },

    ///MSG_BAD_TSTOP
    #[error(
        "The value tstop = {tstop} is behind current t = {t} in the direction of integration."
    )]
    BadStopTime { tstop: f64, t: f64 },

    /// IDA_TOO_MUCH_WORK
    /// The solver took mxstep internal steps but could not reach tout.
    #[error("At t = {t:.5e}, the solver took mxstep internal steps ({mxstep}) but could not reach tout.")]
    TooMuchWork { t: f64, mxstep: usize },

    /// MSG_TOO_MUCH_ACC
    /// The solver could not satisfy the accuracy demanded by the user for some internal step.
    #[error("At t = {t} too much accuracy requested.")]
    TooMuchAccuracy { t: f64 },

    ///MSG_CLOSE_ROOTS
    #[error("Root found at and very near {t}.")]
    CloseRoots { t: f64 },

    #[error("Linear solver setup failed")]
    LinearSetupFailed {
        #[from]
        source: linear::Error,
    },

    #[error(transparent)]
    Nonlinear(#[from] nonlinear::Error),
}
