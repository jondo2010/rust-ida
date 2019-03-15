use failure::Fail;

#[derive(Debug, Fail)]
pub enum IdaError {

    // ERROR_TEST_FAIL
    #[fail(display="Error Test Failed")]
    TestFail,

    // LSETUP_ERROR_NONRECVR
    // IDA_ERR_FAIL
    /// IDA_REP_RES_ERR:
    #[fail(
        display = "The user's residual function repeatedly returned a recoverable error flag, \
                   but the solver was unable to recover"
    )]
    RepeatedResidualError {},

    /// IDA_ILL_INPUT
    #[fail(display = "One of the input arguments was illegal. See printed message")]
    IllegalInput { msg: String },

    /// IDA_LINIT_FAIL
    #[fail(display = "The linear solver's init routine failed")]
    LinearInitFail {},

    /// IDA_BAD_EWT
    #[fail(
        display = "Some component of the error weight vector is zero (illegal), either for the \
                   input value of y0 or a corrected value"
    )]
    BadErrorWeightVector {},

    /// IDA_RES_FAIL
    #[fail(display = "The user's residual routine returned a non-recoverable error flag")]
    ResidualFail {},

    /// IDA_FIRST_RES_FAIL
    #[fail(
        display = "The user's residual routine returned a recoverable error flag on the first \
                   call, but IDACalcIC was unable to recover"
    )]
    FirstResidualFail {},

    /// IDA_LSETUP_FAIL
    #[fail(display = "The linear solver's setup routine had a non-recoverable error")]
    LinearSetupFail {},

    /// IDA_LSOLVE_FAIL
    #[fail(display = "The linear solver's solve routine had a non-recoverable error")]
    LinearSolveFail {},

    /// IDA_NO_RECOVERY
    #[fail(
        display = "The user's residual routine, or the linear solver's setup or solve routine \
                   had a recoverable error, but IDACalcIC was unable to recover"
    )]
    NoRecovery {},

    /// IDA_CONSTR_FAIL
    /// The inequality constraints were violated, and the solver was unable to recover.
    #[fail(
        display = "IDACalcIC was unable to find a solution satisfying the inequality constraints"
    )]
    ConstraintFail {},

    /// IDA_LINESEARCH_FAIL
    #[fail(
        display = "The Linesearch algorithm failed to find a solution with a step larger than \
                   steptol in weighted RMS norm"
    )]
    LinesearchFail {},

    /// IDA_CONV_FAIL
    #[fail(display = "IDACalcIC failed to get convergence of the Newton iterations")]
    ConvergenceFail {},

    ///MSG_BAD_K
    #[fail(display = "Illegal value for k.")]
    BadK {},

    //MSG_NULL_DKY       "dky = NULL illegal."
    ///MSG_BAD_T
    #[fail(
        display = "Illegal value for t: t = {} is not between tcur - hu = {} and tcur = {}.",
        t, tdiff, tcurr
    )]
    BadTimeValue { t: f64, tdiff: f64, tcurr: f64 },

    #[fail(
        display = "At t = {}, the rootfinding routine failed in an unrecoverable manner.",
        t
    )]
    RootFunctionFail { t: f64 },

    ///MSG_BAD_TSTOP
    #[fail(
        display = "The value tstop = {} is behind current t = {} in the direction of integration.",
        tstop, t
    )]
    BadStopTime { tstop: f64, t: f64 },

    ///MSG_TOO_MUCH_ACC
    #[fail(display = "At t = {} too much accuracy requested.", t)]
    TooMuchAccuracy { t: f64 },
}
