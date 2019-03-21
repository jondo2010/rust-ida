use ndarray::prelude::*;

use super::constants::IdaConst;
use super::linear::{LSolver, LSolverType};

/// State variables involved in the linear problem
#[derive(Clone, Debug)]
struct IdaLProblem<Scalar, LS>
where
    Scalar: IdaConst,
    LS: LSolver<Scalar>,
{
    // Linear solver, matrix and vector objects/pointers
    /// generic linear solver object
    ls: LS,
    /// J = dF/dy + cj*dF/dy'
    J: Array<Scalar, Ix2>,

    //ytemp;       /// temp vector used by IDAAtimesDQ
    //yptemp;      /// temp vector used by IDAAtimesDQ
    /// temp vector used by the solve function        
    x: Array<Scalar, Ix1>,
    //ycur;        /// current y vector in Newton iteration
    //ypcur;       /// current yp vector in Newton iteration
    //rcur;        /// rcur = F(tn, ycur, ypcur)

    // Iterative solver tolerance
    /// sqrt(N)                                      
    sqrtN: Scalar,
    /// eplifac = linear convergence factor          
    eplifac: Scalar,

    /// dqincfac = optional increment factor in Jv
    dqincfac: Scalar,
    /// nje = no. of calls to jac
    nje: usize,
    /// npe = total number of precond calls
    npe: usize,
    /// nli = total number of linear iterations
    nli: usize,
    /// nps = total number of psolve calls
    nps: usize,
    /// ncfl = total number of convergence failures
    ncfl: usize,
    /// nreDQ = total number of calls to res
    nreDQ: usize,
    /// njtsetup = total number of calls to jtsetup
    njtsetup: usize,
    /// njtimes = total number of calls to jtimes
    njtimes: usize,
    /// nst0 = saved nst (for performance monitor)
    nst0: usize,
    /// nni0 = saved nni (for performance monitor)
    nni0: usize,
    /// ncfn0 = saved ncfn (for performance monitor)
    ncfn0: usize,
    /// ncfl0 = saved ncfl (for performance monitor)
    ncfl0: usize,
    /// nwarn = no. of warnings (for perf. monitor)
    nwarn: usize,
    /*
        long int last_flag; // last error return flag

        // Preconditioner computation
        // (a) user-provided:
        //     - pdata == user_data
        //     - pfree == NULL (the user dealocates memory)
        // (b) internal preconditioner module
        //     - pdata == ida_mem
        //     - pfree == set by the prec. module and called in idaLsFree
        IDALsPrecSetupFn pset;
        IDALsPrecSolveFn psolve;
        int (*pfree)(IDAMem IDA_mem);
        void *pdata;

        // Jacobian times vector compuation
        // (a) jtimes function provided by the user:
        //     - jt_data == user_data
        //     - jtimesDQ == SUNFALSE
        // (b) internal jtimes
        //     - jt_data == ida_mem
        //     - jtimesDQ == SUNTRUE
        booleantype jtimesDQ;
        IDALsJacTimesSetupFn jtsetup;
        IDALsJacTimesVecFn jtimes;
        void *jt_data;
    */
}

impl<Scalar, LS> IdaLProblem<Scalar, LS>
where
    Scalar: IdaConst,
    LS: LSolver<Scalar>,
{
    /// idaLsSetup
    ///
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS `setup` routine to prepare for subsequent calls to the LS 'solve' routine.
    fn setup<S1, S2, S3>(
        &mut self,
        y: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix2>,
        r: ArrayBase<S3, Ix3>,
    ) where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::Data<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>,
    {
        // Set IDALs N_Vector pointers to inputs
        //self.gycur  = y;
        //self.gypcur = yp;
        //self.grcur  = r;

        /* recompute if J if it is non-NULL */
        //if (self.gJ) {

        /* Increment nje counter. */
        self.nje += 1;

        // Zero out J; call Jacobian routine jac; return if it failed.
        self.J.fill(Scalar::zero());

        // Call Jacobian routine
        //retval = self.gjac(IDA_mem->ida_tn, IDA_mem->ida_cj, y, yp, r, idals_mem->J, idals_mem->J_data, vt1, vt2, vt3);

        if (retval < 0) {
            IDAProcessError(
                IDA_mem,
                IDALS_JACFUNC_UNRECVR,
                "IDALS",
                "idaLsSetup",
                MSG_LS_JACFUNC_FAILED,
            );
            //self.glast_flag = IDALS_JACFUNC_UNRECVR;
            //return(-1);
        }

        //if (retval > 0) { self.glast_flag = IDALS_JACFUNC_RECVR; return(1); }

        //}

        // Call LS setup routine -- the LS will call idaLsPSetup if applicable
        //self.glast_flag = SUNLinSolSetup(idals_mem->LS, idals_mem->J);
        //return(self.glast_flag);
    }

    /// idaLsSolve
    ///
    /// This routine interfaces between IDA and the generic LinearSolver object LS, by setting the appropriate tolerance and scaling vectors, calling the solver, accumulating statistics from the solve for use/reporting by IDA, and scaling the result if using a non-NULL Matrix and cjratio does not equal one.
    fn solve<S1, S2, S3, S4, S5>(
        &mut self,
        b: &ArrayBase<S1, Ix1>,
        weight: &ArrayBAse<S2, Ix1>,
        ycur: &ArrayBase<S3, Ix1>,
        ypcur: &ArrayBase<S4, Ix1>,
        rescur: &ArrayBase<S5, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::Data<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>,
        S4: ndarray::Data<Elem = Scalar>,
        S5: ndarray::Data<Elem = Scalar>,
    {
        //int      nli_inc, retval;
        //realtype tol, w_mean, LSType;

        // Retrieve the LS type
        let LSType = self.ls.get_type();

        // If the linear solver is iterative: set convergence test constant tol, in terms of the
        // Newton convergence test constant epsNewt and safety factors. The factor sqrt(Neq)
        // assures that the convergence test is applied to the WRMS norm of the residual vector,
        // rather than the weighted L2 norm.

        let tol = match LSType {
            LSolverType::Iterative | LSolverType::MatrixIterative => {
                //self.gsqrtN * idals_mem->eplifac * IDA_mem->ida_epsNewt
            }
            _ => Scalar::zero(),
        };

        /* Set vectors ycur, ypcur and rcur for use by the Atimes and Psolve interface routines */
        //self.gycur  = ycur;
        //self.gypcur = ypcur;
        //self.grcur  = rescur;

        // Set initial guess x = 0 to LS
        x.fill(Scalar::zero());

        // Set scaling vectors for LS to use (if applicable)
        self.ls.set_scaling_vectors(&weight, &weight);
        /*
        retval = SUNLinSolSetScalingVectors(self.gLS, weight, weight);
        if (retval != SUNLS_SUCCESS) {
            IDAProcessError(IDA_mem, IDALS_SUNLS_FAIL, "IDALS", "idaLsSolve", "Error in calling SUNLinSolSetScalingVectors");
            self.glast_flag = IDALS_SUNLS_FAIL;
            return(self.glast_flag);
        }
        */

        // If solver is iterative and does not support scaling vectors, update the tolerance in an attempt to account for weight vector.  We make the following assumptions:
        //   1. w_i = w_mean, for i=0,...,n-1 (i.e. the weights are homogeneous)
        //   2. the linear solver uses a basic 2-norm to measure convergence
        // Hence (using the notation from sunlinsol_spgmr.h, with S = diag(w)),
        //       || bbar - Abar xbar ||_2 < tol
        //   <=> || S b - S A x ||_2 < tol
        //   <=> || S (b - A x) ||_2 < tol
        //   <=> \sum_{i=0}^{n-1} (w_i (b - A x)_i)^2 < tol^2
        //   <=> w_mean^2 \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2
        //   <=> \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2 / w_mean^2
        //   <=> || b - A x ||_2 < tol / w_mean
        // So we compute w_mean = ||w||_RMS = ||w||_2 / sqrt(n), and scale the desired tolerance accordingly.
        if let LSolverType::Iterative | LSolverType::MatrixIterative = LSType {}

        match LSType {
            LSolverType::Iterative | LSolverType::MatrixIterative => {
                //let w_mean = SUNRsqrt( N_VDotProd(weight, weight) ) / self.gsqrtN;
                //tol /= w_mean;
            }
        }

        // If a user-provided jtsetup routine is supplied, call that here
        /*
        if (self.gjtsetup) {
            self.glast_flag = idals_mem->jtsetup(IDA_mem->ida_tn, ycur, ypcur, rescur,
                                                    IDA_mem->ida_cj, self.gjt_data);
            self.gnjtsetup++;
            if (self.glast_flag != 0) {
            IDAProcessError(IDA_mem, retval, "IDALS",
                            "idaLsSolve", MSG_LS_JTSETUP_FAILED);
            return(self.glast_flag);
            }
        }
        */

        // Call solver
        //retval = SUNLinSolSolve(self.gLS, self.J, self.gx, b, tol);

        /*
        // Copy appropriate result to b (depending on solver type)
        if ( (LSType == SUNLINEARSOLVER_ITERATIVE) || (LSType == SUNLINEARSOLVER_MATRIX_ITERATIVE) ) {

            // Retrieve solver statistics
            nli_inc = SUNLinSolNumIters(self.gLS);

            // Copy x (or preconditioned residual vector if no iterations required) to b
            if (nli_inc == 0) {N_VScale(ONE, SUNLinSolResid(self.gLS), b);}
            else N_VScale(ONE, self.gx, b);

            // Increment nli counter
            self.gnli += nli_inc;

        } else {

            // Copy x to b
            N_VScale(ONE, self.gx, b);

        }

        // If using a direct or matrix-iterative solver, scale the correction to account for change in cj
        if ( ((LSType == SUNLINEARSOLVER_DIRECT) || (LSType == SUNLINEARSOLVER_MATRIX_ITERATIVE)) && (IDA_mem->ida_cjratio != ONE) ) {
            N_VScale(TWO/(ONE + IDA_mem->ida_cjratio), b, b);
        }

        // Increment ncfl counter
        if (retval != SUNLS_SUCCESS) {self.gncfl++;}

        // Interpret solver return value
        self.glast_flag = retval;
        */

        /*
        switch(retval) {

        case SUNLS_SUCCESS:
            return(0);
            break;
        case SUNLS_RES_REDUCED:
        case SUNLS_CONV_FAIL:
        case SUNLS_PSOLVE_FAIL_REC:
        case SUNLS_PACKAGE_FAIL_REC:
        case SUNLS_QRFACT_FAIL:
        case SUNLS_LUFACT_FAIL:
            return(1);
            break;
        case SUNLS_MEM_NULL:
        case SUNLS_ILL_INPUT:
        case SUNLS_MEM_FAIL:
        case SUNLS_GS_FAIL:
        case SUNLS_QRSOL_FAIL:
            return(-1);
            break;
        case SUNLS_PACKAGE_FAIL_UNREC:
            IDAProcessError(IDA_mem, SUNLS_PACKAGE_FAIL_UNREC, "IDALS",
                            "idaLsSolve",
                            "Failure in SUNLinSol external package");
            return(-1);
            break;
        case SUNLS_PSOLVE_FAIL_UNREC:
            IDAProcessError(IDA_mem, SUNLS_PSOLVE_FAIL_UNREC, "IDALS",
                            "idaLsSolve", MSG_LS_PSOLVE_FAILED);
            return(-1);
            break;
        }
        */

        //return(0);
    }
}
