use log::warn;
use ndarray::prelude::*;

use super::constants::IdaConst;
use super::linear::{LSolver, LSolverType};
use super::traits::IdaProblem;
use super::IdaCounters;

use serde::Serialize;

/// State variables involved in the linear problem
#[derive(Clone, Debug, Serialize)]
pub struct IdaLProblem<P, LS>
where
    P: IdaProblem,
    //P::Scalar: IdaConst,
    LS: LSolver<P::Scalar>,
{
    // Linear solver, matrix and vector objects/pointers
    /// generic linear solver object
    ls: LS,

    /// J = dF/dy + cj*dF/dy'
    mat_j: Array<P::Scalar, Ix2>,

    //ytemp;       /// temp vector used by IDAAtimesDQ
    //yptemp;      /// temp vector used by IDAAtimesDQ
    /// temp vector used by the solve function        
    x: Array<P::Scalar, Ix1>,
    //ycur;        /// current y vector in Newton iteration
    //ypcur;       /// current yp vector in Newton iteration
    //rcur;        /// rcur = F(tn, ycur, ypcur)

    // Iterative solver tolerance
    /// sqrt(N)                                      
    sqrt_n: P::Scalar,
    /// eplifac = linear convergence factor          
    eplifac: P::Scalar,

    /// dqincfac = optional increment factor in Jv
    dqincfac: P::Scalar,
    /// nje = no. of calls to jac
    pub(super) nje: usize,
    /// npe = total number of precond calls
    npe: usize,
    /// nli = total number of linear iterations
    nli: usize,
    /// nps = total number of psolve calls
    nps: usize,
    /// ncfl = total number of convergence failures
    ncfl: usize,
    /// total number of calls to res
    pub(super) nre_dq: usize,
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
    /// current value of scalar (-alphas/hh) in Jacobian
    pub(super) ida_cj: P::Scalar,
    /// cj value saved from last call to lsetup
    pub(super) ida_cjold: P::Scalar,
    /// ratio of cj values: cj/cjold
    pub(super) ida_cjratio: P::Scalar,

    /// IDA problem
    pub(super) problem: P,
}

impl<P, LS> IdaLProblem<P, LS>
where
    P: IdaProblem,
    P::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + IdaConst<Scalar = P::Scalar>,
    LS: LSolver<P::Scalar>,
{
    pub fn new(problem: P) -> Self {
        use num_traits::identities::{One, Zero};
        use num_traits::Float;
        use num_traits::NumCast;
        // Retrieve the LS type */
        //LSType = SUNLinSolGetType(LS);

        /* Set four main system linear solver function fields in IDA_mem */
        //IDA_mem->ida_linit  = idaLsInitialize;
        //IDA_mem->ida_lsetup = idaLsSetup;
        //IDA_mem->ida_lsolve = idaLsSolve;
        //IDA_mem->ida_lfree  = idaLsFree;

        /* Set ida_lperf if using an iterative SUNLinearSolver object */
        //IDA_mem->ida_lperf = ( (LSType == SUNLINEARSOLVER_ITERATIVE) || (LSType == SUNLINEARSOLVER_MATRIX_ITERATIVE) ) ? idaLsPerf : NULL;

        /* Set defaults for Jacobian-related fields */
        /*
                idals_mem->J = A;
                if (A != NULL) {
                    idals_mem->jacDQ     = SUNTRUE;
                    idals_mem->jac       = idaLsDQJac;
                    idals_mem->J_data    = IDA_mem;
                } else {
                    idals_mem->jacDQ     = SUNFALSE;
                    idals_mem->jac       = NULL;
                    idals_mem->J_data    = NULL;
                }
        */
        //self.jtimesDQ = true;
        //self.jtsetup  = NULL;
        //self.jtimes   = idaLsDQJtimes;
        //self.jt_data  = IDA_mem;

        /* If LS supports ATimes, attach IDALs routine */
        /*
                if (LS->ops->setatimes) {
                    retval = SUNLinSolSetATimes(LS, IDA_mem, idaLsATimes);
                    if (retval != SUNLS_SUCCESS) {
                    IDAProcessError(IDA_mem, IDALS_SUNLS_FAIL, "IDALS",
                                    "IDASetLinearSolver",
                                    "Error in calling SUNLinSolSetATimes");
                    free(idals_mem); idals_mem = NULL;
                    return(IDALS_SUNLS_FAIL);
                    }
                }
        */

        /* If LS supports preconditioning, initialize pset/psol to NULL */
        /*
                if (LS->ops->setpreconditioner) {
                    retval = SUNLinSolSetPreconditioner(LS, IDA_mem, NULL, NULL);
                    if (retval != SUNLS_SUCCESS) {
                    IDAProcessError(IDA_mem, IDALS_SUNLS_FAIL, "IDALS",
                                    "IDASetLinearSolver",
                                    "Error in calling SUNLinSolSetPreconditioner");
                    free(idals_mem); idals_mem = NULL;
                    return(IDALS_SUNLS_FAIL);
                    }
                }
        */

        /* Allocate memory for ytemp, yptemp and x */
        //idals_mem->ytemp = N_VClone(IDA_mem->ida_tempv1);
        //idals_mem->yptemp = N_VClone(IDA_mem->ida_tempv1);
        //idals_mem->x = N_VClone(IDA_mem->ida_tempv1);

        /* Compute sqrtN from a dot product */
        //N_VConst(ONE, idals_mem->ytemp);
        //idals_mem->sqrtN = SUNRsqrt( N_VDotProd(idals_mem->ytemp, idals_mem->ytemp) );

        Self {
            ls: LS::new(problem.model_size()),

            // Initialize counters
            nje: 0,
            nre_dq: 0,
            npe: 0,
            nli: 0,
            nps: 0,
            ncfl: 0,
            njtsetup: 0,
            njtimes: 0,

            nst0: 0,
            ncfl0: 0,
            ncfn0: 0,
            nni0: 0,
            nwarn: 0,

            // Set default values for the rest of the Ls parameters
            eplifac: P::Scalar::pt05(),
            dqincfac: P::Scalar::one(),
            //last_flag : IDALS_SUCCESS
            sqrt_n: <P::Scalar as NumCast>::from(problem.model_size())
                .unwrap()
                .sqrt(),

            ida_cj: P::Scalar::zero(),
            ida_cjold: P::Scalar::zero(),
            ida_cjratio: P::Scalar::zero(),

            mat_j: Array::zeros((problem.model_size(), problem.model_size())),
            x: Array::zeros(problem.model_size()),

            problem,
        }
    }

    /// idaLsSetup
    ///
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS `setup` routine to prepare for subsequent calls to the LS 'solve' routine.
    pub fn setup<S1, S2, S3>(
        &mut self,
        y: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        r: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
        S3: ndarray::Data<Elem = P::Scalar>,
    {
        use num_traits::identities::Zero;

        // Set IDALs N_Vector pointers to inputs
        //self.ycur  = y;
        //self.ypcur = yp;
        //self.rcur  = r;

        // recompute if J if it is non-NULL
        if !self.mat_j.is_empty() {
            // Increment nje counter.
            self.nje += 1;

            // Zero out J; call Jacobian routine jac; return if it failed.
            self.mat_j.fill(P::Scalar::zero());

            // Call Jacobian routine
            //retval = self.jac(IDA_mem->ida_tn, IDA_mem->ida_cj, y, yp, r, idals_mem->J, idals_mem->J_data, vt1, vt2, vt3);
            //TODO fix
            self.problem.jac(
                P::Scalar::zero(),
                self.ida_cj,
                y.view(),
                yp.view(),
                r.view(),
                self.mat_j.view_mut(),
            );

            /*
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
            */

            //if (retval > 0) { self.glast_flag = IDALS_JACFUNC_RECVR; return(1); }
        }

        // Call LS setup routine -- the LS will call idaLsPSetup if applicable
        self.ls.setup(self.mat_j.view_mut()).unwrap();
        //self.last_flag = SUNLinSolSetup(idals_mem->LS, idals_mem->J);
        //return(self.last_flag);
    }

    /// idaLsSolve
    ///
    /// This routine interfaces between IDA and the generic LinearSolver object LS, by setting the
    /// appropriate tolerance and scaling vectors, calling the solver, accumulating statistics from
    /// the solve for use/reporting by IDA, and scaling the result if using a non-NULL Matrix and
    /// cjratio does not equal one.
    pub fn solve<S1, S2, S3, S4, S5>(
        &mut self,
        mut b: ArrayBase<S1, Ix1>,
        weight: ArrayBase<S2, Ix1>,
        _ycur: ArrayBase<S3, Ix1>,
        _ypcur: ArrayBase<S4, Ix1>,
        _rescur: ArrayBase<S5, Ix1>,
    ) where
        S1: ndarray::DataMut<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
        S3: ndarray::Data<Elem = P::Scalar>,
        S4: ndarray::Data<Elem = P::Scalar>,
        S5: ndarray::Data<Elem = P::Scalar>,
        ArrayBase<S2, Ix1>: ndarray::linalg::Dot<ArrayBase<S2, Ix1>>,
    {
        use num_traits::identities::{One, Zero};

        // Retrieve the LS type
        let ls_type = self.ls.get_type();

        // If the linear solver is iterative: set convergence test constant tol, in terms of the
        // Newton convergence test constant epsNewt and safety factors. The factor sqrt(Neq)
        // assures that the convergence test is applied to the WRMS norm of the residual vector,
        // rather than the weighted L2 norm.

        let tol = match ls_type {
            LSolverType::Iterative | LSolverType::MatrixIterative => {
                self.sqrt_n * self.eplifac
                //self.sqrtN * idals_mem->eplifac * IDA_mem->ida_epsNewt
            }
            _ => P::Scalar::zero(),
        };

        /* Set vectors ycur, ypcur and rcur for use by the Atimes and Psolve interface routines */
        //self.ycur  = ycur;
        //self.ypcur = ypcur;
        //self.rcur  = rescur;

        // Set initial guess x = 0 to LS
        self.x.fill(P::Scalar::zero());

        // Set scaling vectors for LS to use (if applicable)
        self.ls.set_scaling_vectors(weight.view(), weight.view());
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
        if let LSolverType::Iterative | LSolverType::MatrixIterative = ls_type {
            //let w_mean = weight.dot(&weight).sqrt() / self.sqrtN;
            //tol /= w_mean;
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
        let retval = self
            .ls
            .solve(self.mat_j.view(), self.x.view_mut(), b.view(), tol);

        // Copy appropriate result to b (depending on solver type)
        if let LSolverType::Iterative | LSolverType::MatrixIterative = ls_type {
            // Retrieve solver statistics
            let nli_inc = self.ls.num_iters();

            // Copy x (or preconditioned residual vector if no iterations required) to b
            if nli_inc == 0 {
                //N_VScale(ONE, SUNLinSolResid(self.gLS), b);
            } else {
                b.assign(&self.x);
            }

            // Increment nli counter
            self.nli += nli_inc;
        } else {
            // Copy x to b
            b.assign(&self.x);
        }

        // If using a direct or matrix-iterative solver, scale the correction to account for change in cj
        if let LSolverType::Direct | LSolverType::MatrixIterative = ls_type {
            if self.ida_cjratio != P::Scalar::one() {
                b *= P::Scalar::two() / (P::Scalar::one() + self.ida_cjratio);
            }
        }

        // Increment ncfl counter
        if retval.is_err() {
            self.ncfl += 1;
        }

        /*
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
            IDAProcessError(IDA_mem, SUNLS_PACKAGE_FAIL_UNREC, "IDALS", "idaLsSolve", "Failure in SUNLinSol external package");
            return(-1);
            break;
        case SUNLS_PSOLVE_FAIL_UNREC:
            IDAProcessError(IDA_mem, SUNLS_PSOLVE_FAIL_UNREC, "IDALS", "idaLsSolve", MSG_LS_PSOLVE_FAILED);
            return(-1);
            break;
        }
        */

        //return(0);
    }

    /// idaLsPerf: accumulates performance statistics information for IDA
    pub fn ls_perf(&mut self, counters: &IdaCounters, perftask: bool) {
        // when perftask == 0, store current performance statistics
        if !perftask {
            self.nst0 = counters.ida_nst;
            self.nni0 = counters.ida_nni;
            self.ncfn0 = counters.ida_ncfn;
            self.ncfl0 = self.ncfl;
            self.nwarn = 0;
            return;
        }

        // Compute statistics since last call
        //
        // Note: the performance monitor that checked whether the average
        // number of linear iterations was too close to maxl has been
        // removed, since the 'maxl' value is no longer owned by the
        // IDALs interface.

        let nstd = counters.ida_nst - self.nst0;
        let nnid = counters.ida_nni - self.nni0;
        if nstd == 0 || nnid == 0 {
            return;
        };

        let rcfn = (counters.ida_ncfn - self.ncfn0) as f64 / (nstd as f64);
        let rcfl = (self.ncfl - self.ncfl0) as f64 / (nnid as f64);
        let lcfn = rcfn > 0.9;
        let lcfl = rcfl > 0.9;
        if !(lcfn || lcfl) {
            return;
        }
        self.nwarn += 1;
        if self.nwarn > 10 {
            return;
        }
        if lcfn {
            warn!("Warning: at t = {}, poor iterative algorithm performance. Nonlinear convergence failure rate is {}.", 0.0, rcfn);
        }
        if lcfl {
            warn!("Warning: at t = {}, poor iterative algorithm performance. Linear convergence failure rate is {}.", 0.0, rcfl);
        }
    }
}
