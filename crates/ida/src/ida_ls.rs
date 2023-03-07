use log::warn;

use linear::{LSolver, LSolverType};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimName, Matrix, OMatrix, OVector, Storage,
    StorageMut, U1,
};

use crate::{
    traits::{IdaProblem, IdaReal},
    IdaCounters,
};
//use super::IdaCounters;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Statistics and associated parameters for the linear solver
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct IdaLProblemCounters {
    /// dqincfac = optional increment factor in Jv
    //pub dqincfac: T,
    /// nje = no. of calls to jac
    pub nje: usize,
    /// npe = total number of precond calls
    pub npe: usize,
    /// nli = total number of linear iterations
    pub nli: usize,
    /// nps = total number of psolve calls
    pub nps: usize,
    /// ncfl = total number of convergence failures
    pub ncfl: usize,
    /// total number of calls to res
    pub nre_dq: usize,
    /// njtsetup = total number of calls to jtsetup
    pub njtsetup: usize,
    /// njtimes = total number of calls to jtimes
    pub njtimes: usize,
    /// nst0 = saved nst (for performance monitor)
    pub nst0: usize,
    /// nni0 = saved nni (for performance monitor)
    pub nni0: usize,
    /// ncfn0 = saved ncfn (for performance monitor)
    pub ncfn0: usize,
    /// ncfl0 = saved ncfl (for performance monitor)
    pub ncfl0: usize,
    /// nwarn = no. of warnings (for perf. monitor)
    pub nwarn: usize,
}

/// State variables involved in the linear problem
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, D>: Serialize, OMatrix<T, D, D>: Serialize, LS: Serialize, P: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "T: Deserialize<'de>, OVector<T, D>: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>, LS: Deserialize<'de>, P: Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug)]
pub struct IdaLProblem<T, D, P, LS>
where
    T: IdaReal,
    D: Dim,
    P: IdaProblem<T, D>,
    LS: LSolver<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    // Linear solver, matrix and vector objects/pointers
    /// generic linear solver object
    pub(crate) ls: LS,

    /// J = dF/dy + cj*dF/dy'
    pub(crate) mat_j: OMatrix<T, D, D>,

    //ytemp;       /// temp vector used by IDAAtimesDQ
    //yptemp;      /// temp vector used by IDAAtimesDQ
    /// temp vector used by the solve function        
    pub(crate) x: OVector<T, D>,
    //ycur;        /// current y vector in Newton iteration
    //ypcur;       /// current yp vector in Newton iteration
    //rcur;        /// rcur = F(tn, ycur, ypcur)

    // Iterative solver tolerance
    /// eplifac = linear convergence factor          
    pub eplifac: T,
    /// integrator -> LS norm conversion factor
    pub nrmfac: T,

    /// current value of scalar (-alphas/hh) in Jacobian
    pub(super) ida_cj: T,
    /// cj value saved from last call to lsetup
    pub(super) ida_cjold: T,
    /// ratio of cj values: cj/cjold
    pub(super) ida_cjratio: T,

    /// IDA problem
    pub(super) problem: P,
    /// Statistics and associated parameters for the linear solver
    pub(super) counters: IdaLProblemCounters,
}

impl<T, D, P, LS> IdaLProblem<T, D, P, LS>
where
    T: IdaReal,
    D: DimName,
    P: IdaProblem<T, D>,
    LS: LSolver<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub fn new(problem: P, ls: LS) -> Self {
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
            ls,
            // Set default values for the rest of the Ls parameters
            eplifac: T::pt05(),
            //last_flag : IDALS_SUCCESS
            nrmfac: T::from(D::try_to_usize().unwrap() as u32).unwrap().sqrt(),

            ida_cj: T::zero(),
            ida_cjold: T::zero(),
            ida_cjratio: T::zero(),

            mat_j: OMatrix::<T, D, D>::zeros(),
            x: OVector::<T, D>::zeros(),

            problem,

            // Initialize counters
            //dqincfac: T::one(),
            counters: IdaLProblemCounters::default(),
        }
    }

    /// idaLsSetup
    ///
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS `setup` routine to prepare for subsequent calls to the LS 'solve' routine.
    pub fn setup<SA, SB, SC>(
        &mut self,
        y: &Matrix<T, D, U1, SA>,
        yp: &Matrix<T, D, U1, SB>,
        r: &Matrix<T, D, U1, SC>,
    ) where
        SA: Storage<T, D>,
        SB: Storage<T, D>,
        SC: Storage<T, D>,
    {
        // Set IDALs N_Vector pointers to inputs
        //self.ycur  = y;
        //self.ypcur = yp;
        //self.rcur  = r;

        // recompute if J if it is non-NULL
        if !self.mat_j.is_empty() {
            // Increment nje counter.
            self.counters.nje += 1;

            // Zero out J; call Jacobian routine jac; return if it failed.
            self.mat_j.fill(T::zero());

            // Call Jacobian routine
            //retval = self.jac(IDA_mem->ida_tn, IDA_mem->ida_cj, y, yp, r, idals_mem->J, idals_mem->J_data, vt1, vt2, vt3);
            //TODO fix
            self.problem
                .jac(T::zero(), self.ida_cj, &y, &yp, &r, &mut self.mat_j);

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
        self.ls.setup(&mut self.mat_j).unwrap();
        //self.last_flag = SUNLinSolSetup(idals_mem->LS, idals_mem->J);
        //return(self.last_flag);
    }

    /// idaLsSolve
    ///
    /// This routine interfaces between IDA and the generic LinearSolver object LS, by setting the
    /// appropriate tolerance and scaling vectors, calling the solver, accumulating statistics from
    /// the solve for use/reporting by IDA, and scaling the result if using a non-NULL Matrix and
    /// cjratio does not equal one.
    pub fn solve<SA, SB, SC, SD, SE>(
        &mut self,
        b: &mut Matrix<T, D, U1, SA>,
        weight: &Matrix<T, D, U1, SB>,
        _ycur: &Matrix<T, D, U1, SC>,
        _ypcur: &Matrix<T, D, U1, SD>,
        _rescur: &Matrix<T, D, U1, SE>,
    ) where
        SA: StorageMut<T, D>,
        SB: Storage<T, D>,
        SC: Storage<T, D>,
        SD: Storage<T, D>,
        SE: Storage<T, D>,
    {
        // Retrieve the LS type
        let ls_type = self.ls.get_type();

        // If the linear solver is iterative: set convergence test constant tol, in terms of the Newton convergence test
        // constant epsNewt and safety factors. The factor nrmlfac assures that the convergence test is applied to the
        // WRMS norm of the residual vector, rather than the weighted L2 norm.

        let tol = match ls_type {
            LSolverType::Iterative | LSolverType::MatrixIterative => {
                //TODO(fix)
                self.nrmfac * self.eplifac //* self.ida_eps_newt
            }
            _ => T::zero(),
        };

        /* Set vectors ycur, ypcur and rcur for use by the Atimes and Psolve interface routines */
        //self.ycur  = ycur;
        //self.ypcur = ypcur;
        //self.rcur  = rescur;

        // Set initial guess x = 0 to LS
        self.x.fill(T::zero());

        // Set scaling vectors for LS to use (if applicable)
        self.ls.set_scaling_vectors(weight, weight);
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
        let retval = self.ls.solve(&self.mat_j, &mut self.x, &b, tol);

        // Copy appropriate result to b (depending on solver type)
        if let LSolverType::Iterative | LSolverType::MatrixIterative = ls_type {
            // Retrieve solver statistics
            let nli_inc = self.ls.num_iters();

            // Copy x (or preconditioned residual vector if no iterations required) to b
            if nli_inc == 0 {
                //N_VScale(ONE, SUNLinSolResid(self.gLS), b);
            } else {
                b.copy_from(&self.x);
            }

            // Increment nli counter
            self.counters.nli += nli_inc;
        } else {
            // Copy x to b
            b.copy_from(&self.x);
        }

        // If using a direct or matrix-iterative solver, scale the correction to account for change in cj
        if let LSolverType::Direct | LSolverType::MatrixIterative = ls_type {
            if self.ida_cjratio != T::one() {
                *b *= T::two() / (T::one() + self.ida_cjratio);
            }
        }

        // Increment ncfl counter
        if retval.is_err() {
            self.counters.ncfl += 1;
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
            self.counters.nst0 = counters.ida_nst;
            self.counters.nni0 = counters.ida_nni;
            self.counters.ncfn0 = counters.ida_ncfn;
            self.counters.ncfl0 = self.counters.ncfl;
            self.counters.nwarn = 0;
            return;
        }

        // Compute statistics since last call
        //
        // Note: the performance monitor that checked whether the average
        // number of linear iterations was too close to maxl has been
        // removed, since the 'maxl' value is no longer owned by the
        // IDALs interface.

        let nstd = counters.ida_nst - self.counters.nst0;
        let nnid = counters.ida_nni - self.counters.nni0;
        if nstd == 0 || nnid == 0 {
            return;
        };

        let rcfn = (counters.ida_ncfn - self.counters.ncfn0) as f64 / (nstd as f64);
        let rcfl = (self.counters.ncfl - self.counters.ncfl0) as f64 / (nnid as f64);
        let lcfn = rcfn > 0.9;
        let lcfl = rcfl > 0.9;
        if !(lcfn || lcfl) {
            return;
        }
        self.counters.nwarn += 1;
        if self.counters.nwarn > 10 {
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
