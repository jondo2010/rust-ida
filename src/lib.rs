//! The `ida` crate is a pure Rust port of the Implicit Differential-Algebraic solver from the Sundials suite.
//!
//! IDA is a general purpose solver for the initial value problem (IVP) for systems of
//! differential-algebraic equations (DAEs). The name IDA stands for Implicit
//! Differential-Algebraic solver.

mod constants;
mod error;
mod ida_io;
mod ida_ls;
mod ida_nls;

pub mod linear;
pub mod nonlinear;
mod norm_rms;
pub mod tol_control;
pub mod traits;
use constants::*;
use error::{IdaError, Recoverable};
use ida_nls::IdaNLProblem;
use norm_rms::{NormRms, NormRmsMasked};
use tol_control::TolControl;
use traits::*;

use profiler::profile_scope;

use log::{error, trace};
use ndarray::{prelude::*, s, Slice};
use num_traits::{
    cast::{NumCast, ToPrimitive},
    identities::{One, Zero},
    Float,
};

#[derive(Copy, Clone)]
pub enum IdaTask {
    Normal,
    OneStep,
}

pub enum IdaSolveStatus {
    ContinueSteps,
    Success,
    TStop,
    Root,
}

/// Counters
#[derive(Debug, Clone)]
pub struct IdaCounters {
    /// number of internal steps taken
    ida_nst: usize,
    /// number of function (res) calls
    ida_nre: usize,
    /// number of corrector convergence failures
    ida_ncfn: usize,
    /// number of error test failures
    ida_netf: usize,
    /// number of Newton iterations performed
    ida_nni: usize,
}

/// This structure contains fields to keep track of problem state.
#[derive(Debug, Clone)]
pub struct Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
{
    //ida_itol: ToleranceType,
    /// relative tolerance
    //ida_rtol: P::Scalar,
    /// scalar absolute tolerance
    //ida_Satol: P::Scalar,
    /// vector absolute tolerance
    //ida_Vatol: Array1<P::Scalar>,
    ida_setup_done: bool,
    tol_control: TolC,

    /// constraints vector present: do constraints calc
    ida_constraints_set: bool,
    /// SUNTRUE means suppress algebraic vars in local error tests
    ida_suppressalg: bool,

    // Divided differences array and associated minor arrays
    /// phi = (maxord+1) arrays of divided differences
    ida_phi: Array<P::Scalar, Ix2>,
    /// differences in t (sums of recent step sizes)
    ida_psi: Array1<P::Scalar>,
    /// ratios of current stepsize to psi values
    ida_alpha: Array1<P::Scalar>,
    /// ratios of current to previous product of psi's
    ida_beta: Array1<P::Scalar>,
    /// product successive alpha values and factorial
    ida_sigma: Array1<P::Scalar>,
    /// sum of reciprocals of psi values
    ida_gamma: Array1<P::Scalar>,

    // Vectors
    /// error weight vector
    //ida_ewt: Array<P::Scalar, Ix1>,
    /// residual vector
    ida_delta: Array<P::Scalar, Ix1>,
    /// bit vector for diff./algebraic components
    ida_id: Array<bool, Ix1>,
    /// vector of inequality constraint options
    ida_constraints: Array<P::Scalar, Ix1>,
    /// accumulated corrections to y vector, but set equal to estimated local errors upon successful return
    ida_ee: Array<P::Scalar, Ix1>,

    //ida_mm;          /* mask vector in constraints tests (= tempv2)    */
    //ida_tempv1;      /* work space vector                              */
    //ida_tempv2;      /* work space vector                              */
    //ida_tempv3;      /* work space vector                              */
    //ida_ynew;        /* work vector for y in IDACalcIC (= tempv2)      */
    //ida_ypnew;       /* work vector for yp in IDACalcIC (= ee)         */
    //ida_delnew;      /* work vector for delta in IDACalcIC (= phi[2])  */
    //ida_dtemp;       /* work vector in IDACalcIC (= phi[3])            */

    // Tstop information
    ida_tstop: Option<P::Scalar>,

    // Step Data
    /// current BDF method order
    ida_kk: usize,
    /// method order used on last successful step
    ida_kused: usize,
    /// order for next step from order decrease decision
    ida_knew: usize,
    /// flag to trigger step doubling in first few steps
    ida_phase: usize,
    /// counts steps at fixed stepsize and order
    ida_ns: usize,

    /// initial step
    ida_hin: P::Scalar,
    /// actual initial stepsize
    ida_h0u: P::Scalar,
    /// current step size h
    ida_hh: P::Scalar,
    /// step size used on last successful step
    ida_hused: P::Scalar,
    /// rr = hnext / hused
    ida_rr: P::Scalar,

    /// value of tret previously returned by IDASolve
    ida_tretlast: P::Scalar,
    /// current value of scalar (-alphas/hh) in Jacobian
    //pub(super) ida_cj: P::Scalar,
    /// cj value saved from last successful step
    ida_cjlast: P::Scalar,

    /// test constant in Newton convergence test
    ida_eps_newt: P::Scalar,

    /// coeficient of the Newton covergence test
    ida_epcon: P::Scalar,

    // Limits
    /// max numer of convergence failures
    ida_maxncf: u64,
    /// max number of error test failures
    ida_maxnef: u64,
    /// max value of method order k:
    ida_maxord: usize,
    /// value of maxord used when allocating memory
    //ida_maxord_alloc: u64,
    /// max number of internal steps for one user call
    ida_mxstep: u64,
    /// inverse of max. step size hmax (default = 0.0)
    ida_hmax_inv: P::Scalar,

    //// Counters
    counters: IdaCounters,

    /// number of function (res) calls
    //pub(super) ida_nre: u64,
    /// number of corrector convergence failures
    //ida_ncfn: u64,
    /// number of error test failures
    //ida_netf: u64,
    /// number of Newton iterations performed
    //ida_nni: u64,

    // Arrays for Fused Vector Operations
    ida_cvals: Array1<P::Scalar>,
    ida_dvals: Array1<P::Scalar>,

    /// tolerance scale factor (saved value)
    ida_tolsf: P::Scalar,

    // Rootfinding Data

    //IDARootFn ida_gfun;       /* Function g for roots sought                     */
    /// number of components of g
    ida_nrtfn: usize,
    //int *ida_iroots;          /* array for root information                      */
    //int *ida_rootdir;         /* array specifying direction of zero-crossing     */
    /// nearest endpoint of interval in root search
    //ida_tlo: P::Scalar,
    /// farthest endpoint of interval in root search
    //ida_thi: P::Scalar,
    /// t return value from rootfinder routine
    //ida_trout: P::Scalar,
    /// saved array of g values at t = tlo
    //ida_glo: Array1<P::Scalar>,
    //realtype *ida_ghi;        /* saved array of g values at t = thi              */
    //realtype *ida_grout;      /* array of g values at t = trout                  */
    //realtype ida_toutc;       /* copy of tout (if NORMAL mode)                   */
    /// tolerance on root location
    //ida_ttol: P::Scalar,
    //int ida_taskc;            /* copy of parameter itask                         */
    //int ida_irfnd;            /* flag showing whether last step had a root       */
    /// counter for g evaluations
    ida_nge: u64,
    //booleantype *ida_gactive; /* array with active/inactive event functions      */
    //int ida_mxgnull;          /* number of warning messages about possible g==0  */

    // Arrays for Fused Vector Operations
    ida_zvecs: Array<P::Scalar, Ix2>,

    /// Nonlinear Solver
    nls: NLS,

    /// Nonlinear problem
    nlp: IdaNLProblem<P, LS>,
}

impl<P, LS, NLS, TolC> Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
    <P as ModelSpec>::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + std::fmt::LowerExp
        + IdaConst<Scalar = P::Scalar>,
{
    /// Creates a new IdaProblem given a ModelSpec, initial Arrays of yy0 and yyp
    ///
    /// *Panics" if ModelSpec::Scalar is unable to convert any constant initialization value.
    pub fn new(
        problem: P,
        yy0: Array<P::Scalar, Ix1>,
        yp0: Array<P::Scalar, Ix1>,
        tol_control: TolC,
    ) -> Self {
        assert_eq!(problem.model_size(), yy0.len());

        // Initialize the phi array
        let mut ida_phi = Array::zeros(problem.model_size())
            .broadcast([&[MXORDP1], yy0.shape()].concat())
            .unwrap()
            .into_dimensionality::<_>()
            .unwrap()
            .to_owned();

        ida_phi.index_axis_mut(Axis(0), 0).assign(&yy0);
        ida_phi.index_axis_mut(Axis(0), 1).assign(&yp0);

        //IDAResFn res, realtype t0, N_Vector yy0, N_Vector yp0
        Self {
            ida_setup_done: false,

            // Set unit roundoff in IDA_mem
            // NOTE: Use P::Scalar::epsilon() instead!
            //ida_uround: UNIT_ROUNDOFF,
            tol_control: tol_control,

            // Set default values for integrator optional inputs
            //ida_ehfun       = IDAErrHandler;
            //ida_eh_data     = IDA_mem;
            //ida_errfp       = stderr;
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as u64,
            ida_hmax_inv: NumCast::from(HMAX_INV_DEFAULT).unwrap(),
            ida_hin: P::Scalar::zero(),
            ida_eps_newt: P::Scalar::zero(),
            ida_epcon: NumCast::from(EPCON).unwrap(),
            ida_maxnef: MXNEF as u64,
            ida_maxncf: MXNCF as u64,
            ida_suppressalg: false,
            //ida_id          = NULL;
            ida_constraints: Array::zeros(problem.model_size()),
            ida_constraints_set: false,

            ida_cjlast: P::Scalar::zero(),

            // set the saved value maxord_alloc
            //ida_maxord_alloc = MAXORD_DEFAULT;

            // Set default values for IC optional inputs
            //ida_epiccon : P::Scalar::from(0.01 * EPCON).unwrap(),
            //ida_maxnh   : MAXNH,
            //ida_maxnj   = MAXNJ;
            //ida_maxnit  = MAXNI;
            //ida_maxbacks  = MAXBACKS;
            //ida_lsoff   = SUNFALSE;
            //ida_steptol = SUNRpowerR(self.ida_uround, TWOTHIRDS);

            /* Initialize lrw and liw */
            //ida_lrw = 25 + 5*MXORDP1;
            //ida_liw = 38;
            ida_phi: ida_phi,

            ida_psi: Array::zeros(MXORDP1),
            ida_alpha: Array::zeros(MXORDP1),
            ida_beta: Array::zeros(MXORDP1),
            ida_sigma: Array::zeros(MXORDP1),
            ida_gamma: Array::zeros(MXORDP1),

            ida_delta: Array::zeros(problem.model_size()),
            ida_id: Array::from_elem(problem.model_size(), false),

            // Initialize all the counters and other optional output values
            counters: IdaCounters {
                ida_nst: 0,
                ida_ncfn: 0,
                ida_netf: 0,
                ida_nni: 0,

                ida_nre: 0,
            },

            ida_kused: 0,
            ida_hused: P::Scalar::zero(),
            ida_tolsf: P::Scalar::one(),
            ida_nge: 0,

            //ida_irfnd = 0;

            // Initialize root-finding variables

            //ida_glo: Array::zeros(),
            //ida_glo     = NULL;
            //ida_ghi     = NULL;
            //ida_grout   = NULL;
            //ida_iroots  = NULL;
            //ida_rootdir = NULL;
            //ida_gfun    = NULL;
            ida_nrtfn: 0,
            //ida_gactive  = NULL;
            //ida_mxgnull  = 1;

            // Not from ida.c...
            ida_ee: Array::zeros(problem.model_size()),

            ida_tstop: None,

            ida_kk: 0,
            ida_knew: 0,
            ida_phase: 0,
            ida_ns: 0,

            ida_rr: P::Scalar::zero(),
            ida_tretlast: P::Scalar::zero(),
            ida_h0u: P::Scalar::zero(),
            ida_hh: P::Scalar::zero(),
            //ida_hused: <P::Scalar as AssociatedReal>::Real::from_f64(0.0),
            ida_cvals: Array::zeros(MXORDP1),
            ida_dvals: Array::zeros(MAXORD_DEFAULT),

            ida_zvecs: Array::zeros((MXORDP1, yy0.shape()[0])),

            //ida_rtol: P::Scalar::zero(),
            //ida_Satol: P::Scalar::zero(),
            //ida_Vatol: Array::zeros(MXORDP1),

            // Initialize nonlinear solver
            nls: NLS::new(yy0.len(), MAXNLSIT),
            nlp: IdaNLProblem::new(problem),
        }
    }

    //-----------------------------------------------------------------
    // Main solver function
    //-----------------------------------------------------------------

    /// This routine is the main driver of the IDA package.
    ///
    /// This is the central step in the solution process, the call to perform the integration of
    /// the DAE. One of the input arguments (itask) specifies one of two modes as to where ida is
    /// to return a solution. But these modes are modified if the user has set a stop time (with
    /// `set_stop_time()`) or requested rootfinding.
    ///
    /// It integrates over an independent variable interval defined by the user, by calling `step()`
    /// to take internal independent variable steps.
    ///
    /// The first time that `solve()` is called for a successfully initialized problem, it computes
    /// a tentative initial step size.
    ///
    /// `solve()` supports two modes, specified by `itask`:
    /// * In the `Normal` mode, the solver steps until it passes `tout` and then interpolates to obtain
    /// `y(tout)` and `yp(tout)`.
    /// * In the `OneStep` mode, it takes one internal step and returns.
    ///
    /// # Arguments
    ///
    /// * `tout` The next time at which a computed solution is desired.
    /// * `tret` The time reached by the solver (output).
    /// * `itask` A flag indicating the job of the solver for the next user step. The IDA NORMAL task is to have the solver take internal steps until it has reached or just passed the user specified tout parameter. The solver then interpolates in order to return approximate values of y(tout) and ˙y(tout). The IDA ONE STEP option tells the solver to just take one internal step and return the solution at the point reached by that step
    ///
    /// # Returns
    ///
    /// * `IdaSolveStatus::Success` - general success.
    /// * `IdaSolveStatus::TStop` - `solve()` succeeded by reaching the stop point specified through
    /// the optional input function `set_stop_time()`.
    /// * `IdaSolveStatus::Root` -  `solve()` succeeded and found one or more roots. In this case,
    /// tret is the location of the root. If nrtfn > 1, call `get_root_info()` to see which gi were
    /// found to have a root.
    ///
    /// # Errors
    ///
    /// IDA_ILL_INPUT
    /// IDA_TOO_MUCH_WORK
    /// IDA_MEM_NULL
    /// IDA_TOO_MUCH_ACC
    /// IDA_CONV_FAIL
    /// IDA_LSETUP_FAIL
    /// IDA_LSOLVE_FAIL
    /// IDA_CONSTR_FAIL
    /// IDA_ERR_FAIL
    /// IDA_REP_RES_ERR
    /// IDA_RES_FAIL
    pub fn solve(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        profile_scope!(format!("solve(tout={:?})", tout));

        if self.counters.ida_nst == 0 {
            // This is the first call

            // Check inputs to IDA for correctness and consistency */
            if !self.ida_setup_done {
                self.initial_setup();
                self.ida_setup_done = true;
            }

            // On first call, check for tout - tn too small, set initial hh, check for approach to
            // tstop, and scale phi[1] by hh.
            // Also check for zeros of root function g at and near t0.

            let tdist = (tout - self.nlp.ida_tn).abs();
            if tdist == P::Scalar::zero() {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }
            let troundoff =
                P::Scalar::two() * P::Scalar::epsilon() * (self.nlp.ida_tn.abs() + tout.abs());
            if tdist < troundoff {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }

            self.ida_hh = self.ida_hin;
            if (self.ida_hh != P::Scalar::zero())
                && ((tout - self.nlp.ida_tn) * self.ida_hh < P::Scalar::zero())
            {
                Err(IdaError::IllegalInput {
                    msg: format!("Initial step is not towards tout."),
                })?
            }

            if self.ida_hh == P::Scalar::zero() {
                self.ida_hh = P::Scalar::pt001() * tdist;
                let ypnorm = self.wrms_norm(
                    &self.ida_phi.index_axis(Axis(0), 1),
                    &self.nlp.ida_ewt,
                    self.ida_suppressalg,
                );
                if ypnorm > P::Scalar::two() / self.ida_hh {
                    self.ida_hh = P::Scalar::half() / ypnorm;
                }
                if tout < self.nlp.ida_tn {
                    self.ida_hh = -self.ida_hh;
                }
            }

            let rh = (self.ida_hh).abs() * self.ida_hmax_inv;
            if rh > P::Scalar::one() {
                self.ida_hh /= rh;
            }

            if let Some(tstop) = self.ida_tstop {
                if (tstop - self.nlp.ida_tn) * self.ida_hh <= P::Scalar::zero() {
                    Err(IdaError::IllegalInput {
                        msg: format!(
                            "The value tstop = {:?} \
                             is behind current t = {:?} \
                             in the direction of integration.",
                            tstop, self.nlp.ida_tn
                        ),
                    })?
                }
                if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                    self.ida_hh = (tstop - self.nlp.ida_tn)
                        * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                }
            }

            self.ida_h0u = self.ida_hh;
            self.ida_kk = 0;
            self.ida_kused = 0; // set in case of an error return before a step

            /* Check for exact zeros of the root functions at or near t0. */
            //if self.ida_nrtfn > 0 {
            self.r_check1()?;
            //  ier = IDARcheck1(IDA_mem);
            //  if (ier == IDA_RTFUNC_FAIL) {
            //    IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck1", MSG_RTFUNC_FAILED, self.nlp.ida_tn);
            //    return(IDA_RTFUNC_FAIL);
            //  }
            //}

            //N_VScale(self.ida_hh, self.ida_phi[1], self.ida_phi[1]);  /* set phi[1] = hh*y' */
            let mut phi = self.ida_phi.index_axis_mut(Axis(0), 1);
            phi *= self.ida_hh;

            // Set the convergence test constants epsNewt and toldel
            self.ida_eps_newt = self.ida_epcon;
            self.nlp.ida_toldel = P::Scalar::pt0001() * self.ida_eps_newt;
        } // end of first-call block.

        // Call lperf function and set nstloc for later performance testing.
        self.nlp.lp.ls_perf(&self.counters, false);
        let mut nstloc = 0;

        // If not the first call, perform all stopping tests.

        if self.counters.ida_nst > 0 {
            // First, check for a root in the last step taken, other than the last root found, if
            // any.  If itask = IDA_ONE_STEP and y(tn) was not returned because of an intervening
            // root, return y(tn) now.

            if self.ida_nrtfn > 0 {
                /*

                irfndp = self.ida_irfnd;

                ier = IDARcheck2(IDA_mem);

                if (ier == CLOSERT) {
                  IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDARcheck2", MSG_CLOSE_ROOTS, self.ida_tlo);
                  return(IDA_ILL_INPUT);
                } else if (ier == IDA_RTFUNC_FAIL) {
                  IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck2", MSG_RTFUNC_FAILED, self.ida_tlo);
                  return(IDA_RTFUNC_FAIL);
                } else if (ier == RTFOUND) {
                  self.ida_tretlast = *tret = self.ida_tlo;
                  return(IDA_ROOT_RETURN);
                }

                /* If tn is distinct from tretlast (within roundoff),
                   check remaining interval for roots */
                troundoff = HUNDRED * self.ida_uround * (SUNRabs(self.nlp.ida_tn) + SUNRabs(self.ida_hh));
                if ( SUNRabs(self.nlp.ida_tn - self.ida_tretlast) > troundoff ) {
                ier = IDARcheck3(IDA_mem);
                if (ier == IDA_SUCCESS) {     /* no root found */
                self.ida_irfnd = 0;
                if ((irfndp == 1) && (itask == IDA_ONE_STEP)) {
                self.ida_tretlast = *tret = self.nlp.ida_tn;
                ier = IDAGetSolution(IDA_mem, self.nlp.ida_tn, yret, ypret);
                return(IDA_SUCCESS);
                }
                } else if (ier == RTFOUND) {  /* a new root was found */
                self.ida_irfnd = 1;
                self.ida_tretlast = *tret = self.ida_tlo;
                return(IDA_ROOT_RETURN);
                } else if (ier == IDA_RTFUNC_FAIL) {  /* g failed */
                IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck3", MSG_RTFUNC_FAILED, self.ida_tlo);
                return(IDA_RTFUNC_FAIL);
                }
                }

                */
            } // end of root stop check

            // Now test for all other stop conditions.

            let istate = self.stop_test1(tout, tret, itask);
            match istate {
                Err(_)
                | Ok(IdaSolveStatus::Root)
                | Ok(IdaSolveStatus::Success)
                | Ok(IdaSolveStatus::TStop) => {
                    return istate;
                }
                _ => {}
            }
        }

        // Looping point for internal steps.
        profile_scope!(format!("solve loop"));
        loop {
            // Check for too many steps taken.

            if (self.ida_mxstep > 0) && (nstloc >= self.ida_mxstep) {
                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                // Here yy=yret and yp=ypret already have the current solution.
                Err(IdaError::IllegalInput {
                    msg: format!(
                        "At t = {:.5e}, mxstep steps ({}) taken before reaching tout.",
                        self.nlp.ida_tn, self.ida_mxstep
                    ),
                })?
                //istate = IDA_TOO_MUCH_WORK;
            }

            // Call lperf to generate warnings of poor performance.
            self.nlp.lp.ls_perf(&self.counters, true);

            // Reset and check ewt (if not first call).
            if self.counters.ida_nst > 0 {
                self.tol_control.ewt_set(
                    self.ida_phi.index_axis(Axis(0), 0),
                    self.nlp.ida_ewt.view_mut(),
                );

                if self.nlp.ida_ewt.iter().any(|&x| x <= P::Scalar::zero()) {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                    *tret = self.nlp.ida_tn;
                    self.ida_tretlast = self.nlp.ida_tn;

                    //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_EWT_NOW_BAD, self.nlp.ida_tn);
                    Err(IdaError::IllegalInput {
                        msg: format!(
                            "At t = {:?} some ewt component has become <= 0.0.",
                            self.nlp.ida_tn
                        ),
                    })?
                }
            }

            // Check for too much accuracy requested.

            let nrm = self.wrms_norm(
                &self.ida_phi.index_axis(Axis(0), 0),
                &self.nlp.ida_ewt,
                self.ida_suppressalg,
            );

            /*
            trace!(
                "At t = {:.5e}, nstloc={}; nrm = {:.5e}",
                self.nlp.ida_tn,
                nstloc,
                nrm
            );
            */

            self.ida_tolsf = P::Scalar::epsilon() * nrm;
            if self.ida_tolsf > P::Scalar::one() {
                self.ida_tolsf *= P::Scalar::ten();

                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                if self.counters.ida_nst > 0 {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                }
                //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_TOO_MUCH_ACC, self.nlp.ida_tn);
                Err(IdaError::TooMuchAccuracy {
                    t: self.nlp.ida_tn.to_f64().unwrap(),
                })?
            }

            // Call IDAStep to take a step.
            let sflag = self.step();

            // Process all failed-step cases, and exit loop.
            sflag.map_err(|err| {
                let ier = self.get_solution(self.nlp.ida_tn);
                match ier {
                    Ok(_) => {
                        *tret = self.nlp.ida_tn;
                        self.ida_tretlast = self.nlp.ida_tn;
                    }
                    Err(e2) => {
                        error!("Error occured with get_solution: {:?}", e2.as_fail());
                    }
                }
                // Forward the error
                err
            })?;

            nstloc += 1;

            // After successful step, check for stop conditions; continue or break.

            // First check for root in the last step taken.

            if self.ida_nrtfn > 0 {
                /*
                ier = IDARcheck3(IDA_mem);

                if (ier == RTFOUND) {  /* A new root was found */
                self.ida_irfnd = 1;
                istate = IDA_ROOT_RETURN;
                self.ida_tretlast = *tret = self.ida_tlo;
                break;
                } else if (ier == IDA_RTFUNC_FAIL) { /* g failed */
                IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck3", MSG_RTFUNC_FAILED, self.ida_tlo);
                istate = IDA_RTFUNC_FAIL;
                break;
                }

                /* If we are at the end of the first step and we still have
                 * some event functions that are inactive, issue a warning
                 * as this may indicate a user error in the implementation
                 * of the root function. */
                if (self.counters.ida_nst==1) {
                inactive_roots = SUNFALSE;
                for (ir=0; ir<self.ida_nrtfn; ir++) {
                if (!self.ida_gactive[ir]) {
                inactive_roots = SUNTRUE;
                break;
                }
                }
                if ((self.ida_mxgnull > 0) && inactive_roots) {
                IDAProcessError(IDA_mem, IDA_WARNING, "IDA", "IDASolve", MSG_INACTIVE_ROOTS);
                }
                }
                 */
            }

            // Now check all other stop conditions.

            let istate = self.stop_test2(tout, tret, itask);
            match istate {
                Err(_)
                | Ok(IdaSolveStatus::Root)
                | Ok(IdaSolveStatus::Success)
                | Ok(IdaSolveStatus::TStop) => {
                    return istate;
                }
                _ => {}
            }
        } // End of step loop

        //return(istate);
    }

    //-----------------------------------------------------------------
    // Interpolated output
    //-----------------------------------------------------------------

    /// IDAGetDky
    ///
    /// This routine evaluates the k-th derivative of y(t) as the value of
    /// the k-th derivative of the interpolating polynomial at the independent
    /// variable t, and stores the results in the vector dky.  It uses the current
    /// independent variable value, tn, and the method order last used, kused.
    ///
    /// The return values are:
    ///   IDA_SUCCESS       if t is legal
    ///   IDA_BAD_T         if t is not within the interval of the last step taken
    ///   IDA_BAD_DKY       if the dky vector is NULL
    ///   IDA_BAD_K         if the requested k is not in the range [0,order used]
    ///   IDA_VTolCTOROP_ERR  if the fused vector operation fails
    pub fn get_dky<S>(
        &mut self,
        t: P::Scalar,
        k: usize,
        dky: &mut ArrayBase<S, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S: ndarray::DataMut<Elem = P::Scalar>,
    {
        /*
        IDAMem IDA_mem;
        realtype tfuzz, tp, delt, psij_1;
        int i, j, retval;
        realtype cjk  [MXORDP1];
        realtype cjk_1[MXORDP1];
        */

        if (k < 0) || (k > self.ida_kused) {
            Err(IdaError::BadK {})?
            //IDAProcessError(IDA_mem, IDA_BAD_K, "IDA", "IDAGetDky", MSG_BAD_K);
            //return(IDA_BAD_K);
        }

        // Check t for legality.  Here tn - hused is t_{n-1}.

        let tfuzz = P::Scalar::hundred()
            * P::Scalar::epsilon()
            * (self.nlp.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();
        let tp = self.nlp.ida_tn - self.ida_hused - tfuzz;
        if (t - tp) * self.ida_hh < P::Scalar::zero() {
            Err(IdaError::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.nlp.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.nlp.ida_tn.to_f64().unwrap(),
            })?
            //IDAProcessError(IDA_mem, IDA_BAD_T, "IDA", "IDAGetDky", MSG_BAD_T, t, self.nlp.ida_tn-self.ida_hused, self.nlp.ida_tn);
            //return(IDA_BAD_T);
        }

        // Initialize the c_j^(k) and c_k^(k-1)
        let mut cjk = Array::zeros(MXORDP1);
        let mut cjk_1 = Array::zeros(MXORDP1);

        let delt = t - self.nlp.ida_tn;
        let mut psij_1 = P::Scalar::zero();

        for i in 0..k + 1 {
            let scalar_i: P::Scalar = NumCast::from(i as f64).unwrap();
            // The below reccurence is used to compute the k-th derivative of the solution:
            //    c_j^(k) = ( k * c_{j-1}^(k-1) + c_{j-1}^{k} (Delta+psi_{j-1}) ) / psi_j
            //
            //    Translated in indexes notation:
            //    cjk[j] = ( k*cjk_1[j-1] + cjk[j-1]*(delt+psi[j-2]) ) / psi[j-1]
            //
            //    For k=0, j=1: c_1 = c_0^(-1) + (delt+psi[-1]) / psi[0]
            //
            //    In order to be able to deal with k=0 in the same way as for k>0, the
            //    following conventions were adopted:
            //      - c_0(t) = 1 , c_0^(-1)(t)=0
            //      - psij_1 stands for psi[-1]=0 when j=1
            //                      for psi[j-2]  when j>1
            if i == 0 {
                cjk[i] = P::Scalar::one();
            } else {
                //                                                i       i-1          1
                // c_i^(i) can be always updated since c_i^(i) = -----  --------  ... -----
                //                                               psi_j  psi_{j-1}     psi_1
                cjk[i] = cjk[i - 1] * scalar_i / self.ida_psi[i - 1];
                psij_1 = self.ida_psi[i - 1];
            }

            // update c_j^(i)
            //j does not need to go till kused
            //for(j=i+1; j<=self.ida_kused-k+i; j++) {
            for j in i + 1..self.ida_kused - k + 1 + 1 {
                cjk[j] =
                    (scalar_i * cjk_1[j - 1] + cjk[j - 1] * (delt + psij_1)) / self.ida_psi[j - 1];
                psij_1 = self.ida_psi[j - 1];
            }

            // save existing c_j^(i)'s
            //for(j=i+1; j<=self.ida_kused-k+i; j++)
            for j in i + 1..self.ida_kused - k + 1 + 1 {
                cjk_1[j] = cjk[j];
            }
        }

        // Compute sum (c_j(t) * phi(t))
        // Sum j=k to j<=self.ida_kused
        //retval = N_VLinearCombination( self.ida_kused - k + 1, cjk + k, self.ida_phi + k, dky);
        let phi = self
            .ida_phi
            .slice_axis(Axis(0), Slice::from(k..self.ida_kused + 1));

        // We manually broadcast here so we can turn it into a column vec
        let cvals = cjk.slice(s![k..self.ida_kused + 1]);
        let cvals = cvals
            .broadcast((phi.len_of(Axis(1)), phi.len_of(Axis(0))))
            .unwrap()
            .reversed_axes();

        dky.assign(&(&phi * &cvals).sum_axis(Axis(0)));

        Ok(())
    }

    /// IDAInitialSetup
    ///
    /// This routine is called by `solve` once at the first step. It performs all checks on optional
    /// inputs and inputs to `init`/`reinit` that could not be done before.
    ///
    /// If no error is encountered, IDAInitialSetup returns IDA_SUCCESS. Otherwise, it returns an error flag and reported to the error handler function.
    fn initial_setup(&mut self) {
        //booleantype conOK;
        //int ier;

        // Initial error weight vector
        self.tol_control.ewt_set(
            self.ida_phi.index_axis(Axis(0), 0),
            self.nlp.ida_ewt.view_mut(),
        );

        /*
        // Check to see if y0 satisfies constraints.
        if (IDA_mem->ida_constraintsSet) {
          conOK = N_VConstrMask(IDA_mem->ida_constraints, IDA_mem->ida_phi[0], IDA_mem->ida_tempv2);
          if (!conOK) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_Y0_FAIL_CONSTR);
            return(IDA_ILL_INPUT);
          }
        }

        // Call linit function if it exists.
        if (IDA_mem->ida_linit != NULL) {
          ier = IDA_mem->ida_linit(IDA_mem);
          if (ier != 0) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_LINIT_FAIL);
            return(IDA_LINIT_FAIL);
          }
        }
        */

        //return(IDA_SUCCESS);
    }

    //-----------------------------------------------------------------
    //Stopping tests
    //-----------------------------------------------------------------

    /// IDAStopTest1
    ///
    /// This routine tests for stop conditions before taking a step.
    /// The tests depend on the value of itask.
    /// The variable tretlast is the previously returned value of tret.
    ///
    /// The return values are:
    /// CONTINUE_STEPS       if no stop conditions were found
    /// IDA_SUCCESS          for a normal return to the user
    /// IDA_TSTOP_RETURN     for a tstop-reached return to the user
    /// IDA_ILL_INPUT        for an illegal-input return to the user
    ///
    /// In the tstop cases, this routine may adjust the stepsize hh to cause
    /// the next step to reach tstop exactly.
    fn stop_test1(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        if let Some(tstop) = self.ida_tstop {
            // Test for tn past tstop, tn = tretlast, tn past tout, tn near tstop.
            if ((self.nlp.ida_tn - tstop) * self.ida_hh) > P::Scalar::zero() {
                Err(IdaError::BadStopTime {
                    tstop: tstop.to_f64().unwrap(),
                    t: self.nlp.ida_tn.to_f64().unwrap(),
                })?
            }
        }
        match itask {
            IdaTask::Normal => {
                // Test for tout = tretlast, and for tn past tout.
                if tout == self.ida_tretlast {
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if (self.nlp.ida_tn - tout) * self.ida_hh >= P::Scalar::zero() {
                    self.get_solution(tout)?;
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        self.get_solution(tstop)
                            .map_err(|_| IdaError::BadStopTime {
                                tstop: tstop.to_f64().unwrap(),
                                t: self.nlp.ida_tn.to_f64().unwrap(),
                            })?;

                        self.ida_tretlast = tstop;
                        *tret = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }

            IdaTask::OneStep => {
                // Test for tn past tretlast.
                if (self.nlp.ida_tn - self.ida_tretlast) * self.ida_hh > P::Scalar::zero() {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                    self.ida_tretlast = self.nlp.ida_tn;
                    *tret = self.nlp.ida_tn;
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        self.get_solution(tstop)?;
                        self.ida_tretlast = tstop;
                        *tret = tstop;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }
        }
    }

    /// IDAStopTest2
    ///
    /// This routine tests for stop conditions after taking a step.
    /// The tests depend on the value of itask.
    ///
    /// The return values are:
    ///  CONTINUE_STEPS     if no stop conditions were found
    ///  IDA_S>UCCESS        for a normal return to the user
    ///  IDA_TSTOP_RETURN   for a tstop-reached return to the user
    ///  IDA_ILL_INPUT      for an illegal-input return to the user
    ///
    /// In the two cases with tstop, this routine may reset the stepsize hh
    /// to cause the next step to reach tstop exactly.
    ///
    /// In the two cases with ONE_STEP mode, no interpolation to tn is needed
    /// because yret and ypret already contain the current y and y' values.
    ///
    /// Note: No test is made for an error return from IDAGetSolution here,
    /// because the same test was made prior to the step.
    fn stop_test2(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        match itask {
            IdaTask::Normal => {
                // Test for tn past tout.
                if (self.nlp.ida_tn - tout) * self.ida_hh >= P::Scalar::zero() {
                    // /* ier = */ IDAGetSolution(IDA_mem, tout, yret, ypret);
                    *tret = tout;
                    self.ida_tretlast = tout;
                    let _ier = self.get_solution(tout);
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        let _ier = self.get_solution(tstop);
                        *tret = tstop;
                        self.ida_tretlast = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                return Ok(IdaSolveStatus::ContinueSteps);
            }

            IdaTask::OneStep => {
                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        let _ier = self.get_solution(tstop);
                        *tret = tstop;
                        self.ida_tretlast = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }
                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                return Ok(IdaSolveStatus::Success);
            }
        }
    }

    /// This routine performs one internal IDA step, from `tn` to `tn + hh`. It calls other
    /// routines to do all the work.
    ///
    /// It solves a system of differential/algebraic equations of the form `F(t,y,y') = 0`, for
    /// one step.
    /// In IDA, `tt` is used for `t`, `yy` is used for `y`, and `yp` is used for `y'`. The function
    /// F is supplied as 'res' by the user.
    ///
    /// The methods used are modified divided difference, fixed leading coefficient forms of
    /// backward differentiation formulas. The code adjusts the stepsize and order to control the
    /// local error per step.
    ///
    /// The main operations done here are as follows:
    /// * initialize various quantities;
    /// * setting of multistep method coefficients;
    /// * solution of the nonlinear system for yy at t = tn + hh;
    /// * deciding on order reduction and testing the local error;
    /// * attempting to recover from failure in nonlinear solver or error test;
    /// * resetting stepsize and order for the next step.
    /// * updating phi and other state data if successful;
    ///
    /// On a failure in the nonlinear system solution or error test, the step may be reattempted,
    /// depending on the nature of the failure.
    ///
    /// Variables or arrays (all in the IDAMem structure) used in IDAStep are:
    ///
    /// tt -- Independent variable.
    /// yy -- Solution vector at tt.
    /// yp -- Derivative of solution vector after successful stelp.
    /// res -- User-supplied function to evaluate the residual. See the description given in file ida.h
    /// lsetup -- Routine to prepare for the linear solver call. It may either save or recalculate
    ///   quantities used by lsolve. (Optional)
    /// lsolve -- Routine to solve a linear system. A prior call to lsetup may be required.
    /// hh  -- Appropriate step size for next step.
    /// ewt -- Vector of weights used in all convergence tests.
    /// phi -- Array of divided differences used by IDAStep. This array is composed of (maxord+1)
    ///   nvectors (each of size Neq). (maxord+1) is the maximum order for the problem, maxord, plus 1.
    ///
    /// Return values are:
    ///       IDA_SUCCESS   IDA_RES_FAIL      LSETUP_ERROR_NONRTolCVR
    ///                     IDA_LSOLVE_FAIL   IDA_ERR_FAIL
    ///                     IDA_CONSTR_FAIL   IDA_CONV_FAIL
    ///                     IDA_REP_RES_ERR
    fn step(&mut self) -> Result<(), failure::Error> {
        profile_scope!(format!("step(), nst={}", self.counters.ida_nst));
        trace!(
            "step(), nst={}, tn={:.6e}",
            self.counters.ida_nst,
            self.nlp.ida_tn
        );

        let saved_t = self.nlp.ida_tn;

        if self.counters.ida_nst == 0 {
            self.ida_kk = 1;
            self.ida_kused = 0;
            self.ida_hused = P::Scalar::one();
            self.ida_psi[0] = self.ida_hh;
            self.nlp.lp.ida_cj = self.ida_hh.recip();
            self.ida_phase = 0;
            self.ida_ns = 0;
        }

        let mut ncf = 0; // local counter for convergence failures
        let mut nef = 0; // local counter for error test failures

        // Looping point for attempts to take a step

        let (ck, err_k, err_km1) = loop {
            //-----------------------
            // Set method coefficients
            //-----------------------

            let ck = self.set_coeffs();

            //kflag = IDA_SUCCESS;

            //----------------------------------------------------
            // If tn is past tstop (by roundoff), reset it to tstop.
            //-----------------------------------------------------

            self.nlp.ida_tn += self.ida_hh;
            if let Some(tstop) = self.ida_tstop {
                if ((self.nlp.ida_tn - tstop) * self.ida_hh) > P::Scalar::one() {
                    self.nlp.ida_tn = tstop;
                }
            }

            //-----------------------
            // Advance state variables
            //-----------------------

            // Compute predicted values for yy and yp
            self.predict();

            // Nonlinear system solution
            let (err_k, err_km1, converged) = self
                .nonlinear_solve()
                .map_err(|err| (P::Scalar::zero(), P::Scalar::zero(), err))
                .and_then(|_| {
                    // If NLS was successful, perform error test
                    self.test_error(ck)
                })
                // Test for convergence or error test failures
                .or_else(|(err_k, err_km1, err)| {
                    // restore and decide what to do
                    self.restore(saved_t);

                    self.handle_n_flag(err, err_k, err_km1, &mut ncf, &mut nef)
                        .map(|_| {
                            // recoverable error; predict again
                            if self.counters.ida_nst == 0 {
                                self.reset();
                            }

                            (err_k, err_km1, false)
                        })
                })?;

            if converged {
                break (ck, err_k, err_km1);
            }
        };

        // Nonlinear system solve and error test were both successful;
        // update data, and consider change of step and/or order

        self.complete_step(err_k, err_km1);

        //  Rescale ee vector to be the estimated local error
        //  Notes:
        //    (1) altering the value of ee is permissible since it will be overwritten by
        //        solve()->step()->nonlinear_solve() before it is needed again
        //    (2) the value of ee is only valid if IDAHandleNFlag() returns either
        //        PREDICT_AGAIN or IDA_SUCCESS

        self.ida_ee *= ck;

        Ok(())
    }

    /// This routine computes the coefficients relevant to the current step.
    ///
    /// The counter ns counts the number of consecutive steps taken at constant stepsize h and order
    /// k, up to a maximum of k + 2.
    /// Then the first ns components of beta will be one, and on a step with ns = k + 2, the
    /// coefficients alpha, etc. need not be reset here.
    /// Also, complete_step() prohibits an order increase until ns = k + 2.
    ///
    /// Returns the 'variable stepsize error coefficient ck'
    fn set_coeffs(&mut self) -> P::Scalar {
        profile_scope!(format!("set_coeffs()"));

        // Set coefficients for the current stepsize h
        if (self.ida_hh != self.ida_hused) || (self.ida_kk != self.ida_kused) {
            self.ida_ns = 0;
        }
        self.ida_ns = std::cmp::min(self.ida_ns + 1, self.ida_kused + 2);
        if self.ida_kk + 1 >= self.ida_ns {
            self.ida_beta[0] = P::Scalar::one();
            self.ida_alpha[0] = P::Scalar::one();
            let mut temp1 = self.ida_hh;
            self.ida_gamma[0] = P::Scalar::zero();
            self.ida_sigma[0] = P::Scalar::one();
            for i in 1..self.ida_kk + 1 {
                let scalar_i: P::Scalar = NumCast::from(i).unwrap();
                let temp2 = self.ida_psi[i - 1];
                self.ida_psi[i - 1] = temp1;
                self.ida_beta[i] = self.ida_beta[i - 1] * (self.ida_psi[i - 1] / temp2);
                temp1 = temp2 + self.ida_hh;
                self.ida_alpha[i] = self.ida_hh / temp1;
                self.ida_sigma[i] = self.ida_sigma[i - 1] * self.ida_alpha[i] * scalar_i;
                self.ida_gamma[i] = self.ida_gamma[i - 1] + self.ida_alpha[i - 1] / self.ida_hh;
            }
            self.ida_psi[self.ida_kk] = temp1;
        }
        // compute alphas, alpha0
        let mut alphas = P::Scalar::zero();
        let mut alpha0 = P::Scalar::zero();
        for i in 0..self.ida_kk {
            let scalar_i: P::Scalar = NumCast::from(i + 1).unwrap();
            alphas -= P::Scalar::one() / scalar_i;
            alpha0 -= self.ida_alpha[i];
        }

        // compute leading coefficient cj
        self.ida_cjlast = self.nlp.lp.ida_cj;
        self.nlp.lp.ida_cj = -alphas / self.ida_hh;

        // compute variable stepsize error coefficient ck
        let mut ck = (self.ida_alpha[self.ida_kk] + alphas - alpha0).abs();
        ck = ck.max(self.ida_alpha[self.ida_kk]);

        // change phi to phi-star
        // Scale i=self.ida_ns to i<=self.ida_kk
        if self.ida_ns <= self.ida_kk {
            //N_VScaleVectorArray( self.ida_kk - self.ida_ns + 1, self.ida_beta + self.ida_ns, self.ida_phi + self.ida_ns, self.ida_phi + self.ida_ns,);
            let mut phi = self
                .ida_phi
                .slice_axis_mut(Axis(0), Slice::from(self.ida_ns..self.ida_kk + 1));
            let beta = self.ida_beta.slice(s![self.ida_ns..self.ida_kk + 1]);
            let beta = beta
                .broadcast((phi.len_of(Axis(1)), phi.len_of(Axis(0))))
                .unwrap()
                .reversed_axes();
            phi *= &beta;
        }

        return ck;
    }

    /// IDANls
    /// This routine attempts to solve the nonlinear system using the linear solver specified.
    /// NOTE: this routine uses N_Vector ee as the scratch vector tempv3 passed to lsetup.
    fn nonlinear_solve(&mut self) -> Result<(), failure::Error> {
        profile_scope!(format!("nonlinear_solve()"));

        // Initialize if the first time called
        let mut call_lsetup = false;

        if self.counters.ida_nst == 0 {
            self.nlp.lp.ida_cjold = self.nlp.lp.ida_cj;
            self.nlp.ida_ss = P::Scalar::twenty();
            //if (self.ida_lsetup) { callLSetup = true; }
            call_lsetup = true;
        }

        // Decide if lsetup is to be called

        //if self.ida_lsetup {
        self.nlp.lp.ida_cjratio = self.nlp.lp.ida_cj / self.nlp.lp.ida_cjold;
        let temp1: P::Scalar = NumCast::from((1.0 - XRATE) / (1.0 + XRATE)).unwrap();
        let temp2 = temp1.recip();
        if self.nlp.lp.ida_cjratio < temp1 || self.nlp.lp.ida_cjratio > temp2 {
            call_lsetup = true;
        }
        if self.nlp.lp.ida_cj != self.ida_cjlast {
            self.nlp.ida_ss = P::Scalar::hundred();
        }
        //}

        // initial guess for the correction to the predictor
        self.ida_delta.fill(P::Scalar::zero());

        // call nonlinear solver setup if it exists
        self.nls.setup(&mut self.ida_delta)?;
        /*
        if ((self.NLS)->ops->setup) {
          retval = SUNNonlinSolSetup(self.NLS, self.ida_delta, IDA_mem);
          if (retval < 0) return(IDA_NLS_SETUP_FAIL);
          if (retval > 0) return(IDA_NLS_SETUP_RTolCVR);
        }
        */

        let w = self.nlp.ida_ewt.clone();

        // solve the nonlinear system
        let retval = self.nls.solve(
            &mut self.nlp,
            self.ida_delta.view(),
            self.ida_ee.view_mut(),
            w,
            self.ida_eps_newt,
            call_lsetup,
        );

        // update yy and yp based on the final correction from the nonlinear solve
        self.nlp.ida_yy = &self.nlp.ida_yypredict + &self.ida_ee;
        //N_VLinearSum( ONE, self.ida_yppredict, self.ida_cj, self.ida_ee, self.ida_yp,);
        //self.ida_yp = &self.ida_yppredict + (&self.ida_ee * self.ida_cj);
        self.nlp.ida_yp.assign(&self.nlp.ida_yppredict);
        self.nlp.ida_yp.scaled_add(self.nlp.lp.ida_cj, &self.ida_ee);

        // return if nonlinear solver failed */
        retval?;

        // If otherwise successful, check and enforce inequality constraints.
        //trace!("nst={}, yy={:.5e}", self.counters.ida_nst, self.nlp.ida_yy);

        // Check constraints and get mask vector mm, set where constraints failed
        if self.ida_constraints_set {
            unimplemented!();
            /*
            self.ida_mm = self.ida_tempv2;
            let constraintsPassed = N_VConstrMask(self.ida_constraints, self.ida_yy, self.ida_mm);
            if (constraintsPassed) {
                return (IDA_SUCCESS);
            } else {
                N_VCompare(ONEPT5, self.ida_constraints, self.ida_tempv1);
                /* a , where a[i] =1. when |c[i]| = 2 ,  c the vector of constraints */
            N_VProd(self.ida_tempv1, self.ida_constraints, self.ida_tempv1); /* a * c */
            N_VDiv(self.ida_tempv1, self.ida_ewt, self.ida_tempv1); /* a * c * wt */
            N_VLinearSum(ONE, self.ida_yy, -PT1, self.ida_tempv1, self.ida_tempv1); /* y - 0.1 * a * c * wt */
            N_VProd(self.ida_tempv1, self.ida_mm, self.ida_tempv1); /*  v = mm*(y-.1*a*c*wt) */
            vnorm = IDAWrmsNorm(IDA_mem, self.ida_tempv1, self.ida_ewt, SUNFALSE); /*  ||v|| */

            // If vector v of constraint corrections is small in norm, correct and accept this step
            if vnorm <= self.ida_eps_newt {
            N_VLinearSum(ONE, self.ida_ee, -ONE, self.ida_tempv1, self.ida_ee); /* ee <- ee - v */
            return (IDA_SUCCESS);
            } else {
            /* Constraints not met -- reduce h by computing rr = h'/h */
            N_VLinearSum(ONE, self.ida_phi[0], -ONE, self.ida_yy, self.ida_tempv1);
            N_VProd(self.ida_mm, self.ida_tempv1, self.ida_tempv1);
            self.ida_rr = PT9 * N_VMinQuotient(self.ida_phi[0], self.ida_tempv1);
            self.ida_rr = SUNMAX(self.ida_rr, PT1);
            return (IDA_CONSTR_RTolCVR);
            }
            }
             */
        }

        Ok(())
    }

    /// IDAPredict
    /// This routine predicts the new values for vectors yy and yp.
    fn predict(&mut self) -> () {
        profile_scope!(format!("predict()"));

        // yypredict = cvals * phi[0..kk+1]
        //(void) N_VLinearCombination(self.ida_kk+1, self.ida_cvals, self.ida_phi, self.ida_yypredict);
        {
            self.nlp.ida_yypredict.assign(
                &self
                    .ida_phi
                    .slice_axis(Axis(0), Slice::from(0..self.ida_kk + 1))
                    .sum_axis(Axis(0)),
            );
        }

        // yppredict = gamma[1..kk+1] * phi[1..kk+1]
        //(void) N_VLinearCombination(self.ida_kk, self.ida_gamma+1, self.ida_phi+1, self.ida_yppredict);
        {
            let phi = self
                .ida_phi
                .slice_axis(Axis(0), Slice::from(1..self.ida_kk + 1));

            // We manually broadcast here so we can turn it into a column vec
            let gamma = self.ida_gamma.slice(s![1..self.ida_kk + 1]);
            let gamma = gamma
                .broadcast((phi.len_of(Axis(1)), phi.len_of(Axis(0))))
                .unwrap()
                .reversed_axes();

            self.nlp
                .ida_yppredict
                .assign(&(&phi * &gamma).sum_axis(Axis(0)));
        }
        /*
        trace!(
            "predict() yypredict={:.6e} yppredict={:.6e}",
            self.nlp.ida_yypredict,
            self.nlp.ida_yppredict
        );
        */
    }

    /// IDATestError
    ///
    /// This routine estimates errors at orders k, k-1, k-2, decides whether or not to suggest an order
    /// decrease, and performs the local error test.
    ///
    /// Returns a tuple of (err_k, err_km1, nflag)
    fn test_error(
        &mut self,
        ck: P::Scalar,
    ) -> Result<
        (
            P::Scalar, // err_k
            P::Scalar, // err_km1
            bool,      // converged
        ),
        (P::Scalar, P::Scalar, failure::Error),
    > {
        //trace!("test_error phi={:.5e}", self.ida_phi);
        //trace!("test_error ee={:.5e}", self.ida_ee);

        // Compute error for order k.
        let enorm_k = self.wrms_norm(&self.ida_ee, &self.nlp.ida_ewt, self.ida_suppressalg);
        let err_k = self.ida_sigma[self.ida_kk] * enorm_k; // error norms

        // local truncation error norm
        let terr_k = err_k * <P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap();


        let (err_km1, knew) = if self.ida_kk > 1 {
            // Compute error at order k-1
            self.ida_delta = &self.ida_phi.index_axis(Axis(0), self.ida_kk) + &self.ida_ee;
            let enorm_km1 =
                self.wrms_norm(&self.ida_delta, &self.nlp.ida_ewt, self.ida_suppressalg);
            // estimated error at k-1
            let err_km1 = self.ida_sigma[self.ida_kk - 1] * enorm_km1;
            let terr_km1 = err_km1 * <P::Scalar as NumCast>::from(self.ida_kk).unwrap();

            let knew = if self.ida_kk > 2 {
                // Compute error at order k-2
                // ida_delta = ida_phi[ida_kk - 1] + ida_delta
                self.ida_delta
                    .assign(&self.ida_phi.index_axis(Axis(0), self.ida_kk - 1));
                self.ida_delta.scaled_add(P::Scalar::one(), &self.ida_ee);

                let enorm_km2 =
                    self.wrms_norm(&self.ida_delta, &self.nlp.ida_ewt, self.ida_suppressalg);
                // estimated error at k-2
                let err_km2 = self.ida_sigma[self.ida_kk - 2] * enorm_km2;
                let terr_km2 = err_km2 * <P::Scalar as NumCast>::from(self.ida_kk - 1).unwrap();

                // Decrease order if errors are reduced
                if terr_km1.max(terr_km2) <= terr_k {
                    self.ida_kk - 1
                } else {
                    self.ida_kk
                }
            } else {
                // Decrease order to 1 if errors are reduced by at least 1/2
                if terr_km1 <= (terr_k * P::Scalar::half()) {
                    self.ida_kk - 1
                } else {
                    self.ida_kk
                }
            };

            (err_km1, knew)
        } else {
            (P::Scalar::zero(), self.ida_kk)
        };

        self.ida_knew = knew;

        // Perform error test
        let converged = (ck * enorm_k) <= P::Scalar::one();

        if converged {
            Ok((err_k, err_km1, true))
        } else {
            Err((err_k, err_km1, failure::Error::from(IdaError::TestFail)))
        }
    }

    /// IDARestore
    /// This routine restores tn, psi, and phi in the event of a failure.
    /// It changes back `phi-star` to `phi` (changed in `set_coeffs()`)
    fn restore(&mut self, saved_t: P::Scalar) -> () {
        trace!("restore(saved_t={:.6e})", saved_t);
        self.nlp.ida_tn = saved_t;

        // Restore psi[0 .. kk] = psi[1 .. kk + 1] - hh
        for j in 1..self.ida_kk + 1 {
            self.ida_psi[j - 1] = self.ida_psi[j] - self.ida_hh;
        }

        //Zip::from(&mut self.ida_psi.slice_mut(s![0..self.ida_kk]))
        //.and(&self.ida_psi.slice(s![1..self.ida_kk+1]));
        //ida_psi -= &self.ida_psi.slice(s![1..self.ida_kk+1]);

        if self.ida_ns <= self.ida_kk {
            // cvals[0 .. kk-ns+1] = 1 / beta[ns .. kk+1]
            ndarray::Zip::from(
                &mut self
                    .ida_cvals
                    .slice_mut(s![0..self.ida_kk - self.ida_ns + 1]),
            )
            .and(&self.ida_beta.slice(s![self.ida_ns..self.ida_kk + 1]))
            .apply(|cvals, &beta| {
                *cvals = beta.recip();
            });

            // phi[ns .. (kk + 1)] *= cvals[ns .. (kk + 1)]
            let mut ida_phi = self
                .ida_phi
                .slice_axis_mut(Axis(0), Slice::from(self.ida_ns..self.ida_kk + 1));

            // We manually broadcast cvals here so we can turn it into a column vec
            let cvals = self.ida_cvals.slice(s![0..self.ida_kk - self.ida_ns + 1]);
            let cvals = cvals
                .broadcast((1, ida_phi.len_of(Axis(0))))
                .unwrap()
                .reversed_axes();

            ida_phi *= &cvals;
        }
    }

    /// IDAHandleNFlag
    ///
    /// This routine handles failures indicated by the input variable nflag. Positive values
    /// indicate various recoverable failures while negative values indicate nonrecoverable
    /// failures. This routine adjusts the step size for recoverable failures.
    ///
    ///  Possible nflag values (input):
    ///
    ///   --convergence failures--
    ///   IDA_RES_RTolCVR              > 0
    ///   IDA_LSOLVE_RTolCVR           > 0
    ///   IDA_CONSTR_RTolCVR           > 0
    ///   SUN_NLS_CONV_RTolCV          > 0
    ///   IDA_RES_FAIL               < 0
    ///   IDA_LSOLVE_FAIL            < 0
    ///   IDA_LSETUP_FAIL            < 0
    ///
    ///   --error test failure--
    ///   ERROR_TEST_FAIL            > 0
    ///
    ///  Possible kflag values (output):
    /// # Returns
    ///
    /// * Ok(()), Recoverable, PREDICT_AGAIN
    ///
    /// * IdaError
    ///
    ///   --nonrecoverable--
    ///   IDA_CONSTR_FAIL
    ///   IDA_REP_RES_ERR
    ///   IDA_ERR_FAIL
    ///   IDA_CONV_FAIL
    ///   IDA_RES_FAIL
    ///   IDA_LSETUP_FAIL
    ///   IDA_LSOLVE_FAIL
    fn handle_n_flag(
        &mut self,
        error: failure::Error,
        err_k: P::Scalar,
        err_km1: P::Scalar,
        ncf_ptr: &mut u64,
        nef_ptr: &mut u64,
    ) -> Result<(), failure::Error> {
        use failure::format_err;

        self.ida_phase = 1;

        // Try and convert the error into an IdaError
        error
            .downcast::<IdaError>()
            .map_err(|e| {
                // nonrecoverable failure
                *ncf_ptr += 1; // local counter for convergence failures
                self.counters.ida_ncfn += 1; // global counter for convergence failures
                e
            })
            .and_then(|error: IdaError| {
                match error {
                    IdaError::TestFail => {
                        // -----------------
                        // Error Test failed
                        //------------------

                        *nef_ptr += 1; // local counter for error test failures
                        self.counters.ida_netf += 1; // global counter for error test failures

                        if *nef_ptr == 1 {
                            // On first error test failure, keep current order or lower order by one.
                            // Compute new stepsize based on differences of the solution.

                            let err_knew = if self.ida_kk == self.ida_knew {
                                err_k
                            } else {
                                err_km1
                            };

                            self.ida_kk = self.ida_knew;
                            // rr = 0.9 * (2 * err_knew + 0.0001)^(-1/(kk+1))
                            self.ida_rr = P::Scalar::pt9()
                                * (P::Scalar::two() * err_knew + P::Scalar::pt0001()).powf(
                                    -(<P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap())
                                        .recip(),
                                );
                            self.ida_rr =
                                P::Scalar::quarter().max(P::Scalar::pt9().min(self.ida_rr));
                            self.ida_hh *= self.ida_rr;

                            //return(PREDICT_AGAIN);
                            Ok(())
                        } else if *nef_ptr == 2 {
                            // On second error test failure, use current order or decrease by one.
                            // Reduce stepsize by factor of 1/4.

                            self.ida_kk = self.ida_knew;
                            self.ida_rr = P::Scalar::quarter();
                            self.ida_hh *= self.ida_rr;

                            //return(PREDICT_AGAIN);
                            Ok(())
                        } else if *nef_ptr < self.ida_maxnef {
                            // On third and subsequent error test failures, set order to 1. Reduce
                            // stepsize by factor of 1/4.
                            self.ida_kk = 1;
                            self.ida_rr = P::Scalar::quarter();
                            self.ida_hh *= self.ida_rr;
                            //return(PREDICT_AGAIN);
                            Ok(())
                        } else {
                            // Too many error test failures
                            //return(IDA_ERR_FAIL);
                            Err(format_err!("IDA_ERR_FAIL"))
                        }
                    }

                    IdaError::RecoverableFail { rec_type } => {
                        // recoverable failure
                        *ncf_ptr += 1; // local counter for convergence failures
                        self.counters.ida_ncfn += 1; // global counter for convergence failures

                        // Reduce step size for a new prediction
                        match rec_type {
                            // Note that if nflag=IDA_CONSTR_RECVR then rr was already set in IDANls
                            Recoverable::Constraint => {}
                            _ => {
                                self.ida_rr = P::Scalar::quarter();
                            }
                        }

                        self.ida_hh *= self.ida_rr;

                        // Test if there were too many convergence failures
                        if *ncf_ptr < self.ida_maxncf {
                            //return(PREDICT_AGAIN);
                            Ok(())
                        } else {
                            match rec_type {
                                // return (IDA_REP_RES_ERR);
                                Recoverable::Residual => {
                                    Err(failure::Error::from(IdaError::ResidualFail {}))
                                }

                                // return (IDA_CONSTR_FAIL);
                                Recoverable::Constraint => {
                                    Err(failure::Error::from(IdaError::ConstraintFail {}))
                                }

                                // return (IDA_CONV_FAIL);
                                _ => Err(failure::Error::from(IdaError::ConvergenceFail {})),
                            }
                        }
                    }

                    _ => {
                        unimplemented!("Should never happen");
                    }
                }
            })
    }

    /// IDAReset
    /// This routine is called only if we need to predict again at the very first step. In such a case,
    /// reset phi[1] and psi[0].
    fn reset(&mut self) -> () {
        self.ida_psi[0] = self.ida_hh;
        self.ida_phi *= self.ida_rr;
    }

    /// IDACompleteStep
    /// This routine completes a successful step.  It increments nst, saves the stepsize and order
    /// used, makes the final selection of stepsize and order for the next step, and updates the phi
    /// array.
    fn complete_step(&mut self, err_k: P::Scalar, err_km1: P::Scalar) -> () {
        profile_scope!(format!("complete_step()"));
        trace!(
            "complete_step(err_k={:.5e}, err_km1={:.5e}), nst={}, phase={}",
            err_k,
            err_km1,
            self.counters.ida_nst,
            self.ida_phase
        );

        self.counters.ida_nst += 1;
        let kdiff = (self.ida_kk as isize) - (self.ida_kused as isize);
        self.ida_kused = self.ida_kk;
        self.ida_hused = self.ida_hh;

        if (self.ida_knew == self.ida_kk - 1) || (self.ida_kk == self.ida_maxord) {
            self.ida_phase = 1;
        }

        // For the first few steps, until either a step fails, or the order is reduced, or the
        // order reaches its maximum, we raise the order and double the stepsize. During these
        // steps, phase = 0. Thereafter, phase = 1, and stepsize and order are set by the usual
        // local error algorithm.
        //
        // Note that, after the first step, the order is not increased, as not all of the
        // neccessary information is available yet.

        if self.ida_phase == 0 {
            if self.counters.ida_nst > 1 {
                self.ida_kk += 1;
                let mut hnew = P::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > P::Scalar::one() {
                    hnew /= tmp;
                }
                self.ida_hh = hnew;
            }
        } else {
            #[derive(Debug)]
            enum Action {
                Lower,
                Maintain,
                Raise,
            }

            // Set action = LOWER/MAINTAIN/RAISE to specify order decision

            let (action, err_kp1) = if self.ida_knew == (self.ida_kk - 1) {
                (Action::Lower, P::Scalar::zero())
            } else if self.ida_kk == self.ida_maxord {
                (Action::Maintain, P::Scalar::zero())
            } else if (self.ida_kk + 1) >= self.ida_ns || (kdiff == 1) {
                (Action::Maintain, P::Scalar::zero())
            } else {
                // Estimate the error at order k+1, unless already decided to reduce order, or already using
                // maximum order, or stepsize has not been constant, or order was just raised.

                // tempv1 = ee - phi[kk+1]
                let enorm = {
                    let temp = &self.ida_ee - &self.ida_phi.index_axis(Axis(0), self.ida_kk + 1);
                    self.wrms_norm(&temp, &self.nlp.ida_ewt, self.ida_suppressalg)
                };
                let err_kp1 = enorm / <P::Scalar as NumCast>::from(self.ida_kk + 2).unwrap();

                // Choose among orders k-1, k, k+1 using local truncation error norms.

                let terr_k = <P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap() * err_k;
                let terr_kp1 = <P::Scalar as NumCast>::from(self.ida_kk + 2).unwrap() * err_kp1;

                if self.ida_kk == 1 {
                    if terr_kp1 >= (P::Scalar::half() * terr_k) {
                        (Action::Maintain, err_kp1)
                    } else {
                        (Action::Raise, err_kp1)
                    }
                } else {
                    let terr_km1 = <P::Scalar as NumCast>::from(self.ida_kk).unwrap() * err_km1;
                    if terr_km1 <= terr_k.min(terr_kp1) {
                        (Action::Lower, err_kp1)
                    } else if terr_kp1 >= terr_k {
                        (Action::Maintain, err_kp1)
                    } else {
                        (Action::Raise, err_kp1)
                    }
                }
            };

            // Set the estimated error norm and, on change of order, reset kk.
            let err_knew = match action {
                Action::Raise => {
                    self.ida_kk += 1;
                    err_kp1
                }
                Action::Lower => {
                    self.ida_kk -= 1;
                    err_km1
                }
                _ => err_k,
            };
            trace!(
                "    nst={}, {:#?}, kk={}, err_knew={:.5e}",
                self.counters.ida_nst,
                action,
                self.ida_kk,
                err_knew
            );

            // Compute rr = tentative ratio hnew/hh from error norm estimate.
            // Reduce hh if rr <= 1, double hh if rr >= 2, else leave hh as is.
            // If hh is reduced, hnew/hh is restricted to be between .5 and .9.

            let mut hnew = self.ida_hh;
            //ida_rr = SUNRpowerR( TWO * err_knew + PT0001, -ONE/(self.ida_kk + 1) );
            self.ida_rr = {
                let base = P::Scalar::two() * err_knew + P::Scalar::pt0001();
                let arg = -(<P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap()).recip();
                base.powf(arg)
            };

            if self.ida_rr >= P::Scalar::two() {
                hnew = P::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > P::Scalar::one() {
                    hnew /= tmp;
                }
            } else if self.ida_rr <= P::Scalar::one() {
                //ida_rr = SUNMAX(HALF, SUNMIN(PT9,self.ida_rr));
                self.ida_rr = P::Scalar::half().max(self.ida_rr.min(P::Scalar::pt9()));
                hnew = self.ida_hh * self.ida_rr;
            }

            self.ida_hh = hnew;
        }
        trace!("    next hh={:.5e}", self.ida_hh);
        // end of phase if block

        // Save ee for possible order increase on next step
        if self.ida_kused < self.ida_maxord {
            self.ida_phi
                .index_axis_mut(Axis(0), self.ida_kused + 1)
                .assign(&self.ida_ee);
        }

        // Update phi arrays

        // To update phi arrays compute X += Z where
        // X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ]
        // Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ]

        // Note: this is a recurrence relation, and needs to be performed as below

        let mut z_view = self
            .ida_zvecs
            .slice_axis_mut(Axis(0), Slice::from(0..self.ida_kused + 1));

        for (i, mut z_row) in z_view.genrows_mut().into_iter().enumerate() {
            // z[i] = ee + phi[kused] + phi[kused-1] + .. + phi[i]
            z_row.assign(&self.ida_ee);
            z_row += &self
                .ida_phi
                .slice_axis(Axis(0), Slice::from(i..self.ida_kused + 1))
                .sum_axis(Axis(0));
        }

        self.ida_phi
            .slice_axis_mut(Axis(0), Slice::from(0..self.ida_kused + 1))
            .assign(&z_view);
    }

    /// This routine evaluates `y(t)` and `y'(t)` as the value and derivative of the interpolating
    /// polynomial at the independent variable t, and stores the results in the vectors yret and ypret.
    /// It uses the current independent variable value, tn, and the method order last used, kused.
    /// This function is called by `solve` with `t = tout`, `t = tn`, or `t = tstop`.
    ///
    /// If `kused = 0` (no step has been taken), or if `t = tn`, then the order used here is taken
    /// to be 1, giving `yret = phi[0]`, `ypret = phi[1]/psi[0]`.
    ///
    /// # Arguments
    ///
    /// * `t` - requested independent variable (time)
    /// * `yret` - return value of `y(t)`
    /// * `ypret` - return value of `y'(t)`
    ///
    /// # Returns
    ///
    /// * () if `t` was legal. Outputs placed into `yret` and `ypret`
    ///
    /// # Errors
    ///
    /// * `IdaError::BadTimeValue` if `t` is not within the interval of the last step taken.
    pub fn get_solution(&mut self, t: P::Scalar) -> Result<(), failure::Error> {
        profile_scope!(format!("get_solution(t={:.5e})", t));
        // Check t for legality.  Here tn - hused is t_{n-1}.

        let tfuzz = P::Scalar::hundred()
            * P::Scalar::epsilon()
            * (self.nlp.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();

        let tp = self.nlp.ida_tn - self.ida_hused - tfuzz;
        if ((t - tp) * self.ida_hh) < P::Scalar::zero() {
            Err(IdaError::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.nlp.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.nlp.ida_tn.to_f64().unwrap(),
            })?;
        }

        // Initialize kord = (kused or 1).
        let kord = if self.ida_kused == 0 {
            1
        } else {
            self.ida_kused
        };

        // Accumulate multiples of columns phi[j] into yret and ypret.
        let delt = t - self.nlp.ida_tn;
        let mut c = P::Scalar::one();
        let mut d = P::Scalar::zero();
        let mut gam = delt / self.ida_psi[0];

        self.ida_cvals[0] = c;
        for j in 1..kord {
            d = d * gam + c / self.ida_psi[j - 1];
            c = c * gam;
            gam = (delt + self.ida_psi[j - 1]) / self.ida_psi[j];

            self.ida_cvals[j] = c;
            self.ida_dvals[j - 1] = d;
        }

        //retval = N_VLinearCombination(kord+1, self.ida_cvals, self.ida_phi,  yret);
        let cvals = self.ida_cvals.slice(s![0..kord + 1]);
        ndarray::Zip::from(self.nlp.ida_yy.view_mut())
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(0..kord + 1))
                    .lanes(Axis(0)),
            )
            .apply(|z, row| {
                *z = (&row * &cvals).sum();
            });

        //retval = N_VLinearCombination(kord, self.ida_dvals, self.ida_phi+1, ypret);
        let dvals = self.ida_dvals.slice(s![0..kord]);
        ndarray::Zip::from(&mut self.nlp.ida_yp)
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(1..kord + 1))
                    .lanes(Axis(0)),
            )
            .apply(|z, row| {
                *z = (&row * &dvals).sum();
            });

        Ok(())
    }

    /// Returns the WRMS norm of vector x with weights w.
    /// If mask = SUNTRUE, the weight vector w is masked by id, i.e.,
    ///      nrm = N_VWrmsNormMask(x,w,id);
    ///  Otherwise,
    ///      nrm = N_VWrmsNorm(x,w);
    ///
    /// mask = SUNFALSE       when the call is made from the nonlinear solver.
    /// mask = suppressalg otherwise.
    pub fn wrms_norm<S1, S2>(
        &self,
        x: &ArrayBase<S1, Ix1>,
        w: &ArrayBase<S2, Ix1>,
        mask: bool,
    ) -> P::Scalar
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
    {
        profile_scope!(format!("wrms_norm()"));
        if mask {
            x.norm_wrms_masked(w, &self.ida_id)
        } else {
            x.norm_wrms(w)
        }
    }

    /// IDARcheck1
    ///
    /// This routine completes the initialization of rootfinding memory
    /// information, and checks whether g has a zero both at and very near
    /// the initial point of the IVP.
    ///
    /// This routine returns an int equal to:
    ///  IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///  IDA_SUCCESS     = 0 otherwise.
    fn r_check1(&mut self) -> Result<(), IdaError> {
        //int i, retval;
        //realtype smallh, hratio, tplus;
        //booleantype zroot;

        //for (i = 0; i < self.ida_nrtfn; i++)
        //  self.ida_iroots[i] = 0;
        //self.ida_tlo = self.nlp.ida_tn;
        //self.ida_ttol = ((self.nlp.ida_tn).abs() + (self.ida_hh).abs())
        //    * P::Scalar::epsilon()
        //    * P::Scalar::from(100.0).unwrap();

        // Evaluate g at initial t and check for zero values.
        /*
        let retval: Result<(), IdaError> = self.ida_gfun(
            self.ida_tlo,
            self.ida_phi[0],
            self.ida_phi[1],
            self.ida_glo,
            self.ida_user_data,
        );
        */
        self.ida_nge = 1;
        //retval.map_err(|e| IdaError::RootFunctionFail{t: self.nlp.ida_tn.to_f64().unwrap() })?;

        /*
        zroot = SUNFALSE;
        for (i = 0; i < self.ida_nrtfn; i++) {
          if (SUNRabs(self.ida_glo[i]) == ZERO) {
            zroot = SUNTRUE;
            self.ida_gactive[i] = SUNFALSE;
          }
        }
        if (!zroot) return(IDA_SUCCESS);

        /* Some g_i is zero at t0; look at g at t0+(small increment). */
        hratio = SUNMAX(self.ida_ttol/SUNRabs(self.ida_hh), PT1);
        smallh = hratio * self.ida_hh;
        tplus = self.ida_tlo + smallh;
        N_VLinearSum(ONE, self.ida_phi[0], smallh, self.ida_phi[1], self.ida_yy);
        retval = self.ida_gfun(tplus, self.ida_yy, self.ida_phi[1],
        self.ida_ghi, self.ida_user_data);
        self.ida_nge++;
        if (retval != 0) return(IDA_RTFUNC_FAIL);

        /* We check now only the components of g which were exactly 0.0 at t0
         * to see if we can 'activate' them. */
        for (i = 0; i < self.ida_nrtfn; i++) {
        if (!self.ida_gactive[i] && SUNRabs(self.ida_ghi[i]) != ZERO) {
        self.ida_gactive[i] = SUNTRUE;
        self.ida_glo[i] = self.ida_ghi[i];
        }
        }
        return(IDA_SUCCESS);
         */
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;
    use nearly_eq::*;
    use tol_control::*;

    #[derive(Clone, Copy, Debug)]
    struct Dummy {}

    impl ModelSpec for Dummy {
        type Scalar = f64;
        type Dim = Ix1;
        fn model_size(&self) -> usize {
            3
        }
    }

    impl Residual for Dummy {
        fn res<S1, S2, S3>(
            &self,
            _tres: Self::Scalar,
            _yy: ArrayBase<S1, Ix1>,
            _yp: ArrayBase<S2, Ix1>,
            mut _resval: ArrayBase<S3, Ix1>,
        ) where
            S1: ndarray::Data<Elem = Self::Scalar>,
            S2: ndarray::Data<Elem = Self::Scalar>,
            S3: ndarray::DataMut<Elem = Self::Scalar>,
        {
        }
    }

    impl Jacobian for Dummy {
        fn jac<S1, S2, S3, S4>(
            &self,
            _tt: Self::Scalar,
            _cj: Self::Scalar,
            _yy: ArrayBase<S1, Ix1>,
            _yp: ArrayBase<S2, Ix1>,
            _rr: ArrayBase<S3, Ix1>,
            mut _j: ArrayBase<S4, Ix2>,
        ) where
            S1: ndarray::Data<Elem = Self::Scalar>,
            S2: ndarray::Data<Elem = Self::Scalar>,
            S3: ndarray::Data<Elem = Self::Scalar>,
            S4: ndarray::DataMut<Elem = Self::Scalar>,
        {
        }
    }

    #[test]
    fn test_set_coeffs() {
        //Before
        #[rustfmt::skip]
        let ida_phi = array![
            [ 4.1295003522440181e-07, 1.6518008147114031e-12, 9.9999958704831304e-01, ],
            [ -6.4049734044789205e-08, -2.5619916159829551e-13, 6.4049990326726996e-08, ],
            [ 2.1135440604995772e-08, 8.4541889872000439e-14, -2.1135525197726480e-08, ],
            [ -2.2351400807868742e-08, -8.9405756031743853e-14, 2.2351489636470618e-08, ],
            [ 1.8323105973439385e-08, 7.3292641194159994e-14, -1.8323176512520801e-08, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
        let ida_psi = array![
            6.6874844417638421e+08,
            1.4118022710390334e+09,
            1.8407375671333179e+09,
            1.8153920670983608e+09,
            2.1446764804714236e+09,
            2.6020582487631597e+07,
        ];
        let ida_alpha = array![
            1.0000000000000000e+00,
            4.7368421052631576e-01,
            3.6330461012857090e-01,
            4.0930763129879277e-01,
            3.9999999999999997e-01,
            3.6363636363636365e-01,
        ];
        let ida_beta = array![
            1.0000000000000000e+00,
            9.0000000000000002e-01,
            1.0841585634594841e+00,
            3.5332089881864119e+00,
            7.1999999999999993e+00,
            1.0285714285714285e+01,
        ];
        let ida_sigma = array![
            1.0000000000000000e+00,
            4.7368421052631576e-01,
            3.4418331485864612e-01,
            7.2268199139687761e-01,
            1.4222222222222223e+00,
            2.5858585858585861e+00,
        ];
        let ida_gamma = array![
            0.0000000000000000e+00,
            1.4953305816383288e-09,
            2.2036450676775371e-09,
            2.8236868704168917e-09,
            3.0437121109953610e-09,
            3.1823098347208659e-07,
        ];
        let kk = 2;
        let kused = 2;
        let ns = 1;
        let hh = 6.6874844417638421e+08;
        let hused = 6.6874844417638421e+08;
        let cj = 2.2429958724574930e-09;
        let cjlast = 2.4672954597032423e-09;

        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        // Set preconditions:
        ida.ida_hh = hh;
        ida.ida_hused = hused;
        ida.ida_ns = ns;
        ida.ida_kused = kused;
        ida.ida_kk = kk;
        ida.ida_beta.assign(&ida_beta);
        ida.ida_alpha.assign(&ida_alpha);
        ida.ida_gamma.assign(&ida_gamma);
        ida.ida_sigma.assign(&ida_sigma);
        ida.ida_phi.assign(&ida_phi);
        ida.ida_psi.assign(&ida_psi);
        ida.ida_cjlast = cjlast;
        ida.nlp.lp.ida_cj = cj;

        // Call the function under test
        let ck = ida.set_coeffs();

        //--- IDASetCoeffs After
        let ck_expect = 0.3214285714285713969;
        #[rustfmt::skip]
        let ida_phi = array![
            [ 4.1295003522440181e-07, 1.6518008147114031e-12, 9.9999958704831304e-01, ],
            [ -6.4049734044789205e-08, -2.5619916159829551e-13, 6.4049990326726996e-08, ],
            [ 2.0023048994206519e-08, 8.0092316720842518e-14, -2.0023129134688242e-08, ],
            [ -2.2351400807868742e-08, -8.9405756031743853e-14, 2.2351489636470618e-08, ],
            [ 1.8323105973439385e-08, 7.3292641194159994e-14, -1.8323176512520801e-08, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
        let ida_psi = array![
            6.6874844417638421e+08,
            1.3374968883527684e+09,
            2.0805507152154176e+09,
            1.8153920670983608e+09,
            2.1446764804714236e+09,
            2.6020582487631597e+07,
        ];
        let ida_alpha = array![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.2142857142857140e-01,
            4.0930763129879277e-01,
            3.9999999999999997e-01,
            3.6363636363636365e-01,
        ];
        let ida_beta = array![
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            9.4736842105263153e-01,
            3.5332089881864119e+00,
            7.1999999999999993e+00,
            1.0285714285714285e+01,
        ];
        let ida_sigma = array![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.2142857142857140e-01,
            7.2268199139687761e-01,
            1.4222222222222223e+00,
            2.5858585858585861e+00,
        ];
        let ida_gamma = array![
            0.0000000000000000e+00,
            1.4953305816383288e-09,
            2.2429958724574930e-09,
            2.8236868704168917e-09,
            3.0437121109953610e-09,
            3.1823098347208659e-07,
        ];
        let kk = 2;
        let kused = 2;
        let ns = 2;
        let hh = 6.6874844417638421e+08;
        let hused = 6.6874844417638421e+08;
        let cj = 2.2429958724574930e-09;
        let cjlast = 2.2429958724574930e-09;

        assert_nearly_eq!(ida.ida_hh, hh);
        assert_nearly_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_ns, ns);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_kk, kk);
        assert_nearly_eq!(ida.ida_beta, ida_beta);
        assert_nearly_eq!(ida.ida_alpha, ida_alpha);
        assert_nearly_eq!(ida.ida_gamma, ida_gamma);
        assert_nearly_eq!(ida.ida_sigma, ida_sigma);
        assert_nearly_eq!(ida.ida_phi, ida_phi);
        assert_nearly_eq!(ida.ida_psi, ida_psi);
        assert_nearly_eq!(ida.ida_cjlast, cjlast);
        assert_nearly_eq!(ida.nlp.lp.ida_cj, cj);
        assert_nearly_eq!(ck, ck_expect);
    }

    #[test]
    fn test_predict() {}

    #[test]
    fn test_test_error1() {
        let ck = 1.091414141414142;
        let suppressalg = 0;
        let kk = 5;
        #[rustfmt::skip]
        let ida_phi = array![
            [ 3.634565317158998e-05, 1.453878335134203e-10, 0.9999636542014404, ],
            [ -6.530333550677049e-06, -2.612329458968465e-11, 6.530359673556191e-06, ],
            [ 1.946442728026142e-06, 7.786687275994346e-12, -1.946450515496441e-06, ],
            [ -8.097632208221231e-07, -3.239585549038764e-12, 8.097664556005615e-07, ],
            [ 3.718130977075839e-07, 1.487573462300438e-12, -3.71814615793545e-07, ],
            [ -3.24421895454213e-07, -1.297915245220823e-12, 3.244230624265827e-07, ],
        ];
        let ida_ee = array![
            2.65787533317467e-07,
            1.063275845801634e-12,
            -2.657884288386138e-07,
        ];
        let ida_ewt = array![73343005.56993243, 999999.985461217, 9901.346408259429];
        let ida_sigma = array![
            1.0,
            0.6666666666666666,
            0.6666666666666666,
            0.888888888888889,
            1.422222222222222,
            2.585858585858586,
        ];
        let knew = 4;
        let err_k = 29.10297975314245;
        let err_km1 = 3.531162835377502;

        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_suppressalg = suppressalg > 0;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_ee.assign(&ida_ee);
        ida.nlp.ida_ewt.assign(&ida_ewt);
        ida.ida_sigma.assign(&ida_sigma);

        // Call the function under test
        let (err_k_new, err_km1_new, nflag_new) = ida.test_error(ck).expect_err("Should be TestFail");

        assert_eq!(ida.ida_knew, knew);
        assert_nearly_eq!(err_k_new, err_k);
        assert_nearly_eq!(err_km1_new, err_km1);
    }

    #[test]
    fn test_test_error2() {
        //--- IDATestError Before:
        let ck = 0.2025812352167927;
        let suppressalg = 0;
        let kk = 4;
        #[rustfmt::skip]
        let ida_phi = array![
            [ 3.051237735052657e-05, 1.220531905117091e-10, 0.9999694875005963, ],
            [ -2.513114849098281e-06, -1.005308974226734e-11, 2.513124902721765e-06, ],
            [ 4.500284453718991e-07, 1.800291970640913e-12, -4.500302448499092e-07, ],
            [ -1.366709389821433e-07, -5.467603693902342e-13, 1.366714866794709e-07, ],
            [ 7.278821769100639e-08, 2.911981566628798e-13, -7.278850816613011e-08, ],
            [ -8.304741244343501e-09, -3.324587131187576e-14, 8.304772990651073e-09, ],
        ];
        let ida_ee = array![
            -2.981302228744271e-08,
            -1.192712676406388e-13,
            2.981313872620108e-08,
        ];
        let ida_ewt = array![76621085.31777237, 999999.9877946811, 9901.289220872719,];
        let ida_sigma = array![
            1.0,
            0.5,
            0.3214285714285715,
            0.2396514200444849,
            0.1941955227762807,
            2.585858585858586,
        ];
        //--- IDATestError After:
        let knew = 4;
        let err_k = 0.2561137489433976;
        let err_km1 = 0.455601916633899;
        let nflag = true;

        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_suppressalg = suppressalg > 0;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_ee.assign(&ida_ee);
        ida.nlp.ida_ewt.assign(&ida_ewt);
        ida.ida_sigma.assign(&ida_sigma);

        // Call the function under test
        let (err_k_new, err_km1_new, nflag_new) = ida.test_error(ck).unwrap();

        assert_eq!(ida.ida_knew, knew);
        assert_nearly_eq!(err_k_new, err_k);
        assert_nearly_eq!(err_km1_new, err_km1);
        assert_eq!(nflag_new, nflag);
    }

    #[test]
    fn test_predict1() {
        #[rustfmt::skip]
        let ida_phi = array![
            [ 1.0570152037228958e-07, 4.2280612558303261e-13, 9.9999989429805680e-01, ],
            [ -3.3082196412696304e-08, -1.3232881828710420e-13, 3.3082328676061534e-08, ],
            [ 1.8675273859330434e-08, 7.4701128706323864e-14, -1.8675348801050254e-08, ],
            [ -1.9956501813542136e-08, -7.9826057803058290e-14, 1.9956580862443821e-08, ],
            [ 1.2851942479612096e-09, 5.1407743965993651e-15, -1.2851948368212051e-09, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
        let ida_gamma = array![
            0.0000000000000000e+00,
            2.6496925453439462e-10,
            3.8862188959925182e-10,
            8.0997073172076138e-10,
            3.0437121109953610e-09,
            3.1823098347208659e-07,
        ];
        let ida_yypredict = array![
            1.2565802218583172e-07,
            5.0263218338609083e-13,
            9.9999987434147597e-01,
        ];
        let ida_yppredict = array![
            1.5848602690328082e-18,
            6.3394566628399208e-24,
            -1.5848663595269871e-18,
        ];
        let kk = 2;

        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_gamma.assign(&ida_gamma);
        ida.nlp.ida_yypredict.assign(&ida_yypredict);
        ida.nlp.ida_yppredict.assign(&ida_yppredict);

        // Call the function under test
        ida.predict();

        //--- IDAPredict After
        #[rustfmt::skip]
        let ida_phi = array![
            [ 1.0570152037228958e-07, 4.2280612558303261e-13, 9.9999989429805680e-01, ],
            [ -3.3082196412696304e-08, -1.3232881828710420e-13, 3.3082328676061534e-08, ],
            [ 1.8675273859330434e-08, 7.4701128706323864e-14, -1.8675348801050254e-08, ],
            [ -1.9956501813542136e-08, -7.9826057803058290e-14, 1.9956580862443821e-08, ],
            [ 1.2851942479612096e-09, 5.1407743965993651e-15, -1.2851948368212051e-09, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
        let ida_yypredict = array![
            9.1294597818923714e-08,
            3.6517843600225230e-13,
            9.9999990870503663e-01,
        ];
        let ida_yppredict = array![
            -1.5081447058360581e-18,
            -6.0325745419028739e-24,
            1.5081506275685795e-18,
        ];

        assert_eq!(ida.ida_kk, kk);
        assert_nearly_eq!(ida.ida_phi, ida_phi);
        assert_nearly_eq!(ida.nlp.ida_yypredict, ida_yypredict);
        assert_nearly_eq!(ida.nlp.ida_yppredict, ida_yppredict);
    }

    #[test]
    fn test_restore1() {
        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        let saved_t = 1.2123749601869230e-03;

        // Set preconditions:
        {
            ida.nlp.ida_tn = 1.9051606517223076e-03;
            ida.ida_ns = 1;
            ida.ida_kk = 2;
            ida.ida_hh = 6.9278569153538464e-04;
            #[rustfmt::skip]
            ida.ida_phi.assign(&array![
                [ 9.9995150785077214e-01, 3.2239271870584518e-05, 1.6252877357362511e-05, ],
                [ -2.7707635503825098e-05, 9.7313256739903617e-06, 1.7976309829765466e-05, ],
                [ 3.4935651154088256e-09, -1.0880124117276491e-05, 1.0876630551341415e-05, ],
                [ 5.6694097544747856e-10, 2.1378758881496850e-08, -2.1945701078189652e-08, ],
                [ -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06, ],
                [ -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07, ]
            ]);
            ida.ida_psi.assign(&array![
                6.9278569153538464e-04,
                1.0391785373030770e-03,
                1.3855713830707693e-03,
                7.7938390297730776e-04,
                3.4639284576769232e-04,
                0.0000000000000000e+00
            ]);
            ida.ida_cvals.assign(&array![
                1.0000000000000000e+00,
                1.0000000000000000e+00,
                1.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00,
                0.0000000000000000e+00
            ]);
            ida.ida_beta.assign(&array![
                1.0000000000000000e+00,
                2.0000000000000000e+00,
                3.0000000000000000e+00,
                4.8000000000000007e+00,
                1.5000000000000000e+01,
                0.0000000000000000e+00
            ]);
        }

        // Call the function under test
        ida.restore(saved_t);

        //--- IDARestore After: saved_t=1.2123749601869230e-03 tn=1.2123749601869230e-03 ns=1 kk=2
        #[rustfmt::skip]
        let phi_after = array![
            [ 9.9995150785077214e-01, 3.2239271870584518e-05, 1.6252877357362511e-05, ],
            [ -1.3853817751912549e-05, 4.8656628369951808e-06, 8.9881549148827332e-06, ],
            [ 1.1645217051362752e-09, -3.6267080390921635e-06, 3.6255435171138049e-06, ],
            [ 5.6694097544747856e-10, 2.1378758881496850e-08, -2.1945701078189652e-08, ],
            [ -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06, ],
            [ -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07, ]
        ];
        let psi_after = array![
            3.4639284576769237e-04,
            6.9278569153538464e-04,
            1.3855713830707693e-03,
            7.7938390297730776e-04,
            3.4639284576769232e-04,
            0.0000000000000000e+00
        ];
        let cvals_after = array![
            5.0000000000000000e-01,
            3.3333333333333331e-01,
            1.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ];
        let beta_after = array![
            1.0000000000000000e+00,
            2.0000000000000000e+00,
            3.0000000000000000e+00,
            4.8000000000000007e+00,
            1.5000000000000000e+01,
            0.0000000000000000e+00
        ];

        assert_nearly_eq!(ida.nlp.ida_tn, saved_t);
        assert_eq!(ida.ida_ns, 1);
        assert_eq!(ida.ida_kk, 2);
        assert_nearly_eq!(ida.ida_cvals, cvals_after);
        assert_nearly_eq!(ida.ida_beta, beta_after);
        assert_nearly_eq!(ida.ida_psi, psi_after);
        assert_nearly_eq!(ida.ida_phi, phi_after);
    }

    #[test]
    fn test_complete_step1() {
        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        let err_k = 0.0001987733462845937;
        let err_km1 = 0.0;

        // Set preconditions:
        {
            let ida_phi = array![
                [1.0, 0.0, 0.0,],
                [-8.6598211441923077e-07, 8.6598211441923077e-07, 0.0,],
                [0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0,],
                [0.0, 0.0, 0.0,]
            ];
            let ida_ee = array![
                7.5001558608301906e-13,
                -4.8726813621044346e-10,
                4.8651812062436036e-10,
            ];
            let ida_ewt = array![
                9.9990000999900003e+03,
                1.0000000000000000e+06,
                1.0000000000000000e+06,
            ];
            let kk = 1;
            let kused = 0;
            let knew = 1;
            let phase = 0;
            let hh = 2.1649552860480770e-05;
            let hused = 0.0000000000000000e+00;
            let rr = 0.0000000000000000e+00;
            let hmax_inv = 0.0000000000000000e+00;
            let nst = 0;
            let maxord = 5;

            ida.counters.ida_nst = nst;
            ida.ida_kk = kk;
            ida.ida_hh = hh;
            ida.ida_rr = rr;
            ida.ida_kused = kused;
            ida.ida_hused = hused;
            ida.ida_knew = knew;
            ida.ida_maxord = maxord;
            ida.ida_phase = phase;
            ida.ida_hmax_inv = hmax_inv;
            ida.ida_ee.assign(&ida_ee);
            ida.ida_phi.assign(&ida_phi);
            ida.nlp.ida_ewt.assign(&ida_ewt);
        }

        ida.complete_step(err_k, err_km1);

        #[rustfmt::skip]
        let ida_phi = array![
            [ 9.9999913401863560e-01, 8.6549484628302034e-07, 4.8651812062436036e-10, ],
            [ -8.6598136440364466e-07, 8.6549484628302034e-07, 4.8651812062436036e-10, ],
            [ 7.5001558608301906e-13, -4.8726813621044346e-10, 4.8651812062436036e-10, ],
            [0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0,],
            [0.0, 0.0, 0.0,],
        ];
        let ida_ee = array![
            7.5001558608301906e-13,
            -4.8726813621044346e-10,
            4.8651812062436036e-10,
        ];
        let ida_ewt = array![
            9.9990000999900003e+03,
            1.0000000000000000e+06,
            1.0000000000000000e+06,
        ];
        let kk = 1;
        let kused = 1;
        let knew = 1;
        let phase = 0;
        let hh = 2.1649552860480770e-05;
        let hused = 2.1649552860480770e-05;
        let rr = 0.0000000000000000e+00;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 1;
        let maxord = 5;

        assert_eq!(ida.counters.ida_nst, nst);
        assert_eq!(ida.ida_kk, kk);
        assert_eq!(ida.ida_hh, hh);
        assert_nearly_eq!(ida.ida_rr, rr);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_knew, knew);
        assert_eq!(ida.ida_maxord, maxord);
        assert_eq!(ida.ida_phase, phase);
        assert_eq!(ida.ida_hmax_inv, hmax_inv);
        assert_nearly_eq!(ida.ida_ee, ida_ee);
        assert_nearly_eq!(ida.ida_phi, ida_phi);
        assert_nearly_eq!(ida.nlp.ida_ewt, ida_ewt);
    }

    #[test]
    fn test_complete_step2() {
        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        let err_k = 0.001339349356604325;
        let err_km1 = 0.003720519687081918;

        // Set preconditions:
        {
            #[rustfmt::skip]
            ida.ida_phi.assign(&array![
                [ 9.9999826803802172e-01, 1.7295310279504897e-06, 2.4309503863111873e-09, ],
                [ -1.7319612278663124e-06, 1.7280723633349389e-06, 3.8888645313736536e-09, ],
                [ 2.2514114651871690e-12, -4.3759938466525865e-09, 4.3737424351873994e-09, ],
                [ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, ],
                [ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, ],
                [ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, ],
            ]);
            ida.ida_ee.assign(&array![
                -4.2122294839452673e-13,
                -2.4605442771223734e-09,
                2.4609655000707684e-09,
            ]);
            ida.nlp.ida_ewt.assign(&array![
                9.9990174161763662e+03,
                9.9982707680480811e+05,
                9.9999975690502045e+05,
            ]);

            ida.ida_kk = 2;
            ida.ida_kused = 1;
            ida.ida_knew = 2;
            ida.ida_phase = 0;
            ida.ida_hh = 4.3299105720961540e-05;
            ida.ida_hused = 2.1649552860480770e-05;
            ida.ida_rr = 0.0000000000000000e+00;
            ida.ida_hmax_inv = 0.0000000000000000e+00;
            ida.counters.ida_nst = 2;
            ida.ida_maxord = 5;
        }

        ida.complete_step(err_k, err_km1);

        #[rustfmt::skip]
        let ida_phi = array![
            [ 9.9999653607862404e-01, 3.4507668531616537e-06, 1.3154522852943008e-08, ],
            [ -1.7319593976777956e-06, 1.7212358252111640e-06, 1.0723572466631820e-08, ],
            [ 1.8301885167926423e-12, -6.8365381237749594e-09, 6.8347079352581675e-09, ],
            [ -4.2122294839452673e-13, -2.4605442771223734e-09, 2.4609655000707684e-09, ],
            [ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, ],
            [ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, ],
        ];
        let ida_ee = array![
            -4.2122294839452673e-13,
            -2.4605442771223734e-09,
            2.4609655000707684e-09,
        ];
        let ida_ewt = array![
            9.9990174161763662e+03,
            9.9982707680480811e+05,
            9.9999975690502045e+05,
        ];
        let kk = 3;
        let kused = 2;
        let knew = 2;
        let phase = 0;
        let hh = 8.6598211441923079e-05;
        let hused = 4.3299105720961540e-05;
        let rr = 0.0000000000000000e+00;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 3;
        let maxord = 5;

        assert_eq!(ida.counters.ida_nst, nst);
        assert_eq!(ida.ida_kk, kk);
        assert_eq!(ida.ida_hh, hh);
        assert_nearly_eq!(ida.ida_rr, rr);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_knew, knew);
        assert_eq!(ida.ida_maxord, maxord);
        assert_eq!(ida.ida_phase, phase);
        assert_eq!(ida.ida_hmax_inv, hmax_inv);
        assert_nearly_eq!(ida.ida_ee, ida_ee);
        assert_nearly_eq!(ida.ida_phi, ida_phi);
        assert_nearly_eq!(ida.nlp.ida_ewt, ida_ewt);
    }

    #[test]
    fn test_complete_step3() {
        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        let err_k = 0.04158869255496026;
        let err_km1 = 0.0434084688121079;
        // Set preconditions:
        {
            #[rustfmt::skip]
            let ida_phi = array![
                [ 9.9989651723441231e-01, 3.6526684773526134e-05, 6.6956080814055887e-05, ],
                [ -1.3744883185162313e-05, 1.6198423621611769e-07, 1.3582898948830698e-05, ],
                [ 1.8985246772161757e-09, -3.1828827656675421e-07, 3.1638975173052525e-07, ],
                [ 6.0782494622797157e-11, 3.6508292328524681e-07, -3.6514370600191418e-07, ],
                [ -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06, ],
                [ -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07, ],
            ];
            let ida_ee = array![
                -3.0358000609489579e-11,
                -2.0478050395116282e-07,
                2.0481086265121282e-07,
            ];
            let ida_ewt = array![
                1.0000034827777174e+04,
                9.9636062495315843e+05,
                9.9334892491071229e+05,
            ];
            let kk = 3;
            let kused = 2;
            let knew = 2;
            let phase = 1;
            let hh = 3.4384304814216195e-04;
            let hused = 3.4384304814216195e-04;
            let rr = 1.7379509697128959e+00;
            let hmax_inv = 0.0000000000000000e+00;
            let nst = 12;
            let maxord = 5;

            ida.counters.ida_nst = nst;
            ida.ida_kk = kk;
            ida.ida_hh = hh;
            ida.ida_rr = rr;
            ida.ida_kused = kused;
            ida.ida_hused = hused;
            ida.ida_knew = knew;
            ida.ida_maxord = maxord;
            ida.ida_phase = phase;
            ida.ida_hmax_inv = hmax_inv;
            ida.ida_ee.assign(&ida_ee);
            ida.ida_phi.assign(&ida_phi);
            ida.nlp.ida_ewt.assign(&ida_ewt);
        }

        ida.complete_step(err_k, err_km1);

        #[rustfmt::skip]
        let ida_phi = array![
            [ 9.9988277428017636e-01, 3.6530683152509582e-05, 8.0695036671266405e-05, ],
            [ -1.3742954235991083e-05, 3.9983789834474729e-09, 1.3738955857210521e-05, ],
            [ 1.9289491712294831e-09, -1.5798585723267022e-07, 1.5605690837982388e-07, ],
            [ 3.0424494013307578e-11, 1.6030241933408400e-07, -1.6033284335070136e-07, ],
            [ -3.0358000609489579e-11, -2.0478050395116282e-07, 2.0481086265121282e-07, ],
            [ -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07, ],
        ];
        let ida_ee = array![
            -3.0358000609489579e-11,
            -2.0478050395116282e-07,
            2.0481086265121282e-07,
        ];
        let ida_ewt = array![
            1.0000034827777174e+04,
            9.9636062495315843e+05,
            9.9334892491071229e+05,
        ];
        let kk = 2;
        let kused = 3;
        let knew = 2;
        let phase = 1;
        let hh = 6.8768609628432390e-04;
        let hused = 3.4384304814216195e-04;
        let rr = 2.2575213239991561e+00;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 13;
        let maxord = 5;

        assert_eq!(ida.counters.ida_nst, nst);
        assert_eq!(ida.ida_kk, kk);
        assert_eq!(ida.ida_hh, hh);
        assert_nearly_eq!(ida.ida_rr, rr);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_knew, knew);
        assert_eq!(ida.ida_maxord, maxord);
        assert_eq!(ida.ida_phase, phase);
        assert_eq!(ida.ida_hmax_inv, hmax_inv);
        assert_nearly_eq!(ida.ida_ee, ida_ee);
        assert_nearly_eq!(ida.ida_phi, ida_phi);
        assert_nearly_eq!(ida.nlp.ida_ewt, ida_ewt);
    }

    /*
    #[test]
    fn test_nonlinear_solve() {
        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        ida.ida_phi = array![
            [0.999993, 0.000007, 0.000000,],
            [-0.000007, 0.000007, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, 0.000000, 0.000000,],
        ];
        ida.ida_psi = array![
            1.7319642288384616e-04,
            2.5979463432576925e-04,
            3.0309374004673076e-04,
            3.2474329290721156e-04,
            3.4639284576769232e-04,
            0.0000000000000000e+00,
        ];
        ida.ida_alpha = array![
            1.0000000000000000e+00,
            6.6666666666666663e-01,
            5.7142857142857151e-01,
            5.3333333333333333e-01,
            5.0000000000000000e-01,
            0.0000000000000000e+00,
        ];
        ida.ida_beta = array![
            1.0000000000000000e+00,
            2.0000000000000000e+00,
            4.0000000000000000e+00,
            8.0000000000000000e+00,
            1.5000000000000000e+01,
            0.0000000000000000e+00,
        ];
        ida.ida_sigma = array![
            1.0000000000000000e+00,
            6.6666666666666663e-01,
            7.6190476190476197e-01,
            1.2190476190476192e+00,
            2.4380952380952383e+00,
            0.0000000000000000e+00,
        ];
        ida.ida_gamma = array![
            0.0000000000000000e+00,
            5.7737913020908527e+03,
            9.6229855034847533e+03,
            1.2922294818965242e+04,
            1.6001650180080363e+04,
            0.0000000000000000e+00,
        ];
        ida.nlp.ida_yy = array![0.999993, 0.000007, 0.000000];
        ida.nlp.ida_yp = array![-0.040000, 0.038555, 0.001444];
        ida.nlp.ida_yypredict = array![0.999986, 0.000013, 0.000001];
        ida.nlp.ida_yppredict = array![-0.039998, 0.032311, 0.007687];
        let ida_delta = [0.000000, -0.000000, 0.000000];
        let ida_saves = [0.000000, -0.000188, 0.000000];
        let ida_ee = [0.000000, -0.000000, 0.000000];
        let ida_ewt = [9999.069365, 999316.472678, 999991.211854];
        let hh = 0.0001731964228838;
        let kk = 4;
        let kused = 3;
        let knew = 3;
        let phase = 0;
        let ns = 1;
        let hin = 0.0000000000000000e+00;
        let h0u = 2.1649552860480770e-05;
        let hh = 1.7319642288384616e-04;
        let hused = 8.6598211441923079e-05;
        let rr = 0.0000000000000000e+00;
        let tn = 3.4639284576769232e-04;
        let tretlast = 0.0000000000000000e+00;
        let cj = 1.2028731879355941e+04;
        let cjlast = 2.1170568107666459e+04;
        let cjold = 2.1170568107666459e+04;
        let cjratio = 1.0000000000000000e+00;
        let ss = 2.0000000000000000e+01;
        let oldnrm = 7.1165780617896508e-03;
        let epsNewt = 3.3000000000000002e-01;
        let epcon = 3.3000000000000002e-01;
        let toldel = 3.3000000000000003e-05;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 4;
        let nre = 4;
        let ncfn = 0;
        let netf = 0;
        let nni = 0;
        let nsetups = 2;
        let maxord = 5;

        //---
        let ida_phi = array![
            [0.999993, 0.000007, 0.000000,],
            [-0.000007, 0.000007, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, -0.000000, 0.000000,],
            [0.000000, 0.000000, 0.000000,],
        ];
        let ida_psi = array![
            1.7319642288384616e-04,
            2.5979463432576925e-04,
            3.0309374004673076e-04,
            3.2474329290721156e-04,
            3.4639284576769232e-04,
            0.0000000000000000e+00,
        ];
        let ida_alpha = array![
            1.0000000000000000e+00,
            6.6666666666666663e-01,
            5.7142857142857151e-01,
            5.3333333333333333e-01,
            5.0000000000000000e-01,
            0.0000000000000000e+00,
        ];
        let ida_beta = array![
            1.0000000000000000e+00,
            2.0000000000000000e+00,
            4.0000000000000000e+00,
            8.0000000000000000e+00,
            1.5000000000000000e+01,
            0.0000000000000000e+00,
        ];
        let ida_sigma = array![
            1.0000000000000000e+00,
            6.6666666666666663e-01,
            7.6190476190476197e-01,
            1.2190476190476192e+00,
            2.4380952380952383e+00,
            0.0000000000000000e+00,
        ];
        let ida_gamma = array![
            0.0000000000000000e+00,
            5.7737913020908527e+03,
            9.6229855034847533e+03,
            1.2922294818965242e+04,
            1.6001650180080363e+04,
            0.0000000000000000e+00,
        ];
        let ida_yy = [0.999986, 0.000013, 0.000001];
        let ida_yp = [-0.039999, 0.034719, 0.005280];
        let ida_yypredict = [0.999986, 0.000013, 0.000001];
        let ida_yppredict = [-0.039998, 0.032311, 0.007687];
        let ida_delta = [0.000000, 0.000000, 0.000000];
        let ida_saves = [-0.000000, -0.000001, 0.000000];
        let ida_ee = [-0.000000, 0.000000, -0.000000];
        let ida_ewt = [9999.069365, 999316.472678, 999991.211854];
        let hh = 0.0001731964228838;
        let kk = 4;
        let kused = 3;
        let knew = 3;
        let phase = 0;
        let ns = 1;
        let hin = 0.0000000000000000e+00;
        let h0u = 2.1649552860480770e-05;
        let hh = 1.7319642288384616e-04;
        let hused = 8.6598211441923079e-05;
        let rr = 0.0000000000000000e+00;
        let tn = 3.4639284576769232e-04;
        let tretlast = 0.0000000000000000e+00;
        let cj = 1.2028731879355941e+04;
        let cjlast = 2.1170568107666459e+04;
        let cjold = 1.2028731879355941e+04;
        let cjratio = 1.0000000000000000e+00;
        let ss = 4.6930097883603612e-04;
        let oldnrm = 1.6345169955559977e-01;
        let epsNewt = 3.3000000000000002e-01;
        let epcon = 3.3000000000000002e-01;
        let toldel = 3.3000000000000003e-05;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 4;
        let nre = 6;
        let ncfn = 0;
        let netf = 0;
        let nni = 0;
        let nsetups = 3;
        let maxord = 5;
    }
    */

    #[test]
    fn test_get_solution() {
        // --- IDAGetSolution Before:
        let t = 3623118336.24244;
        let hh = 857870592.1885694;
        let tn = 3623118336.24244;
        let kused = 4;
        let hused = 428935296.0942847;

        #[rustfmt::skip]
        let ida_phi = array![
            [ 5.716499633245077e-07, 2.286601144610028e-12, 0.9999994283477499, ],
            [ -7.779233860067279e-08, -3.111697299545603e-13, 7.779264957586927e-08, ],
            [ 2.339417551980491e-08, 9.35768837422748e-14, -2.33942692332846e-08, ],
            [ -9.503346432581604e-09, -3.801349575270522e-14, 9.503383895634436e-09, ],
            [ 7.768373161310588e-09, 3.107357755532867e-14, -7.768407422476745e-09, ],
            [ -2.242367216194777e-10, -8.970915966733762e-16, 2.242247401239887e-10, ],
        ];
        let ida_psi = array![
            428935296.0942847,
            857870592.1885694,
            1072338240.235712,
            1286805888.282854,
            1501273536.329997,
            26020582.4876316,
        ];

        //--- IDAGetSolution After:
        let yret_expect = array![
            5.716499633245077e-07,
            2.286601144610028e-12,
            0.9999994283477499,
        ];
        let ypret_expect = array![
            -1.569167478317552e-16,
            -6.276676917262037e-22,
            1.569173718962504e-16,
        ];

        let problem = Dummy {};
        let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
            problem,
            array![0., 0., 0.],
            array![0., 0., 0.],
            TolControlSS::new(1e-4, 1e-4),
        );

        ida.ida_hh = hh;
        ida.nlp.ida_tn = tn;
        ida.ida_kused = kused;
        ida.ida_hused = hused;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_psi.assign(&ida_psi);

        ida.get_solution(t).unwrap();

        assert_nearly_eq!(ida.get_yy(), yret_expect);
        assert_nearly_eq!(ida.get_yp(), ypret_expect);
    }
}
