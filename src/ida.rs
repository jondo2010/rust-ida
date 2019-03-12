use log::error;
use ndarray::*;

use num_traits::{
    cast::{FromPrimitive, NumCast, ToPrimitive},
    identities::{One, Zero},
    Float, NumAssignRef, NumRef,
};

/*
num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ScalarOperand
        */

use crate::constants::*;
use crate::ida_ls::*;
use crate::nonlinear::*;
use crate::traits::*;

pub use crate::error::IdaError;

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

/// Structure containing the parameters for the numerical integration.
#[derive(Debug, Clone)]
pub struct Ida<
    F: IdaModel,
    /*NLS: NLSolver*/
> {
    f: F,
    //dt: <F::Scalar as AssociatedReal>::Real,
    //x: Array<F::Scalar, Ix1>,
    ida_itol: ToleranceType,
    /// relative tolerance
    ida_rtol: F::Scalar,
    /// scalar absolute tolerance
    ida_Satol: F::Scalar,
    /// vector absolute tolerance
    ida_Vatol: Array1<F::Scalar>,

    /// constraints vector present: do constraints calc
    ida_constraintsSet: bool,
    /// SUNTRUE means suppress algebraic vars in local error tests
    ida_suppressalg: bool,

    // Divided differences array and associated minor arrays
    /// phi = (maxord+1) arrays of divided differences
    ida_phi: Array<F::Scalar, Ix2>,
    /// differences in t (sums of recent step sizes)
    ida_psi: Array1<F::Scalar>,
    /// ratios of current stepsize to psi values
    ida_alpha: Array1<F::Scalar>,
    /// ratios of current to previous product of psi's
    ida_beta: Array1<F::Scalar>,
    /// product successive alpha values and factorial
    ida_sigma: Array1<F::Scalar>,
    /// sum of reciprocals of psi values
    ida_gamma: Array1<F::Scalar>,

    // N_Vectors
    /// error weight vector
    ida_ewt: Array<F::Scalar, Ix1>,
    /// work space for y vector (= user's yret)
    //ida_yy: Array<F::Scalar, Ix1>,
    /// work space for y' vector (= user's ypret)
    //ida_yp: Array<F::Scalar, Ix1>,
    /// predicted y vector
    ida_yypredict: Array<F::Scalar, Ix1>,
    /// predicted y' vector
    ida_yppredict: Array<F::Scalar, Ix1>,
    /// residual vector
    ida_delta: Array<F::Scalar, Ix1>,
    /// bit vector for diff./algebraic components
    ida_id: Array<bool, Ix1>,
    /// vector of inequality constraint options
    //ida_constraints: Array1<<F::Scalar as AssociatedReal>::Real>,
    /// saved residual vector
    //ida_savres: Array1<<F::Scalar as AssociatedReal>::Real>,
    /// accumulated corrections to y vector, but set equal to estimated local errors upon successful return
    ida_ee: Array<F::Scalar, Ix1>,

    //ida_mm;          /* mask vector in constraints tests (= tempv2)    */
    //ida_tempv1;      /* work space vector                              */
    //ida_tempv2;      /* work space vector                              */
    //ida_tempv3;      /* work space vector                              */
    //ida_ynew;        /* work vector for y in IDACalcIC (= tempv2)      */
    //ida_ypnew;       /* work vector for yp in IDACalcIC (= ee)         */
    //ida_delnew;      /* work vector for delta in IDACalcIC (= phi[2])  */
    //ida_dtemp;       /* work vector in IDACalcIC (= phi[3])            */

    // Tstop information
    ida_tstopset: bool,
    ida_tstop: F::Scalar,

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
    ida_hin: F::Scalar,
    /// actual initial stepsize
    ida_h0u: F::Scalar,
    /// current step size h
    ida_hh: F::Scalar,
    /// step size used on last successful step
    ida_hused: F::Scalar,
    /// rr = hnext / hused
    ida_rr: F::Scalar,
    /// current internal value of t
    ida_tn: F::Scalar,
    /// value of tret previously returned by IDASolve
    ida_tretlast: F::Scalar,
    /// current value of scalar (-alphas/hh) in Jacobian
    ida_cj: F::Scalar,
    /// cj value saved from last successful step
    ida_cjlast: F::Scalar,
    /// cj value saved from last call to lsetup
    ida_cjold: F::Scalar,
    /// ratio of cj values: cj/cjold
    ida_cjratio: F::Scalar,
    /// scalar used in Newton iteration convergence test
    ida_ss: F::Scalar,
    /// norm of previous nonlinear solver update
    ida_oldnrm: F::Scalar,
    /// test constant in Newton convergence test
    ida_epsNewt: F::Scalar,
    /// coeficient of the Newton covergence test
    ida_epcon: F::Scalar,
    /// tolerance in direct test on Newton corrections
    ida_toldel: F::Scalar,

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
    ida_hmax_inv: F::Scalar,

    // Counters
    /// number of internal steps taken
    ida_nst: u64,
    /// number of function (res) calls
    ida_nre: u64,
    /// number of corrector convergence failures
    ida_ncfn: u64,
    /// number of error test failures
    ida_netf: u64,
    /// number of Newton iterations performed
    ida_nni: u64,
    /// number of lsetup calls
    ida_nsetups: u64,
    // Arrays for Fused Vector Operations
    ida_cvals: Array1<F::Scalar>,
    ida_dvals: Array1<F::Scalar>,

    /// tolerance scale factor (saved value)
    ida_tolsf: F::Scalar,

    //IDARootFn ida_gfun;       /* Function g for roots sought                     */
    /// number of components of g
    ida_nrtfn: usize,
    //int *ida_iroots;          /* array for root information                      */
    //int *ida_rootdir;         /* array specifying direction of zero-crossing     */
    /// nearest endpoint of interval in root search
    //ida_tlo: F::Scalar,
    /// farthest endpoint of interval in root search
    //ida_thi: F::Scalar,
    /// t return value from rootfinder routine
    //ida_trout: F::Scalar,
    /// saved array of g values at t = tlo
    //ida_glo: Array1<F::Scalar>,
    //realtype *ida_ghi;        /* saved array of g values at t = thi              */
    //realtype *ida_grout;      /* array of g values at t = trout                  */
    //realtype ida_toutc;       /* copy of tout (if NORMAL mode)                   */
    /// tolerance on root location
    //ida_ttol: F::Scalar,
    //int ida_taskc;            /* copy of parameter itask                         */
    //int ida_irfnd;            /* flag showing whether last step had a root       */
    /// counter for g evaluations
    ida_nge: u64,
    //booleantype *ida_gactive; /* array with active/inactive event functions      */
    //int ida_mxgnull;          /* number of warning messages about possible g==0  */

    // Arrays for Fused Vector Operations
    ida_Xvecs: Array<F::Scalar, Ix2>,
    ida_Zvecs: Array<F::Scalar, Ix2>,
}

impl<F /*, NLS*/> Ida<F /*, NLS*/>
where
    F: IdaModel,
    <F as ModelSpec>::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ScalarOperand
        + std::fmt::Debug
        + IdaConst,
{
    /// Creates a new IdaModel given a ModelSpec, initial Arrays of yy0 and yyp
    ///
    /// *Panics" if ModelSpec::Scalar is unable to convert any constant initialization value.
    pub fn new(f: F, yy0: Array<F::Scalar, Ix1>, yp0: Array<F::Scalar, Ix1>) -> Self {
        // Initialize the phi array
        let mut ida_phi = Array::zeros(f.model_size())
            .broadcast([&[MXORDP1], yy0.shape()].concat())
            .unwrap()
            .into_dimensionality::<_>()
            .unwrap()
            .to_owned();

        ida_phi.index_axis_mut(Axis(0), 0).assign(&yy0);
        ida_phi.index_axis_mut(Axis(0), 1).assign(&yp0);

        //IDAResFn res, realtype t0, N_Vector yy0, N_Vector yp0
        Self {
            f: f,
            // Set unit roundoff in IDA_mem
            // NOTE: Use F::Scalar::epsilon() instead!
            //ida_uround: UNIT_ROUNDOFF,

            // Set default values for integrator optional inputs
            //ida_res:         = NULL,
            //ida_user_data:   = NULL,
            ida_itol: ToleranceType::TolNN,
            //ida_user_efun   = SUNFALSE;
            //ida_efun        = NULL;
            //ida_edata       = NULL;
            //ida_ehfun       = IDAErrHandler;
            //ida_eh_data     = IDA_mem;
            //ida_errfp       = stderr;
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as u64,
            ida_hmax_inv: NumCast::from(HMAX_INV_DEFAULT).unwrap(),
            ida_hin: F::Scalar::zero(),
            ida_epsNewt: F::Scalar::zero(),
            ida_epcon: NumCast::from(EPCON).unwrap(),
            ida_toldel: F::Scalar::zero(),
            ida_maxnef: MXNEF as u64,
            ida_maxncf: MXNCF as u64,
            ida_suppressalg: false,
            //ida_id          = NULL;
            //ida_constraints: Array::zeros(yy0.raw_dim()),
            ida_constraintsSet: false,
            ida_tstopset: false,

            // set the saved value maxord_alloc
            //ida_maxord_alloc = MAXORD_DEFAULT;

            // Set default values for IC optional inputs
            //ida_epiccon : F::Scalar::from(0.01 * EPCON).unwrap(),
            //ida_maxnh   : MAXNH,
            //ida_maxnj   = MAXNJ;
            //ida_maxnit  = MAXNI;
            //ida_maxbacks  = MAXBACKS;
            //ida_lsoff   = SUNFALSE;
            //ida_steptol = SUNRpowerR(self.ida_uround, TWOTHIRDS);

            /* Initialize lrw and liw */
            //ida_lrw = 25 + 5*MXORDP1;
            //ida_liw = 38;

            /* Initialize nonlinear solver pointer */
            //self.NLS    = NULL;
            //self.ownNLS = SUNFALSE;
            ida_phi: ida_phi,

            ida_psi: Array::zeros(MXORDP1),
            ida_alpha: Array::zeros(MXORDP1),
            ida_beta: Array::zeros(MXORDP1),
            ida_sigma: Array::zeros(MXORDP1),
            ida_gamma: Array::zeros(MXORDP1),

            ida_delta: Array::zeros(yy0.raw_dim()),
            ida_id: Array::from_elem(yy0.raw_dim(), false),

            // Initialize all the counters and other optional output values
            ida_nst: 0,
            ida_nre: 0,
            ida_ncfn: 0,
            ida_netf: 0,
            ida_nni: 0,
            ida_nsetups: 0,
            ida_kused: 0,
            ida_hused: F::Scalar::zero(),
            ida_tolsf: F::Scalar::one(),
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
            ida_ewt: Array::zeros(yy0.raw_dim()),
            ida_ee: Array::zeros(yy0.raw_dim()),

            ida_tstop: F::Scalar::zero(),

            ida_kk: 0,
            //ida_kused: 0,
            ida_knew: 0,
            ida_phase: 0,
            ida_ns: 0,

            ida_rr: F::Scalar::zero(),
            ida_tn: F::Scalar::zero(),
            ida_tretlast: F::Scalar::zero(),
            ida_h0u: F::Scalar::zero(),
            ida_hh: F::Scalar::zero(),
            //ida_hused: <F::Scalar as AssociatedReal>::Real::from_f64(0.0),
            ida_cj: F::Scalar::zero(),
            ida_cjlast: F::Scalar::zero(),
            ida_cjold: F::Scalar::zero(),
            ida_cjratio: F::Scalar::zero(),
            ida_oldnrm: F::Scalar::zero(),
            ida_ss: F::Scalar::zero(),

            ida_cvals: Array::zeros(MXORDP1),
            ida_dvals: Array::zeros(MAXORD_DEFAULT),

            ida_Xvecs: Array::zeros((MXORDP1, yy0.shape()[0])),
            ida_Zvecs: Array::zeros((MXORDP1, yy0.shape()[0])),

            ida_yypredict: Array::zeros(yy0.raw_dim()),
            ida_yppredict: Array::zeros(yy0.raw_dim()),

            ida_rtol: F::Scalar::zero(),
            ida_Satol: F::Scalar::zero(),
            ida_Vatol: Array::zeros(MXORDP1),
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
    /// * `yret` The computed solution vector y (output).
    /// * `ypret` The computed solution vector ˙y (output).
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
    fn solve<S1, S2>(
        &mut self,
        tout: F::Scalar,
        tret: &mut F::Scalar,
        yret: &mut ArrayBase<S1, Ix1>,
        ypret: &mut ArrayBase<S2, Ix1>,
        itask: &IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error>
    where
        S1: DataMut<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
        ArrayBase<S1, Ix1>: IntoNdProducer,
        ArrayBase<S2, Ix1>: IntoNdProducer,
    {
        if self.ida_nst == 0 {
            // This is the first call

            // Check inputs to IDA for correctness and consistency */
            /*
            if (self.ida_SetupDone == SUNFALSE) {
              ier = IDAInitialSetup(IDA_mem);
              if (ier != IDA_SUCCESS) return(IDA_ILL_INPUT);
              self.ida_SetupDone = SUNTRUE;
            }
            */

            // On first call, check for tout - tn too small, set initial hh, check for approach to tstop, and scale phi[1] by hh. Also check for zeros of root function g at and near t0.

            let tdist = (tout - self.ida_tn).abs();
            if tdist == F::Scalar::zero() {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }
            let troundoff =
                F::Scalar::two() * F::Scalar::epsilon() * (self.ida_tn.abs() + tout.abs());
            if tdist < troundoff {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }

            self.ida_hh = self.ida_hin;
            if (self.ida_hh != F::Scalar::zero())
                && ((tout - self.ida_tn) * self.ida_hh < F::Scalar::zero())
            {
                Err(IdaError::IllegalInput {
                    msg: format!("Initial step is not towards tout."),
                })?
            }

            if self.ida_hh == F::Scalar::zero() {
                self.ida_hh = F::Scalar::pt001() * tdist;
                let ypnorm = self.wrms_norm(
                    &self.ida_phi.index_axis(Axis(0), 1),
                    &self.ida_ewt,
                    self.ida_suppressalg,
                );
                if ypnorm > F::Scalar::two() / self.ida_hh {
                    self.ida_hh = F::Scalar::half() / ypnorm;
                }
                if tout < self.ida_tn {
                    self.ida_hh = -self.ida_hh;
                }
            }

            let rh = (self.ida_hh).abs() * self.ida_hmax_inv;
            if rh > F::Scalar::one() {
                self.ida_hh /= rh;
            }

            if self.ida_tstopset {
                if (self.ida_tstop - self.ida_tn) * self.ida_hh <= F::Scalar::zero() {
                    Err(IdaError::IllegalInput {
                        msg: format!(
                            "The value tstop = {:?} \
                             is behind current t = {:?} \
                             in the direction of integration.",
                            self.ida_tstop, self.ida_tn
                        ),
                    })?
                }
                if (self.ida_tn + self.ida_hh - self.ida_tstop) * self.ida_hh > F::Scalar::zero() {
                    self.ida_hh = (self.ida_tstop - self.ida_tn)
                        * (F::Scalar::one() - F::Scalar::four() * F::Scalar::epsilon());
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
            //    IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck1", MSG_RTFUNC_FAILED, self.ida_tn);
            //    return(IDA_RTFUNC_FAIL);
            //  }
            //}

            //N_VScale(self.ida_hh, self.ida_phi[1], self.ida_phi[1]);  /* set phi[1] = hh*y' */
            let mut phi = self.ida_phi.index_axis_mut(Axis(0), 1);
            phi *= self.ida_hh;

            // Set the convergence test constants epsNewt and toldel
            self.ida_epsNewt = self.ida_epcon;
            self.ida_toldel = F::Scalar::pt001() * self.ida_epsNewt;
        } // end of first-call block.

        // Call lperf function and set nstloc for later performance testing.
        //if self.ida_lperf != NULL {self.ida_lperf(IDA_mem, 0);}
        let mut nstloc = 0;

        // If not the first call, perform all stopping tests.

        if self.ida_nst > 0 {
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
                troundoff = HUNDRED * self.ida_uround * (SUNRabs(self.ida_tn) + SUNRabs(self.ida_hh));
                if ( SUNRabs(self.ida_tn - self.ida_tretlast) > troundoff ) {
                ier = IDARcheck3(IDA_mem);
                if (ier == IDA_SUCCESS) {     /* no root found */
                self.ida_irfnd = 0;
                if ((irfndp == 1) && (itask == IDA_ONE_STEP)) {
                self.ida_tretlast = *tret = self.ida_tn;
                ier = IDAGetSolution(IDA_mem, self.ida_tn, yret, ypret);
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

            let istate = self.stop_test1(tout, tret, yret, ypret, itask)?;
            //let istate = IDAStopTest1(tout, tret, yret, ypret, itask);
            //if istate != CONTINUE_STEPS { return (istate); }
        }

        // Looping point for internal steps.

        loop {
            // Check for too many steps taken.

            if (self.ida_mxstep > 0) && (nstloc >= self.ida_mxstep) {
                *tret = self.ida_tn;
                self.ida_tretlast = self.ida_tn;
                // Here yy=yret and yp=ypret already have the current solution.
                Err(IdaError::IllegalInput {
                    msg: format!(
                        "At t = {:?}, mxstep steps taken before reaching tout.",
                        self.ida_tn
                    ),
                })?
                //istate = IDA_TOO_MUCH_WORK;
            }

            // Call lperf to generate warnings of poor performance.

            //if (self.ida_lperf != NULL)
            //  self.ida_lperf(IDA_mem, 1);

            // Reset and check ewt (if not first call).

            if self.ida_nst > 0 {
                //ier = self.ida_efun(self.ida_phi[0], self.ida_ewt, self.ida_edata);
                let ier = 0;

                if ier != 0 {
                    self.get_solution(self.ida_tn, yret, ypret);
                    *tret = self.ida_tn;
                    self.ida_tretlast = self.ida_tn;

                    match self.ida_itol {
                        ToleranceType::TolWF => {
                            //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_EWT_NOW_FAIL, self.ida_tn);
                            Err(IdaError::IllegalInput {
                                msg: format!(
                                    "At t = {:?} the user-provide EwtSet function failed.",
                                    self.ida_tn
                                ),
                            })?
                        }
                        _ => {
                            //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_EWT_NOW_BAD, self.ida_tn);
                            Err(IdaError::IllegalInput {
                                msg: format!(
                                    "At t = {:?} some ewt component has become <= 0.0.",
                                    self.ida_tn
                                ),
                            })?
                        }
                    }
                    //istate = IDA_ILL_INPUT;
                    //break;
                }
            }

            // Check for too much accuracy requested.

            let nrm = self.wrms_norm(
                &self.ida_phi.index_axis(Axis(0), 0),
                &self.ida_ewt,
                self.ida_suppressalg,
            );

            self.ida_tolsf = F::Scalar::epsilon() * nrm;
            if self.ida_tolsf > F::Scalar::one() {
                self.ida_tolsf *= F::Scalar::ten();

                *tret = self.ida_tn;
                self.ida_tretlast = self.ida_tn;
                if self.ida_nst > 0 {
                    let ier = self.get_solution(self.ida_tn, yret, ypret);
                }
                //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_TOO_MUCH_ACC, self.ida_tn);
                //istate = IDA_TOO_MUCH_ACC;
                //break;
                Err(IdaError::TooMuchAccuracy {
                    t: self.ida_tn.to_f64().unwrap(),
                })?
            }

            // Call IDAStep to take a step.

            let sflag = self.step();

            // Process all failed-step cases, and exit loop.

            sflag.map_err(|err| {
                let ier = self.get_solution(self.ida_tn, yret, ypret);
                match ier {
                    Ok(_) => {
                        *tret = self.ida_tn;
                        self.ida_tretlast = self.ida_tn;
                    }
                    Err(e2) => {
                        error!("Error occured with get_solution: {:?}", e2.as_fail());
                        //err.context(e2)
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
                if (self.ida_nst==1) {
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

            self.stop_test2(tout, tret, yret, ypret, itask)?;
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
    ///   IDA_VECTOROP_ERR  if the fused vector operation fails
    pub fn get_dky<S>(
        &mut self,
        t: F::Scalar,
        k: usize,
        dky: &mut ArrayBase<S, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S: DataMut<Elem = F::Scalar>,
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

        let tfuzz = F::Scalar::hundred()
            * F::Scalar::epsilon()
            * (self.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();
        let tp = self.ida_tn - self.ida_hused - tfuzz;
        if (t - tp) * self.ida_hh < F::Scalar::zero() {
            Err(IdaError::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.ida_tn.to_f64().unwrap(),
            })?
            //IDAProcessError(IDA_mem, IDA_BAD_T, "IDA", "IDAGetDky", MSG_BAD_T, t, self.ida_tn-self.ida_hused, self.ida_tn);
            //return(IDA_BAD_T);
        }

        // Initialize the c_j^(k) and c_k^(k-1)
        let mut cjk = Array::zeros(MXORDP1);
        let mut cjk_1 = Array::zeros(MXORDP1);

        let delt = t - self.ida_tn;
        let mut psij_1 = F::Scalar::zero();

        for i in 0..k + 1 {
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
                cjk[i] = F::Scalar::one();
            } else {
                //                                                i       i-1          1
                // c_i^(i) can be always updated since c_i^(i) = -----  --------  ... -----
                //                                               psi_j  psi_{j-1}     psi_1
                cjk[i] = cjk[i - 1] * NumCast::from(i as f64).unwrap() / self.ida_psi[i - 1];
                psij_1 = self.ida_psi[i - 1];
            }

            // update c_j^(i)
            //j does not need to go till kused
            //for(j=i+1; j<=self.ida_kused-k+i; j++) {
            for j in i + 1..self.ida_kused - k + 1 + 1 {
                cjk[j] = (NumCast::from(i as f64).unwrap() * cjk_1[j - 1]
                    + cjk[j - 1] * (delt + psij_1))
                    / self.ida_psi[j - 1];
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

    /// IDAEwtSet
    ///
    /// This routine is responsible for loading the error weight vector
    /// ewt, according to itol, as follows:
    /// (1) ewt[i] = 1 / (rtol * SUNRabs(ycur[i]) + atol), i=0,...,Neq-1
    ///     if itol = IDA_SS
    /// (2) ewt[i] = 1 / (rtol * SUNRabs(ycur[i]) + atol[i]), i=0,...,Neq-1
    ///     if itol = IDA_SV
    ///
    ///  `ewt_set` returns true if ewt is successfully set as above to a
    ///  positive vector and false otherwise. In the latter case, ewt is
    ///  considered undefined.
    ///
    /// All the real work is done in the routines `ewt_set_ss`, `ewt_set_sv`.
    pub fn ewt_set<S1, S2>(
        &mut self,
        ycur: &ArrayBase<S1, Ix1>,
        weight: &mut ArrayBase<S2, Ix1>,
    ) -> bool
    where
        S1: Data<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
    {
        match self.ida_itol {
            ToleranceType::TolSS => self.ewt_set_ss(ycur, weight),
            ToleranceType::TolSV => self.ewt_set_sv(ycur, weight),
            _ => false,
        }
    }

    /// IDAEwtSetSS
    ///
    /// This routine sets ewt as decribed above in the case itol=IDA_SS.
    /// It tests for non-positive components before inverting. IDAEwtSetSS
    /// returns 0 if ewt is successfully set to a positive vector
    /// and -1 otherwise. In the latter case, ewt is considered
    /// undefined.
    fn ewt_set_ss<S1, S2>(
        &mut self,
        ycur: &ArrayBase<S1, Ix1>,
        weight: &mut ArrayBase<S2, Ix1>,
    ) -> bool
    where
        S1: Data<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
    {
        let tempv1 = ycur.mapv(|x| (self.ida_rtol * x.abs()) + self.ida_Satol);
        //self.ida_tempv1.zip_mut_with(ycur, |z, &a| { *z = });
        /*
        N_VAbs(ycur, self.ida_tempv1);
        N_VScale(self.ida_rtol, self.ida_tempv1, self.ida_tempv1);
        N_VAddConst(self.ida_tempv1, self.ida_Satol, self.ida_tempv1);
        */
        if tempv1.fold(F::Scalar::max_value(), |acc, &x| acc.min(x)) <= F::Scalar::zero() {
            //if tempv1.iter().min().unwrap() <= &F::Scalar::zero() {
            false
        } else {
            weight.zip_mut_with(&tempv1, |w, &t| *w = t.recip());
            true
        }

        //if (N_VMin(self.ida_tempv1) <= ZERO) {return false}
        //N_VInv(self.ida_tempv1, weight);
        //return true;
    }

    /// IDAEwtSetSV
    ///
    /// This routine sets ewt as decribed above in the case itol=IDA_SV.
    /// It tests for non-positive components before inverting. IDAEwtSetSV
    /// returns 0 if ewt is successfully set to a positive vector
    /// and -1 otherwise. In the latter case, ewt is considered
    /// undefined.
    fn ewt_set_sv<S1, S2>(
        &mut self,
        ycur: &ArrayBase<S1, Ix1>,
        weight: &mut ArrayBase<S2, Ix1>,
    ) -> bool
    where
        S1: Data<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
    {
        /*
        N_VAbs(ycur, self.ida_tempv1);
        N_VLinearSum(self.ida_rtol, self.ida_tempv1, ONE, self.ida_Vatol, self.ida_tempv1);
        if (N_VMin(self.ida_tempv1) <= ZERO) {return false;}
        N_VInv(self.ida_tempv1, weight);
        return true;
        */
        unimplemented!();
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
    fn stop_test1<S1, S2>(
        &mut self,
        tout: F::Scalar,
        tret: &mut F::Scalar,
        yret: &mut ArrayBase<S1, Ix1>,
        ypret: &mut ArrayBase<S2, Ix1>,
        itask: &IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error>
    where
        S1: DataMut<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
        ArrayBase<S1, Ix1>: IntoNdProducer,
        ArrayBase<S2, Ix1>: IntoNdProducer,
    {
        match itask {
            IdaTask::Normal => {
                if self.ida_tstopset {
                    // Test for tn past tstop, tn = tretlast, tn past tout, tn near tstop.
                    if (self.ida_tn - self.ida_tstop) * self.ida_hh > F::Scalar::zero() {
                        //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_BAD_TSTOP, self.ida_tstop, self.ida_tn);
                        //return(IDA_ILL_INPUT);
                        Err(IdaError::BadStopTime {
                            tstop: self.ida_tstop.to_f64().unwrap(),
                            t: self.ida_tn.to_f64().unwrap(),
                        })?
                    }
                }

                // Test for tout = tretlast, and for tn past tout.
                if tout == self.ida_tretlast {
                    //*tret = self.ida_tretlast = tout;
                    //return(IDA_SUCCESS);
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if (self.ida_tn - tout) * self.ida_hh >= F::Scalar::zero() {
                    self.get_solution(tout, yret, ypret)?;
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if self.ida_tstopset {
                    let troundoff = F::Scalar::hundred()
                        * F::Scalar::epsilon()
                        * (self.ida_tn.abs() + self.ida_hh.abs());
                    if (self.ida_tn - self.ida_tstop).abs() <= troundoff {
                        /*
                        self.get_solution(self.ida_tstop, yret, ypret)
                            .map_err(|e| IdaError::BadStopTime {
                                tstop: self.ida_tstop.to_f64().unwrap(),
                                t: self.ida_tn.to_f64().unwrap(),
                            })
                            .and_then(|_| {
                                self.ida_tretlast = self.ida_tstop;
                                *tret = self.ida_tstop;
                                self.ida_tstopset = false;
                            })
                        */
                        //if (ier != IDA_SUCCESS) {
                        //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_BAD_TSTOP, self.ida_tstop, self.ida_tn);
                        //return(IDA_ILL_INPUT);
                        //}
                        //return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.ida_tn + self.ida_hh - self.ida_tstop) * self.ida_hh
                        > F::Scalar::zero()
                    {
                        self.ida_hh = (self.ida_tstop - self.ida_tn)
                            * (F::Scalar::one() - F::Scalar::four() * F::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }
            IdaTask::OneStep => Ok(IdaSolveStatus::Success),
        }
    }

    /// IDAStopTest2
    ///
    /// This routine tests for stop conditions after taking a step.
    /// The tests depend on the value of itask.
    ///
    /// The return values are:
    ///  CONTINUE_STEPS     if no stop conditions were found
    ///  IDA_SUCCESS        for a normal return to the user
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
    fn stop_test2<S1, S2>(
        &mut self,
        tout: F::Scalar,
        tret: &mut F::Scalar,
        yret: &ArrayBase<S1, Ix1>,
        ypret: &mut ArrayBase<S2, Ix1>,
        itask: &IdaTask,
    ) -> Result<(), IdaError>
    where
        S1: Data<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
    {
        unimplemented!()
        /*
        match itask {
            IdaTask::Normal => {
                // Test for tn past tout.
                if (self.ida_tn - tout) * self.ida_hh >= F::Scalar::zero() {
                    // /* ier = */ IDAGetSolution(IDA_mem, tout, yret, ypret);
                    *tret = tout;
                    self.ida_tretlast = tout;
                    self.get_solution(tout, yret, ypret)?;
                    return Ok(IdaSolveStatus::Success);
                }

                if (self.ida_tstopset) {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = F::Scalar::hundred()
                        * F::Scalar::epsilon()
                        * (self.ida_tn.abs() + self.ida_hh.abs());
                    if (self.ida_tn - self.ida_tstop).abs() <= troundoff {
                        /* ier = */
        self.get_solution(self.ida_tstop, yret, ypret);
         *tret = self.ida_tretlast = self.ida_tstop;
        self.ida_tstopset = false;
        //return(IDA_TSTOP_RETURN);

        //Err(IdaError::BadStopTime { tstop: self.ida_tstop.to_f64().unwrap(), t: self.ida_tn.to_f64().unwrap(), })?
        }
        if (self.ida_tn + self.ida_hh - self.ida_tstop) * self.ida_hh
        > F::Scalar::zero()
        {
        self.ida_hh = (self.ida_tstop - self.ida_tn)
         * (F::Scalar::one() - F::Scalar::four() * self.ida_uround);
        }
        }

        //return(CONTINUE_STEPS);
        }

        IdaTask::OneStep => {
        if self.ida_tstopset {
        /* Test for tn at tstop and for tn near tstop */
        let troundoff = F::Scalar::hundred()
         * F::Scalar::epsilon()
         * (self.ida_tn.abs() + self.ida_hh.abs());
        if (self.ida_tn - self.ida_tstop).abs() <= troundoff {
        /* ier = */
        //IDAGetSolution(IDA_mem, self.ida_tstop, yret, ypret);
         *tret = self.ida_tretlast = self.ida_tstop;
        self.ida_tstopset = false;

        self.get_solution(tout, yret, ypret)?;
        //return(IDA_TSTOP_RETURN);
        }
        if (self.ida_tn + self.ida_hh - self.ida_tstop) * self.ida_hh
        > F::Scalar::zero()
        {
        self.ida_hh = (self.ida_tstop - self.ida_tn)
         * (F::Scalar::one() - F::Scalar::four() * F::Scalar::epsilon());
        }
        }

        *tret = self.ida_tretlast = self.ida_tn;
        //return (IDA_SUCCESS);
        }
        }

        //return IDA_ILL_INPUT;  /* This return should never happen. */
         */
    }

    /// This routine performs one internal IDA step, from tn to tn + hh. It calls other routines to do all the work.
    ///
    /// It solves a system of differential/algebraic equations of the form F(t,y,y') = 0, for one step.
    /// In IDA, tt is used for t, yy is used for y, and yp is used for y'. The function F is supplied
    /// as 'res' by the user.
    ///
    /// The methods used are modified divided difference, fixed leading coefficient forms of backward
    /// differentiation formulas. The code adjusts the stepsize and order to control the local error
    /// per step.
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
    ///       IDA_SUCCESS   IDA_RES_FAIL      LSETUP_ERROR_NONRECVR
    ///                     IDA_LSOLVE_FAIL   IDA_ERR_FAIL
    ///                     IDA_CONSTR_FAIL   IDA_CONV_FAIL
    ///                     IDA_REP_RES_ERR
    fn step(&mut self) -> Result<(), failure::Error> {
        //realtype saved_t, ck;
        //realtype err_k, err_km1;
        //int ncf, nef;
        //int nflag, kflag;
        let mut ck = F::Scalar::one();

        let saved_t = self.ida_tn;
        //ncf = nef = 0;

        if self.ida_nst == 0 {
            self.ida_kk = 1;
            self.ida_kused = 0;
            self.ida_hused = F::Scalar::one();
            self.ida_psi[0] = self.ida_hh;
            self.ida_cj = F::Scalar::one() / self.ida_hh;
            self.ida_phase = 0;
            self.ida_ns = 0;
        }

        /* To prevent 'unintialized variable' warnings */
        //err_k = ZERO;
        //err_km1 = ZERO;

        // Looping point for attempts to take a step

        loop {
            //-----------------------
            // Set method coefficients
            //-----------------------

            ck = self.set_coeffs();

            //kflag = IDA_SUCCESS;

            //----------------------------------------------------
            // If tn is past tstop (by roundoff), reset it to tstop.
            //-----------------------------------------------------

            self.ida_tn += self.ida_hh;
            if self.ida_tstopset
                && ((self.ida_tn - self.ida_tstop) * self.ida_hh > F::Scalar::one())
            {
                self.ida_tn = self.ida_tstop;
            }

            //-----------------------
            // Advance state variables
            //-----------------------

            // Compute predicted values for yy and yp
            self.predict();

            // Nonlinear system solution
            let nflag = self.nonlinear_solve();

            // If NLS was successful, perform error test
            if nflag.is_ok() {
                let nflag2 = self.test_error(ck);
            }

            let err_flags = nflag.map(|_| self.test_error(ck));

            //let (err_k, err_km1, nflag) = self.test_error(ck);
            let nflag = false;

            // Test for convergence or error test failures
            if nflag != true {
                // restore and decide what to do
                self.restore(saved_t);
                //kflag = self.handle_n_flag(nflag, err_k, err_km1, &(self.ida_ncfn), &ncf, &(self.ida_netf), &nef);

                // exit on nonrecoverable failure
                //if kflag != PREDICT_AGAIN {
                //    return (kflag);
                //}

                // recoverable error; predict again
                if self.ida_nst == 0 {
                    self.reset();
                }
                continue;
            }

            /* kflag == IDA_SUCCESS */
            break;
        }

        // Nonlinear system solve and error test were both successful;
        // update data, and consider change of step and/or order

        //self.complete_step(err_k, err_km1);

        //  Rescale ee vector to be the estimated local error
        //  Notes:
        //    (1) altering the value of ee is permissible since
        //        it will be overwritten by
        //        IDASolve()->IDAStep()->IDANls()
        //        before it is needed again
        //    (2) the value of ee is only valid if IDAHandleNFlag()
        //        returns either PREDICT_AGAIN or IDA_SUCCESS

        //N_VScale(ck, self.ida_ee, self.ida_ee);
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
    fn set_coeffs(&mut self) -> F::Scalar {
        // Set coefficients for the current stepsize h
        if (self.ida_hh != self.ida_hused) || (self.ida_kk != self.ida_kused) {
            self.ida_ns = 0;
        }
        self.ida_ns = std::cmp::min(self.ida_ns + 1, self.ida_kused + 2);
        if self.ida_kk + 1 >= self.ida_ns {
            self.ida_beta[0] = F::Scalar::one();
            self.ida_alpha[0] = F::Scalar::one();
            let mut temp1 = self.ida_hh;
            self.ida_gamma[0] = F::Scalar::zero();
            self.ida_sigma[0] = F::Scalar::one();
            for i in 1..self.ida_kk + 1 {
                let temp2 = self.ida_psi[i - 1];
                self.ida_psi[i - 1] = temp1;
                self.ida_beta[i] = self.ida_beta[i - 1] * (self.ida_psi[i - 1] / temp2);
                temp1 = temp2 + self.ida_hh;
                self.ida_alpha[i] = self.ida_hh / temp1;
                self.ida_sigma[i] =
                    self.ida_sigma[i - 1] * self.ida_alpha[i] * NumCast::from(i).unwrap();
                self.ida_gamma[i] = self.ida_gamma[i - 1] + self.ida_alpha[i - 1] / self.ida_hh;
            }
            self.ida_psi[self.ida_kk] = temp1;
        }
        // compute alphas, alpha0
        let mut alphas = F::Scalar::zero();
        let mut alpha0 = F::Scalar::zero();
        for i in 0..self.ida_kk {
            alphas -= F::Scalar::one() / NumCast::from(i + 1).unwrap();
            alpha0 -= self.ida_alpha[i];
        }

        // compute leading coefficient cj
        self.ida_cjlast = self.ida_cj;
        self.ida_cj = -alphas / self.ida_hh;

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
        // Initialize if the first time called

        if self.ida_nst == 0 {
            self.ida_cjold = self.ida_cj;
            self.ida_ss = F::Scalar::twenty();
            //if (self.ida_lsetup) { callLSetup = true; }
        }

        // Decide if lsetup is to be called

        if self.ida_lsetup {
            self.ida_cjratio = self.ida_cj / self.ida_cjold;
            let temp1 = NumCast::from((1.0 - XRATE) / (1.0 + XRATE)).unwrap();
            let temp2 = temp1.recip();
            if self.ida_cjratio < temp1 || self.ida_cjratio > temp2 {
                callLSetup = true;
            }
            if self.ida_cj != self.ida_cjlast {
                self.ida_ss = F::Scalar::hundred();
            }
        }

        // initial guess for the correction to the predictor
        //N_VConst(ZERO, self.ida_delta);
        self.ida_delta = Array::zeros(self.f.model_size());

        // call nonlinear solver setup if it exists
        /*
        if ((self.NLS)->ops->setup) {
          retval = SUNNonlinSolSetup(self.NLS, self.ida_delta, IDA_mem);
          if (retval < 0) return(IDA_NLS_SETUP_FAIL);
          if (retval > 0) return(IDA_NLS_SETUP_RECVR);
        }
        */

        // solve the nonlinear system
        retval = SUNNonlinSolSolve(
            self.NLS,
            self.ida_delta,
            self.ida_ee,
            self.ida_ewt,
            self.ida_epsNewt,
            callLSetup,
            IDA_mem,
        );

        // update yy and yp based on the final correction from the nonlinear solve
        N_VLinearSum(ONE, self.ida_yypredict, ONE, self.ida_ee, self.ida_yy);
        N_VLinearSum(
            ONE,
            self.ida_yppredict,
            self.ida_cj,
            self.ida_ee,
            self.ida_yp,
        );

        // return if nonlinear solver failed */
        //if (retval != IDA_SUCCESS) { return (retval); };

        /* If otherwise successful, check and enforce inequality constraints. */
    }

    /// IDAPredict
    /// This routine predicts the new values for vectors yy and yp.
    fn predict(&mut self) -> () {
        self.ida_cvals.assign(&Array::ones(self.ida_cvals.shape()));

        // yypredict = cvals * phi[0..kk+1]
        //(void) N_VLinearCombination(self.ida_kk+1, self.ida_cvals, self.ida_phi, self.ida_yypredict);
        {
            let phi = self
                .ida_phi
                .slice_axis(Axis(0), Slice::from(0..self.ida_kk + 1));

            // We manually broadcast here so we can turn it into a column vec
            let cvals = self.ida_cvals.slice(s![0..self.ida_kk + 1]);
            let cvals = cvals
                .broadcast((phi.len_of(Axis(1)), phi.len_of(Axis(0))))
                .unwrap()
                .reversed_axes();

            let mut yypredict = self.ida_yypredict.slice_axis_mut(Axis(0), Slice::from(0..));

            yypredict.assign(&(&phi * &cvals).sum_axis(Axis(0)));
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

            let mut yppredict = self.ida_yppredict.slice_axis_mut(Axis(0), Slice::from(0..));

            yppredict.assign(&(&phi * &gamma).sum_axis(Axis(0)));
        }
    }

    /// IDATestError
    ///
    /// This routine estimates errors at orders k, k-1, k-2, decides whether or not to suggest an order
    /// decrease, and performs the local error test.
    ///
    /// Returns a tuple of (err_k, err_km1, nflag)
    fn test_error(
        &mut self,
        ck: F::Scalar,
    ) -> Option<(
        F::Scalar, // err_k
        F::Scalar, // err_km1
        bool,      // nflag
    )> {
        //realtype enorm_k, enorm_km1, enorm_km2;   /* error norms */
        //realtype terr_k, terr_km1, terr_km2;      /* local truncation error norms */
        // Compute error for order k.
        let enorm_k = self.wrms_norm(&self.ida_ee, &self.ida_ewt, self.ida_suppressalg);
        let err_k = self.ida_sigma[self.ida_kk] * enorm_k;
        let terr_k = err_k * NumCast::from(self.ida_kk + 1).unwrap();

        let mut err_km1 = F::Scalar::zero(); // estimated error at k-1
        let mut err_km2 = F::Scalar::zero(); // estimated error at k-2

        self.ida_knew = self.ida_kk;

        if self.ida_kk > 1 {
            // Compute error at order k-1
            self.ida_delta = &self.ida_phi.index_axis(Axis(0), self.ida_kk) + &self.ida_ee;
            let enorm_km1 = self.wrms_norm(&self.ida_delta, &self.ida_ewt, self.ida_suppressalg);
            err_km1 = self.ida_sigma[self.ida_kk - 1] * enorm_km1;
            let terr_km1: F::Scalar = err_km1 * NumCast::from(self.ida_kk).unwrap();

            if self.ida_kk > 2 {
                // Compute error at order k-2
                // ida_delta = ida_phi[ida_kk - 1] + ida_delta
                self.ida_delta
                    .assign(&self.ida_phi.index_axis(Axis(0), self.ida_kk - 1));
                self.ida_delta.scaled_add(F::Scalar::one(), &self.ida_ee);

                let enorm_km2 =
                    self.wrms_norm(&self.ida_delta, &self.ida_ewt, self.ida_suppressalg);
                err_km2 = self.ida_sigma[self.ida_kk - 2] * enorm_km2;
                let terr_km2 = err_km2 * NumCast::from(self.ida_kk - 1).unwrap();

                // Decrease order if errors are reduced
                if terr_km1.max(terr_km2) <= terr_k {
                    self.ida_knew = self.ida_kk - 1;
                }
            } else {
                // Decrease order to 1 if errors are reduced by at least 1/2
                if terr_km1 <= (terr_k * F::Scalar::half()) {
                    self.ida_knew = self.ida_kk - 1;
                }
            }
        };
        // Perform error test
        if (ck * enorm_k) > F::Scalar::one() {
            Some((err_k, err_km1, true))
        } else {
            None
        }
    }

    /// IDARestore
    /// This routine restores tn, psi, and phi in the event of a failure.
    /// It changes back `phi-star` to `phi` (changed in `set_coeffs()`)
    ///
    ///
    fn restore(&mut self, saved_t: F::Scalar) -> () {
        self.ida_tn = saved_t;

        // Restore psi[0 .. kk] = psi[1 .. kk + 1] - hh
        for j in 1..self.ida_kk + 1 {
            self.ida_psi[j - 1] = self.ida_psi[j] - self.ida_hh;
        }

        //Zip::from(&mut self.ida_psi.slice_mut(s![0..self.ida_kk]))
        //.and(&self.ida_psi.slice(s![1..self.ida_kk+1]));
        //ida_psi -= &self.ida_psi.slice(s![1..self.ida_kk+1]);

        if self.ida_ns <= self.ida_kk {
            // cvals[0 .. kk-ns+1] = 1 / beta[ns .. kk+1]
            Zip::from(
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
    /// This routine handles failures indicated by the input variable nflag. Positive values indicate various recoverable failures while negative values indicate nonrecoverable failures. This routine adjusts the step size for recoverable failures.
    ///
    ///  Possible nflag values (input):
    ///
    ///   --convergence failures--
    ///   IDA_RES_RECVR              > 0
    ///   IDA_LSOLVE_RECVR           > 0
    ///   IDA_CONSTR_RECVR           > 0
    ///   SUN_NLS_CONV_RECV          > 0
    ///   IDA_RES_FAIL               < 0
    ///   IDA_LSOLVE_FAIL            < 0
    ///   IDA_LSETUP_FAIL            < 0
    ///
    ///   --error test failure--
    ///   ERROR_TEST_FAIL            > 0
    ///
    ///  Possible kflag values (output):
    ///
    ///   --recoverable--
    ///   PREDICT_AGAIN
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
        nflag: u32,
        err_k: F::Scalar,
        err_km1: F::Scalar, //long int *ncfnPtr,
                            //int *ncfPtr,
                            //long int *netfPtr,
                            //int *nefPtr
    ) -> () {
        unimplemented!();
    }

    /// IDAReset
    /// This routine is called only if we need to predict again at the very first step. In such a case,
    /// reset phi[1] and psi[0].
    fn reset(&mut self) -> () {
        self.ida_psi[0] = self.ida_hh;
        //N_VScale(self.ida_rr, self.ida_phi[1], self.ida_phi[1]);
        self.ida_phi *= self.ida_rr;
    }

    /// IDACompleteStep
    /// This routine completes a successful step.  It increments nst, saves the stepsize and order
    /// used, makes the final selection of stepsize and order for the next step, and updates the phi
    /// array.
    fn complete_step(&mut self, err_k: F::Scalar, err_km1: F::Scalar) -> () {
        self.ida_nst += 1;
        let kdiff = self.ida_kk - self.ida_kused;
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
            if self.ida_nst > 1 {
                self.ida_kk += 1;
                let mut hnew = F::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > F::Scalar::one() {
                    hnew /= tmp;
                }
                self.ida_hh = hnew;
            }
        } else {
            enum Action {
                None,
                Lower,
                Maintain,
                Raise,
            }

            let mut action = Action::None;

            // Set action = LOWER/MAINTAIN/RAISE to specify order decision

            if self.ida_knew == (self.ida_kk - 1) {
                action = Action::Lower;
            } else if self.ida_kk == self.ida_maxord {
                action = Action::Maintain;
            } else if (self.ida_kk + 1) >= self.ida_ns || (kdiff == 1) {
                action = Action::Maintain;
            }

            // Estimate the error at order k+1, unless already decided to reduce order, or already using
            // maximum order, or stepsize has not been constant, or order was just raised.

            let mut err_kp1 = F::Scalar::zero();

            if let Action::None = action {
                //N_VLinearSum(ONE, self.ida_ee, -ONE, self.ida_phi[self.ida_kk + 1], self.ida_tempv1);
                let ida_tempv1 = &self.ida_ee - &self.ida_phi.index_axis(Axis(0), self.ida_kk + 1);
                let enorm = self.wrms_norm(&ida_tempv1, &self.ida_ewt, self.ida_suppressalg);
                err_kp1 = enorm / NumCast::from(self.ida_kk + 2).unwrap();

                // Choose among orders k-1, k, k+1 using local truncation error norms.

                let terr_k: F::Scalar = NumCast::from(self.ida_kk + 1).unwrap() * err_k;
                let terr_kp1 = NumCast::from(self.ida_kk + 2).unwrap() * err_kp1;

                if self.ida_kk == 1 {
                    if terr_kp1 >= F::Scalar::half() * terr_k {
                        action = Action::Maintain;
                    } else {
                        action = Action::Raise;
                    }
                } else {
                    let terr_km1 = NumCast::from(self.ida_kk).unwrap() * err_km1;
                    if terr_km1 <= terr_k.min(terr_kp1) {
                        action = Action::Lower;
                    } else if terr_kp1 >= terr_k {
                        action = Action::Maintain;
                    } else {
                        action = Action::Raise;
                    }
                }
            }
            //takeaction:

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

            // Compute rr = tentative ratio hnew/hh from error norm estimate.
            // Reduce hh if rr <= 1, double hh if rr >= 2, else leave hh as is.
            // If hh is reduced, hnew/hh is restricted to be between .5 and .9.

            let mut hnew = self.ida_hh;
            //ida_rr = SUNRpowerR( TWO * err_knew + PT0001, -ONE/(self.ida_kk + 1) );
            self.ida_rr = {
                let base = F::Scalar::two() * err_knew + F::Scalar::pt0001();
                let arg =
                    -F::Scalar::one() / (NumCast::from(self.ida_kk).unwrap() + F::Scalar::one());
                base.powf(arg)
            };

            if self.ida_rr >= F::Scalar::two() {
                hnew = F::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > F::Scalar::one() {
                    hnew /= tmp;
                }
            } else if self.ida_rr <= F::Scalar::one() {
                //ida_rr = SUNMAX(HALF, SUNMIN(PT9,self.ida_rr));
                self.ida_rr = F::Scalar::half().max(self.ida_rr.min(F::Scalar::pt9()));
                hnew = self.ida_hh * self.ida_rr;
            }

            self.ida_hh = hnew;
        }
        // end of phase if block

        // Save ee for possible order increase on next step
        if self.ida_kused < self.ida_maxord {
            //N_VScale(ONE, self.ida_ee, self.ida_phi[self.ida_kused + 1]);
            self.ida_phi
                .index_axis_mut(Axis(0), self.ida_kused + 1)
                .assign(&self.ida_ee);
        }

        // Update phi arrays

        // To update phi arrays compute X += Z where                  */
        // X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ] */
        // Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ] */
        self.ida_Zvecs
            .index_axis_mut(Axis(0), 0)
            .assign(&self.ida_ee);
        self.ida_Zvecs
            .slice_mut(s![1..self.ida_kused + 1, ..])
            .assign(&self.ida_phi.slice(s![1..self.ida_kused + 1;-1, ..]));
        self.ida_Xvecs
            .slice_mut(s![1..self.ida_kused + 1, ..])
            .assign(&self.ida_phi.slice(s![0..self.ida_kused;-1, ..]));

        let mut sliceXvecs = self
            .ida_Xvecs
            .slice_axis_mut(Axis(0), Slice::from(0..self.ida_kused + 1));
        let sliceZvecs = self
            .ida_Zvecs
            .slice_axis(Axis(0), Slice::from(0..self.ida_kused + 1));
        sliceXvecs += &sliceZvecs;
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
    pub fn get_solution<'a, S1, S2>(
        &mut self,
        t: F::Scalar,
        yret: &'a mut ArrayBase<S1, Ix1>,
        ypret: &'a mut ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: DataMut<Elem = F::Scalar>,
        S2: DataMut<Elem = F::Scalar>,
        ArrayBase<S1, Ix1>: IntoNdProducer,
        ArrayBase<S2, Ix1>: IntoNdProducer,
    {
        // Check t for legality.  Here tn - hused is t_{n-1}.

        //tfuzz = HUNDRED * self.ida_uround * (SUNRabs(self.ida_tn) + SUNRabs(self.ida_hh));

        let mut tfuzz = F::Scalar::hundred()
            * F::Scalar::epsilon()
            * (self.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();
        //if self.ida_hh < F::Scalar::zero() { tfuzz = -tfuzz; }
        let tp = self.ida_tn - self.ida_hused - tfuzz;
        if (t - tp) * self.ida_hh < F::Scalar::zero() {
            Err(IdaError::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.ida_tn.to_f64().unwrap(),
            })?;
        }

        // Initialize kord = (kused or 1).
        let kord = if self.ida_kused == 0 {
            1
        } else {
            self.ida_kused
        };

        // Accumulate multiples of columns phi[j] into yret and ypret.
        let delt = t - self.ida_tn;
        let mut c = F::Scalar::one();
        let mut d = F::Scalar::zero();
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
        ndarray::Zip::from(yret)
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(0..kord + 1))
                    .lanes(Axis(0)),
            )
            .apply(|z, row| {
                *z = (&row * &self.ida_cvals.slice(s![0..kord + 1])).sum();
            });

        //retval = N_VLinearCombination(kord, self.ida_dvals, self.ida_phi+1, ypret);
        ndarray::Zip::from(ypret)
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(1..kord + 1))
                    .lanes(Axis(0)),
            )
            .apply(|z, row| {
                *z = (&row * &self.ida_dvals.slice(s![0..kord])).sum();
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
    pub fn wrms_norm<S1: Data<Elem = F::Scalar>, S2: Data<Elem = F::Scalar>>(
        &self,
        x: &ArrayBase<S1, Ix1>,
        w: &ArrayBase<S2, Ix1>,
        mask: bool,
    ) -> F::Scalar {
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
        //self.ida_tlo = self.ida_tn;
        //self.ida_ttol = ((self.ida_tn).abs() + (self.ida_hh).abs())
        //    * F::Scalar::epsilon()
        //    * F::Scalar::from(100.0).unwrap();

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
        //retval.map_err(|e| IdaError::RootFunctionFail{t: self.ida_tn.to_f64().unwrap() })?;

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
    use crate::ida::Ida;
    use crate::lorenz63::Lorenz63;
    use ndarray::*;
    use nearly_eq::*;

    #[test]
    fn test_set_coeffs() {
        //Before
        let ida_phi = array![
            [
                4.1295003522440181e-07,
                1.6518008147114031e-12,
                9.9999958704831304e-01,
            ],
            [
                -6.4049734044789205e-08,
                -2.5619916159829551e-13,
                6.4049990326726996e-08,
            ],
            [
                2.1135440604995772e-08,
                8.4541889872000439e-14,
                -2.1135525197726480e-08,
            ],
            [
                -2.2351400807868742e-08,
                -8.9405756031743853e-14,
                2.2351489636470618e-08,
            ],
            [
                1.8323105973439385e-08,
                7.3292641194159994e-14,
                -1.8323176512520801e-08,
            ],
            [
                -2.2423672161947766e-10,
                -8.9709159667337618e-16,
                2.2422474012398869e-10,
            ],
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

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

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
        ida.ida_cj = cj;

        // Call the function under test
        let ck = ida.set_coeffs();

        //--- IDASetCoeffs After
        let ck_expect = 0.3214285714285713969;
        let ida_phi = array![
            [
                4.1295003522440181e-07,
                1.6518008147114031e-12,
                9.9999958704831304e-01,
            ],
            [
                -6.4049734044789205e-08,
                -2.5619916159829551e-13,
                6.4049990326726996e-08,
            ],
            [
                2.0023048994206519e-08,
                8.0092316720842518e-14,
                -2.0023129134688242e-08,
            ],
            [
                -2.2351400807868742e-08,
                -8.9405756031743853e-14,
                2.2351489636470618e-08,
            ],
            [
                1.8323105973439385e-08,
                7.3292641194159994e-14,
                -1.8323176512520801e-08,
            ],
            [
                -2.2423672161947766e-10,
                -8.9709159667337618e-16,
                2.2422474012398869e-10,
            ],
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
        assert_nearly_eq!(ida.ida_cj, cj);
        assert_nearly_eq!(ck, ck_expect);
    }

    #[test]
    fn test_predict() {}

    #[test]
    fn test_test_error1() {
        let ck = 1.091414141414142;
        let suppressalg = 0;
        let kk = 5;
        let ida_phi = array![
            [
                3.634565317158998e-05,
                1.453878335134203e-10,
                0.9999636542014404,
            ],
            [
                -6.530333550677049e-06,
                -2.612329458968465e-11,
                6.530359673556191e-06,
            ],
            [
                1.946442728026142e-06,
                7.786687275994346e-12,
                -1.946450515496441e-06,
            ],
            [
                -8.097632208221231e-07,
                -3.239585549038764e-12,
                8.097664556005615e-07,
            ],
            [
                3.718130977075839e-07,
                1.487573462300438e-12,
                -3.71814615793545e-07,
            ],
            [
                -3.24421895454213e-07,
                -1.297915245220823e-12,
                3.244230624265827e-07,
            ],
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
        let nflag = true;

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_suppressalg = suppressalg > 0;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_ee.assign(&ida_ee);
        ida.ida_ewt.assign(&ida_ewt);
        ida.ida_sigma.assign(&ida_sigma);

        // Call the function under test
        let (err_k_new, err_km1_new, nflag_new) = ida.test_error(ck).unwrap();

        assert_eq!(ida.ida_knew, knew);
        assert_nearly_eq!(err_k_new, err_k);
        assert_nearly_eq!(err_km1_new, err_km1);
        assert_eq!(nflag_new, nflag);
    }

    #[test]
    #[ignore]
    fn test_test_error2() {
        //--- IDATestError Before:
        let ck = 0.2025812352167927;
        let suppressalg = 0;
        let kk = 4;
        let ida_phi = array![
            [
                3.051237735052657e-05,
                1.220531905117091e-10,
                0.9999694875005963,
            ],
            [
                -2.513114849098281e-06,
                -1.005308974226734e-11,
                2.513124902721765e-06,
            ],
            [
                4.500284453718991e-07,
                1.800291970640913e-12,
                -4.500302448499092e-07,
            ],
            [
                -1.366709389821433e-07,
                -5.467603693902342e-13,
                1.366714866794709e-07,
            ],
            [
                7.278821769100639e-08,
                2.911981566628798e-13,
                -7.278850816613011e-08,
            ],
            [
                -8.304741244343501e-09,
                -3.324587131187576e-14,
                8.304772990651073e-09,
            ],
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
        let nflag = false;

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_suppressalg = suppressalg > 0;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_ee.assign(&ida_ee);
        ida.ida_ewt.assign(&ida_ewt);
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
        let ida_phi = array![
            [
                1.0570152037228958e-07,
                4.2280612558303261e-13,
                9.9999989429805680e-01,
            ],
            [
                -3.3082196412696304e-08,
                -1.3232881828710420e-13,
                3.3082328676061534e-08,
            ],
            [
                1.8675273859330434e-08,
                7.4701128706323864e-14,
                -1.8675348801050254e-08,
            ],
            [
                -1.9956501813542136e-08,
                -7.9826057803058290e-14,
                1.9956580862443821e-08,
            ],
            [
                1.2851942479612096e-09,
                5.1407743965993651e-15,
                -1.2851948368212051e-09,
            ],
            [
                -2.2423672161947766e-10,
                -8.9709159667337618e-16,
                2.2422474012398869e-10,
            ],
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

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_kk = kk;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_gamma.assign(&ida_gamma);
        ida.ida_yypredict.assign(&ida_yypredict);
        ida.ida_yppredict.assign(&ida_yppredict);

        // Call the function under test
        ida.predict();

        //--- IDAPredict After
        let ida_phi = array![
            [
                1.0570152037228958e-07,
                4.2280612558303261e-13,
                9.9999989429805680e-01,
            ],
            [
                -3.3082196412696304e-08,
                -1.3232881828710420e-13,
                3.3082328676061534e-08,
            ],
            [
                1.8675273859330434e-08,
                7.4701128706323864e-14,
                -1.8675348801050254e-08,
            ],
            [
                -1.9956501813542136e-08,
                -7.9826057803058290e-14,
                1.9956580862443821e-08,
            ],
            [
                1.2851942479612096e-09,
                5.1407743965993651e-15,
                -1.2851948368212051e-09,
            ],
            [
                -2.2423672161947766e-10,
                -8.9709159667337618e-16,
                2.2422474012398869e-10,
            ],
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
        assert_nearly_eq!(ida.ida_phi, ida_phi, 1e-9);
        assert_nearly_eq!(ida.ida_yypredict, ida_yypredict, 1e-9);
        assert_nearly_eq!(ida.ida_yppredict, ida_yppredict, 1e-9);
    }

    #[test]
    fn test_restore1() {
        let saved_t = 717553.4942644858;
        #[rustfmt::skip]
        let phi_before = array![[0.00280975951420059, 1.125972706132338e-08, 0.9971902292261264], [-0.0001926545663078034, -7.857235149861102e-10,0.0001926553520857565], [2.945636347837807e-05, 1.066748079583829e-10,-2.945647009050819e-05], [-5.518529121250618e-06, -4.529997656241677e-11,5.518574540464112e-06], [2.822681468681011e-06, -4.507342025411469e-11,-2.822636100488049e-06], [-8.124641701620927e-08,-8.669560754165103e-11,8.133355922669991e-08], ];
        #[rustfmt::skip]
        let psi_before = array![ 47467.05706123715, 94934.1141224743, 142401.1711837114, 166134.69971433, 189868.2282449486, 107947.0192373629 ];
        let cvals_before = array![1., 1., 1., 1., 1., 0.];
        let beta_before = array![1., 1., 1., 1.2, 1.4, 1.];

        #[rustfmt::skip]
        let phi_after = array![[0.00280975951420059,1.125972706132338e-08, 0.9971902292261264,], [-0.0001926545663078034,-7.857235149861102e-10,0.0001926553520857565,], [2.945636347837807e-05,1.066748079583829e-10,-2.945647009050819e-05,], [-4.598774267708849e-06,-3.774998046868064e-11,4.598812117053426e-06,], [2.016201049057865e-06,-3.219530018151049e-11,-2.016168643205749e-06,], [-8.124641701620927e-08,-8.669560754165103e-11,8.133355922669991e-08,], ];
        #[rustfmt::skip]
        let psi_after = array![ 47467.05706123715, 94934.11412247429, 118667.6426530929, 142401.1711837114, 189868.2282449486, 107947.0192373629 ];
        let cvals_after = array![0.8333333333333334, 0.7142857142857142, 1., 1., 1., 0.];
        let beta_after = array![1., 1., 1., 1.2, 1.4, 1.];

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_tn = 765020.5513257229;
        ida.ida_ns = 3;
        ida.ida_kk = 4;
        ida.ida_hh = 47467.05706123715;
        ida.ida_phi.assign(&phi_before);
        ida.ida_psi.assign(&psi_before);
        ida.ida_cvals.assign(&cvals_before);
        ida.ida_beta.assign(&beta_before);

        // Call the function under test
        ida.restore(saved_t);

        assert_nearly_eq!(ida.ida_tn, saved_t);
        assert_eq!(ida.ida_ns, 3);
        assert_eq!(ida.ida_kk, 4);
        assert_nearly_eq!(ida.ida_cvals, cvals_after, 1e-6);
        assert_nearly_eq!(ida.ida_beta, beta_after, 1e-6);
        assert_nearly_eq!(ida.ida_psi, psi_after, 1e-6);
        assert_nearly_eq!(ida.ida_phi, phi_after, 1e-6);
    }

    #[test]
    fn test_restore2() {
        let saved_t = 3623118336.24244;
        #[rustfmt::skip]
        let phi_before = array![ [5.716499633245077e-07,2.286601144610028e-12, 0.9999994283477499,], [-1.555846772013456e-07,-6.223394599091205e-13,1.555852991517385e-07,], [7.018252655941472e-08,2.807306512268244e-13,-7.01828076998538e-08,], [-4.56160628763917e-08,-1.824647796129851e-13,4.561624269904529e-08,], [5.593228676143622e-08,2.237297583983664e-13,-5.593253344183256e-08,], [-2.242367216194777e-10,-8.970915966733762e-16,2.242247401239887e-10,], ];
        #[rustfmt::skip]
        let psi_before = array![  857870592.1885694,   1286805888.282854,   1715741184.377139,   1930208832.424281,   2144676480.471424,    26020582.4876316];
        #[rustfmt::skip]
        let cvals_before = array![1., 1., 1., 1., 1., 1.];
        #[rustfmt::skip]
        let beta_before = array![1., 2., 3., 4.8, 7.199999999999999, 10.28571428571428];
        //--- IDARestore After: saved_t=   3623118336.24244 tn=   3623118336.24244 ns=1 kk=4
        #[rustfmt::skip]
        let phi_after = array![ [5.716499633245077e-07,2.286601144610028e-12, 0.9999994283477499,], [-7.779233860067279e-08,-3.111697299545603e-13,7.779264957586927e-08,], [2.339417551980491e-08,9.35768837422748e-14,-2.33942692332846e-08,], [-9.503346432581604e-09,-3.801349575270522e-14,9.503383895634436e-09,], [7.768373161310588e-09,3.107357755532867e-14,-7.768407422476745e-09,], [-2.242367216194777e-10,-8.970915966733762e-16,2.242247401239887e-10,], ];
        #[rustfmt::skip]
        let psi_after= array![  428935296.0942847,   857870592.1885694,   1072338240.235712,   1286805888.282854,   2144676480.471424,    26020582.4876316];
        #[rustfmt::skip]
        let cvals_after = array![ 0.5, 0.3333333333333333, 0.2083333333333333, 0.1388888888888889, 1., 1. ];
        #[rustfmt::skip]
        let beta_after = array![1., 2., 3., 4.8, 7.199999999999999, 10.28571428571428];

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_tn = 4480988928.431009;
        ida.ida_ns = 1;
        ida.ida_kk = 4;
        ida.ida_hh = 857870592.1885694;
        ida.ida_phi.assign(&phi_before);
        ida.ida_psi.assign(&psi_before);
        ida.ida_cvals.assign(&cvals_before);
        ida.ida_beta.assign(&beta_before);

        // Call the function under test
        ida.restore(saved_t);

        assert_nearly_eq!(ida.ida_tn, saved_t);
        assert_eq!(ida.ida_ns, 1);
        assert_eq!(ida.ida_kk, 4);
        assert_nearly_eq!(ida.ida_cvals, cvals_after, 1e-6);
        assert_nearly_eq!(ida.ida_beta, beta_after, 1e-6);
        assert_nearly_eq!(ida.ida_psi, psi_after, 1e-6);
        assert_nearly_eq!(ida.ida_phi, phi_after, 1e-6);
    }

    #[test]
    fn test_restore3() {
        let saved_t = 13638904.64873992;
        let phi_before = array![
            [
                0.0001523741818966069,
                6.095884948264652e-10,
                0.9998476252085154,
            ],
            [
                -1.964117218731689e-05,
                -7.858910051867137e-11,
                1.964125077907938e-05,
            ],
            [
                4.048658569496216e-06,
                1.620249912028008e-11,
                -4.048674765925692e-06,
            ],
            [
                -1.215165175266232e-06,
                -4.863765573523665e-12,
                1.21517004866448e-06,
            ],
            [
                4.909710408845208e-07,
                1.965778579990634e-12,
                -4.909729965008022e-07,
            ],
            [
                -2.529640523993838e-07,
                -1.012593011825966e-12,
                2.529650614751456e-07,
            ],
        ];
        let psi_before = array![
            1656116.685489699,
            2484175.028234549,
            3312233.370979399,
            4140291.713724249,
            5060356.538996303,
            5520388.951632331
        ];
        let cvals_before = array![1., 1., 1., 1., 1., 1.];
        let beta_before = array![1., 2., 3., 4., 4.864864864864866, 6.370656370656372];
        //--- IDARestore After: saved_t=  13638904.64873992 tn=  13638904.64873992 ns=1 kk=5
        let phi_after = array![
            [
                0.0001523741818966069,
                6.095884948264652e-10,
                0.9998476252085154,
            ],
            [
                -9.820586093658443e-06,
                -3.929455025933569e-11,
                9.820625389539692e-06,
            ],
            [
                1.349552856498739e-06,
                5.400833040093358e-12,
                -1.349558255308564e-06,
            ],
            [
                -3.037912938165579e-07,
                -1.215941393380916e-12,
                3.0379251216612e-07,
            ],
            [
                1.009218250707071e-07,
                4.040767081091857e-13,
                -1.009222270584982e-07,
            ],
            [
                -3.970769064935782e-08,
                -1.589464182199546e-13,
                3.970784904367437e-08,
            ],
        ];
        let psi_after = array![
            828058.3427448499,
            1656116.685489699,
            2484175.02823455,
            3404239.853506604,
            3864272.266142632,
            5520388.951632331,
        ];
        let cvals_after = array![
            0.5,
            0.3333333333333333,
            0.25,
            0.2055555555555555,
            0.156969696969697,
            1.,
        ];
        let beta_after = array![1., 2., 3., 4., 4.864864864864866, 6.370656370656372];

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);

        // Set preconditions:
        ida.ida_tn = 15295021.33422961;
        ida.ida_ns = 1;
        ida.ida_kk = 5;
        ida.ida_hh = 1656116.685489699;
        ida.ida_phi.assign(&phi_before);
        ida.ida_psi.assign(&psi_before);
        ida.ida_cvals.assign(&cvals_before);
        ida.ida_beta.assign(&beta_before);

        // Call the function under test
        ida.restore(saved_t);

        assert_nearly_eq!(ida.ida_tn, saved_t);
        assert_eq!(ida.ida_ns, 1);
        assert_eq!(ida.ida_kk, 5);
        assert_nearly_eq!(ida.ida_cvals, cvals_after, 1e-6);
        assert_nearly_eq!(ida.ida_beta, beta_after, 1e-6);
        assert_nearly_eq!(ida.ida_psi, psi_after, 1e-6);
        assert_nearly_eq!(ida.ida_phi, phi_after, 1e-6);
    }

    #[test]
    fn test_complete_step() {
        let err_k = 0.1022533962984153;
        let err_km1 = 0.3638660854770704;
        let ida_phi = array![
            [0.0000001057015204, 0.0000000000004228, 0.9999998942980568,],
            [-0.0000000330821964, -0.0000000000001323, 0.0000000330823287,],
            [0.0000000186752739, 0.0000000000000747, -0.0000000186753488,],
            [-0.0000000199565018, -0.0000000000000798, 0.0000000199565809,],
            [0.0000000012851942, 0.0000000000000051, -0.0000000012851948,],
            [-0.0000000002242367, -0.0000000000000009, 0.0000000002242247,],
        ];
        let ida_ee = array![-0.0000000051560075, -0.0000000000000206, 0.0000000051560285,];
        let ida_ewt = array![
            99894410.0897681862115860,
            999999.9999577193520963,
            9900.9911352019826154,
        ];
        let kk = 2;
        let kused = 2;
        let knew = 2;
        let phase = 1;
        let hh = 3774022770.1406540870666504;
        let hused = 4313148194.5176315307617188;
        let rr = 0.8750041964562566;
        let hmax_inv = 0.0000000000000000;
        let nst = 357;
        let maxord = 5;

        // Set preconditions:
        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![0., 0., 0.], array![0., 0., 0.]);
        ida.ida_nst = nst;
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
        ida.ida_ewt.assign(&ida_ewt);
        //ida.ida_Xvecs.assign(&ida_Xvecs);
        //ida.ida_Zvecs.assign(&ida_Zvecs);

        ida.complete_step(err_k, err_km1);

        let ida_phi = array![
            [0.0000000861385903, 0.0000000000003446, 0.9999999138610652,],
            [-0.0000000195629300, -0.0000000000000783, 0.0000000195630084,],
            [0.0000000135192664, 0.0000000000000541, -0.0000000135193203,],
            [-0.0000000051560075, -0.0000000000000206, 0.0000000051560285,],
            [0.0000000012851942, 0.0000000000000051, -0.0000000012851948,],
            [-0.0000000002242367, -0.0000000000000009, 0.0000000002242247,],
        ];
        let ida_ee = array![-0.0000000051560075, -0.0000000000000206, 0.0000000051560285,];
        let ida_ewt = array![
            99894410.0897681862115860,
            999999.9999577193520963,
            9900.9911352019826154,
        ];
        let kk = 2;
        let kused = 2;
        let knew = 2;
        let phase = 1;
        let hh = 3774022770.1406540870666504;
        let hused = 3774022770.1406540870666504;
        let rr = 1.6970448397793398;
        let hmax_inv = 0.0000000000000000;
        let nst = 358;
        let maxord = 5;

        assert_eq!(ida.ida_nst, nst);
        assert_eq!(ida.ida_kk, kk);
        assert_eq!(ida.ida_hh, hh);
        assert_nearly_eq!(ida.ida_rr, rr, 1e-6);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_knew, knew);
        assert_eq!(ida.ida_maxord, maxord);
        assert_eq!(ida.ida_phase, phase);
        assert_eq!(ida.ida_hmax_inv, hmax_inv);
        assert_nearly_eq!(ida.ida_ee, ida_ee, 1e-6);
        assert_nearly_eq!(ida.ida_phi, ida_phi, 1e-6);
        assert_nearly_eq!(ida.ida_ewt, ida_ewt, 1e-6);
    }

    #[test]
    fn test_get_solution() {
        // --- IDAGetSolution Before:
        let t = 3623118336.24244;
        let hh = 857870592.1885694;
        let tn = 3623118336.24244;
        let kused = 4;
        let hused = 428935296.0942847;
        let ida_phi = array![
            [
                5.716499633245077e-07,
                2.286601144610028e-12,
                0.9999994283477499,
            ],
            [
                -7.779233860067279e-08,
                -3.111697299545603e-13,
                7.779264957586927e-08,
            ],
            [
                2.339417551980491e-08,
                9.35768837422748e-14,
                -2.33942692332846e-08,
            ],
            [
                -9.503346432581604e-09,
                -3.801349575270522e-14,
                9.503383895634436e-09,
            ],
            [
                7.768373161310588e-09,
                3.107357755532867e-14,
                -7.768407422476745e-09,
            ],
            [
                -2.242367216194777e-10,
                -8.970915966733762e-16,
                2.242247401239887e-10,
            ],
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

        let f = Lorenz63::default();
        let mut ida = Ida::new(f, array![1., 2., 3.], array![4., 5., 6.]);
        //println!("{:#?}", i);

        ida.ida_hh = hh;
        ida.ida_tn = tn;
        ida.ida_kused = kused;
        ida.ida_hused = hused;
        ida.ida_phi.assign(&ida_phi);
        ida.ida_psi.assign(&ida_psi);

        let mut yret = Array::zeros((3));
        let mut ypret = Array::zeros((3));

        ida.get_solution(t, &mut yret.view_mut(), &mut ypret.view_mut())
            .unwrap();

        assert_nearly_eq!(yret, yret_expect, 1e-6);
        assert_nearly_eq!(ypret, ypret_expect, 1e-6);
    }
}
