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
mod impl_complete_step;
mod impl_r_check;
mod impl_solve;
mod impl_stop_test;
mod norm_rms;

#[cfg(test)]
mod tests;

pub mod linear;
pub mod nonlinear;
pub mod sample_problems;
pub mod tol_control;
pub mod traits;
pub use norm_rms::{NormRms, NormRmsMasked};

use constants::*;
use error::{IdaError, Recoverable};
use ida_nls::IdaNLProblem;
use impl_r_check::RootStatus;
use tol_control::TolControl;
use traits::*;

#[cfg(feature = "profiler")]
use profiler::profile_scope;

use log::{error, trace};
use ndarray::{prelude::*, s, Slice};
use num_traits::{
    cast::{NumCast, ToPrimitive},
    identities::{One, Zero},
    Float,
};

#[cfg(feature = "data_trace")]
use serde::Serialize;
use std::io::Write;

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
pub enum IdaTask {
    Normal,
    OneStep,
}

#[derive(PartialEq, Debug)]
pub enum IdaSolveStatus {
    ContinueSteps,
    Success,
    TStop,
    Root,
}

enum IdaConverged {
    Converged,
    NotConverged,
}

/// Counters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
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
#[derive(Debug)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
pub struct Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
{
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
    /// max number of internal steps for one user call
    ida_mxstep: u64,
    /// inverse of max. step size hmax (default = 0.0)
    ida_hmax_inv: P::Scalar,

    //// Counters
    counters: IdaCounters,

    // Arrays for Fused Vector Operations
    ida_cvals: Array1<P::Scalar>,
    ida_dvals: Array1<P::Scalar>,

    /// tolerance scale factor (saved value)
    ida_tolsf: P::Scalar,

    // Rootfinding Data
    /// number of components of g
    ida_nrtfn: usize,
    /// array for root information
    ida_iroots: Array1<P::Scalar>,
    /// array specifying direction of zero-crossing
    ida_rootdir: Array1<u8>,
    /// nearest endpoint of interval in root search
    ida_tlo: P::Scalar,

    /// farthest endpoint of interval in root search
    ida_thi: P::Scalar,
    /// t return value from rootfinder routine
    ida_trout: P::Scalar,

    /// saved array of g values at t = tlo
    ida_glo: Array1<P::Scalar>,
    /// saved array of g values at t = thi
    ida_ghi: Array1<P::Scalar>,
    /// array of g values at t = trout
    ida_grout: Array1<P::Scalar>,
    /// copy of tout (if NORMAL mode)
    ida_toutc: P::Scalar,
    /// tolerance on root location
    ida_ttol: P::Scalar,
    /// copy of parameter itask
    ida_taskc: IdaTask,
    /// flag showing whether last step had a root
    ida_irfnd: bool,
    /// counter for g evaluations
    ida_nge: usize,

    /// array with active/inactive event functions
    ida_gactive: Array1<bool>,
    /// number of warning messages about possible g==0
    ida_mxgnull: usize,

    // Arrays for Fused Vector Operations
    ida_zvecs: Array<P::Scalar, Ix2>,

    /// Nonlinear Solver
    nls: NLS,

    /// Nonlinear problem
    nlp: IdaNLProblem<P, LS>,

    #[cfg(feature = "data_trace")]
    data_trace: std::fs::File,
}

#[cfg(feature = "data_trace")]
impl<P, LS, NLS, TolC> Drop for Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
{
    fn drop(&mut self) {
        use std::io::Write;
        self.data_trace.write_all(b"]}\n").unwrap();
    }
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

        #[cfg(feature = "data_trace")]
        {
            let mut data_trace = std::fs::File::create("roberts_rs.json").unwrap();
            data_trace.write_all(b"{\"data\":[\n").unwrap();
        }

        //IDAResFn res, realtype t0, N_Vector yy0, N_Vector yp0
        Self {
            ida_setup_done: false,

            tol_control,

            // Set default values for integrator optional inputs
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
            ida_phi,

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

            // Initialize root-finding variables
            ida_irfnd: false,
            ida_glo: Array::zeros(problem.num_roots()),
            ida_ghi: Array::zeros(problem.num_roots()),
            ida_grout: Array::zeros(problem.num_roots()),
            ida_iroots: Array::zeros(problem.num_roots()),
            ida_rootdir: Array::zeros(problem.num_roots()),
            ida_nrtfn: problem.num_roots(),
            ida_mxgnull: 1,
            ida_tlo: P::Scalar::zero(),
            ida_ttol: P::Scalar::zero(),
            ida_gactive: Array::from_elem(problem.num_roots(), false),
            ida_taskc: IdaTask::Normal,
            ida_toutc: P::Scalar::zero(),
            ida_thi: P::Scalar::zero(),
            ida_trout: P::Scalar::zero(),

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
            ida_cvals: Array::zeros(MXORDP1),
            ida_dvals: Array::zeros(MAXORD_DEFAULT),

            ida_zvecs: Array::zeros((MXORDP1, yy0.shape()[0])),

            // Initialize nonlinear solver
            nls: NLS::new(yy0.len(), MAXNLSIT),
            nlp: IdaNLProblem::new(problem, yy0.view(), yp0.view()),

            #[cfg(feature = "data_trace")]
            data_trace,
        }
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
        #[cfg(feature = "profiler")]
        profile_scope!(format!("step(), nst={}", self.counters.ida_nst));

        let saved_t = self.nlp.ida_tn;

        if self.counters.ida_nst == 0 {
            self.ida_kk = 1;
            self.ida_kused = 0;
            self.ida_hused = P::Scalar::zero();
            self.ida_psi[0] = self.ida_hh;
            self.nlp.lp.ida_cj = self.ida_hh.recip();
            self.ida_phase = 0;
            self.ida_ns = 0;
        }

        let mut ncf = 0; // local counter for convergence failures
        let mut nef = 0; // local counter for error test failures

        // Looping point for attempts to take a step

        let (ck, err_k, err_km1) = loop {
            #[cfg(feature = "data_trace")]
            {
                serde_json::to_writer(&self.data_trace, self).unwrap();
                self.data_trace.write_all(b",\n").unwrap();
            }

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
        #[cfg(feature = "profiler")]
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
            for i in 1..=self.ida_kk {
                let scalar_i: P::Scalar = NumCast::from(i).unwrap();
                let temp2 = self.ida_psi[i - 1];
                self.ida_psi[i - 1] = temp1;
                self.ida_beta[i] = self.ida_beta[i - 1] * self.ida_psi[i - 1] / temp2;
                temp1 = temp2 + self.ida_hh;
                self.ida_alpha[i] = self.ida_hh / temp1;
                self.ida_sigma[i] = scalar_i * self.ida_sigma[i - 1] * self.ida_alpha[i];
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
        #[cfg(feature = "profiler")]
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

        //trace!("\"ewt\":{:.6e}", w);

        // solve the nonlinear system
        let retval = self.nls.solve(
            &mut self.nlp,
            self.ida_delta.view(),
            self.ida_ee.view_mut(),
            w,
            self.ida_eps_newt,
            call_lsetup,
        );

        //trace!("\"ee\":{:.6e}", self.ida_ee);

        // update yy and yp based on the final correction from the nonlinear solve
        self.nlp.ida_yy = &self.nlp.ida_yypredict + &self.ida_ee;
        //N_VLinearSum( ONE, self.ida_yppredict, self.ida_cj, self.ida_ee, self.ida_yp,);
        //self.ida_yp = &self.ida_yppredict + (&self.ida_ee * self.ida_cj);
        self.nlp.ida_yp.assign(&self.nlp.ida_yppredict);
        self.nlp.ida_yp.scaled_add(self.nlp.lp.ida_cj, &self.ida_ee);

        // return if nonlinear solver failed */
        retval?;

        // If otherwise successful, check and enforce inequality constraints.

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
    fn predict(&mut self) {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("predict()"));

        // yypredict = cvals * phi[0..kk+1]
        //(void) N_VLinearCombination(self.ida_kk+1, self.ida_cvals, self.ida_phi, self.ida_yypredict);
        {
            /*
            self.nlp.ida_yypredict.assign(
                &self
                    .ida_phi
                    .slice_axis(Axis(0), Slice::from(0..=self.ida_kk))
                    .sum_axis(Axis(0)),
            );
            */

            self.nlp.ida_yypredict.fill(P::Scalar::zero());
            for phi in self
                .ida_phi
                .slice_axis(Axis(0), Slice::from(0..=self.ida_kk))
                .genrows()
            {
                self.nlp.ida_yypredict += &phi;
            }
        }

        // yppredict = gamma[1..kk+1] * phi[1..kk+1]
        //(void) N_VLinearCombination(self.ida_kk, self.ida_gamma+1, self.ida_phi+1, self.ida_yppredict);
        /*
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
        */

        let &mut Ida {
            ida_kk,
            ref ida_phi,
            ref ida_gamma,
            ref mut nlp,
            ..
        } = self;

        nlp.ida_yppredict.fill(P::Scalar::zero());
        ndarray::Zip::from(&ida_gamma.slice(s![1..=ida_kk]))
            .and(
                ida_phi
                    .slice_axis(Axis(0), Slice::from(1..=ida_kk))
                    .genrows(),
            )
            .apply(|gamma, phi| {
                nlp.ida_yppredict.scaled_add(*gamma, &phi);
            });
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
        let scalar_kk = <P::Scalar as NumCast>::from(self.ida_kk).unwrap();

        // Compute error for order k.
        let enorm_k = self.wrms_norm(&self.ida_ee, &self.nlp.ida_ewt, self.ida_suppressalg);
        let err_k = self.ida_sigma[self.ida_kk] * enorm_k; // error norms

        // local truncation error norm
        let terr_k = err_k * (scalar_kk + P::Scalar::one());

        let (err_km1, knew) = if self.ida_kk > 1 {
            // Compute error at order k-1
            // delta = phi[ida_kk - 1] + ee
            self.ida_delta = &self.ida_phi.index_axis(Axis(0), self.ida_kk) + &self.ida_ee;
            let enorm_km1 =
                self.wrms_norm(&self.ida_delta, &self.nlp.ida_ewt, self.ida_suppressalg);
            // estimated error at k-1
            let err_km1 = self.ida_sigma[self.ida_kk - 1] * enorm_km1;
            let terr_km1 = scalar_kk * err_km1;

            let knew = if self.ida_kk > 2 {
                // Compute error at order k-2
                // delta += phi[ida_kk - 1]
                self.ida_delta += &self.ida_phi.index_axis(Axis(0), self.ida_kk - 1);
                let enorm_km2 =
                    self.wrms_norm(&self.ida_delta, &self.nlp.ida_ewt, self.ida_suppressalg);
                // estimated error at k-2
                let err_km2 = self.ida_sigma[self.ida_kk - 2] * enorm_km2;
                let terr_km2 = (scalar_kk - P::Scalar::one()) * err_km2;

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
                            self.ida_rr = {
                                let base = P::Scalar::two() * err_knew + P::Scalar::pt0001();
                                let arg = <P::Scalar as NumCast>::from(self.ida_kk + 1)
                                    .unwrap()
                                    .recip();
                                P::Scalar::pt9() * base.powf(-arg)
                            };
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
    ///
    /// # Returns
    ///
    /// * () if `t` was legal. Outputs placed into `yy` and `yp`, and can be accessed using
    ///     `get_yy()` and `get_yp()`.
    ///
    /// # Errors
    ///
    /// * `IdaError::BadTimeValue` if `t` is not within the interval of the last step taken.
    pub fn get_solution(&mut self, t: P::Scalar) -> Result<(), failure::Error> {
        #[cfg(feature = "profiler")]
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
        for j in 1..=kord {
            d = d * gam + c / self.ida_psi[j - 1];
            c = c * gam;
            gam = (delt + self.ida_psi[j - 1]) / self.ida_psi[j];

            self.ida_cvals[j] = c;
            self.ida_dvals[j - 1] = d;
        }

        //retval = N_VLinearCombination(kord+1, self.ida_cvals, self.ida_phi,  yret);
        let ida_yy = &mut self.nlp.ida_yy;
        ida_yy.fill(P::Scalar::zero());
        ndarray::Zip::from(self.ida_cvals.slice(s![0..=kord]))
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(0..=kord))
                    .genrows(),
            )
            .apply(|&c, phi| {
                ida_yy.scaled_add(c, &phi);
            });

        //retval = N_VLinearCombination(kord, self.ida_dvals, self.ida_phi+1, ypret);
        let ida_yp = &mut self.nlp.ida_yp;
        ida_yp.fill(P::Scalar::zero());
        ndarray::Zip::from(self.ida_dvals.slice(s![0..kord]))
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(1..=kord))
                    .genrows(),
            )
            .apply(|&d, phi| {
                ida_yp.scaled_add(d, &phi);
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
        #[cfg(feature = "profiler")]
        profile_scope!(format!("wrms_norm()"));
        if mask {
            x.norm_wrms_masked(w, &self.ida_id)
        } else {
            x.norm_wrms(w)
        }
    }
}
