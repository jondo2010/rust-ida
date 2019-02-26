use ndarray::*;
//use ndarray_linalg::*;

use failure::Fail;

use crate::traits::*;

/// hmax_inv default value
const HMAX_INV_DEFAULT: f64 = 0.0;
/// maxord default value
const MAXORD_DEFAULT: usize = 5;
/// max. number of N_Vectors in phi
const MXORDP1: usize = 6;
/// mxstep default value
const MXSTEP_DEFAULT: usize = 500;

/// max number of convergence failures allowed
const MXNCF: u32 = 10;
/// max number of error test failures allowed
const MXNEF: u32 = 10;
/// max. number of h tries in IC calc.
const MAXNH: u32 = 5;
/// max. number of J tries in IC calc.
const MAXNJ: u32 = 4;
/// max. Newton iterations in IC calc.
const MAXNI: u32 = 10;
/// Newton convergence test constant
const EPCON: f64 = 0.33;
/// max backtracks per Newton step in IDACalcIC
const MAXBACKS: u32 = 100;
/// constant for updating Jacobian/preconditioner
const XRATE: f64 = 0.25;

#[derive(Debug, Fail)]
enum IdaError {
    // LSETUP_ERROR_NONRECVR
    // IDA_ERR_FAIL
    /// IDA_REP_RES_ERR:
    #[fail(
        display = "The user's residual function repeatedly returned a recoverable error flag, but the solver was unable to recover"
    )]
    RepeatedResidualError {},

    /// IDA_ILL_INPUT
    #[fail(display = "One of the input arguments was illegal. See printed message")]
    IllegalInput {},

    /// IDA_LINIT_FAIL
    #[fail(display = "The linear solver's init routine failed")]
    LinearInitFail {},

    /// IDA_BAD_EWT
    #[fail(
        display = "Some component of the error weight vector is zero (illegal), either for the input value of y0 or a corrected value"
    )]
    BadErrorWeightVector {},

    /// IDA_RES_FAIL
    #[fail(display = "The user's residual routine returned a non-recoverable error flag")]
    ResidualFail {},

    /// IDA_FIRST_RES_FAIL
    #[fail(
        display = "The user's residual routine returned a recoverable error flag on the first call, but IDACalcIC was unable to recover"
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
        display = "The user's residual routine, or the linear solver's setup or solve routine had a recoverable error, but IDACalcIC was unable to recover"
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
        display = "The Linesearch algorithm failed to find a  solution with a step larger than steptol   in weighted RMS norm"
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
}

/// Structure containing the parameters for the numerical integration.
#[derive(Debug, Clone)]
pub struct Ida<F: IdaModel> {
    f: F,
    //dt: <F::Scalar as AssociatedReal>::Real,
    //x: Array<F::Scalar, Ix1>,
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
    //ida_yy: Array1<<F::Scalar as AssociatedReal>::Real>,
    /// work space for y' vector (= user's ypret)
    //ida_yp: Array1<<F::Scalar as AssociatedReal>::Real>,
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
    //ida_hh: <F::Scalar as AssociatedReal>::Real,
    ida_hh: F::Scalar,
    /// step size used on last successful step
    ida_hused: F::Scalar,
    /// rr = hnext / hused
    ida_rr: F::Scalar,
    //ida_rr: <F::Scalar as AssociatedReal>::Real,
    /// current internal value of t
    ida_tn: F::Scalar,
    /// value of tret previously returned by IDASolve
    ida_tretlast: F::Scalar,
    /// current value of scalar (-alphas/hh) in Jacobian
    ida_cj: F::Scalar,
    /// cj value saved from last successful step
    ida_cjlast: F::Scalar,
    //realtype ida_cjold;    /* cj value saved from last call to lsetup           */
    //realtype ida_cjratio;  /* ratio of cj values: cj/cjold                      */
    //realtype ida_ss;       /* scalar used in Newton iteration convergence test  */
    //realtype ida_oldnrm;   /* norm of previous nonlinear solver update          */
    //realtype ida_epsNewt;  /* test constant in Newton convergence test          */
    //realtype ida_epcon;    /* coeficient of the Newton covergence test          */
    //realtype ida_toldel;   /* tolerance in direct test on Newton corrections    */

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
    //realtype ida_cvals[MXORDP1];
    //realtype ida_dvals[MAXORD_DEFAULT];
}

impl<
        F: IdaModel<
            Scalar = impl num_traits::Float
                         + num_traits::float::FloatConst
                         + num_traits::NumRef
                         + num_traits::NumAssignRef
                         + ScalarOperand
                         + std::fmt::Debug,
        >,
    > Ida<F>
//where
//num_traits::float::Float + num_traits::float::FloatConst + num_traits::NumAssignRef + ScalarOperand
//<<F as ModelSpec>::Dim as Dimension>::Larger: RemoveAxis,
{
    /// Creates a new IdaModel given a ModelSpec, initial Arrays of yy0 and yyp
    ///
    /// Can throw if ModelSpec::Scalar is unable to convert any constant initialization value.
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
            //ida_itol        = IDA_NN;
            //ida_user_efun   = SUNFALSE;
            //ida_efun        = NULL;
            //ida_edata       = NULL;
            //ida_ehfun       = IDAErrHandler;
            //ida_eh_data     = IDA_mem;
            //ida_errfp       = stderr;
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as u64,
            ida_hmax_inv: F::Scalar::from(HMAX_INV_DEFAULT).unwrap(),
            ida_hin: F::Scalar::zero(),
            //ida_epcon       = EPCON;
            ida_maxnef: MXNEF as u64,
            ida_maxncf: MXNCF as u64,
            //ida_suppressalg = SUNFALSE;
            //ida_id          = NULL;
            //ida_constraints: Array::zeros(yy0.raw_dim()),
            ida_constraintsSet: false,
            ida_tstopset: false,

            // set the saved value maxord_alloc
            //ida_maxord_alloc = MAXORD_DEFAULT;

            // Set default values for IC optional inputs
            //ida_epiccon = PT01 * EPCON;
            //ida_maxnh   = MAXNH;
            //ida_maxnj   = MAXNJ;
            //ida_maxnit  = MAXNI;
            //ida_maxbacks  = MAXBACKS;
            //ida_lsoff   = SUNFALSE;
            //ida_steptol = SUNRpowerR(IDA_mem->ida_uround, TWOTHIRDS);

            /* Initialize lrw and liw */
            //ida_lrw = 25 + 5*MXORDP1;
            //ida_liw = 38;

            /* Initialize nonlinear solver pointer */
            //IDA_mem->NLS    = NULL;
            //IDA_mem->ownNLS = SUNFALSE;
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
            //ida_tolsf: <F::Scalar as AssociatedReal>::Real::from_f64(1.0),

            //ida_nge = 0;

            //ida_irfnd = 0;

            // Initialize root-finding variables

            //ida_glo     = NULL;
            //ida_ghi     = NULL;
            //ida_grout   = NULL;
            //ida_iroots  = NULL;
            //ida_rootdir = NULL;
            //ida_gfun    = NULL;
            //ida_nrtfn   = 0;
            //ida_gactive  = NULL;
            //ida_mxgnull  = 1;

            // Not from ida.c...
            ida_ewt: Array::zeros(yy0.raw_dim()),
            ida_ee: Array::zeros(yy0.raw_dim()),
            ida_suppressalg: false,

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

            ida_cvals: Array::zeros(MXORDP1),
            ida_dvals: Array::zeros(MXORDP1),

            ida_yypredict: Array::zeros(yy0.raw_dim()),
            ida_yppredict: Array::zeros(yy0.raw_dim()),
        }
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
            //self.ida_cj = into_scalar(1.0) / self.ida_hh;
            self.ida_phase = 0;
            self.ida_ns = 0;
        }

        /* To prevent 'unintialized variable' warnings */
        //err_k = ZERO;
        //err_km1 = ZERO;

        /* Looping point for attempts to take a step */

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
            if self.ida_tstopset {
                if (self.ida_tn - self.ida_tstop) * self.ida_hh > F::Scalar::one() {
                    self.ida_tn = self.ida_tstop;
                }
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
                let (err_k, err_km1, nflag) = self.test_error(ck);
            }

            // Test for convergence or error test failures
            //if nflag != IDA_SUCCESS {
            // restore and decide what to do
            self.restore(saved_t);
            //kflag = handle_n_flag(IDA_mem, nflag, err_k, err_km1, &(self.ida_ncfn), &ncf, &(self.ida_netf), &nef);

            // exit on nonrecoverable failure
            //if kflag != PREDICT_AGAIN {
            //    return (kflag);
            //}

            // recoverable error; predict again
            if self.ida_nst == 0 {
                self.reset();
            }
            continue;
            //}

            /* kflag == IDA_SUCCESS */
            break;
        }

        /* Nonlinear system solve and error test were both successful;
        update data, and consider change of step and/or order */

        //self.complete_step(err_k, err_km1);

        /*
          Rescale ee vector to be the estimated local error
          Notes:
            (1) altering the value of ee is permissible since
                it will be overwritten by
                IDASolve()->IDAStep()->IDANls()
                before it is needed again
            (2) the value of ee is only valid if IDAHandleNFlag()
                returns either PREDICT_AGAIN or IDA_SUCCESS
        */
        //N_VScale(ck, IDA_mem->ida_ee, IDA_mem->ida_ee);
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
    pub fn set_coeffs(&mut self) -> F::Scalar {
        // Set coefficients for the current stepsize h
        if self.ida_hh != self.ida_hused || self.ida_kk != self.ida_kused {
            self.ida_ns = 0;
        }
        self.ida_ns = std::cmp::min(self.ida_ns + 1, self.ida_kused + 2);
        if self.ida_kk + 1 >= self.ida_ns {
            self.ida_beta[0] = F::Scalar::one();
            self.ida_alpha[0] = F::Scalar::one();
            let mut temp1 = self.ida_hh;
            self.ida_gamma[0] = F::Scalar::zero();
            self.ida_sigma[0] = F::Scalar::one();
            for i in 1..self.ida_kk {
                let temp2 = self.ida_psi[i - 1];
                self.ida_psi[i - 1] = temp1;
                self.ida_beta[i] = self.ida_beta[i - 1] * (self.ida_psi[i - 1] / temp2);
                temp1 = temp2 + self.ida_hh;
                self.ida_alpha[i] = self.ida_hh / temp1;
                self.ida_sigma[i] =
                    self.ida_sigma[i - 1] * self.ida_alpha[i] * F::Scalar::from(i).unwrap();
                self.ida_gamma[i] = self.ida_gamma[i - 1] + self.ida_alpha[i - 1] / self.ida_hh;
            }
            self.ida_psi[self.ida_kk] = temp1;
        }
        // compute alphas, alpha0
        let mut alphas = F::Scalar::zero();
        let mut alpha0 = F::Scalar::zero();
        for i in 0..self.ida_kk {
            alphas -= F::Scalar::one() / F::Scalar::from(i + 1).unwrap();
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
            let nv = self.ida_kk - self.ida_ns + 1;
            let c = self.ida_beta.slice(s![self.ida_ns..]);

            let ix1 = s![self.ida_ns..];
            let ix2 = SliceOrIndex::from(0..self.ida_ns);
            //self.ida_phi.slice(ix1);
            //&SliceInfo<<<<F as ModelSpec>::Dim as Dimension>::Larger as Dimension>::SliceArg, _>
            //let ix: <<<F as ModelSpec>::Dim as Dimension>::Larger as Dimension>::SliceArg = &[SliceOrIndex::from(0..self.ida_ns)];

            //<<<<F as ModelSpec>::Dim as Dimension>::Larger as Dimension>::SliceArg as AsRef<[SliceOrIndex]>>::from(0..1);

            //let z = self.ida_phi.slice_mut(s![self.ida_ns..]);
            //self.ida_phi.index_axis_mut(Axis(0), 0).assign(&yy0);
            /*
            N_VScaleVectorArray(
              self.ida_kk - self.ida_ns + 1,
              self.ida_beta + self.ida_ns,
              self.ida_phi + self.ida_ns,
              self.ida_phi + self.ida_ns,
            );
            */
        }

        return ck;
    }

    /// IDANls
    /// This routine attempts to solve the nonlinear system using the linear solver specified.
    /// NOTE: this routine uses N_Vector ee as the scratch vector tempv3 passed to lsetup.
    pub fn nonlinear_solve(&mut self) -> Result<(), failure::Error> {
        unimplemented!();
    }

    /// IDAPredict
    /// This routine predicts the new values for vectors yy and yp.
    pub fn predict(&mut self) -> () {
        for j in 0..self.ida_kk {
            self.ida_cvals[j] = F::Scalar::one();
        }

        let cv = self.ida_cvals.slice(s![self.ida_kk + 1..]);
        let ph = self.ida_phi.index_axis(Axis(0), self.ida_kk + 1);
        //let x = self.ida_phi.slice(s![self.ida_kk + 1.., ..]);

        // ida_delta = ida_phi[ida_kk] + self.ida_ee;
        //self.ida_delta .assign(&self.ida_phi.index_axis(Axis(0), self.ida_kk));
        let v = self.ida_phi.index_axis(Axis(0), self.ida_kk);
        //Zip::from(&mut self.ida_delta).and(&v).and(&self.ida_ee).apply(|delta, phi, ee| {});
        self.ida_delta.assign(&v);
        self.ida_delta += &self.ida_ee;

        //self.ida_delta = &v + &self.ida_ee;

        // ida_yypredict = sum 0..kk (cvals[k] * phi[k])
        /*
        for i = 0..n
            for j = 0..nv

        */

        let c = self.ida_cvals.slice(s![self.ida_kk + 1..]);
        let x = self
            .ida_phi
            .slice_axis(Axis(0), Slice::from(self.ida_kk + 1..));
        //let mut z = self.ida_yypredict.slice_axis_mut(Axis(0), Slice::from(self.ida_kk+1..));

        ndarray::Zip::from(&mut self.ida_yypredict)
            .and(x.lanes(Axis(0)))
            .apply(|z, row| {
                *z = (&row * &c).sum();
            });

        //N_VLinearCombination(&c, &x, &mut z);
        //(void) N_VLinearCombination(IDA_mem->ida_kk+1, IDA_mem->ida_cvals, IDA_mem->ida_phi, IDA_mem->ida_yypredict);
        //(void) N_VLinearCombination(IDA_mem->ida_kk, IDA_mem->ida_gamma+1, IDA_mem->ida_phi+1, IDA_mem->ida_yppredict);
    }

    /// IDATestError
    ///
    /// This routine estimates errors at orders k, k-1, k-2, decides whether or not to suggest an order
    /// decrease, and performs the local error test.
    ///
    /// Returns a tuple of (err_k, err_km1, nflag)
    pub fn test_error(
        &mut self,
        ck: F::Scalar,
    ) -> (
        F::Scalar, // err_k
        F::Scalar, // err_km1
        bool,      // nflag
    ) {
        //realtype enorm_k, enorm_km1, enorm_km2;   /* error norms */
        //realtype terr_k, terr_km1, terr_km2;      /* local truncation error norms */
        // Compute error for order k.
        let enorm_k = self.wrms_norm(&self.ida_ee, &self.ida_ewt, self.ida_suppressalg);
        let err_k = self.ida_sigma[self.ida_kk] * enorm_k;
        let terr_k = err_k * F::Scalar::from(self.ida_kk + 1).unwrap();

        let mut err_km1 = F::Scalar::zero(); // estimated error at k-1
        let mut err_km2 = F::Scalar::zero(); // estimated error at k-2

        self.ida_knew = self.ida_kk;

        if self.ida_kk > 1 {
            // Compute error at order k-1
            // ida_delta = ida_phi[ida_kk] + self.ida_ee;
            self.ida_delta
                .assign(&self.ida_phi.index_axis(Axis(0), self.ida_kk));
            self.ida_delta.scaled_add(F::Scalar::one(), &self.ida_ee);

            let enorm_km1 = self.wrms_norm(&self.ida_delta, &self.ida_ewt, self.ida_suppressalg);
            err_km1 = self.ida_sigma[self.ida_kk - 1] * enorm_km1;
            let terr_km1 = err_km1 * F::Scalar::from(self.ida_kk).unwrap();

            if self.ida_kk > 2 {
                // Compute error at order k-2
                // ida_delta = ida_phi[ida_kk - 1] + ida_delta
                self.ida_delta
                    .assign(&self.ida_phi.index_axis(Axis(0), self.ida_kk - 1));
                self.ida_delta.scaled_add(F::Scalar::one(), &self.ida_ee);

                let enorm_km2 =
                    self.wrms_norm(&self.ida_delta, &self.ida_ewt, self.ida_suppressalg);
                err_km2 = self.ida_sigma[self.ida_kk - 2] * enorm_km2;
                let terr_km2 = err_km2 * F::Scalar::from(self.ida_kk - 1).unwrap();

                // Decrease order if errors are reduced
                if terr_km1.max(terr_km2) <= terr_k {
                    self.ida_knew = self.ida_kk - 1;
                }
            } else {
                // Decrease order to 1 if errors are reduced by at least 1/2
                if terr_km1 <= (terr_k * F::Scalar::from(0.5).unwrap()) {
                    self.ida_knew = self.ida_kk - 1;
                }
            }
        };

        (
            err_k,
            err_km1,
            (ck * enorm_k) > F::Scalar::one(), // Perform error test
        )
    }

    /// IDARestore
    /// This routine restores tn, psi, and phi in the event of a failure. It changes back phi-star to
    /// phi (changed in IDASetCoeffs)
    pub fn restore(&mut self, saved_t: F::Scalar) -> () {
        //int j;

        self.ida_tn = saved_t;

        //for (j = 1; j <= IDA_mem->ida_kk; j++)
        for j in 1..self.ida_kk {
            self.ida_psi[j - 1] = self.ida_psi[j] - self.ida_hh;
        }

        if self.ida_ns <= self.ida_kk {
            //for (j = IDA_mem->ida_ns; j <= IDA_mem->ida_kk; j++)
            for j in self.ida_ns..self.ida_kk {
                self.ida_cvals[j - self.ida_ns] = F::Scalar::one() / self.ida_beta[j];
            }

            /*
            N_VScaleVectorArray(IDA_mem->ida_kk-IDA_mem->ida_ns+1,
                                       IDA_mem->ida_cvals,
                                       IDA_mem->ida_phi+IDA_mem->ida_ns,
                                       IDA_mem->ida_phi+IDA_mem->ida_ns);
                                       */
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
    pub fn handle_n_flag(
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
    pub fn reset(&mut self) -> () {
        self.ida_psi[0] = self.ida_hh;
        //N_VScale(IDA_mem->ida_rr, IDA_mem->ida_phi[1], IDA_mem->ida_phi[1]);
        self.ida_phi *= self.ida_rr;
    }

    /// IDACompleteStep
    /// This routine completes a successful step.  It increments nst, saves the stepsize and order
    /// used, makes the final selection of stepsize and order for the next step, and updates the phi
    /// array.
    pub fn complete_step(&mut self, err_k: F::Scalar, err_km1: F::Scalar) -> () {
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
                let mut hnew = F::Scalar::from(2.0).unwrap() * self.ida_hh;
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

            if action == Action::None {
                //N_VLinearSum(ONE, IDA_mem->ida_ee, -ONE, IDA_mem->ida_phi[IDA_mem->ida_kk + 1], IDA_mem->ida_tempv1);
                let enorm = self.wrms_norm(&self.ida_tempv1, &self.ida_ewt, self.ida_suppressalg);
                let err_kp1 = enorm / F::Scalar::from(self.ida_kk + 2).unwrap();

                // Choose among orders k-1, k, k+1 using local truncation error norms.

                let terr_k = F::Scalar::from(self.ida_kk + 1).unwrap() * err_k;
                let terr_kp1 = F::Scalar::from(self.ida_kk + 2).unwrap() * err_kp1;

                if self.ida_kk == 1 {
                    if terr_kp1 >= F::Scalar::from(0.5).unwrap() * terr_k {
                        action = Action::Maintain;
                    } else {
                        action = Action::Raise;
                    }
                } else {
                    let terr_km1 = F::Scalar::from(self.ida_kk).unwrap() * err_km1;
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
            //self.ida_rr = SUNRpowerR( TWO * err_knew + PT0001, -ONE/(IDA_mem->ida_kk + 1) );

            if self.ida_rr >= F::Scalar::from(2.0).unwrap() {
                hnew = F::Scalar::from(2.0).unwrap() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > F::Scalar::one() {
                    hnew /= tmp;
                }
            } else if self.ida_rr <= F::Scalar::one() {
                self.ida_rr = F::Scalar::from(0.5)
                    .unwrap()
                    .max(self.ida_rr.min(F::Scalar::from(0.9).unwrap()));
                //self.ida_rr = SUNMAX(HALF, SUNMIN(PT9,IDA_mem->ida_rr));
                hnew = self.ida_hh * self.ida_rr;
            }

            self.ida_hh = hnew;
        } /* end of phase if block */

        /* Save ee for possible order increase on next step */
        if (self.ida_kused < self.ida_maxord) {
            //N_VScale(ONE, IDA_mem->ida_ee, IDA_mem->ida_phi[IDA_mem->ida_kused + 1]);
        }

        /* Update phi arrays */

        /* To update phi arrays compute X += Z where                  */
        /* X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ] */
        /* Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ] */

        //self.ida_Zvecs[0] = self.ida_ee;
        //self.ida_Xvecs[0] = self.ida_phi[self.ida_kused];
        //for (j=1; j<=IDA_mem->ida_kused; j++) {
        for j in 1..self.ida_kused {
            //self.ida_Zvecs[j] = self.ida_phi[self.ida_kused-j+1];
            //self.ida_Xvecs[j] = self.ida_phi[self.ida_kused-j];
        }

        /*
        (void) N_VLinearSumVectorArray(IDA_mem->ida_kused+1,
                                       ONE, IDA_mem->ida_Xvecs,
                                       ONE, IDA_mem->ida_Zvecs,
                                       IDA_mem->ida_Xvecs);
                                       */
    }

    /// This routine evaluates y(t) and y'(t) as the value and derivative of the interpolating
    /// polynomial at the independent variable t, and stores the results in the vectors yret and ypret.
    /// It uses the current independent variable value, tn, and the method order last used, kused.
    /// This function is called by `solve` with `t = tout`, `t = tn`, or `t = tstop`.
    ///
    /// If `kused = 0` (no step has been taken), or if `t = tn`, then the order used here is taken
    /// to be 1, giving `yret = phi[0]`, `ypret = phi[1]/psi[0]`.
    ///
    /// The return values are:
    ///   IDA_SUCCESS  if t is legal, or
    ///   IDA_BAD_T    if t is not within the interval of the last step taken.
    pub fn get_solution(
        &mut self,
        t: F::Scalar,
        yret: &mut Array<F::Scalar, Ix1>,
        ypret: &mut Array<F::Scalar, Ix1>,
    ) -> Result<(), failure::Error> {
        // Check t for legality.  Here tn - hused is t_{n-1}.

        //tfuzz = HUNDRED * IDA_mem->ida_uround * (SUNRabs(IDA_mem->ida_tn) + SUNRabs(IDA_mem->ida_hh));

        let mut tfuzz = F::Scalar::from(100.0).unwrap()
            * F::Scalar::epsilon()
            * (self.ida_tn.abs() + self.ida_hh.abs());
        if self.ida_hh < F::Scalar::zero() {
            tfuzz = -tfuzz;
        }
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

        //retval = N_VLinearCombination(kord+1, IDA_mem->ida_cvals, IDA_mem->ida_phi,  yret);
        ndarray::Zip::from(yret)
            .and(
                self.ida_phi
                    .slice_axis(Axis(0), Slice::from(0..kord + 1))
                    .lanes(Axis(0)),
            )
            .apply(|z, row| {
                *z = (&row * &self.ida_cvals.slice(s![0..kord + 1])).sum();
            });

        //retval = N_VLinearCombination(kord, IDA_mem->ida_dvals, IDA_mem->ida_phi+1, ypret);
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
    pub fn wrms_norm(
        &self,
        x: &Array<F::Scalar, Ix1>,
        w: &Array<F::Scalar, Ix1>,
        mask: bool,
    ) -> F::Scalar {
        if mask {
            //x.norm_wrms_masked(w, self.ida_id)
            x.norm_wrms_masked(w, &self.ida_id)
        } else {
            x.norm_wrms(w)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_get_solution() {
        use crate::ida::Ida;
        use crate::lorenz63::Lorenz63;
        use ndarray::*;

        let f = Lorenz63::default();
        let mut i = Ida::new(f, array![1., 2., 3.], array![4., 5., 6.]);
        //println!("{:#?}", i);

        i.ida_cvals += 2.0;
        i.ida_dvals += 2.0;
        i.ida_phi += 1.0;
        i.ida_tn = 1e-3;
        i.ida_hh = 1e-6;

        let mut yret = Array::zeros((3));
        let mut ypret = Array::zeros((3));

        i.get_solution(0.0, &mut yret, &mut ypret).unwrap();

        dbg!(&yret);
        dbg!(&ypret);

        i.get_solution(10.0, &mut yret, &mut ypret).unwrap();
    }
}
