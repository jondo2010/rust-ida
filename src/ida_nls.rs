use ndarray::prelude::*;

use super::constants::IdaConst;
use super::error::IdaError;
use super::ida_ls::IdaLProblem;
use super::linear::LSolver;
use super::nonlinear::{Error, NLProblem, NLSolver};
use super::traits::IdaProblem;

#[cfg(feature = "data_trace")]
use serde::Serialize;

// nonlinear solver parameters
/// max convergence rate used in divergence check
const RATEMAX: f64 = 0.9;

/// State variables involved in the Non-linear problem
#[derive(Debug, Clone)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
pub struct IdaNLProblem<P, LS>
where
    P: IdaProblem,
    LS: LSolver<P::Scalar>,
{
    // Vectors
    /// work space for y vector (= user's yret)
    pub(super) ida_yy: Array1<P::Scalar>,
    /// work space for y' vector (= user's ypret)
    pub(super) ida_yp: Array1<P::Scalar>,
    /// predicted y vector
    pub(super) ida_yypredict: Array1<P::Scalar>,
    /// predicted y' vector
    pub(super) ida_yppredict: Array1<P::Scalar>,

    /// error weight vector
    pub(super) ida_ewt: Array1<P::Scalar>,

    /// saved residual vector
    pub(super) ida_savres: Array1<P::Scalar>,
    /// current internal value of t
    pub(super) ida_tn: P::Scalar,

    /// scalar used in Newton iteration convergence test
    pub(super) ida_ss: P::Scalar,
    /// norm of previous nonlinear solver update
    pub(super) ida_oldnrm: P::Scalar,
    /// tolerance in direct test on Newton corrections
    pub(super) ida_toldel: P::Scalar,

    /// number of function (res) calls
    pub(super) ida_nre: usize,

    /// number of lsetup calls
    pub(super) ida_nsetups: usize,

    /// Linear Problem
    pub(super) lp: IdaLProblem<P, LS>,

    a: Array2<P::Scalar>,
}

impl<P, LS> IdaNLProblem<P, LS>
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
    /// * `size` - The problem size
    pub fn new<S1, S2>(problem: P, yy0: ArrayBase<S1, Ix1>, yp0: ArrayBase<S2, Ix1>) -> Self
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
    {
        use num_traits::identities::Zero;
        IdaNLProblem {
            ida_yy: yy0.to_owned(),
            ida_yp: yp0.to_owned(),
            ida_yypredict: Array::zeros(problem.model_size()),
            ida_yppredict: Array::zeros(problem.model_size()),

            ida_ewt: Array::zeros(problem.model_size()),

            ida_savres: Array::zeros(problem.model_size()),
            ida_tn: P::Scalar::zero(),

            ida_ss: P::Scalar::zero(),
            ida_oldnrm: P::Scalar::zero(),
            ida_toldel: P::Scalar::zero(),
            ida_nre: 0,
            ida_nsetups: 0,

            a: Array::zeros((problem.model_size(), problem.model_size())),

            lp: IdaLProblem::new(problem),
        }
    }
}

impl<P, LS> NLProblem<P> for IdaNLProblem<P, LS>
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
    /// idaNlsResidual
    fn sys<S1, S2>(
        &mut self,
        ycor: ArrayBase<S1, Ix1>,
        mut res: ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::DataMut<Elem = P::Scalar>,
    {
        // update yy and yp based on the current correction
        //N_VLinearSum(ONE, self.ida_yypredict, ONE, ycor, self.ida_yy);
        //N_VLinearSum(ONE, self.ida_yppredict, self.ida_cj, ycor, self.ida_yp);
        self.ida_yy = &self.ida_yypredict + &ycor;
        //self.ida_yp = &self.ida_yppredict + ycor * self.ida_cj;
        self.ida_yp.assign(&self.ida_yppredict);
        self.ida_yp.scaled_add(self.lp.ida_cj, &ycor);

        // evaluate residual
        self.lp.problem.res(
            self.ida_tn,
            self.ida_yy.view(),
            self.ida_yp.view(),
            res.view_mut(),
        );

        // increment the number of residual evaluations
        self.ida_nre += 1;

        // save a copy of the residual vector in savres
        self.ida_savres.assign(&res);

        //if (retval < 0) return(IDA_RES_FAIL);
        //if (retval > 0) return(IDA_RES_RECVR);

        Ok(())
    }

    /// idaNlsLSetup
    fn setup<S1, S2>(
        &mut self,
        _ycor: ArrayBase<S1, Ix1>,
        res: ArrayBase<S2, Ix1>,
        _jbad: bool,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
    {
        use num_traits::identities::One;

        self.ida_nsetups += 1;
        // ida_lsetup() aka idaLsSetup()
        self.lp
            .setup(self.ida_yy.view(), self.ida_yp.view(), res.view());

        // update Jacobian status
        //*jcur = SUNTRUE;

        // update convergence test constants
        self.lp.ida_cjold = self.lp.ida_cj;
        self.lp.ida_cjratio = P::Scalar::one();
        self.ida_ss = P::Scalar::twenty();

        //if (retval < 0) return(IDA_LSETUP_FAIL);
        //if (retval > 0) return(IDA_LSETUP_RECVR);

        //return(IDA_SUCCESS);

        Ok(true)
    }

    /// idaNlsLSolve
    fn solve<S1, S2>(
        &mut self,
        _ycor: ArrayBase<S1, Ix1>,
        mut delta: ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::DataMut<Elem = P::Scalar>,
    {
        self.lp.solve(
            delta.view_mut(),
            self.ida_ewt.view(),
            self.ida_yy.view(),
            self.ida_yp.view(),
            self.ida_savres.view(),
        );

        // lp solved, delta = [-0.00000000000000001623748801571458, -0.0000000004872681362104435, 0.0000000004865181206243604]
        //[7.5001558608301906e-13,-4.8726813621044346e-10,4.8651812062436036e-10,]
        //retval = IDA_mem->ida_lsolve(IDA_mem, delta, IDA_mem->ida_ewt, IDA_mem->ida_yy, IDA_mem->ida_yp, IDA_mem->ida_savres);

        //if (retval < 0) return(IDA_LSOLVE_FAIL);
        //if (retval > 0) return(IDA_LSOLVE_RECVR);

        Ok(())
    }

    /// idaNlsConvTest
    fn ctest<S1, S2, S3, NLS>(
        &mut self,
        solver: &NLS,
        _y: ArrayBase<S1, Ix1>,
        del: ArrayBase<S2, Ix1>,
        tol: P::Scalar,
        ewt: ArrayBase<S3, Ix1>,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
        S3: ndarray::Data<Elem = P::Scalar>,
        NLS: NLSolver<P>,
    {
        use crate::norm_rms::NormRms;
        use num_traits::identities::One;
        use num_traits::{Float, NumCast};

        // compute the norm of the correction
        let delnrm = del.norm_wrms(&ewt);

        // get the current nonlinear solver iteration count
        let m = solver.get_cur_iter();

        // test for convergence, first directly, then with rate estimate.
        if m == 0 {
            self.ida_oldnrm = delnrm;
            if delnrm <= P::Scalar::pt0001() * self.ida_toldel {
                return Ok(true);
            }
        } else {
            let rate = {
                let base = delnrm / self.ida_oldnrm;
                let arg = <P::Scalar as NumCast>::from(m).unwrap().recip();
                base.powf(arg)
            };
            if rate > <P::Scalar as NumCast>::from(RATEMAX).unwrap() {
                return Err(failure::Error::from(Error::ConvergenceRecover {}));
            }
            self.ida_ss = rate / (P::Scalar::one() - rate);
        }

        if self.ida_ss * delnrm <= tol {
            return Ok(true);
        }

        // not yet converged
        return Ok(false);
    }
}
