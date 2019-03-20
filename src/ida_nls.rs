use ndarray::prelude::*;

use super::constants::IdaConst;
use super::linear::LSolver;
use super::nonlinear::NLProblem;
use super::traits::{IdaProblem};



/*
impl<F> NLProblem for Ida<F>
where
    F: IdaModel,
    <F as ModelSpec>::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + IdaConst,
{
}
*/

/// State variables involved in the Non-linear problem
#[derive(Debug, Clone)]
pub struct IdaNLProblem<P, LS>
where
    P: IdaProblem,
    LS: LSolver<P::Scalar>,
{
    // Vectors
    /// work space for y vector (= user's yret)
    pub(super) ida_yy: Array<P::Scalar, Ix1>,
    /// work space for y' vector (= user's ypret)
    pub(super) ida_yp: Array<P::Scalar, Ix1>,
    /// predicted y vector
    pub(super) ida_yypredict: Array<P::Scalar, Ix1>,
    /// predicted y' vector
    pub(super) ida_yppredict: Array<P::Scalar, Ix1>,

    /// saved residual vector
    pub(super) ida_savres: Array<P::Scalar, Ix1>,
    /// current internal value of t
    pub(super) ida_tn: P::Scalar,

    /// current value of scalar (-alphas/hh) in Jacobian
    pub(super) ida_cj: P::Scalar,
    /// cj value saved from last call to lsetup
    pub(super) ida_cjold: P::Scalar,
    /// ratio of cj values: cj/cjold
    pub(super) ida_cjratio: P::Scalar,
    /// scalar used in Newton iteration convergence test
    pub(super) ida_ss: P::Scalar,
    /// norm of previous nonlinear solver update
    pub(super) ida_oldnrm: P::Scalar,
    /// tolerance in direct test on Newton corrections
    pub(super) ida_toldel: P::Scalar,

    /// number of function (res) calls
    pub(super) ida_nre: u64,

    /// number of lsetup calls
    pub(super) ida_nsetups: u64,

    /// Linear Solver
    pub(super) ls: LS,

    pub(super) problem: P,


    a: Array<P::Scalar, Ix2>,
}

impl<P, LS> IdaNLProblem<P, LS>
where
    P: IdaProblem,
    P::Scalar: IdaConst,
    LS: LSolver<P::Scalar>,
{
    /// * `size` - The problem size
    pub fn new(problem: P) -> Self {
        use num_traits::identities::Zero;
        IdaNLProblem {
            ida_yp: Array::zeros(problem.model_size()),
            ida_yy: Array::zeros(problem.model_size()),
            ida_yypredict: Array::zeros(problem.model_size()),
            ida_yppredict: Array::zeros(problem.model_size()),

            ida_savres: Array::zeros(problem.model_size()),
            ida_tn: P::Scalar::zero(),

            ida_cj: P::Scalar::zero(),

            ida_cjold: P::Scalar::zero(),
            ida_cjratio: P::Scalar::zero(),
            ida_ss: P::Scalar::zero(),
            ida_oldnrm: P::Scalar::zero(),
            ida_toldel: P::Scalar::zero(),
            ida_nre: 0,
            ida_nsetups: 0,

            ls: LS::new(problem.model_size()),

            a: Array::zeros((problem.model_size(),problem.model_size())),

            problem,
        }
    }
}

impl<P, LS> NLProblem<P> for IdaNLProblem<P, LS>
where
    P: IdaProblem,
    P::Scalar: IdaConst + ndarray::ScalarOperand,
    LS: LSolver<P::Scalar>,
{
    fn sys<S1, S2>(
        &mut self,
        ycor: &ArrayBase<S1, Ix1>,
        res: &mut ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::DataMut<Elem = P::Scalar>,
    {
        // update yy and yp based on the current correction
        //N_VLinearSum(ONE, self.ida_yypredict, ONE, ycor, self.ida_yy);
        self.ida_yy = &self.ida_yypredict + ycor;
        //N_VLinearSum(ONE, self.ida_yppredict, self.ida_cj, ycor, self.ida_yp);
        //self.ida_yp = &self.ida_yppredict + ycor * self.ida_cj;
        self.ida_yp.assign(&self.ida_yppredict);
        self.ida_yp.scaled_add(self.ida_cj, ycor);

        // evaluate residual
        self.problem
            .res(self.ida_tn, &self.ida_yy, &self.ida_yp, res);

        // increment the number of residual evaluations
        self.ida_nre += 1;

        // save a copy of the residual vector in savres
        self.ida_savres.assign(res);

        //if (retval < 0) return(IDA_RES_FAIL);
        //if (retval > 0) return(IDA_RES_RECVR);

        Ok(())
    }

    fn setup<S1>(
        &mut self,
        ycor: &ArrayBase<S1, Ix1>,
        res: &ArrayView<P::Scalar, Ix1>,
        jbad: bool,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
    {
        use num_traits::identities::One;

        self.ida_nsetups += 1;
        //self.ls.setup(&self.ida_yy, &self.ida_yp, res);

        //Self::jac(0.0, y, &Array::zeros(self.model_size()), &mut self.a).map(|_| true)
        self.problem.jac(
            self.ida_tn,
            self.ida_cj,
            &self.ida_yy,
            &self.ida_yp,
            res,
            &mut self.a.view_mut()
        );

        // update Jacobian status
        //*jcur = SUNTRUE;

        // update convergence test constants
        self.ida_cjold = self.ida_cj;
        self.ida_cjratio = P::Scalar::one();
        self.ida_ss = P::Scalar::twenty();

        //if (retval < 0) return(IDA_LSETUP_FAIL);
        //if (retval > 0) return(IDA_LSETUP_RECVR);

        //return(IDA_SUCCESS);

        Ok(true)
    }

    fn solve<S1, S2>(
        &mut self,
        ycor: &ArrayBase<S1, Ix1>,
        delta: &mut ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::DataMut<Elem = P::Scalar>,
    {
        /*
        self.ls.ls_solve(
            delta,
            &self.ida_ewt,
            &self.ida_yy,
            &self.ida_yp,
            &self.ida_savres,
        );
        */
        //retval = IDA_mem->ida_lsolve(IDA_mem, delta, IDA_mem->ida_ewt, IDA_mem->ida_yy, IDA_mem->ida_yp, IDA_mem->ida_savres);

        //if (retval < 0) return(IDA_LSOLVE_FAIL);
        //if (retval > 0) return(IDA_LSOLVE_RECVR);

        Ok(())
    }

    fn ctest<S1, S2, S3>(
        &mut self,
        y: &ArrayBase<S1, Ix1>,
        del: &ArrayBase<S2, Ix1>,
        tol: P::Scalar,
        ewt: &ArrayBase<S3, Ix1>,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = P::Scalar>,
        S2: ndarray::Data<Elem = P::Scalar>,
        S3: ndarray::Data<Elem = P::Scalar>,
    {
        //realtype delnrm;
        //realtype rate;

        use crate::norm_rms::NormRms;
        use num_traits::identities::One;
        use num_traits::{Float, NumCast};
        // compute the norm of the correction
        let delnrm = del.norm_wrms(ewt);

        // get the current nonlinear solver iteration count
        //let m = self.nls.get_cur_iter();
        let m = 0;

        // test for convergence, first directly, then with rate estimate.
        if m == 0 {
            self.ida_oldnrm = delnrm;
            if delnrm <= P::Scalar::pt0001() * self.ida_toldel {
                return Ok(true);
            }
        } else {
            let rate = (delnrm / self.ida_oldnrm)
                .powf(P::Scalar::one() / <P::Scalar as NumCast>::from(m).unwrap());
            //rate = SUNRpowerR(delnrm / self.ida_oldnrm, P::Scalar::one() / m);
            //if (rate > RATEMAX) return(SUN_NLS_CONV_RECVR);
            self.ida_ss = rate / (P::Scalar::one() - rate);
        }

        if self.ida_ss * delnrm <= tol {
            return Ok(true);
        }

        // not yet converged
        return Ok(false);
    }
}
