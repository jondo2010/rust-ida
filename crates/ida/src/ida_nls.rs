use linear::LSolver;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, Matrix, OVector, Storage, StorageMut, U1,
};
use nonlinear::{norm_wrms::NormWRMS, NLProblem, NLSolver};

use crate::{
    ida_ls::IdaLProblem,
    traits::{IdaProblem, IdaReal},
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

// nonlinear solver parameters
/// max convergence rate used in divergence check
const RATEMAX: f64 = 0.9;

/// State variables involved in the Non-linear problem
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, D>: Serialize, IdaLProblem<T, D, P, LS>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "T: Deserialize<'de>,  OVector<T, D>: Deserialize<'de>, IdaLProblem<T, D, P, LS>: Deserialize<'de>"
    ))
)]
#[derive(Debug, Clone)]
pub struct IdaNLProblem<T, D, P, LS>
where
    T: IdaReal,
    D: DimName,
    P: IdaProblem<T, D>,
    LS: LSolver<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    // Vectors
    /// work space for y vector (= user's yret)
    pub(super) ida_yy: OVector<T, D>,
    /// work space for y' vector (= user's ypret)
    pub(super) ida_yp: OVector<T, D>,
    /// predicted y vector
    pub(super) ida_yypredict: OVector<T, D>,
    /// predicted y' vector
    pub(super) ida_yppredict: OVector<T, D>,

    /// error weight vector
    pub(super) ida_ewt: OVector<T, D>,

    /// saved residual vector
    pub(super) ida_savres: OVector<T, D>,
    /// current internal value of t
    pub(super) ida_tn: T,

    /// scalar used in Newton iteration convergence test
    pub(super) ida_ss: T,
    /// norm of previous nonlinear solver update
    pub(super) ida_oldnrm: T,
    /// tolerance in direct test on Newton corrections
    pub(super) ida_toldel: T,

    /// number of function (res) calls
    pub(super) ida_nre: usize,

    /// number of lsetup calls
    pub(super) ida_nsetups: usize,

    /// Linear Problem
    pub(super) lp: IdaLProblem<T, D, P, LS>,
}

impl<T, D, P, LS> IdaNLProblem<T, D, P, LS>
where
    T: IdaReal,
    D: DimName,
    P: IdaProblem<T, D>,
    LS: LSolver<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    pub fn new<SA, SB>(
        problem: P,
        ls: LS,
        yy0: &Matrix<T, D, U1, SA>,
        yp0: &Matrix<T, D, U1, SB>,
    ) -> Self
    where
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
    {
        IdaNLProblem {
            ida_yy: yy0.clone_owned(),
            ida_yp: yp0.clone_owned(),
            ida_yypredict: OVector::zeros(),
            ida_yppredict: OVector::zeros(),

            ida_ewt: OVector::zeros(),

            ida_savres: OVector::zeros(),
            ida_tn: T::zero(),

            ida_ss: T::zero(),
            ida_oldnrm: T::zero(),
            ida_toldel: T::zero(),
            ida_nre: 0,
            ida_nsetups: 0,

            lp: IdaLProblem::new(problem, ls),
        }
    }
}

impl<T, D, P, LS> NLProblem<T, D> for IdaNLProblem<T, D, P, LS>
where
    T: IdaReal,
    D: DimName,
    P: IdaProblem<T, D>,
    LS: LSolver<T, D>,
    DefaultAllocator: Allocator<T, D> + Allocator<T, D, D>,
{
    /// idaNlsResidual
    fn sys<SA, SB>(
        &mut self,
        ycor: &Matrix<T, D, U1, SA>,
        res: &mut Matrix<T, D, U1, SB>,
    ) -> Result<(), nonlinear::Error>
    where
        SA: Storage<T, D, U1>,
        SB: StorageMut<T, D, U1>,
    {
        // update yy and yp based on the current correction
        //N_VLinearSum(ONE, self.ida_yypredict, ONE, ycor, self.ida_yy);
        //N_VLinearSum(ONE, self.ida_yppredict, self.ida_cj, ycor, self.ida_yp);
        self.ida_yy = &self.ida_yypredict + ycor;

        //self.ida_yp = &self.ida_yppredict + ycor * self.ida_cj;
        self.ida_yp.copy_from(&self.ida_yppredict);
        self.ida_yp.axpy(self.lp.ida_cj, ycor, T::one());

        // evaluate residual
        self.lp
            .problem
            .res(self.ida_tn, &self.ida_yy, &self.ida_yp, res);

        // increment the number of residual evaluations
        self.ida_nre += 1;

        // save a copy of the residual vector in savres
        self.ida_savres.copy_from(res);

        //if (retval < 0) return(IDA_RES_FAIL);
        //if (retval > 0) return(IDA_RES_RECVR);

        Ok(())
    }

    /// idaNlsLSetup
    fn setup<SA, SB>(
        &mut self,
        _ycor: &Matrix<T, D, U1, SA>,
        res: &Matrix<T, D, U1, SB>,
        _jbad: bool,
    ) -> Result<bool, nonlinear::Error>
    where
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
    {
        self.ida_nsetups += 1;
        // ida_lsetup() aka idaLsSetup()
        self.lp.setup(&self.ida_yy, &self.ida_yp, res);

        // update Jacobian status
        //*jcur = SUNTRUE;

        // update convergence test constants
        self.lp.ida_cjold = self.lp.ida_cj;
        self.lp.ida_cjratio = T::one();
        self.ida_ss = T::twenty();

        //if (retval < 0) return(IDA_LSETUP_FAIL);
        //if (retval > 0) return(IDA_LSETUP_RECVR);

        //return(IDA_SUCCESS);

        Ok(true)
    }

    /// idaNlsLSolve
    fn solve<SA, SB>(
        &mut self,
        _ycor: &Matrix<T, D, U1, SA>,
        delta: &mut Matrix<T, D, U1, SB>,
    ) -> Result<(), nonlinear::Error>
    where
        SA: Storage<T, D, U1>,
        SB: StorageMut<T, D, U1>,
    {
        self.lp.solve(
            delta,
            &self.ida_ewt,
            &self.ida_yy,
            &self.ida_yp,
            &self.ida_savres,
        );

        // lp solved, delta = [-0.00000000000000001623748801571458, -0.0000000004872681362104435, 0.0000000004865181206243604]
        //[7.5001558608301906e-13,-4.8726813621044346e-10,4.8651812062436036e-10,]
        //retval = IDA_mem->ida_lsolve(IDA_mem, delta, IDA_mem->ida_ewt, IDA_mem->ida_yy, IDA_mem->ida_yp, IDA_mem->ida_savres);

        //if (retval < 0) return(IDA_LSOLVE_FAIL);
        //if (retval > 0) return(IDA_LSOLVE_RECVR);

        Ok(())
    }

    /// idaNlsConvTest
    fn ctest<NLS, SA, SB, SC>(
        &mut self,
        solver: &NLS,
        _y: &Matrix<T, D, U1, SA>,
        del: &Matrix<T, D, U1, SB>,
        tol: T,
        ewt: &Matrix<T, D, U1, SC>,
    ) -> Result<bool, nonlinear::Error>
    where
        NLS: NLSolver<T, D>,
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
        SC: Storage<T, D, U1>,
    {
        // compute the norm of the correction

        let delnrm = del.norm_wrms(&ewt);

        // get the current nonlinear solver iteration count
        let m = solver.get_cur_iter();

        // test for convergence, first directly, then with rate estimate.
        if m == 0 {
            self.ida_oldnrm = delnrm;
            if delnrm <= T::pt0001() * self.ida_toldel {
                return Ok(true);
            }
        } else {
            let rate = {
                let base = delnrm / self.ida_oldnrm;
                let arg = T::from(m).unwrap().recip();
                base.powf(arg)
            };
            if rate > T::from(RATEMAX).unwrap() {
                return Err(nonlinear::Error::ConvergenceRecover {});
            }
            self.ida_ss = rate / (T::one() - rate);
        }

        if self.ida_ss * delnrm <= tol {
            return Ok(true);
        }

        // not yet converged
        return Ok(false);
    }
}
