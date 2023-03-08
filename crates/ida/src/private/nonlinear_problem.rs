//! Interfaces the IDA nonlinear problem to the nonlinear solver

use linear::LSolver;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, OVector, Storage, StorageMut, Vector,
};
use nonlinear::{norm_wrms::NormWRMS, NLProblem, NLSolver};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use super::linear_problem::IdaLProblem;
use crate::traits::{IdaProblem, IdaReal};

// nonlinear solver parameters
/// max convergence rate used in divergence check
const RATEMAX: f64 = 0.9;

/// State variables involved in the Non-linear problem
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, P::D>: Serialize, IdaLProblem<T, P, LS>: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "T: Deserialize<'de>,  OVector<T, P::D>: Deserialize<'de>, IdaLProblem<T, P, LS>: Deserialize<'de>"
    ))
)]
#[derive(Debug, Clone)]
pub struct IdaNLProblem<T, P, LS>
where
    T: IdaReal,
    P: IdaProblem<T>,
    LS: LSolver<T, P::D>,
    DefaultAllocator: Allocator<T, P::D> + Allocator<T, P::D, P::D>,
{
    // Vectors
    /// work space for y vector (= user's yret)
    pub ida_yy: OVector<T, P::D>,
    /// cratespace for y' vector (= user's ypret)
    pub ida_yp: OVector<T, P::D>,
    /// cratected y vector
    pub ida_yypredict: OVector<T, P::D>,
    /// cratected y' vector
    pub ida_yppredict: OVector<T, P::D>,

    /// crate weight vector
    pub ida_ewt: OVector<T, P::D>,

    /// crate residual vector
    pub ida_savres: OVector<T, P::D>,
    /// cratent internal value of t
    pub ida_tn: T,

    /// crater used in Newton iteration convergence test
    pub ida_ss: T,
    /// crateof previous nonlinear solver update
    pub ida_oldnrm: T,
    /// crateance in direct test on Newton corrections
    pub ida_toldel: T,

    /// crater of function (res) calls
    pub ida_nre: usize,

    /// crater of lsetup calls
    pub ida_nsetups: usize,

    /// crater Problem
    pub lp: IdaLProblem<T, P, LS>,
}

impl<T, P, LS> IdaNLProblem<T, P, LS>
where
    T: IdaReal,
    P: IdaProblem<T>,
    P::D: DimName,
    LS: LSolver<T, P::D>,
    DefaultAllocator: Allocator<T, P::D> + Allocator<T, P::D, P::D>,
{
    pub fn new<SA, SB>(
        problem: P,
        ls: LS,
        yy0: &Vector<T, P::D, SA>,
        yp0: &Vector<T, P::D, SB>,
    ) -> Self
    where
        SA: Storage<T, P::D>,
        SB: Storage<T, P::D>,
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

impl<T, P, LS> NLProblem<T, P::D> for IdaNLProblem<T, P, LS>
where
    T: IdaReal,
    P: IdaProblem<T>,
    LS: LSolver<T, P::D>,
    DefaultAllocator: Allocator<T, P::D> + Allocator<T, P::D, P::D>,
{
    /// idaNlsResidual
    fn sys<SA, SB>(
        &mut self,
        ycor: &Vector<T, P::D, SA>,
        res: &mut Vector<T, P::D, SB>,
    ) -> Result<(), nonlinear::Error>
    where
        SA: Storage<T, P::D>,
        SB: StorageMut<T, P::D>,
    {
        tracing::trace!("idaNlsResidual");
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
        _ycor: &Vector<T, P::D, SA>,
        res: &Vector<T, P::D, SB>,
        _jbad: bool,
    ) -> Result<bool, nonlinear::Error>
    where
        SA: Storage<T, P::D>,
        SB: Storage<T, P::D>,
    {
        tracing::trace!("idaNlsLSetup");
        self.ida_nsetups += 1;
        // ida_lsetup() aka idaLsSetup()
        let retval = self.lp.setup(&self.ida_yy, &self.ida_yp, res);

        // update convergence test constants
        self.lp.ida_cjold = self.lp.ida_cj;
        self.lp.ida_cjratio = T::one();
        self.ida_ss = T::twenty();

        retval?;
        Ok(true)
    }

    /// idaNlsLSolve
    fn solve<SA, SB>(
        &mut self,
        _ycor: &Vector<T, P::D, SA>,
        delta: &mut Vector<T, P::D, SB>,
    ) -> Result<(), nonlinear::Error>
    where
        SA: Storage<T, P::D>,
        SB: StorageMut<T, P::D>,
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
        _y: &Vector<T, P::D, SA>,
        del: &Vector<T, P::D, SB>,
        tol: T,
        ewt: &Vector<T, P::D, SC>,
    ) -> Result<bool, nonlinear::Error>
    where
        NLS: NLSolver<T, P::D>,
        SA: Storage<T, P::D>,
        SB: Storage<T, P::D>,
        SC: Storage<T, P::D>,
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
