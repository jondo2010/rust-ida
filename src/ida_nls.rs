use ndarray::prelude::*;

use super::ida::Ida;
use super::traits::{IdaConst, IdaModel};
use crate::nonlinear::{NLProblem, NLSolver};
use crate::traits::ModelSpec;

#[derive(Clone, Debug)]
struct IDANLProblem {
    A: Array<f64, Ix2>,
    x: Array<f64, Ix1>,
    // Linear solver, matrix and vector objects/pointers
    /*
    SUNLinearSolver LS;   // generic linear solver object
    SUNMatrix J;          // J = dF/dy + cj*dF/dy'
    N_Vector ytemp;       // temp vector used by IDAAtimesDQ
    N_Vector yptemp;      // temp vector used by IDAAtimesDQ
    N_Vector x;           // temp vector used by the solve function
    N_Vector ycur;        // current y vector in Newton iteration
    N_Vector ypcur;       // current yp vector in Newton iteration
    N_Vector rcur;        // rcur = F(tn, ycur, ypcur)

    // Iterative solver tolerance
    realtype sqrtN;     // sqrt(N)
    realtype eplifac;   // eplifac = linear convergence factor

    // Statistics and associated parameters
    realtype dqincfac;  // dqincfac = optional increment factor in Jv
    long int nje;       // nje = no. of calls to jac
    long int npe;       // npe = total number of precond calls
    long int nli;       // nli = total number of linear iterations
    long int nps;       // nps = total number of psolve calls
    long int ncfl;      // ncfl = total number of convergence failures
    long int nreDQ;     // nreDQ = total number of calls to res
    long int njtsetup;  // njtsetup = total number of calls to jtsetup
    long int njtimes;   // njtimes = total number of calls to jtimes
    long int nst0;      // nst0 = saved nst (for performance monitor)
    long int nni0;      // nni0 = saved nni (for performance monitor)
    long int ncfn0;     // ncfn0 = saved ncfn (for performance monitor)
    long int ncfl0;     // ncfl0 = saved ncfl (for performance monitor)
    long int nwarn;     // nwarn = no. of warnings (for perf. monitor)

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
}

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

impl<M, NLS> NLProblem<M, NLS> for Ida<M>
where
    M: IdaModel,
    NLS: NLSolver<M>,
    M::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + IdaConst,
{
    //fn idaNlsResidual(N_Vector ycor, N_Vector res) { }
    fn sys<S1, S2>(
        &self,
        y: &ArrayBase<S1, Ix1>,
        f: &mut ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::DataMut<Elem = M::Scalar>,
    {
        // update yy and yp based on the current correction
        N_VLinearSum(ONE, self.ida_yypredict, ONE, ycor, self.ida_yy);
        N_VLinearSum(ONE, self.ida_yppredict, self.ida_cj, ycor, self.ida_yp);

        // evaluate residual
        retval = self.ida_res(
            self.ida_tn,
            self.ida_yy,
            self.ida_yp,
            res,
            self.ida_user_data,
        );

        // increment the number of residual evaluations
        self.ida_nre += 1;

        // save a copy of the residual vector in savres
        N_VScale(ONE, res, self.ida_savres);

        //if (retval < 0) return(IDA_RES_FAIL);
        //if (retval > 0) return(IDA_RES_RECVR);

        Ok(())
    }

    fn lsetup<S1>(
        &mut self,
        y: &ArrayBase<S1, Ix1>,
        F: &ArrayView<M::Scalar, Ix1>,
        jbad: bool,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
    {
        Ok(false)
    }

    fn lsolve<S1, S2>(
        &self,
        y: &ArrayBase<S1, Ix1>,
        b: &mut ArrayBase<S2, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::DataMut<Elem = M::Scalar>,
    {
        Ok(())
    }

    fn ctest<S1, S2, S3>(
        &self,
        nls: &NLS,
        y: &ArrayBase<S1, Ix1>,
        del: &ArrayBase<S2, Ix1>,
        tol: M::Scalar,
        ewt: &ArrayBase<S3, Ix1>,
    ) -> Result<bool, failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {
        //realtype delnrm;
        //realtype rate;

        use crate::traits::NormRms;
        // compute the norm of the correction
        let delnrm = del.norm_wrms(ewt);

        // get the current nonlinear solver iteration count
        let m = nls.get_cur_iter();

        // test for convergence, first directly, then with rate estimate.
        if m == 0 {
            self.ida_oldnrm = delnrm;
            if delnrm <= M::Scalar::pt0001() * self.ida_toldel {
                return Ok(true);
            }
        } else {
            let rate =
                (delnrm / self.ida_oldnrm).powr(M::Scalar::one() / NumCast::from(m).unwrap());
            //rate = SUNRpowerR(delnrm / self.ida_oldnrm, M::Scalar::one() / m);
            //if (rate > RATEMAX) return(SUN_NLS_CONV_RECVR);
            self.ida_ss = rate / (M::Scalar::one() - rate);
        }

        if self.ida_ss*delnrm <= tol {
            return Ok(true);
        }

        // not yet converged
        return Ok(false);
    }
}
