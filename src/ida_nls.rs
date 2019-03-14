use ndarray::prelude::*;

use super::ida::Ida;
use super::traits::{IdaConst, IdaProblem};
use crate::linear::LSolver;
use crate::nonlinear::{NLProblem, NLSolver};
use crate::traits::ModelSpec;

/*
#[derive(Clone, Debug)]
struct IDANLProblem {
    A: Array<f64, Ix2>,
    x: Array<f64, Ix1>,
    // Linear solver, matrix and vector objects/pointers
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
}
*/

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
    LS: LSolver<P>,
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
}

impl<P, LS> IdaNLProblem<P, LS>
where
    P: IdaProblem,
    P::Scalar: IdaConst,
    LS: LSolver<P>,
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

            ls: LS::new(),

            problem,
        }
    }
}

impl<P, LS> NLProblem<P> for IdaNLProblem<P, LS>
where
    P: IdaProblem,
    P::Scalar: IdaConst + ndarray::ScalarOperand,
    LS: LSolver<P>,
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

    fn lsetup<S1>(
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
        self.ls.ls_setup(&self.ida_yy, &self.ida_yp, res);

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

    fn lsolve<S1, S2>(
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