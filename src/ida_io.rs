use super::*;

impl<P, LS, NLS, TolC> Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
{
    /// Return a view of the y vector
    pub fn get_yy(&self) -> ndarray::ArrayView1<P::Scalar> {
        self.nlp.ida_yy.view()
    }

    /// Return a view of the y' vector
    pub fn get_yp(&self) -> ndarray::ArrayView1<P::Scalar> {
        self.nlp.ida_yp.view()
    }

    /// returns the integration method order used during the last internal step.
    pub fn get_last_order(&self) -> usize {
        self.ida_kused
    }

    /// returns the integration method order to be used on the next internal step.
    pub fn get_current_order(&self) -> usize {
        self.ida_kk
    }

    /// returns the value of the integration step size used on the first step.
    pub fn get_actual_init_step(&self) -> P::Scalar {
        self.ida_h0u
    }

    /// returns the integration step size taken on the last internal step (if from `solve()`), or
    /// the last value of the artificial step size h (if from `calc_ic()`).
    pub fn get_last_step(&self) -> P::Scalar {
        self.ida_hused
    }

    /// returns the integration step size to be attempted on the next internal step.
    pub fn get_current_setp(&self) -> P::Scalar {
        self.ida_hh
    }

    /// returns the current internal time reached by the solver.
    pub fn get_current_time(&self) -> P::Scalar {
        self.nlp.ida_tn
    }

    /// returns a suggested factor by which the user’s tolerances should be scaled when too much
    /// accuracy has been requested for some internal step.
    pub fn get_tol_scale_factor(&self) -> P::Scalar {
        self.ida_tolsf
    }

    /// returns the cumulative number of internal steps taken by the solver (total so far).
    pub fn get_num_steps(&self) -> usize {
        self.counters.ida_nst
    }

    /// returns the number of calls to the user’s residual evaluation function.
    /// Note: does not account for calls made to res from a linear solver or preconditioner module
    pub fn get_num_res_evals(&self) -> usize {
        self.nlp.ida_nre
    }

    /// returns the cumulative number of calls made to the linear solver’s setup function (total so
    /// far).
    pub fn get_num_lin_solv_setups(&self) -> usize {
        self.nlp.ida_nsetups
    }

    ///  returns the cumulative number of local error test failures that have occurred (total so far).
    pub fn get_num_err_test_fails(&self) -> usize {
        self.counters.ida_netf
    }

    /// returns the cumulative number of calls to the idals Jacobian approximation function.
    pub fn get_num_jac_evals(&self) -> usize {
        self.nlp.lp.nje
    }

    pub fn get_num_nonlin_solv_iters(&self) -> usize {
        /* get number of iterations for IC calc */
        /* get number of iterations from the NLS */
        self.counters.ida_nni + self.nls.get_num_iters()
    }

    /// returns the number of calls to the DAE residual needed for the DQ Jacobian approximation or
    /// J*v product approximation
    pub fn get_num_lin_res_evals(&self) -> usize {
        self.nlp.lp.nre_dq
    }

    /// returns the cumulative number of linear iterations.
    pub fn get_num_lin_iters(&self) -> usize {
        //self.nlp.lp.nli
        0
    }

    /// returns the cumulative number of linear convergence failures.
    pub fn get_num_nonlin_solv_conv_fails(&self) -> usize {
        self.counters.ida_ncfn
    }

    /// returns the cumulative number of calls to the user root function.
    pub fn get_num_g_evals(&self) -> usize {
        self.ida_nge
    }

    /// returns an array showing which functions were found to have a root.
    /// 
    /// Note that, for the components gi for which a root was found, the sign of rootsfound[i] indicates the direction of zero-crossing. A value of +1 indicates that gi is increasing, while a value of −1 indicates a decreasing gi.
    pub fn get_root_info(&self) -> ndarray::ArrayView1<P::Scalar> {
        self.ida_iroots.view()
    }
}
