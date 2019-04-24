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

    pub fn get_last_order(&self) -> usize {
        self.ida_kused
    }

    pub fn get_current_order(&self) -> usize {
        self.ida_kk
    }

    pub fn get_actual_init_step(&self) -> P::Scalar {
        self.ida_h0u
    }

    pub fn get_last_step(&self) -> P::Scalar {
        self.ida_hused
    }

    pub fn get_current_setp(&self) -> P::Scalar {
        self.ida_hh
    }

    pub fn get_current_time(&self) -> P::Scalar {
        self.nlp.ida_tn
    }

    pub fn get_num_steps(&self) -> usize {
        self.counters.ida_nst
    }

    pub fn get_num_res_evals(&self) -> usize {
        self.nlp.ida_nre
    }

    pub fn get_num_lin_solv_setups(&self) -> usize {
        self.nlp.ida_nsetups
    }

    pub fn get_num_jac_evals(&self) -> usize {
        self.nlp.lp.nje
    }

    pub fn get_num_nonlin_solv_iters(&self) -> usize {
        /* get number of iterations for IC calc */
        /* get number of iterations from the NLS */
        self.counters.ida_nni + self.nls.get_num_iters()
    }

    /// IDAGetNumLinResEvals returns the number of calls to the DAE residual needed for the DQ
    /// Jacobian approximation or J*v product approximation
    pub fn get_num_lin_res_evals(&self) -> usize {
        self.nlp.lp.nreDQ
    }
}
