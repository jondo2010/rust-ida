use super::*;

impl<P, LS, NLS, TolC> Ida<P, LS, NLS, TolC>
where
    P: IdaProblem,
    LS: linear::LSolver<P::Scalar>,
    NLS: nonlinear::NLSolver<P>,
    TolC: TolControl<P::Scalar>,
{
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

    pub fn get_num_res_evals(&self) -> u64 {
        self.nlp.ida_nre
    }

    pub fn get_num_lin_solv_setups(&self) -> u64 {
        self.nlp.ida_nsetups
    }
}