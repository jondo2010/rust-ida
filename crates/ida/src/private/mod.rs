use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim};

use crate::{
    constants::MXORDP1,
    traits::{IdaProblem, IdaReal},
    Error, Ida,
};

mod complete_step;

impl<T, D, P, LS, NLS> Ida<T, D, P, LS, NLS>
where
    T: IdaReal,
    D: Dim,
    P: IdaProblem<T, D>,
    LS: linear::LSolver<T, D>,
    NLS: nonlinear::NLSolver<T, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D> + Allocator<u8, D>,
{
    /// This routine evaluates `y(t)` and `y'(t)` as the value and derivative of the interpolating polynomial at the
    /// independent variable `t`, and stores the results in the vectors `yret` and `ypret`.
    /// It uses the current independent variable value, `tn`, and the method order last used, `kused`.
    /// This function is called by `solve` with `t = tout`, `t = tn`, or `t = tstop`.
    ///
    /// If `kused = 0` (no step has been taken), or if `t = tn`, then the order used here is taken to be 1, giving
    /// `yret = phi[0]`, `ypret = phi[1]/psi[0]`.
    ///
    /// # Arguments
    /// * `t` - requested independent variable (time)
    ///
    /// # Returns
    /// * () if `t` was legal. Outputs placed into `yy` and `yp`, and can be accessed using
    ///     `get_yy()` and `get_yp()`.
    ///
    /// # Errors
    /// * `IdaError::BadTimeValue` if `t` is not within the interval of the last step taken.
    pub fn get_solution(&mut self, t: T) -> Result<(), Error> {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("get_solution(t={:.5e})", t));

        self.check_t(t)?;

        // Initialize kord = (kused or 1).
        let kord = if self.ida_kused == 0 {
            1
        } else {
            self.ida_kused
        };

        // Accumulate multiples of columns phi[j] into yret and ypret.
        let delt = t - self.nlp.ida_tn;
        let mut c = T::one();
        let mut d = T::zero();
        let mut gam = delt / self.ida_psi[0];

        self.ida_cvals[0] = c;
        for j in 1..=kord {
            d = d * gam + c / self.ida_psi[j - 1];
            c = c * gam;
            gam = (delt + self.ida_psi[j - 1]) / self.ida_psi[j];

            self.ida_cvals[j] = c;
            self.ida_dvals[j - 1] = d;
        }

        let ida_yy = &mut self.nlp.ida_yy;
        let cvals = self.ida_cvals.rows(0, kord + 1);
        let phi = self.ida_phi.columns(0, kord + 1);
        phi.mul_to(&cvals, ida_yy);

        let ida_yp = &mut self.nlp.ida_yp;
        let dvals = self.ida_dvals.rows(0, kord);
        let phi = self.ida_phi.columns(1, kord);
        phi.mul_to(&dvals, ida_yp);

        Ok(())
    }
}
