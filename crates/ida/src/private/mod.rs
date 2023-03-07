use std::fmt::LowerExp;

use log::trace;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim};

use crate::{
    constants::MXORDP1,
    traits::{IdaProblem, IdaReal},
    Error, Ida,
};

mod complete_step;

impl<T, D, P, LS, NLS> Ida<T, D, P, LS, NLS>
where
    T: IdaReal + LowerExp,
    D: Dim,
    P: IdaProblem<T, D>,
    LS: linear::LSolver<T, D>,
    NLS: nonlinear::NLSolver<T, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D> + Allocator<u8, D>,
{
    /// This routine computes the coefficients relevant to the current step.
    ///
    /// The counter ns counts the number of consecutive steps taken at constant stepsize h and order k, up to a maximum
    /// of k + 2.
    ///
    /// Then the first ns components of beta will be one, and on a step with ns = k + 2, the
    /// coefficients alpha, etc. need not be reset here.
    /// Also, complete_step() prohibits an order increase until ns = k + 2.
    ///
    /// Returns the 'variable stepsize error coefficient ck'
    pub fn set_coeffs(&mut self) -> T {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("set_coeffs()"));

        // Set coefficients for the current stepsize h
        if (self.ida_hh != self.ida_hused) || (self.ida_kk != self.ida_kused) {
            self.ida_ns = 0;
        }

        self.ida_ns = std::cmp::min(self.ida_ns + 1, self.ida_kused + 2);
        if self.ida_kk + 1 >= self.ida_ns {
            self.ida_beta[0] = T::one();
            self.ida_alpha[0] = T::one();
            self.ida_gamma[0] = T::zero();
            self.ida_sigma[0] = T::one();

            let mut temp1 = self.ida_hh;
            for i in 1..=self.ida_kk {
                let scalar_i = T::from(i).unwrap();
                let temp2 = self.ida_psi[i - 1];
                self.ida_psi[i - 1] = temp1;
                self.ida_beta[i] = self.ida_beta[i - 1] * self.ida_psi[i - 1] / temp2;
                temp1 = temp2 + self.ida_hh;
                self.ida_alpha[i] = self.ida_hh / temp1;
                self.ida_sigma[i] = scalar_i * self.ida_sigma[i - 1] * self.ida_alpha[i];
                self.ida_gamma[i] = self.ida_gamma[i - 1] + self.ida_alpha[i - 1] / self.ida_hh;
            }
            self.ida_psi[self.ida_kk] = temp1;
        }

        // compute alphas, alpha0
        let mut alphas = T::zero();
        let mut alpha0 = T::zero();
        for i in 0..self.ida_kk {
            let scalar_i = T::from(i + 1).unwrap();
            alphas -= T::one() / scalar_i;
            alpha0 -= self.ida_alpha[i];
        }

        // compute leading coefficient cj
        self.ida_cjlast = self.nlp.lp.ida_cj;
        self.nlp.lp.ida_cj = -alphas / self.ida_hh;

        // compute variable stepsize error coefficient ck
        let mut ck = (self.ida_alpha[self.ida_kk] + alphas - alpha0).abs();
        ck = ck.max(self.ida_alpha[self.ida_kk]);

        // change phi to phi-star
        // Scale i=self.ida_ns to i<=self.ida_kk
        if self.ida_ns <= self.ida_kk {
            //N_VScaleVectorArray( self.ida_kk - self.ida_ns + 1, self.ida_beta + self.ida_ns, self.ida_phi + self.ida_ns, self.ida_phi + self.ida_ns,);
            let beta = self.ida_beta.rows_range(self.ida_ns..=self.ida_kk);

            let mut phi = self.ida_phi.columns_range_mut(self.ida_ns..=self.ida_kk);
            phi.column_iter_mut()
                .zip(beta.iter())
                .for_each(|(mut phi, beta)| phi *= *beta);
        }

        return ck;
    }

    /// IDARestore
    /// This routine restores tn, psi, and phi in the event of a failure.
    /// It changes back `phi-star` to `phi` (changed in `set_coeffs()`)
    pub fn restore(&mut self, saved_t: T) -> () {
        trace!("restore(saved_t={:.6e})", saved_t);
        self.nlp.ida_tn = saved_t;

        // Restore psi[0 .. kk] = psi[1 .. kk + 1] - hh
        for j in 1..self.ida_kk + 1 {
            self.ida_psi[j - 1] = self.ida_psi[j] - self.ida_hh;
        }

        if self.ida_ns <= self.ida_kk {
            // cvals[0 .. kk-ns+1] = 1 / beta[ns .. kk+1]
            self.ida_cvals
                .rows_mut(0, self.ida_kk - self.ida_ns + 1)
                .copy_from(
                    &self
                        .ida_beta
                        .rows_range(self.ida_ns..=self.ida_kk)
                        .map(|x| x.recip()),
                );

            // phi[ns .. (kk + 1)] *= cvals[ns .. (kk + 1)]
            let mut phi = self.ida_phi.columns_range_mut(self.ida_ns..=self.ida_kk);
            phi.column_iter_mut()
                .zip(self.ida_cvals.rows(0, self.ida_kk - self.ida_ns + 1).iter())
                .for_each(|(mut phi, cval)| phi *= *cval);
        }
    }

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
