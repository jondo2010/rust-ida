use std::fmt::LowerExp;

use log::trace;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, Storage, Vector};
use nonlinear::norm_wrms::NormWRMS;

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
    DefaultAllocator: Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D>,
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
    pub(crate) fn set_coeffs(&mut self) -> T {
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

    /// IDAPredict
    /// This routine predicts the new values for vectors `yy` and `yp`.
    pub(crate) fn predict(&mut self) {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("predict()"));

        // yypredict = cvals * phi[0..kk+1]
        // N_VLinearCombination(self.ida_kk+1, self.ida_cvals, self.ida_phi, self.ida_yypredict);
        self.nlp
            .ida_yypredict
            .copy_from(&self.ida_phi.columns(0, self.ida_kk + 1).column_sum());

        // yppredict = gamma[1..kk+1] * phi[1..kk+1]
        // N_VLinearCombination(self.ida_kk, self.ida_gamma+1, self.ida_phi+1, self.ida_yppredict);
        let g = self.ida_gamma.rows_range(1..self.ida_kk);
        let phi = self.ida_phi.columns_range(1..self.ida_kk);
        phi.mul_to(&g, &mut self.nlp.ida_yppredict);
    }

    /// IDATestError
    ///
    /// This routine estimates errors at orders k, k-1, k-2, decides whether or not to suggest an order
    /// decrease, and performs the local error test.
    ///
    /// Returns a tuple of (err_k, err_km1, nflag)
    pub(crate) fn test_error(
        &mut self,
        ck: T,
    ) -> Result<
        (
            T,    // err_k
            T,    // err_km1
            bool, // converged
        ),
        (T, T, Error),
    > {
        //trace!("test_error phi={:.5e}", self.ida_phi);
        //trace!("test_error ee={:.5e}", self.ida_ee);
        let scalar_kk = T::from(self.ida_kk).unwrap();

        // Compute error for order k.
        let enorm_k = self.wrms_norm(&self.ida_ee);
        let err_k = self.ida_sigma[self.ida_kk] * enorm_k; // error norms

        // local truncation error norm
        let terr_k = err_k * (scalar_kk + T::one());

        let (err_km1, knew) = if self.ida_kk > 1 {
            // Compute error at order k-1
            // delta = phi[ida_kk - 1] + ee
            self.ida_delta = &self.ida_phi.column(self.ida_kk) + &self.ida_ee;
            let enorm_km1 = self.wrms_norm(&self.ida_delta);
            // estimated error at k-1
            let err_km1 = self.ida_sigma[self.ida_kk - 1] * enorm_km1;
            let terr_km1 = scalar_kk * err_km1;

            let knew = if self.ida_kk > 2 {
                // Compute error at order k-2
                // delta += phi[ida_kk - 1]
                self.ida_delta += &self.ida_phi.column(self.ida_kk - 1);
                let enorm_km2 = self.wrms_norm(&self.ida_delta);
                // estimated error at k-2
                let err_km2 = self.ida_sigma[self.ida_kk - 2] * enorm_km2;
                let terr_km2 = (scalar_kk - T::one()) * err_km2;

                // Decrease order if errors are reduced
                if terr_km1.max(terr_km2) <= terr_k {
                    self.ida_kk - 1
                } else {
                    self.ida_kk
                }
            } else {
                // Decrease order to 1 if errors are reduced by at least 1/2
                if terr_km1 <= (terr_k * T::half()) {
                    self.ida_kk - 1
                } else {
                    self.ida_kk
                }
            };

            (err_km1, knew)
        } else {
            (T::zero(), self.ida_kk)
        };

        self.ida_knew = knew;

        // Perform error test
        let converged = (ck * enorm_k) <= T::one();

        if converged {
            Ok((err_k, err_km1, true))
        } else {
            Err((err_k, err_km1, Error::TestFail))
        }
    }

    /// IDARestore
    /// This routine restores tn, psi, and phi in the event of a failure.
    /// It changes back `phi-star` to `phi` (changed in `set_coeffs()`)
    pub(crate) fn restore(&mut self, saved_t: T) -> () {
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
    pub(crate) fn get_solution(&mut self, t: T) -> Result<(), Error> {
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

    /// Returns the WRMS norm of vector x with weights w.
    pub(crate) fn wrms_norm<S>(&self, x: &Vector<T, D, S>) -> T
    where
        S: Storage<T, D>,
    {
        match self.ida_id {
            Some(ref ida_id) if self.ida_suppressalg => {
                // Mask out the components of ida_ewt using ida_id.
                x.norm_wrms(&self.nlp.ida_ewt.component_mul(ida_id))
            }
            _ => x.norm_wrms(&self.nlp.ida_ewt),
        }
    }
}
