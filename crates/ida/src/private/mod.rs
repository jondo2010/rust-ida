use std::fmt::LowerExp;

use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Storage, Vector};
use nonlinear::norm_wrms::NormWRMS;

use crate::{
    constants::{MXORDP1, XRATE},
    error::RecoverableKind,
    traits::{IdaProblem, IdaReal},
    Error, Ida,
};

mod complete_step;
mod linear_problem;
mod nonlinear_problem;
mod root_finding;
mod solve;
mod stop_test;

pub(crate) use linear_problem::{IdaLProblem, IdaLProblemCounters};
pub(crate) use nonlinear_problem::IdaNLProblem;
pub(crate) use root_finding::RootStatus;

impl<T, P, LS, NLS> Ida<T, P, LS, NLS>
where
    T: IdaReal + LowerExp,
    P: IdaProblem<T>,
    LS: linear::LSolver<T, P::D>,
    NLS: nonlinear::NLSolver<T, P::D>,
    DefaultAllocator: Allocator<T, P::D>
        + Allocator<T, P::R>
        + Allocator<i8, P::R>
        + Allocator<T, P::D, P::D>
        + Allocator<T, P::D, Const<MXORDP1>>,
{
    /// IDAInitialSetup
    ///
    /// This routine is called by `solve` once at the first step. It performs all checks on optional inputs and inputs
    /// to `init`/`reinit` that could not be done before.
    ///
    /// If no error is encountered, IDAInitialSetup returns IDA_SUCCESS. Otherwise, it returns an error flag and
    /// reported to the error handler function.
    fn initial_setup(&mut self) {
        //booleantype conOK;
        //int ier;

        // Initial error weight vector
        self.tol_control
            .ewt_set(&self.ida_phi.column(0), &mut self.nlp.ida_ewt);

        /*
        // Check to see if y0 satisfies constraints.
        if (IDA_mem->ida_constraintsSet) {
          conOK = N_VConstrMask(IDA_mem->ida_constraints, IDA_mem->ida_phi[0], IDA_mem->ida_tempv2);
          if (!conOK) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_Y0_FAIL_CONSTR);
            return(IDA_ILL_INPUT);
          }
        }

        // Call linit function if it exists.
        if (IDA_mem->ida_linit != NULL) {
          ier = IDA_mem->ida_linit(IDA_mem);
          if (ier != 0) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_LINIT_FAIL);
            return(IDA_LINIT_FAIL);
          }
        }
        */

        //return(IDA_SUCCESS);
    }

    /// This routine performs one internal IDA step, from `tn` to `tn + hh`. It calls other
    /// routines to do all the work.
    ///
    /// It solves a system of differential/algebraic equations of the form `F(t,y,y') = 0`, for
    /// one step.
    /// In IDA, `tt` is used for `t`, `yy` is used for `y`, and `yp` is used for `y'`. The function
    /// F is supplied as 'res' by the user.
    ///
    /// The methods used are modified divided difference, fixed leading coefficient forms of
    /// backward differentiation formulas. The code adjusts the stepsize and order to control the
    /// local error per step.
    ///
    /// The main operations done here are as follows:
    /// * initialize various quantities;
    /// * setting of multistep method coefficients;
    /// * solution of the nonlinear system for yy at t = tn + hh;
    /// * deciding on order reduction and testing the local error;
    /// * attempting to recover from failure in nonlinear solver or error test;
    /// * resetting stepsize and order for the next step.
    /// * updating phi and other state data if successful;
    ///
    /// On a failure in the nonlinear system solution or error test, the step may be reattempted,
    /// depending on the nature of the failure.
    ///
    /// Variables or arrays (all in the IDAMem structure) used in IDAStep are:
    ///
    /// tt -- Independent variable.
    /// yy -- Solution vector at tt.
    /// yp -- Derivative of solution vector after successful stelp.
    /// res -- User-supplied function to evaluate the residual. See the description given in file ida.h
    /// lsetup -- Routine to prepare for the linear solver call. It may either save or recalculate
    ///   quantities used by lsolve. (Optional)
    /// lsolve -- Routine to solve a linear system. A prior call to lsetup may be required.
    /// hh  -- Appropriate step size for next step.
    /// ewt -- Vector of weights used in all convergence tests.
    /// phi -- Array of divided differences used by IDAStep. This array is composed of (maxord+1)
    ///   nvectors (each of size Neq). (maxord+1) is the maximum order for the problem, maxord, plus 1.
    ///
    /// Return values are:
    ///       IDA_SUCCESS   IDA_RES_FAIL      LSETUP_ERROR_NONRTolCVR
    ///                     IDA_LSOLVE_FAIL   IDA_ERR_FAIL
    ///                     IDA_CONSTR_FAIL   IDA_CONV_FAIL
    ///                     IDA_REP_RES_ERR
    fn step(&mut self) -> Result<(), Error> {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("step(), nst={}", self.counters.ida_nst));

        let saved_t = self.nlp.ida_tn;

        if self.counters.ida_nst == 0 {
            self.ida_kk = 1;
            self.ida_kused = 0;
            self.ida_hused = T::zero();
            self.ida_psi[0] = self.ida_hh;
            self.nlp.lp.ida_cj = self.ida_hh.recip();
            self.ida_phase = 0;
            self.ida_ns = 0;
        }

        let mut ncf = 0; // local counter for convergence failures
        let mut nef = 0; // local counter for error test failures

        // Looping point for attempts to take a step

        let (ck, err_k, err_km1) = loop {
            #[cfg(feature = "data_trace")]
            {
                serde_json::to_writer(&self.data_trace, self).unwrap();
                self.data_trace.write_all(b",\n").unwrap();
            }

            //-----------------------
            // Set method coefficients
            //-----------------------

            let ck = self.set_coeffs();

            //kflag = IDA_SUCCESS;

            //----------------------------------------------------
            // If tn is past tstop (by roundoff), reset it to tstop.
            //-----------------------------------------------------

            self.nlp.ida_tn += self.ida_hh;
            if let Some(tstop) = self.ida_tstop {
                if ((self.nlp.ida_tn - tstop) * self.ida_hh) > T::one() {
                    self.nlp.ida_tn = tstop;
                }
            }

            //-----------------------
            // Advance state variables
            //-----------------------

            // Compute predicted values for yy and yp
            self.predict();

            // Nonlinear system solution
            let (err_k, err_km1, converged) = self
                .nonlinear_solve()
                .map_err(|err| (T::zero(), T::zero(), err))
                .and_then(|_| {
                    // If NLS was successful, perform error test
                    self.test_error(ck)
                })
                // Test for convergence or error test failures
                .or_else(|(err_k, err_km1, err)| {
                    // restore and decide what to do
                    self.restore(saved_t);

                    self.handle_n_flag(err, err_k, err_km1, &mut ncf, &mut nef)
                        .map(|_| {
                            // recoverable error; predict again
                            if self.counters.ida_nst == 0 {
                                self.reset();
                            }

                            (err_k, err_km1, false)
                        })
                })?;

            if converged {
                break (ck, err_k, err_km1);
            }
        };

        // Nonlinear system solve and error test were both successful;
        // update data, and consider change of step and/or order

        self.complete_step(err_k, err_km1);

        //  Rescale ee vector to be the estimated local error
        //  Notes:
        //    (1) altering the value of ee is permissible since it will be overwritten by
        //        solve()->step()->nonlinear_solve() before it is needed again
        //    (2) the value of ee is only valid if IDAHandleNFlag() returns either
        //        PREDICT_AGAIN or IDA_SUCCESS

        self.ida_ee *= ck;

        Ok(())
    }

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

    /// IDANls
    /// This routine attempts to solve the nonlinear system using the linear solver specified.
    /// NOTE: this routine uses N_Vector ee as the scratch vector tempv3 passed to lsetup.
    pub(crate) fn nonlinear_solve(&mut self) -> Result<(), Error> {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("nonlinear_solve()"));

        let mut call_lsetup = false;

        // Initialize if the first time called
        if self.counters.ida_nst == 0 {
            self.nlp.lp.ida_cjold = self.nlp.lp.ida_cj;
            self.nlp.ida_ss = T::twenty();
            //if (self.ida_lsetup) { callLSetup = true; }
            call_lsetup = true;
        }

        // Decide if lsetup is to be called
        self.nlp.lp.ida_cjratio = self.nlp.lp.ida_cj / self.nlp.lp.ida_cjold;
        let temp1 = T::from((1.0 - XRATE) / (1.0 + XRATE)).unwrap();
        let temp2 = temp1.recip();
        if self.nlp.lp.ida_cjratio < temp1 || self.nlp.lp.ida_cjratio > temp2 {
            call_lsetup = true;
        }
        if self.nlp.lp.ida_cj != self.ida_cjlast {
            self.nlp.ida_ss = T::hundred();
        }

        // initial guess for the correction to the predictor
        self.ida_delta.fill(T::zero());

        // call nonlinear solver setup if it exists
        self.nls.setup(&mut self.ida_delta)?;

        //TODO(FIXME)
        let w = self.nlp.ida_ewt.clone();

        //trace!("\"ewt\":{:.6e}", w);

        // solve the nonlinear system
        let retval = self.nls.solve(
            &mut self.nlp,
            &self.ida_delta,
            &mut self.ida_ee,
            &w,
            self.ida_eps_newt,
            call_lsetup,
        );

        // increment counter
        self.counters.ida_nni += self.nls.get_num_iters();

        // update yy and yp based on the final correction from the nonlinear solve
        self.nlp.ida_yy = &self.nlp.ida_yypredict + &self.ida_ee;
        //N_VLinearSum( ONE, self.ida_yppredict, self.ida_cj, self.ida_ee, self.ida_yp,);
        //self.ida_yp = &self.ida_yppredict + (&self.ida_ee * self.ida_cj);
        self.nlp.ida_yp.copy_from(&self.nlp.ida_yppredict);
        self.nlp
            .ida_yp
            .axpy(self.nlp.lp.ida_cj, &self.ida_ee, T::one());

        // return if nonlinear solver failed */
        retval?;

        // If otherwise successful, check and enforce inequality constraints.

        // Check constraints and get mask vector mm, set where constraints failed
        if let Some(_constraints) = &self.ida_constraints {
            unimplemented!();
            /*
            self.ida_mm = self.ida_tempv2;
            let constraintsPassed = N_VConstrMask(self.ida_constraints, self.ida_yy, self.ida_mm);
            if (constraintsPassed) {
                return (IDA_SUCCESS);
            } else {
                N_VCompare(ONEPT5, self.ida_constraints, self.ida_tempv1);
                /* a , where a[i] =1. when |c[i]| = 2 ,  c the vector of constraints */
            N_VProd(self.ida_tempv1, self.ida_constraints, self.ida_tempv1); /* a * c */
            N_VDiv(self.ida_tempv1, self.ida_ewt, self.ida_tempv1); /* a * c * wt */
            N_VLinearSum(ONE, self.ida_yy, -PT1, self.ida_tempv1, self.ida_tempv1); /* y - 0.1 * a * c * wt */
            N_VProd(self.ida_tempv1, self.ida_mm, self.ida_tempv1); /*  v = mm*(y-.1*a*c*wt) */
            vnorm = IDAWrmsNorm(IDA_mem, self.ida_tempv1, self.ida_ewt, SUNFALSE); /*  ||v|| */

            // If vector v of constraint corrections is small in norm, correct and accept this step
            if vnorm <= self.ida_eps_newt {
            N_VLinearSum(ONE, self.ida_ee, -ONE, self.ida_tempv1, self.ida_ee); /* ee <- ee - v */
            return (IDA_SUCCESS);
            } else {
            /* Constraints not met -- reduce h by computing rr = h'/h */
            N_VLinearSum(ONE, self.ida_phi[0], -ONE, self.ida_yy, self.ida_tempv1);
            N_VProd(self.ida_mm, self.ida_tempv1, self.ida_tempv1);
            self.ida_rr = PT9 * N_VMinQuotient(self.ida_phi[0], self.ida_tempv1);
            self.ida_rr = SUNMAX(self.ida_rr, PT1);
            return (IDA_CONSTR_RTolCVR);
            }
            }
             */
        }

        Ok(())
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
        tracing::trace!("restore(saved_t={:.6e})", saved_t);
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

    /// IDAHandleNFlag
    ///
    /// This routine handles failures indicated by the input variable nflag. Positive values
    /// indicate various recoverable failures while negative values indicate nonrecoverable
    /// failures. This routine adjusts the step size for recoverable failures.
    ///
    ///  Possible nflag values (input):
    ///
    ///   --convergence failures--
    ///   IDA_RES_RTolCVR              > 0
    ///   IDA_LSOLVE_RTolCVR           > 0
    ///   IDA_CONSTR_RTolCVR           > 0
    ///   SUN_NLS_CONV_RTolCV          > 0
    ///   IDA_RES_FAIL               < 0
    ///   IDA_LSOLVE_FAIL            < 0
    ///   IDA_LSETUP_FAIL            < 0
    ///
    ///   --error test failure--
    ///   ERROR_TEST_FAIL            > 0
    ///
    /// # Returns
    /// * Ok(()), Recoverable, PREDICT_AGAIN
    /// * Error
    ///
    ///   --nonrecoverable--
    ///   IDA_CONSTR_FAIL
    ///   IDA_REP_RES_ERR
    ///   IDA_ERR_FAIL
    ///   IDA_CONV_FAIL
    ///   IDA_RES_FAIL
    ///   IDA_LSETUP_FAIL
    ///   IDA_LSOLVE_FAIL
    fn handle_n_flag(
        &mut self,
        error: Error,
        err_k: T,
        err_km1: T,
        ncf_ptr: &mut usize,
        nef_ptr: &mut usize,
    ) -> Result<(), Error> {
        self.ida_phase = 1;

        match error {
            Error::RecoverableFail(recoverable) => {
                //-----------------------
                // Nonlinear solver failed
                //-----------------------

                // recoverable failure
                *ncf_ptr += 1; // local counter for convergence failures
                self.counters.ida_ncfn += 1; // global counter for convergence failures

                // Reduce step size for a new prediction
                if !matches!(recoverable, RecoverableKind::Constraint) {
                    // If nflag=IDA_CONSTR_RECVR then rr was already set in IDANls
                    self.ida_rr = T::quarter();
                }
                self.ida_hh *= self.ida_rr;

                // Test if there were too many convergence failures
                if *ncf_ptr < self.limits.ida_maxncf {
                    Ok(())
                } else {
                    match recoverable {
                        // return (IDA_REP_RES_ERR);
                        RecoverableKind::Residual => Err(Error::ResidualFail),

                        // return (IDA_CONSTR_FAIL);
                        RecoverableKind::Constraint => Err(Error::ConstraintFail),

                        // return (IDA_CONV_FAIL);
                        _ => Err(Error::ConvergenceFail),
                    }
                }
            }

            Error::TestFail => {
                // -----------------
                // Error Test failed
                //------------------

                *nef_ptr += 1; // local counter for error test failures
                self.counters.ida_netf += 1; // global counter for error test failures

                if *nef_ptr == 1 {
                    // On first error test failure, keep current order or lower order by one. Compute new stepsize
                    // based on differences of the solution.

                    let err_knew = if self.ida_kk == self.ida_knew {
                        err_k
                    } else {
                        err_km1
                    };

                    self.ida_kk = self.ida_knew;
                    // rr = 0.9 * (2 * err_knew + 0.0001)^(-1/(kk+1))
                    self.ida_rr = {
                        let base = T::two() * err_knew + T::pt0001();
                        let arg = T::from(self.ida_kk + 1).unwrap().recip();
                        T::pt9() * base.powf(-arg)
                    };
                    self.ida_rr = T::quarter().max(T::pt9().min(self.ida_rr));
                    self.ida_hh *= self.ida_rr;
                    Ok(())
                } else if *nef_ptr == 2 {
                    // On second error test failure, use current order or decrease by one.  Reduce stepsize by factor of 1/4.
                    self.ida_kk = self.ida_knew;
                    self.ida_rr = T::quarter();
                    self.ida_hh *= self.ida_rr;
                    Ok(())
                } else if *nef_ptr < self.limits.ida_maxnef {
                    // On third and subsequent error test failures, set order to 1. Reduce stepsize by factor of 1/4.
                    self.ida_kk = 1;
                    self.ida_rr = T::quarter();
                    self.ida_hh *= self.ida_rr;
                    Ok(())
                } else {
                    // Too many error test failures
                    Err(Error::ErrFail)
                }
            }

            _ => {
                panic!("Should never happen");
            }
        }
    }

    /// IDAReset
    /// This routine is called only if we need to predict again at the very first step. In such a case,
    /// reset phi[1] and psi[0].
    pub(crate) fn reset(&mut self) -> () {
        self.ida_psi[0] = self.ida_hh;
        self.ida_phi *= self.ida_rr;
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
        let kord = self.ida_kused.max(1);

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
    pub(crate) fn wrms_norm<S>(&self, x: &Vector<T, P::D, S>) -> T
    where
        S: Storage<T, P::D>,
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
