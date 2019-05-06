use super::*;

impl<P, LS, NLS, TolC> Ida<P, LS, NLS, TolC>
where
    P: IdaProblem + Serialize,
    LS: linear::LSolver<P::Scalar> + Serialize,
    NLS: nonlinear::NLSolver<P> + Serialize,
    TolC: TolControl<P::Scalar> + Serialize,
    <P as ModelSpec>::Scalar: num_traits::Float
        + num_traits::float::FloatConst
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + ndarray::ScalarOperand
        + std::fmt::Debug
        + std::fmt::LowerExp
        + IdaConst<Scalar = P::Scalar>,
{
    //-----------------------------------------------------------------
    // Main solver function
    //-----------------------------------------------------------------

    /// This routine is the main driver of the IDA package.
    ///
    /// This is the central step in the solution process, the call to perform the integration of
    /// the DAE. One of the input arguments (itask) specifies one of two modes as to where ida is
    /// to return a solution. But these modes are modified if the user has set a stop time (with
    /// `set_stop_time()`) or requested rootfinding.
    ///
    /// It integrates over an independent variable interval defined by the user, by calling `step()`
    /// to take internal independent variable steps.
    ///
    /// The first time that `solve()` is called for a successfully initialized problem, it computes
    /// a tentative initial step size.
    ///
    /// `solve()` supports two modes, specified by `itask`:
    /// * In the `Normal` mode, the solver steps until it passes `tout` and then interpolates to obtain
    /// `y(tout)` and `yp(tout)`.
    /// * In the `OneStep` mode, it takes one internal step and returns.
    ///
    /// # Arguments
    ///
    /// * `tout` The next time at which a computed solution is desired.
    /// * `tret` The time reached by the solver (output).
    /// * `itask` A flag indicating the job of the solver for the next user step. The IDA NORMAL task is to have the solver take internal steps until it has reached or just passed the user specified tout parameter. The solver then interpolates in order to return approximate values of y(tout) and ˙y(tout). The IDA ONE STEP option tells the solver to just take one internal step and return the solution at the point reached by that step
    ///
    /// # Returns
    ///
    /// * `IdaSolveStatus::Success` - general success.
    /// * `IdaSolveStatus::TStop` - `solve()` succeeded by reaching the stop point specified through
    /// the optional input function `set_stop_time()`.
    /// * `IdaSolveStatus::Root` -  `solve()` succeeded and found one or more roots. In this case,
    /// tret is the location of the root. If nrtfn > 1, call `get_root_info()` to see which gi were
    /// found to have a root.
    ///
    /// # Errors
    ///
    /// IDA_ILL_INPUT
    /// IDA_TOO_MUCH_WORK
    /// IDA_MEM_NULL
    /// IDA_TOO_MUCH_ACC
    /// IDA_CONV_FAIL
    /// IDA_LSETUP_FAIL
    /// IDA_LSOLVE_FAIL
    /// IDA_CONSTR_FAIL
    /// IDA_ERR_FAIL
    /// IDA_REP_RES_ERR
    /// IDA_RES_FAIL
    pub fn solve(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        profile_scope!(format!("solve(tout={:?})", tout));

        if self.counters.ida_nst == 0 {
            // This is the first call

            // Check inputs to IDA for correctness and consistency */
            if !self.ida_setup_done {
                self.initial_setup();
                self.ida_setup_done = true;
            }

            // On first call, check for tout - tn too small, set initial hh, check for approach to
            // tstop, and scale phi[1] by hh.
            // Also check for zeros of root function g at and near t0.

            let tdist = (tout - self.nlp.ida_tn).abs();
            if tdist == P::Scalar::zero() {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }
            let troundoff =
                P::Scalar::two() * P::Scalar::epsilon() * (self.nlp.ida_tn.abs() + tout.abs());
            if tdist < troundoff {
                Err(IdaError::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }

            self.ida_hh = self.ida_hin;
            if (self.ida_hh != P::Scalar::zero())
                && ((tout - self.nlp.ida_tn) * self.ida_hh < P::Scalar::zero())
            {
                Err(IdaError::IllegalInput {
                    msg: format!("Initial step is not towards tout."),
                })?
            }

            if self.ida_hh == P::Scalar::zero() {
                self.ida_hh = P::Scalar::pt001() * tdist;
                let ypnorm = self.wrms_norm(
                    &self.ida_phi.index_axis(Axis(0), 1),
                    &self.nlp.ida_ewt,
                    self.ida_suppressalg,
                );
                if ypnorm > P::Scalar::two() / self.ida_hh {
                    self.ida_hh = P::Scalar::half() / ypnorm;
                }
                if tout < self.nlp.ida_tn {
                    self.ida_hh = -self.ida_hh;
                }
            }

            let rh = (self.ida_hh).abs() * self.ida_hmax_inv;
            if rh > P::Scalar::one() {
                self.ida_hh /= rh;
            }

            if let Some(tstop) = self.ida_tstop {
                if (tstop - self.nlp.ida_tn) * self.ida_hh <= P::Scalar::zero() {
                    Err(IdaError::IllegalInput {
                        msg: format!(
                            "The value tstop = {:?} \
                             is behind current t = {:?} \
                             in the direction of integration.",
                            tstop, self.nlp.ida_tn
                        ),
                    })?
                }
                if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                    self.ida_hh = (tstop - self.nlp.ida_tn)
                        * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                }
            }

            self.ida_h0u = self.ida_hh;
            self.ida_kk = 0;
            self.ida_kused = 0; // set in case of an error return before a step

            /* Check for exact zeros of the root functions at or near t0. */
            if self.ida_nrtfn > 0 {
                self.r_check1()?;
                //  ier = IDARcheck1(IDA_mem);
                //  if (ier == IDA_RTFUNC_FAIL) {
                //    IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck1", MSG_RTFUNC_FAILED, self.nlp.ida_tn);
                //    return(IDA_RTFUNC_FAIL);
                //  }
            }

            //N_VScale(self.ida_hh, self.ida_phi[1], self.ida_phi[1]);  /* set phi[1] = hh*y' */
            let mut phi = self.ida_phi.index_axis_mut(Axis(0), 1);
            phi *= self.ida_hh;

            // Set the convergence test constants epsNewt and toldel
            self.ida_eps_newt = self.ida_epcon;
            self.nlp.ida_toldel = P::Scalar::pt0001() * self.ida_eps_newt;
        } // end of first-call block.

        // Call lperf function and set nstloc for later performance testing.
        self.nlp.lp.ls_perf(&self.counters, false);
        let mut nstloc = 0;

        // If not the first call, perform all stopping tests.

        if self.counters.ida_nst > 0 {
            // First, check for a root in the last step taken, other than the last root found, if
            // any.  If itask = IDA_ONE_STEP and y(tn) was not returned because of an intervening
            // root, return y(tn) now.

            if self.ida_nrtfn > 0 {
                /*

                irfndp = self.ida_irfnd;

                ier = IDARcheck2(IDA_mem);

                if (ier == CLOSERT) {
                  IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDARcheck2", MSG_CLOSE_ROOTS, self.ida_tlo);
                  return(IDA_ILL_INPUT);
                } else if (ier == IDA_RTFUNC_FAIL) {
                  IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck2", MSG_RTFUNC_FAILED, self.ida_tlo);
                  return(IDA_RTFUNC_FAIL);
                } else if (ier == RTFOUND) {
                  self.ida_tretlast = *tret = self.ida_tlo;
                  return(IDA_ROOT_RETURN);
                }

                /* If tn is distinct from tretlast (within roundoff),
                   check remaining interval for roots */
                troundoff = HUNDRED * self.ida_uround * (SUNRabs(self.nlp.ida_tn) + SUNRabs(self.ida_hh));
                if ( SUNRabs(self.nlp.ida_tn - self.ida_tretlast) > troundoff ) {
                ier = IDARcheck3(IDA_mem);
                if (ier == IDA_SUCCESS) {     /* no root found */
                self.ida_irfnd = 0;
                if ((irfndp == 1) && (itask == IDA_ONE_STEP)) {
                self.ida_tretlast = *tret = self.nlp.ida_tn;
                ier = IDAGetSolution(IDA_mem, self.nlp.ida_tn, yret, ypret);
                return(IDA_SUCCESS);
                }
                } else if (ier == RTFOUND) {  /* a new root was found */
                self.ida_irfnd = 1;
                self.ida_tretlast = *tret = self.ida_tlo;
                return(IDA_ROOT_RETURN);
                } else if (ier == IDA_RTFUNC_FAIL) {  /* g failed */
                IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck3", MSG_RTFUNC_FAILED, self.ida_tlo);
                return(IDA_RTFUNC_FAIL);
                }
                }

                */
            } // end of root stop check

            // Now test for all other stop conditions.

            let istate = self.stop_test1(tout, tret, itask);
            match istate {
                Err(_)
                | Ok(IdaSolveStatus::Root)
                | Ok(IdaSolveStatus::Success)
                | Ok(IdaSolveStatus::TStop) => {
                    return istate;
                }
                _ => {}
            }
        }

        // Looping point for internal steps.
        profile_scope!(format!("solve loop"));
        loop {
            // Check for too many steps taken.

            if (self.ida_mxstep > 0) && (nstloc >= self.ida_mxstep) {
                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                // Here yy=yret and yp=ypret already have the current solution.
                Err(IdaError::IllegalInput {
                    msg: format!(
                        "At t = {:.5e}, mxstep steps ({}) taken before reaching tout.",
                        self.nlp.ida_tn, self.ida_mxstep
                    ),
                })?
                //istate = IDA_TOO_MUCH_WORK;
            }

            // Call lperf to generate warnings of poor performance.
            self.nlp.lp.ls_perf(&self.counters, true);

            // Reset and check ewt (if not first call).
            if self.counters.ida_nst > 0 {
                self.tol_control.ewt_set(
                    self.ida_phi.index_axis(Axis(0), 0),
                    self.nlp.ida_ewt.view_mut(),
                );

                if self.nlp.ida_ewt.iter().any(|&x| x <= P::Scalar::zero()) {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                    *tret = self.nlp.ida_tn;
                    self.ida_tretlast = self.nlp.ida_tn;

                    //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_EWT_NOW_BAD, self.nlp.ida_tn);
                    Err(IdaError::IllegalInput {
                        msg: format!(
                            "At t = {:?} some ewt component has become <= 0.0.",
                            self.nlp.ida_tn
                        ),
                    })?
                }
            }

            // Check for too much accuracy requested.

            let nrm = self.wrms_norm(
                &self.ida_phi.index_axis(Axis(0), 0),
                &self.nlp.ida_ewt,
                self.ida_suppressalg,
            );

            /*
            trace!(
                "At t = {:.5e}, nstloc={}; nrm = {:.5e}",
                self.nlp.ida_tn,
                nstloc,
                nrm
            );
            */

            self.ida_tolsf = P::Scalar::epsilon() * nrm;
            if self.ida_tolsf > P::Scalar::one() {
                self.ida_tolsf *= P::Scalar::ten();

                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                if self.counters.ida_nst > 0 {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                }
                //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_TOO_MUCH_ACC, self.nlp.ida_tn);
                Err(IdaError::TooMuchAccuracy {
                    t: self.nlp.ida_tn.to_f64().unwrap(),
                })?
            }

            // Call IDAStep to take a step.
            let sflag = self.step();

            // Process all failed-step cases, and exit loop.
            sflag.map_err(|err| {
                let ier = self.get_solution(self.nlp.ida_tn);
                match ier {
                    Ok(_) => {
                        *tret = self.nlp.ida_tn;
                        self.ida_tretlast = self.nlp.ida_tn;
                    }
                    Err(e2) => {
                        error!("Error occured with get_solution: {:?}", e2.as_fail());
                    }
                }
                // Forward the error
                err
            })?;

            nstloc += 1;

            // After successful step, check for stop conditions; continue or break.

            // First check for root in the last step taken.

            if self.ida_nrtfn > 0 {
                /*
                ier = IDARcheck3(IDA_mem);

                if (ier == RTFOUND) {  /* A new root was found */
                self.ida_irfnd = 1;
                istate = IDA_ROOT_RETURN;
                self.ida_tretlast = *tret = self.ida_tlo;
                break;
                } else if (ier == IDA_RTFUNC_FAIL) { /* g failed */
                IDAProcessError(IDA_mem, IDA_RTFUNC_FAIL, "IDA", "IDARcheck3", MSG_RTFUNC_FAILED, self.ida_tlo);
                istate = IDA_RTFUNC_FAIL;
                break;
                }

                /* If we are at the end of the first step and we still have
                 * some event functions that are inactive, issue a warning
                 * as this may indicate a user error in the implementation
                 * of the root function. */
                if (self.counters.ida_nst==1) {
                inactive_roots = SUNFALSE;
                for (ir=0; ir<self.ida_nrtfn; ir++) {
                if (!self.ida_gactive[ir]) {
                inactive_roots = SUNTRUE;
                break;
                }
                }
                if ((self.ida_mxgnull > 0) && inactive_roots) {
                IDAProcessError(IDA_mem, IDA_WARNING, "IDA", "IDASolve", MSG_INACTIVE_ROOTS);
                }
                }
                 */
            }

            // Now check all other stop conditions.

            let istate = self.stop_test2(tout, tret, itask);
            match istate {
                Err(_)
                | Ok(IdaSolveStatus::Root)
                | Ok(IdaSolveStatus::Success)
                | Ok(IdaSolveStatus::TStop) => {
                    return istate;
                }
                _ => {}
            }
        } // End of step loop

        //return(istate);
    }
}
