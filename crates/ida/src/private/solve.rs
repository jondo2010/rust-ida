//! Main solver function

use crate::{private::root_finding::RootStatus, IdaSolveStatus, IdaTask};

use super::*;

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
        tout: T,
        tret: &mut T,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, Error> {
        #[cfg(feature = "profiler")]
        profile_scope!(format!("solve(tout={:?})", tout));

        if let IdaTask::Normal = itask {
            self.roots.ida_toutc = tout;
        }

        self.roots.ida_taskc = itask;

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
            if tdist == T::zero() {
                Err(Error::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }
            let troundoff = T::two() * T::epsilon() * (self.nlp.ida_tn.abs() + tout.abs());
            if tdist < troundoff {
                Err(Error::IllegalInput {
                    msg: format!("tout too close to t0 to start integration."),
                })?
            }

            self.ida_hh = self.ida_hin;
            if (self.ida_hh != T::zero()) && ((tout - self.nlp.ida_tn) * self.ida_hh < T::zero()) {
                Err(Error::IllegalInput {
                    msg: format!("Initial step is not towards tout."),
                })?
            }

            if self.ida_hh == T::zero() {
                self.ida_hh = T::pt001() * tdist;
                let ypnorm = self.wrms_norm(&self.ida_phi.column(1));
                if ypnorm > T::two() / self.ida_hh {
                    self.ida_hh = T::half() / ypnorm;
                }
                if tout < self.nlp.ida_tn {
                    self.ida_hh = -self.ida_hh;
                }
            }

            let rh = (self.ida_hh).abs() * self.limits.ida_hmax_inv;
            if rh > T::one() {
                self.ida_hh /= rh;
            }

            if let Some(tstop) = self.ida_tstop {
                if (tstop - self.nlp.ida_tn) * self.ida_hh <= T::zero() {
                    Err(Error::IllegalInput {
                        msg: format!(
                            "The value tstop = {:?} \
                             is behind current t = {:?} \
                             in the direction of integration.",
                            tstop, self.nlp.ida_tn
                        ),
                    })?
                }
                if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > T::zero() {
                    self.ida_hh = (tstop - self.nlp.ida_tn) * (T::one() - T::four() * T::epsilon());
                }
            }

            self.ida_h0u = self.ida_hh;
            self.ida_kk = 0;
            self.ida_kused = 0; // set in case of an error return before a step

            // Check for exact zeros of the root functions at or near t0.
            if self.roots.num_roots() > 0 {
                self.r_check1()?;
            }

            // set phi[1] = hh*y'
            let mut phi = self.ida_phi.column_mut(1);
            phi *= self.ida_hh;

            // Set the convergence test constants epsNewt and toldel
            self.ida_eps_newt = self.ida_epcon;
            self.nlp.ida_toldel = T::pt0001() * self.ida_eps_newt;
        } // end of first-call block.

        // Call lperf function and set nstloc for later performance testing.
        self.nlp.lp.ls_perf(&self.counters, false);
        let mut nstloc = 0;

        // If not the first call, perform all stopping tests.

        if self.counters.ida_nst > 0 {
            // First, check for a root in the last step taken, other than the last root found, if any.
            // If itask = IDA_ONE_STEP and y(tn) was not returned because of an intervening root, return y(tn) now.

            if self.roots.num_roots() > 0 {
                let irfndp = self.roots.ida_irfnd;

                let ier = self.r_check2()?;

                if let RootStatus::RootFound = ier {
                    self.ida_tretlast = self.roots.ida_tlo;
                    *tret = self.roots.ida_tlo;
                    return Ok(IdaSolveStatus::Root);
                }

                // If tn is distinct from tretlast (within roundoff), check remaining interval for roots
                let troundoff =
                    ((self.nlp.ida_tn).abs() + (self.ida_hh).abs()) * T::epsilon() * T::hundred();

                if (self.nlp.ida_tn - self.ida_tretlast).abs() > troundoff {
                    let ier = self.r_check3()?;
                    match ier {
                        // no root found
                        RootStatus::Continue => {
                            self.roots.ida_irfnd = false;
                            match itask {
                                IdaTask::OneStep if irfndp => {
                                    self.ida_tretlast = self.nlp.ida_tn;
                                    *tret = self.nlp.ida_tn;
                                    let _ier = self.get_solution(self.nlp.ida_tn);
                                    return Ok(IdaSolveStatus::Success);
                                }
                                _ => {}
                            }
                        }
                        // a new root was found
                        RootStatus::RootFound => {
                            self.roots.ida_irfnd = true;
                            self.ida_tretlast = self.roots.ida_tlo;
                            *tret = self.roots.ida_tlo;
                            return Ok(IdaSolveStatus::Root);
                        }
                    }
                }
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
        #[cfg(feature = "profiler")]
        profile_scope!(format!("solve loop"));
        loop {
            // Check for too many steps taken.
            if (self.limits.ida_mxstep > 0) && (nstloc >= self.limits.ida_mxstep) {
                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                // Here yy=yret and yp=ypret already have the current solution.
                Err(Error::TooMuchWork {
                    t: self.nlp.ida_tn.to_f64().unwrap(),
                    mxstep: self.limits.ida_mxstep,
                })?
            }

            // Call lperf to generate warnings of poor performance.
            self.nlp.lp.ls_perf(&self.counters, true);

            // Reset and check ewt (if not first call).
            if self.counters.ida_nst > 0 {
                self.tol_control
                    .ewt_set(&self.ida_phi.column(0), &mut self.nlp.ida_ewt);

                if self.nlp.ida_ewt.iter().any(|&x| x <= T::zero()) {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                    *tret = self.nlp.ida_tn;
                    self.ida_tretlast = self.nlp.ida_tn;

                    //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_EWT_NOW_BAD, self.nlp.ida_tn);
                    Err(Error::IllegalInput {
                        msg: format!(
                            "At t = {:?} some ewt component has become <= 0.0.",
                            self.nlp.ida_tn
                        ),
                    })?
                }
            }

            // Check for too much accuracy requested.
            let nrm = self.wrms_norm(&self.ida_phi.column(0));
            self.ida_tolsf = T::epsilon() * nrm;
            if self.ida_tolsf > T::one() {
                self.ida_tolsf *= T::ten();

                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                if self.counters.ida_nst > 0 {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                }
                //IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDASolve", MSG_TOO_MUCH_ACC, self.nlp.ida_tn);
                Err(Error::TooMuchAccuracy {
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
                        tracing::error!("Error occured with get_solution: {e2:?}");
                    }
                }
                // Forward the error
                err
            })?;

            nstloc += 1;

            // After successful step, check for stop conditions; continue or break.

            // First check for root in the last step taken.

            if self.roots.num_roots() > 0 {
                if (nstloc == 173) {
                    println!("nstloc: {}", nstloc);
                }
                let ier = self.r_check3()?;

                if let RootStatus::RootFound = ier {
                    // A new root was found
                    self.roots.ida_irfnd = true;
                    self.ida_tretlast = self.roots.ida_tlo;
                    *tret = self.roots.ida_tlo;
                    return Ok(IdaSolveStatus::Root);
                }

                // If we are at the end of the first step and we still have some event functions
                // that are inactive, issue a warning as this may indicate a user error in the
                // implementation of the root function.
                if self.counters.ida_nst == 1 {
                    let inactive_roots = self
                        .roots
                        .ida_gactive
                        .iter()
                        .fold(false, |inactive_roots, &gactive| {
                            inactive_roots || gactive > 0
                        });

                    if (self.roots.ida_mxgnull > 0) && inactive_roots {
                        tracing::warn!("At the end of the first step, there are still some root functions identically 0. This warning will not be issued again.");
                    }
                }
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
                Ok(IdaSolveStatus::ContinueSteps) => {
                    // Continue looping
                }
            }
        } // End of step loop

        //return(istate);
    }
}