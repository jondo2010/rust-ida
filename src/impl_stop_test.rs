use super::*;

///-----------------------------------------------------------------
/// Stopping tests
///-----------------------------------------------------------------

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
    /// IDAStopTest1
    ///
    /// This routine tests for stop conditions before taking a step.
    /// The tests depend on the value of itask.
    /// The variable tretlast is the previously returned value of tret.
    ///
    /// The return values are:
    /// CONTINUE_STEPS       if no stop conditions were found
    /// IDA_SUCCESS          for a normal return to the user
    /// IDA_TSTOP_RETURN     for a tstop-reached return to the user
    /// IDA_ILL_INPUT        for an illegal-input return to the user
    ///
    /// In the tstop cases, this routine may adjust the stepsize hh to cause
    /// the next step to reach tstop exactly.
    pub(super) fn stop_test1(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        if let Some(tstop) = self.ida_tstop {
            // Test for tn past tstop, tn = tretlast, tn past tout, tn near tstop.
            if ((self.nlp.ida_tn - tstop) * self.ida_hh) > P::Scalar::zero() {
                Err(IdaError::BadStopTime {
                    tstop: tstop.to_f64().unwrap(),
                    t: self.nlp.ida_tn.to_f64().unwrap(),
                })?
            }
        }
        match itask {
            IdaTask::Normal => {
                // Test for tout = tretlast, and for tn past tout.
                if tout == self.ida_tretlast {
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if (self.nlp.ida_tn - tout) * self.ida_hh >= P::Scalar::zero() {
                    self.get_solution(tout)?;
                    self.ida_tretlast = tout;
                    *tret = tout;
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        self.get_solution(tstop)
                            .map_err(|_| IdaError::BadStopTime {
                                tstop: tstop.to_f64().unwrap(),
                                t: self.nlp.ida_tn.to_f64().unwrap(),
                            })?;

                        self.ida_tretlast = tstop;
                        *tret = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }

            IdaTask::OneStep => {
                // Test for tn past tretlast.
                if (self.nlp.ida_tn - self.ida_tretlast) * self.ida_hh > P::Scalar::zero() {
                    let _ier = self.get_solution(self.nlp.ida_tn);
                    self.ida_tretlast = self.nlp.ida_tn;
                    *tret = self.nlp.ida_tn;
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        self.get_solution(tstop)?;
                        self.ida_tretlast = tstop;
                        *tret = tstop;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }
        }
    }

    /// IDAStopTest2
    ///
    /// This routine tests for stop conditions after taking a step.
    /// The tests depend on the value of itask.
    ///
    /// The return values are:
    ///  CONTINUE_STEPS     if no stop conditions were found
    ///  IDA_S>UCCESS        for a normal return to the user
    ///  IDA_TSTOP_RETURN   for a tstop-reached return to the user
    ///  IDA_ILL_INPUT      for an illegal-input return to the user
    ///
    /// In the two cases with tstop, this routine may reset the stepsize hh
    /// to cause the next step to reach tstop exactly.
    ///
    /// In the two cases with ONE_STEP mode, no interpolation to tn is needed
    /// because yret and ypret already contain the current y and y' values.
    ///
    /// Note: No test is made for an error return from IDAGetSolution here,
    /// because the same test was made prior to the step.
    pub(super) fn stop_test2(
        &mut self,
        tout: P::Scalar,
        tret: &mut P::Scalar,
        itask: IdaTask,
    ) -> Result<IdaSolveStatus, failure::Error> {
        match itask {
            IdaTask::Normal => {
                // Test for tn past tout.
                if (self.nlp.ida_tn - tout) * self.ida_hh >= P::Scalar::zero() {
                    // /* ier = */ IDAGetSolution(IDA_mem, tout, yret, ypret);
                    *tret = tout;
                    self.ida_tretlast = tout;
                    let _ier = self.get_solution(tout);
                    return Ok(IdaSolveStatus::Success);
                }

                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        let _ier = self.get_solution(tstop);
                        *tret = tstop;
                        self.ida_tretlast = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }

                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                Ok(IdaSolveStatus::ContinueSteps)
            }

            IdaTask::OneStep => {
                if let Some(tstop) = self.ida_tstop {
                    // Test for tn at tstop and for tn near tstop
                    let troundoff = P::Scalar::hundred()
                        * P::Scalar::epsilon()
                        * (self.nlp.ida_tn.abs() + self.ida_hh.abs());

                    if (self.nlp.ida_tn - tstop).abs() <= troundoff {
                        let _ier = self.get_solution(tstop);
                        *tret = tstop;
                        self.ida_tretlast = tstop;
                        self.ida_tstop = None;
                        return Ok(IdaSolveStatus::TStop);
                    }
                    if (self.nlp.ida_tn + self.ida_hh - tstop) * self.ida_hh > P::Scalar::zero() {
                        self.ida_hh = (tstop - self.nlp.ida_tn)
                            * (P::Scalar::one() - P::Scalar::four() * P::Scalar::epsilon());
                    }
                }

                *tret = self.nlp.ida_tn;
                self.ida_tretlast = self.nlp.ida_tn;
                Ok(IdaSolveStatus::Success)
            }
        }
    }
}
