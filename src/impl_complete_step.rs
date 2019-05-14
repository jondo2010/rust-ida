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
    /// IDACompleteStep
    /// This routine completes a successful step.  It increments nst, saves the stepsize and order
    /// used, makes the final selection of stepsize and order for the next step, and updates the phi
    /// array.
    pub(super) fn complete_step(&mut self, err_k: P::Scalar, err_km1: P::Scalar) -> () {
        profile_scope!(format!("complete_step()"));
        trace!(
            "complete_step(err_k={:.5e}, err_km1={:.5e}), nst={}, phase={}",
            err_k,
            err_km1,
            self.counters.ida_nst,
            self.ida_phase
        );

        self.counters.ida_nst += 1;
        let kdiff = (self.ida_kk as isize) - (self.ida_kused as isize);
        self.ida_kused = self.ida_kk;
        self.ida_hused = self.ida_hh;

        if (self.ida_knew == self.ida_kk - 1) || (self.ida_kk == self.ida_maxord) {
            self.ida_phase = 1;
        }

        // For the first few steps, until either a step fails, or the order is reduced, or the
        // order reaches its maximum, we raise the order and double the stepsize. During these
        // steps, phase = 0. Thereafter, phase = 1, and stepsize and order are set by the usual
        // local error algorithm.
        //
        // Note that, after the first step, the order is not increased, as not all of the
        // neccessary information is available yet.

        if self.ida_phase == 0 {
            if self.counters.ida_nst > 1 {
                self.ida_kk += 1;
                let mut hnew = P::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > P::Scalar::one() {
                    hnew /= tmp;
                }
                self.ida_hh = hnew;
            }
        } else {
            #[derive(Debug)]
            enum Action {
                Lower,
                Maintain,
                Raise,
            }

            // Set action = LOWER/MAINTAIN/RAISE to specify order decision

            let (action, err_kp1) = if self.ida_knew == (self.ida_kk - 1) {
                (Action::Lower, P::Scalar::zero())
            } else if self.ida_kk == self.ida_maxord {
                (Action::Maintain, P::Scalar::zero())
            } else if (self.ida_kk + 1) >= self.ida_ns || (kdiff == 1) {
                (Action::Maintain, P::Scalar::zero())
            } else {
                // Estimate the error at order k+1, unless already decided to reduce order, or already using
                // maximum order, or stepsize has not been constant, or order was just raised.

                // tempv1 = ee - phi[kk+1]
                let enorm = {
                    let temp = &self.ida_ee - &self.ida_phi.index_axis(Axis(0), self.ida_kk + 1);
                    self.wrms_norm(&temp, &self.nlp.ida_ewt, self.ida_suppressalg)
                };
                let err_kp1 = enorm / <P::Scalar as NumCast>::from(self.ida_kk + 2).unwrap();

                // Choose among orders k-1, k, k+1 using local truncation error norms.

                let terr_k = <P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap() * err_k;
                let terr_kp1 = <P::Scalar as NumCast>::from(self.ida_kk + 2).unwrap() * err_kp1;

                if self.ida_kk == 1 {
                    if terr_kp1 >= (P::Scalar::half() * terr_k) {
                        (Action::Maintain, err_kp1)
                    } else {
                        (Action::Raise, err_kp1)
                    }
                } else {
                    let terr_km1 = <P::Scalar as NumCast>::from(self.ida_kk).unwrap() * err_km1;
                    if terr_km1 <= terr_k.min(terr_kp1) {
                        (Action::Lower, err_kp1)
                    } else if terr_kp1 >= terr_k {
                        (Action::Maintain, err_kp1)
                    } else {
                        (Action::Raise, err_kp1)
                    }
                }
            };

            // Set the estimated error norm and, on change of order, reset kk.
            let err_knew = match action {
                Action::Raise => {
                    self.ida_kk += 1;
                    err_kp1
                }
                Action::Lower => {
                    self.ida_kk -= 1;
                    err_km1
                }
                _ => err_k,
            };
            trace!(
                "    {:#?}, kk={}, err_knew={:.5e}",
                action,
                self.ida_kk,
                err_knew
            );

            // Compute rr = tentative ratio hnew/hh from error norm estimate.
            // Reduce hh if rr <= 1, double hh if rr >= 2, else leave hh as is.
            // If hh is reduced, hnew/hh is restricted to be between .5 and .9.

            let mut hnew = self.ida_hh;
            //ida_rr = SUNRpowerR( TWO * err_knew + PT0001, -ONE/(self.ida_kk + 1) );
            self.ida_rr = {
                let base = P::Scalar::two() * err_knew + P::Scalar::pt0001();
                let arg = -(<P::Scalar as NumCast>::from(self.ida_kk + 1).unwrap()).recip();
                base.powf(arg)
            };

            if self.ida_rr >= P::Scalar::two() {
                hnew = P::Scalar::two() * self.ida_hh;
                let tmp = hnew.abs() * self.ida_hmax_inv;
                if tmp > P::Scalar::one() {
                    hnew /= tmp;
                }
            } else if self.ida_rr <= P::Scalar::one() {
                //ida_rr = SUNMAX(HALF, SUNMIN(PT9,self.ida_rr));
                self.ida_rr = P::Scalar::half().max(self.ida_rr.min(P::Scalar::pt9()));
                hnew = self.ida_hh * self.ida_rr;
            }

            self.ida_hh = hnew;
        }
        trace!("    next hh={:.5e}", self.ida_hh);
        // end of phase if block

        // Save ee for possible order increase on next step
        if self.ida_kused < self.ida_maxord {
            self.ida_phi
                .index_axis_mut(Axis(0), self.ida_kused + 1)
                .assign(&self.ida_ee);
        }

        // Update phi arrays

        // To update phi arrays compute X += Z where
        // X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ]
        // Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ]

        // Note: this is a recurrence relation, and needs to be performed as below

        let mut tmp = self.ida_zvecs.index_axis_mut(Axis(0), 0);
        tmp.assign(&self.ida_ee);

        for mut phi in self
            .ida_phi
            .slice_axis_mut(Axis(0), Slice::from(0..=self.ida_kused).step_by(-1))
            .genrows_mut()
        {
            tmp += &phi;
            phi.assign(&tmp);
        }
    }
}
