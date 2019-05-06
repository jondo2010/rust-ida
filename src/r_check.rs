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
    /// IDARcheck1
    ///
    /// This routine completes the initialization of rootfinding memory
    /// information, and checks whether g has a zero both at and very near
    /// the initial point of the IVP.
    ///
    /// This routine returns an int equal to:
    ///  IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///  IDA_SUCCESS     = 0 otherwise.
    pub(super) fn r_check1(&mut self) -> Result<(), IdaError> {
        //int i, retval;
        //realtype smallh, hratio, tplus;
        //booleantype zroot;

        self.ida_iroots.fill(0);
        self.ida_tlo = self.nlp.ida_tn;
        self.ida_ttol = ((self.nlp.ida_tn).abs() + (self.ida_hh).abs())
            * P::Scalar::epsilon()
            * P::Scalar::hundred();

        // Evaluate g at initial t and check for zero values.
        self.nlp.lp.problem.root(
            self.ida_tlo,
            self.ida_phi.index_axis(Axis(0), 0),
            self.ida_phi.index_axis(Axis(0), 1),
            self.ida_glo.view_mut(),
        );

        self.ida_nge = 1;

        let zroot = ndarray::Zip::from(self.ida_gactive.view_mut())
            .and(&self.ida_glo)
            .fold_while(false, |mut zroot, gactive, glo| {
                if glo.abs() == P::Scalar::zero() {
                    *gactive = false;
                    zroot = true;
                }
                ndarray::FoldWhile::Continue(zroot)
            });

        if zroot.into_inner() {
            // Some g_i is zero at t0; look at g at t0+(small increment).
            let hratio = (self.ida_ttol / self.ida_hh.abs()).max(P::Scalar::pt1());
            let smallh = hratio * self.ida_hh;
            let tplus = self.ida_tlo + smallh;

            //N_VLinearSum(ONE, self.ida_phi[0], smallh, self.ida_phi[1], self.ida_yy);
            self.nlp.ida_yy.assign(&self.ida_phi.index_axis(Axis(0), 0));
            self.nlp
                .ida_yy
                .scaled_add(smallh, &self.ida_phi.index_axis(Axis(0), 1));

            //retval = self.ida_gfun(tplus, self.ida_yy, self.ida_phi[1], self.ida_ghi, self.ida_user_data);

            self.nlp.lp.problem.root(
                tplus,
                self.nlp.ida_yy.view(),
                self.ida_phi.index_axis(Axis(0), 1),
                self.ida_ghi.view_mut(),
            );

            self.ida_nge += 1;
            //if (retval != 0) return(IDA_RTFUNC_FAIL);

            /*
            for (i = 0; i < self.ida_nrtfn; i++) {
                if (!self.ida_gactive[i] && SUNRabs(self.ida_ghi[i]) != ZERO) {
                    self.ida_gactive[i] = SUNTRUE;
                    self.ida_glo[i] = self.ida_ghi[i];
                }
            }
            */

            // We check now only the components of g which were exactly 0.0 at t0 to see if we can 'activate' them.
            ndarray::Zip::from(self.ida_gactive.view_mut())
                .and(self.ida_glo.view_mut())
                .and(self.ida_ghi.view())
                .apply(|gactive, glo, ghi| {
                    *gactive = true;
                    *glo = *ghi;
                });
        }

        Ok(())
    }
}
