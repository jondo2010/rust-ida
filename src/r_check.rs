use super::*;

pub(super) enum RootStatus {
    RootFound,
    Continue,
}

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

        self.ida_iroots.fill(P::Scalar::zero());
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

    /// IDARcheck2
    ///
    /// This routine checks for exact zeros of g at the last root found, if the last return was a
    /// root.  It then checks for a close pair of zeros (an error condition), and for a new root
    /// at a nearby point. The array glo = g(tlo) at the left endpoint of the search interval is
    /// adjusted if necessary to assure that all g_i are nonzero there, before returning to do a
    /// root search in the interval.
    ///
    /// On entry, tlo = tretlast is the last value of tret returned by IDASolve.  This may be the
    /// previous tn, the previous tout value, or the last root location.
    ///
    /// This routine returns an int equal to:
    ///     IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///     CLOSERT         = 3 if a close pair of zeros was found, or
    ///     RTFOUND         = 1 if a new zero of g was found near tlo, or
    ///     IDA_SUCCESS     = 0 otherwise.
    pub(super) fn r_check2(&mut self) -> Result<RootStatus, failure::Error> {
        if !self.ida_irfnd {
            return Ok(RootStatus::Continue);
        }

        self.get_solution(self.ida_tlo);

        //retval = self.ida_gfun(self.ida_tlo, self.ida_yy, self.ida_yp, self.ida_glo, self.ida_user_data);
        self.nlp.lp.problem.root(
            self.ida_tlo,
            self.nlp.ida_yy.view(),
            self.nlp.ida_yp.view(),
            self.ida_glo.view_mut(),
        );
        self.ida_nge += 1;
        //if (retval != 0) return(IDA_RTFUNC_FAIL);

        self.ida_iroots.fill(P::Scalar::zero());
        let zroot = ndarray::Zip::from(self.ida_iroots.view_mut())
            .and(self.ida_gactive.view())
            .and(self.ida_glo.view())
            .fold_while(false, |mut zroot, iroots, &gactive, glo| {
                if gactive && (glo.abs() == P::Scalar::zero()) {
                    zroot = true;
                    *iroots = P::Scalar::one();
                }

                ndarray::FoldWhile::Continue(zroot)
            });

        if zroot.into_inner() {
            // One or more g_i has a zero at tlo.  Check g at tlo+smallh.
            self.ida_ttol = ((self.nlp.ida_tn).abs() + (self.ida_hh).abs())
                * P::Scalar::epsilon()
                * P::Scalar::hundred();
            let smallh = self.ida_ttol * self.ida_hh.signum();
            let tplus = self.ida_tlo + smallh;
            if (tplus - self.nlp.ida_tn) * self.ida_hh >= P::Scalar::zero() {
                let hratio = smallh / self.ida_hh;

                //N_VLinearSum( ONE, self.ida_yy, hratio, self.ida_phi[1], self.ida_yy);
                self.nlp
                    .ida_yy
                    .scaled_add(hratio, &self.ida_phi.index_axis(Axis(0), 1));
            } else {
                self.get_solution(tplus);
            }

            self.nlp.lp.problem.root(
                tplus,
                self.nlp.ida_yy.view(),
                self.nlp.ida_yp.view(),
                self.ida_ghi.view_mut(),
            );
            self.ida_nge += 1;
            //if (retval != 0) return(IDA_RTFUNC_FAIL);

            // Check for close roots (error return), for a new zero at tlo+smallh, and for a g_i
            // that changed from zero to nonzero.
            let zroot = ndarray::Zip::from(self.ida_iroots.view_mut())
                .and(self.ida_gactive.view())
                .and(self.ida_glo.view_mut())
                .and(self.ida_ghi.view())
                .fold_while(false, |mut zroot, iroots, &gactive, glo, ghi| {
                    if gactive {
                        if ghi.abs() == P::Scalar::zero() {
                            if *iroots > P::Scalar::zero() {
                                return ndarray::FoldWhile::Done(false);
                            }
                            zroot = true;
                            *iroots = P::Scalar::one();
                        } else {
                            if *iroots > P::Scalar::zero() {
                                *glo = *ghi;
                            }
                        }
                    }
                    return ndarray::FoldWhile::Continue(zroot);
                });

            if zroot.is_done() {
                Err(IdaError::CloseRoots {
                    t: self.ida_tlo.to_f64().unwrap(),
                })?;
            }

            if zroot.into_inner() {
                return Ok(RootStatus::RootFound);
            }
        }

        Ok(RootStatus::Continue)
    }

    /// IDARcheck3
    ///
    /// This routine interfaces to IDARootfind to look for a root of g between tlo and either tn or
    /// tout, whichever comes first. Only roots beyond tlo in the direction of integration are
    /// sought.
    ///
    /// This routine returns an int equal to:
    ///    IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///    RTFOUND         = 1 if a root of g was found, or
    ///    IDA_SUCCESS     = 0 otherwise.
    pub(super) fn r_check3(&mut self) -> Result<RootStatus, failure::Error> {
        // Set thi = tn or tout, whichever comes first.
        match self.ida_taskc {
            IdaTask::OneStep => self.ida_thi = self.nlp.ida_tn,
            IdaTask::Normal => {
                self.ida_thi =
                    if (self.ida_toutc - self.nlp.ida_tn) * self.ida_hh >= P::Scalar::zero() {
                        self.nlp.ida_tn
                    } else {
                        self.ida_toutc
                    };
            }
        }

        // Get y and y' at thi.
        self.get_solution(self.ida_thi);

        // Set ghi = g(thi) and call IDARootfind to search (tlo,thi) for roots.
        self.nlp.lp.problem.root(
            self.ida_thi,
            self.nlp.ida_yy.view(),
            self.nlp.ida_yp.view(),
            self.ida_ghi.view_mut(),
        );
        self.ida_nge += 1;
        //if (retval != 0) return(IDA_RTFUNC_FAIL);

        self.ida_ttol = (self.nlp.ida_tn.abs() + self.ida_hh.abs())
            * P::Scalar::epsilon()
            * P::Scalar::hundred();

        let ier = self.root_find()?;

        ndarray::Zip::from(self.ida_gactive.view_mut())
            .and(self.ida_grout.view())
            .apply(|gactive, &grout| {
                if !*gactive && (grout != P::Scalar::zero()) {
                    *gactive = true;
                }
            });

        self.ida_tlo = self.ida_trout;
        self.ida_glo.assign(&self.ida_grout);

        // If a root was found, interpolate to get y(trout) and return.
        if let RootStatus::RootFound = ier {
            self.get_solution(self.ida_trout);
        }

        Ok(ier)
    }

    /// IDARootfind
    ///
    /// This routine solves for a root of g(t) between tlo and thi, if one exists.  Only roots of
    /// odd multiplicity (i.e. with a change of sign in one of the g_i), or exact zeros, are found.
    /// Here the sign of tlo - thi is arbitrary, but if multiple roots are found, the one closest
    /// to tlo is returned.
    ///
    /// The method used is the Illinois algorithm, a modified secant method. Reference: Kathie L.
    /// Hiebert and Lawrence F. Shampine, Implicitly Defined Output Points for Solutions of ODEs,
    /// Sandia National Laboratory Report SAND80-0180, February 1980.
    ///
    /// This routine uses the following parameters for communication:
    ///
    /// nrtfn    = number of functions g_i, or number of components of
    ///            the vector-valued function g(t).  Input only.
    ///
    /// gfun     = user-defined function for g(t).  Its form is
    ///            (void) gfun(t, y, yp, gt, user_data)
    ///
    /// rootdir  = in array specifying the direction of zero-crossings.
    ///            If rootdir[i] > 0, search for roots of g_i only if
    ///            g_i is increasing; if rootdir[i] < 0, search for
    ///            roots of g_i only if g_i is decreasing; otherwise
    ///            always search for roots of g_i.
    ///
    /// gactive  = array specifying whether a component of g should
    ///            or should not be monitored. gactive[i] is initially
    ///            set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
    ///            reset to SUNFALSE if at the first step g[i] is 0.0
    ///            both at the I.C. and at a small perturbation of them.
    ///            gactive[i] is then set back on SUNTRUE only after the
    ///            corresponding g function moves away from 0.0.
    ///
    /// nge      = cumulative counter for gfun calls.
    ///
    /// ttol     = a convergence tolerance for trout.  Input only.
    ///            When a root at trout is found, it is located only to
    ///            within a tolerance of ttol.  Typically, ttol should
    ///            be set to a value on the order of
    ///               100 * UROUND * max (SUNRabs(tlo), SUNRabs(thi))
    ///            where UROUND is the unit roundoff of the machine.
    ///
    /// tlo, thi = endpoints of the interval in which roots are sought.
    ///            On input, these must be distinct, but tlo - thi may
    ///            be of either sign.  The direction of integration is
    ///            assumed to be from tlo to thi.  On return, tlo and thi
    ///            are the endpoints of the final relevant interval.
    ///
    /// glo, ghi = arrays of length nrtfn containing the vectors g(tlo)
    ///            and g(thi) respectively.  Input and output.  On input,
    ///            none of the glo[i] should be zero.
    ///
    /// trout    = root location, if a root was found, or thi if not.
    ///            Output only.  If a root was found other than an exact
    ///            zero of g, trout is the endpoint thi of the final
    ///            interval bracketing the root, with size at most ttol.
    ///
    /// grout    = array of length nrtfn containing g(trout) on return.
    ///
    /// iroots   = int array of length nrtfn with root information. Output only. If a root was
    ///            found, iroots indicates which components g_i have a root at trout.
    ///            For i = 0, ..., nrtfn-1, iroots[i] = 1 if g_i has a root and g_i is increasing,
    ///            iroots[i] = -1 if g_i has a root and g_i is decreasing, and iroots[i] = 0 if g_i
    ///            has no roots or g_i varies in the direction opposite to that indicated by
    ///            rootdir[i].
    ///
    /// This routine returns an int equal to:
    ///      IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///      RTFOUND         = 1 if a root of g was found, or
    ///      IDA_SUCCESS     = 0 otherwise.
    fn root_find(&mut self) -> Result<RootStatus, failure::Error> {
        let imax = 0;

        // First check for change in sign in ghi or for a zero in ghi.
        let (zroot, sgnchg, maxfrac, imax) = ndarray::Zip::indexed(self.ida_gactive.view())
            .and(self.ida_ghi.view())
            .and(self.ida_rootdir.view())
            .and(self.ida_glo.view())
            .fold_while(
                (false, false, P::Scalar::zero(), 0),
                |(mut zroot, mut sgnchg, mut maxfrac, mut imax),
                 i,
                 &gactive,
                 &ghi,
                 &rootdir,
                 &glo| {
                    if gactive {
                        let rootdir_glo_neg = <P::Scalar as NumCast>::from(rootdir).unwrap() * glo
                            <= P::Scalar::zero();

                        if ghi.abs() == P::Scalar::zero() {
                            if rootdir_glo_neg {
                                zroot = true;
                            }
                        } else {
                            if (glo * ghi < P::Scalar::zero()) && rootdir_glo_neg {
                                let gfrac = (ghi / (ghi - glo)).abs();
                                if gfrac > maxfrac {
                                    sgnchg = true;
                                    maxfrac = gfrac;
                                    imax = i;
                                }
                            }
                        }
                    }
                    ndarray::FoldWhile::Continue((zroot, sgnchg, maxfrac, imax))
                },
            )
            .into_inner();

        // If no sign change was found, reset trout and grout.  Then return IDA_SUCCESS if no zero
        // was found, or set iroots and return RTFOUND.
        if !sgnchg {
            self.ida_trout = self.ida_thi;
            self.ida_grout.assign(&self.ida_ghi);
            if !zroot {
                return Ok(RootStatus::Continue);
            }

            ndarray::Zip::from(self.ida_iroots.view_mut())
                .and(self.ida_gactive.view())
                .and(self.ida_rootdir.view())
                .and(self.ida_glo.view())
                .and(self.ida_ghi.view())
                .apply(|iroots, &gactive, &rootdir, &glo, &ghi| {
                    *iroots = P::Scalar::zero();
                    if gactive {
                        let rootdir_glo_neg = <P::Scalar as NumCast>::from(rootdir).unwrap() * glo
                            <= P::Scalar::zero();
                        if (ghi.abs() == P::Scalar::zero()) && rootdir_glo_neg {
                            *iroots = glo.signum();
                        }
                    }
                });

            return Ok(RootStatus::RootFound);
        }

        Ok(RootStatus::Continue)
    }
}
