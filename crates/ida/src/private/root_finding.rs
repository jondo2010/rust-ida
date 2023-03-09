//! Functions for root finding.

use std::sync::Arc;

use itertools::{izip, FoldWhile, Itertools};

use crate::IdaTask;

use super::*;

pub(super) enum RootStatus {
    RootFound,
    Continue,
}

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
    /// IDARcheck1
    ///
    /// This routine completes the initialization of rootfinding memory information, and checks whether g has a zero
    /// both at and very near the initial point of the IVP.
    ///
    /// This routine returns an int equal to:
    ///  IDA_RTFUNC_FAIL < 0 if the g function failed, or
    ///  IDA_SUCCESS     = 0 otherwise.
    pub(super) fn r_check1(&mut self) -> Result<(), Error> {
        self.roots.ida_iroots.fill(0);
        self.roots.ida_tlo = self.nlp.ida_tn;
        self.roots.ida_ttol =
            ((self.nlp.ida_tn).abs() + (self.ida_hh).abs()) * T::epsilon() * T::hundred();

        // Evaluate g at initial t and check for zero values.
        self.nlp.lp.problem.root(
            self.roots.ida_tlo,
            &self.ida_phi.column(0),
            &self.ida_phi.column(1),
            &mut self.roots.ida_glo,
        );

        self.roots.ida_nge = 1;

        let zroot = self
            .roots
            .ida_gactive
            .iter_mut()
            .zip(self.roots.ida_glo.iter())
            .fold_while(false, |mut zroot, (gactive, glo)| {
                if glo.abs() == T::zero() {
                    *gactive = 0;
                    zroot = true;
                }
                FoldWhile::Continue(zroot)
            });

        if zroot.into_inner() {
            // Some g_i is zero at t0; look at g at t0+(small increment).
            let hratio = (self.roots.ida_ttol / self.ida_hh.abs()).max(T::pt1());
            let smallh = hratio * self.ida_hh;
            let tplus = self.roots.ida_tlo + smallh;

            //N_VLinearSum(ONE, self.ida_phi[0], smallh, self.ida_phi[1], self.ida_yy);
            self.nlp.ida_yy.copy_from(&self.ida_phi.column(0));
            self.nlp
                .ida_yy
                .axpy(smallh, &self.ida_phi.column(1), T::one());

            self.nlp.lp.problem.root(
                tplus,
                &self.nlp.ida_yy,
                &self.ida_phi.column(1),
                &mut self.roots.ida_ghi,
            );

            self.roots.ida_nge += 1;

            //TODO: add error path to root()
            //if (retval != 0) return(IDA_RTFUNC_FAIL);

            // We check now only the components of g which were exactly 0.0 at t0 to see if we can 'activate' them.
            for (gactive, glo, &ghi) in izip!(
                self.roots.ida_gactive.iter_mut(),
                self.roots.ida_glo.iter_mut(),
                self.roots.ida_ghi.iter()
            ) {
                if *gactive == 0 && ghi.abs() != T::zero() {
                    *gactive = 1;
                    *glo = ghi;
                }
            }
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
    pub(super) fn r_check2(&mut self) -> Result<RootStatus, Error> {
        if !self.roots.ida_irfnd {
            return Ok(RootStatus::Continue);
        }

        self.get_solution(self.roots.ida_tlo);

        //retval = self.ida_gfun(self.ida_tlo, self.ida_yy, self.ida_yp, self.ida_glo, self.ida_user_data);
        self.nlp.lp.problem.root(
            self.roots.ida_tlo,
            &self.nlp.ida_yy,
            &self.nlp.ida_yp,
            &mut self.roots.ida_glo,
        );
        self.roots.ida_nge += 1;
        //if (retval != 0) return(IDA_RTFUNC_FAIL);

        self.roots.ida_iroots.fill(0);

        let zroot = izip!(
            self.roots.ida_iroots.iter_mut(),
            self.roots.ida_gactive.iter(),
            self.roots.ida_glo.iter()
        )
        .fold_while(false, |mut zroot, (iroots, &gactive, glo)| {
            if gactive > 0 && glo.abs() == T::zero() {
                zroot = true;
                *iroots = 1;
            }
            FoldWhile::Continue(zroot)
        });

        if zroot.into_inner() {
            // One or more g_i has a zero at tlo.  Check g at tlo+smallh.
            self.roots.ida_ttol =
                ((self.nlp.ida_tn).abs() + (self.ida_hh).abs()) * T::epsilon() * T::hundred();
            let smallh = self.roots.ida_ttol * self.ida_hh.signum();
            let tplus = self.roots.ida_tlo + smallh;
            if (tplus - self.nlp.ida_tn) * self.ida_hh >= T::zero() {
                let hratio = smallh / self.ida_hh;

                //N_VLinearSum( ONE, self.ida_yy, hratio, self.ida_phi[1], self.ida_yy);
                self.nlp
                    .ida_yy
                    .axpy(hratio, &self.ida_phi.column(1), T::one());
            } else {
                self.get_solution(tplus)?;
            }

            self.nlp.lp.problem.root(
                tplus,
                &self.nlp.ida_yy,
                &self.nlp.ida_yp,
                &mut self.roots.ida_ghi,
            );
            self.roots.ida_nge += 1;
            //if (retval != 0) return(IDA_RTFUNC_FAIL);

            // Check for close roots (error return), for a new zero at tlo+smallh, and for a g_i that changed from zero to nonzero.

            let zroot = izip!(
                self.roots.ida_iroots.iter_mut(),
                self.roots.ida_gactive.iter(),
                self.roots.ida_glo.iter_mut(),
                self.roots.ida_ghi.iter()
            )
            .fold_while(false, |mut zroot, (iroots, &gactive, glo, ghi)| {
                if gactive > 0 {
                    if ghi.abs() == T::zero() {
                        if *iroots > 0 {
                            return FoldWhile::Done(false);
                        }
                        zroot = true;
                        *iroots = 1;
                    } else {
                        if *iroots > 0 {
                            *glo = *ghi;
                        }
                    }
                }
                return FoldWhile::Continue(zroot);
            });

            if zroot.is_done() {
                Err(Error::CloseRoots {
                    t: self.roots.ida_tlo.to_f64().unwrap(),
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
    pub(super) fn r_check3(&mut self) -> Result<RootStatus, Error> {
        // Set thi = tn or tout, whichever comes first.
        match self.roots.ida_taskc {
            IdaTask::OneStep => self.roots.ida_thi = self.nlp.ida_tn,
            IdaTask::Normal => {
                self.roots.ida_thi =
                    if (self.roots.ida_toutc - self.nlp.ida_tn) * self.ida_hh >= T::zero() {
                        self.nlp.ida_tn
                    } else {
                        self.roots.ida_toutc
                    };
            }
        }

        // Get y and y' at thi.
        self.get_solution(self.roots.ida_thi);

        // Set ghi = g(thi) and call IDARootfind to search (tlo,thi) for roots.
        self.nlp.lp.problem.root(
            self.roots.ida_thi,
            &self.nlp.ida_yy,
            &self.nlp.ida_yp,
            &mut self.roots.ida_ghi,
        );
        self.roots.ida_nge += 1;
        //if (retval != 0) return(IDA_RTFUNC_FAIL);

        self.roots.ida_ttol =
            (self.nlp.ida_tn.abs() + self.ida_hh.abs()) * T::epsilon() * T::hundred();

        let ier = self.root_find()?;

        for (gactive, &grout) in izip!(
            self.roots.ida_gactive.iter_mut(),
            self.roots.ida_grout.iter()
        ) {
            if *gactive == 0 && (grout != T::zero()) {
                *gactive = 1;
            }
        }

        self.roots.ida_tlo = self.roots.ida_trout;
        self.roots.ida_glo.copy_from(&self.roots.ida_grout);

        // If a root was found, interpolate to get y(trout) and return.
        if let RootStatus::RootFound = ier {
            self.get_solution(self.roots.ida_trout);
        }

        Ok(ier)
    }

    /// IDARootfind
    ///
    /// This routine solves for a root of g(t) between tlo and thi, if one exists.  Only roots of odd multiplicity (i.e.
    /// with a change of sign in one of the g_i), or exact zeros, are found.  Here the sign of tlo - thi is arbitrary,
    /// but if multiple roots are found, the one closest to tlo is returned.
    ///
    /// The method used is the Illinois algorithm, a modified secant method. Reference: Kathie L. Hiebert and
    /// Lawrence F. Shampine, Implicitly Defined Output Points for Solutions of ODEs, Sandia National Laboratory Report
    /// SAND80-0180, February 1980.
    ///
    /// This routine uses the following parameters for communication:
    ///
    /// nrtfn    = number of functions g_i, or number of components of the vector-valued function g(t).  Input only.
    ///
    /// gfun     = user-defined function for g(t).  Its form is (void) gfun(t, y, yp, gt, user_data)
    ///
    /// rootdir  = in array specifying the direction of zero-crossings.  If rootdir[i] > 0, search for roots of g_i only
    ///            if g_i is increasing; if rootdir[i] < 0, search for roots of g_i only if g_i is decreasing; otherwise
    ///            always search for roots of g_i.
    ///
    /// gactive  = array specifying whether a component of g should
    ///            or should not be monitored. gactive[i] is initially
    ///            set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
    ///            reset to false if at the first step g[i] is 0.0
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
    fn root_find(&mut self) -> Result<RootStatus, Error> {
        let mut imax_loop = 0;

        // First check for change in sign in ghi or for a zero in ghi.
        let (zroot, sgnchg, _maxfrac, imax) = izip!(
            self.roots.ida_gactive.iter(),
            self.roots.ida_ghi.iter(),
            self.roots.ida_rootdir.iter(),
            self.roots.ida_glo.iter()
        )
        .enumerate()
        .fold(
            (false, false, T::zero(), 0),
            |(mut zroot, mut sgnchg, mut maxfrac, mut imax),
             (i, (&gactive, &ghi, &rootdir, &glo))| {
                if gactive > 0 {
                    let rootdir_glo_neg =
                        T::from(rootdir.signum()).unwrap() * glo.signum() < T::zero();

                    if ghi.abs() == T::zero() {
                        if rootdir_glo_neg {
                            zroot = true;
                        }
                    } else {
                        if (glo * ghi < T::zero()) && rootdir_glo_neg {
                            let gfrac = (ghi / (ghi - glo)).abs();
                            if gfrac > maxfrac {
                                sgnchg = true;
                                maxfrac = gfrac;
                                imax = i;
                            }
                        }
                    }
                }
                (zroot, sgnchg, maxfrac, imax)
            },
        );

        imax_loop = imax;

        // If no sign change was found, reset trout and grout. Then return IDA_SUCCESS if no zero was found, or set
        // iroots and return RTFOUND.
        if !sgnchg {
            self.roots.ida_trout = self.roots.ida_thi;
            self.roots.ida_grout.copy_from(&self.roots.ida_ghi);
            if !zroot {
                return Ok(RootStatus::Continue);
            }

            for (iroots, &gactive, &rootdir, &glo, &ghi) in izip!(
                self.roots.ida_iroots.iter_mut(),
                self.roots.ida_gactive.iter(),
                self.roots.ida_rootdir.iter(),
                self.roots.ida_glo.iter(),
                self.roots.ida_ghi.iter(),
            ) {
                *iroots = 0;
                if gactive > 0 {
                    let rootdir_glo_neg = T::from(rootdir).unwrap() * glo <= T::zero();
                    if ghi.abs() == T::zero() && rootdir_glo_neg {
                        *iroots = if glo > T::zero() { -1 } else { 1 };
                    }
                }
            }

            return Ok(RootStatus::RootFound);
        }

        // Initialize alph to avoid compiler warning
        let mut alph = T::one();

        // A sign change was found.  Loop to locate nearest root.

        let mut side = 0;
        let mut sideprev = -1;

        // Looping point
        loop {
            // If interval size is already less than tolerance ttol, break.
            if (self.roots.ida_thi - self.roots.ida_tlo).abs() <= self.roots.ida_ttol {
                break;
            }

            // Set weight alph.
            // On the first two passes, set alph = 1.  Thereafter, reset alph according to the side
            // (low vs high) of the subinterval in which the sign change was found in the previous
            // two passes.

            // If the sides were opposite, set alph = 1.
            // If the sides were the same, then double alph (if high side), or halve alph (if low
            // side). The next guess tmid is the secant method value if alph = 1, but is closer to
            // tlo if alph < 1, and closer to thi if alph > 1.

            alph = if sideprev == side {
                if side == 2 {
                    alph * T::two()
                } else {
                    alph * T::half()
                }
            } else {
                T::one()
            };

            // Set next root approximation tmid and get g(tmid). If tmid is too close to tlo or thi,
            // adjust it inward, by a fractional distance that is between 0.1 and 0.5.
            let mut tmid = self.roots.ida_thi
                - (self.roots.ida_thi - self.roots.ida_tlo) * self.roots.ida_ghi[imax_loop]
                    / (self.roots.ida_ghi[imax_loop] - alph * self.roots.ida_glo[imax_loop]);

            if (tmid - self.roots.ida_tlo).abs() < T::half() * self.roots.ida_ttol {
                let fracint = (self.roots.ida_thi - self.roots.ida_tlo).abs() / self.roots.ida_ttol;
                let fracsub = if fracint > T::five() {
                    T::pt1()
                } else {
                    T::half() / fracint
                };
                tmid = self.roots.ida_tlo + fracsub * (self.roots.ida_thi - self.roots.ida_tlo);
            }
            if (self.roots.ida_thi - tmid).abs() < T::half() * self.roots.ida_ttol {
                let fracint = (self.roots.ida_thi - self.roots.ida_tlo).abs() / self.roots.ida_ttol;
                let fracsub = if fracint > T::five() {
                    T::pt1()
                } else {
                    T::half() / fracint
                };
                tmid = self.roots.ida_thi - fracsub * (self.roots.ida_thi - self.roots.ida_tlo);
            }

            self.get_solution(tmid);
            self.nlp.lp.problem.root(
                tmid,
                &self.nlp.ida_yy,
                &self.nlp.ida_yp,
                &mut self.roots.ida_grout,
            );
            self.roots.ida_nge += 1;
            //if (retval != 0) return(IDA_RTFUNC_FAIL);

            // Check to see in which subinterval g changes sign, and reset imax.
            // Set side = 1 if sign change is on low side, or 2 if on high side.
            sideprev = side;

            let (zroot, sgnchg, _maxfrac, imax) = izip!(
                self.roots.ida_gactive.iter(),
                self.roots.ida_grout.iter(),
                self.roots.ida_rootdir.iter(),
                self.roots.ida_glo.iter()
            )
            .enumerate()
            .fold(
                (false, false, T::zero(), imax_loop),
                |(mut zroot, mut sgnchg, mut maxfrac, mut imax),
                 (i, (&gactive, &grout, &rootdir, &glo))| {
                    if gactive > 0 {
                        let rootdir_glo_neg = T::from(rootdir).unwrap() * glo <= T::zero();

                        if grout.abs() == T::zero() && rootdir_glo_neg {
                            zroot = true;
                        } else {
                            if (glo * grout) < T::zero() && rootdir_glo_neg {
                                let gfrac = (grout / (grout - glo)).abs();
                                if gfrac > maxfrac {
                                    sgnchg = true;
                                    maxfrac = gfrac;
                                    imax = i;
                                }
                            }
                        }
                    }
                    (zroot, sgnchg, maxfrac, imax)
                },
            );

            imax_loop = imax;

            if sgnchg {
                // Sign change found in (tlo,tmid); replace thi with tmid.
                self.roots.ida_thi = tmid;
                self.roots.ida_ghi.copy_from(&self.roots.ida_grout);
                side = 1;
                // Stop at root thi if converged; otherwise loop.
                if (self.roots.ida_thi - self.roots.ida_tlo).abs() <= self.roots.ida_ttol {
                    break;
                }
                // Return to looping point.
                continue;
            }

            if zroot {
                // No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid.
                self.roots.ida_thi = tmid;
                self.roots.ida_ghi.copy_from(&self.roots.ida_grout);
                break;
            }

            // No sign change in (tlo,tmid), and no zero at tmid.
            // Sign change must be in (tmid,thi).  Replace tlo with tmid.
            self.roots.ida_tlo = tmid;
            self.roots.ida_glo.copy_from(&self.roots.ida_grout);
            side = 2;
            // Stop at root thi if converged; otherwise loop back.
            if (self.roots.ida_thi - self.roots.ida_tlo).abs() <= self.roots.ida_ttol {
                break;
            }
        } // End of root-search loop

        // Reset trout and grout, set iroots, and return RTFOUND.
        self.roots.ida_trout = self.roots.ida_thi;
        self.roots.ida_grout.copy_from(&self.roots.ida_ghi);
        for (iroots, &ghi, &glo, &gactive, &rootdir) in izip!(
            self.roots.ida_iroots.iter_mut(),
            self.roots.ida_ghi.iter(),
            self.roots.ida_glo.iter(),
            self.roots.ida_gactive.iter(),
            self.roots.ida_rootdir.iter()
        ) {
            *iroots = 0;

            if gactive > 0 {
                let rootdir_glo_neg = T::from(rootdir).unwrap() * glo <= T::zero();
                if rootdir_glo_neg && (ghi.abs() == T::zero() || (glo * ghi < T::zero())) {
                    *iroots = if glo > T::zero() { -1 } else { 1 };
                }
            }
        }

        Ok(RootStatus::RootFound)
    }
}
