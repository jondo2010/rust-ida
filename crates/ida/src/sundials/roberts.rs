#![allow(non_snake_case, unused)]

//! Roberts example from the SUNDIALS documentation implemented with `sundials-sys`.

use std::ffi::{c_int, c_void};

use crate::sundials::adapter;

use super::{
    adapter::{dense_matrix_view, vector_view},
    IDAGetSolution,
};
use nalgebra::*;

unsafe extern "C" fn resrob(
    _t: f64,
    yy: sundials_sys::N_Vector,
    yp: sundials_sys::N_Vector,
    resval: sundials_sys::N_Vector,
    _user_data: *mut c_void,
) -> c_int {
    let yval = std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer(yy), 3);
    let ypval = std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer(yp), 3);
    let rval = std::slice::from_raw_parts_mut(sundials_sys::N_VGetArrayPointer(resval), 3);
    rval[0] = -0.04 * yval[0] + 1.0e4 * yval[1] * yval[2];
    rval[1] = -rval[0] - 3.0e7 * yval[1] * yval[1] - ypval[1];
    rval[0] -= ypval[0];
    rval[2] = yval[0] + yval[1] + yval[2] - 1.0;
    0
}

unsafe extern "C" fn jacrob(
    _tt: f64,
    cj: f64,
    yy: sundials_sys::N_Vector,
    _yp: sundials_sys::N_Vector,
    _respect: sundials_sys::N_Vector,
    JJ: sundials_sys::SUNMatrix,
    _user_data: *mut c_void,
    _tempv1: sundials_sys::N_Vector,
    _tempv2: sundials_sys::N_Vector,
    _tempv3: sundials_sys::N_Vector,
) -> c_int {
    let yval = vector_view::<U3>(yy);
    let mut jj = dense_matrix_view::<U3, U3>(JJ);
    jj.copy_from(&matrix![
        -0.04 - cj, 0.04, 1.0;
        1.0e4 * yval[2], -1.0e4 * yval[2] - 6.0e7 * yval[1] - cj, 1.0;
        1.0e4 * yval[1], -1.0e4 * yval[1], 1.0;
    ]);
    0
}

unsafe extern "C" fn grob(
    _t: f64,
    yy: sundials_sys::N_Vector,
    _yp: sundials_sys::N_Vector,
    gout: *mut f64,
    _user_data: *mut c_void,
) -> c_int {
    let yval = std::slice::from_raw_parts(sundials_sys::N_VGetArrayPointer(yy), 3);
    let gout = std::slice::from_raw_parts_mut(gout, 2);
    gout[0] = yval[0] - 0.0001;
    gout[1] = yval[2] - 0.01;
    0
}

/// Build an IDA solver for the Roberts problem.
///
/// Returns the initial state vector, the initial derivative vector, and the IDA solver.
#[allow(unused)]
pub unsafe fn build_ida() -> (
    sundials_sys::N_Vector,
    sundials_sys::N_Vector,
    sundials_sys::IDAMem,
) {
    let yy = sundials_sys::N_VNew_Serial(3);
    let mut yval = vector_view::<U3>(yy);
    yval.copy_from(&(vector![1.0, 0.0, 0.0]));

    let yp = sundials_sys::N_VNew_Serial(3);
    let mut ypval = vector_view::<U3>(yp);
    ypval.copy_from(&(vector![-0.04, 0.04, 0.0]));

    let rtol = 1.0e-4;

    let avtol = sundials_sys::N_VNew_Serial(3);
    let mut atval = vector_view::<U3>(avtol);
    atval.copy_from(&vector![1.0e-8, 1.0e-6, 1.0e-6]);

    let t0 = 0.0;

    let mem = sundials_sys::IDACreate();
    assert_eq!(sundials_sys::IDAInit(mem, Some(resrob), t0, yy, yp), 0);

    // Set tolerances
    assert_eq!(sundials_sys::IDASVtolerances(mem, rtol, avtol), 0);

    // Call IDARootInit to specify the root function g with 2 components
    assert_eq!(sundials_sys::IDARootInit(mem, 2, Some(grob)), 0);

    // Create dense SUNMatrix for use in linear solves
    let A = sundials_sys::SUNDenseMatrix(3, 3);

    // Create dense SUNLinearSolver object
    let LS = sundials_sys::SUNDenseLinearSolver(yy, A);

    // Attach the matrix and linear solver
    assert_eq!(sundials_sys::IDASetLinearSolver(mem, LS, A), 0);

    // Set the user-supplied Jacobian routine
    assert_eq!(sundials_sys::IDASetJacFn(mem, Some(jacrob)), 0);

    // Create Newton SUNNonlinearSolver object. IDA uses a Newton SUNNonlinearSolver by default, so it is unecessary to create it and attach it. It is done in this example code solely for demonstration purposes.
    let NLS = sundials_sys::SUNNonlinSol_Newton(yy);

    // Attach the nonlinear solver
    assert_eq!(sundials_sys::IDASetNonlinearSolver(mem, NLS), 0);

    let mem = mem as sundials_sys::IDAMem;

    (yy, yp, mem)
}

pub fn ida_mem_from_ida<P: crate::IdaProblem<f64, D = Dyn, R = Dyn>>(
    ida: &crate::Ida<f64, P, linear::Dense<Dyn>, nonlinear::Newton<f64, Dyn>>,
) -> sundials_sys::IDAMem {
    use std::slice::from_raw_parts_mut;
    unsafe {
        #[allow(non_snake_case)]
        let (yy, yp, IDA_mem) = build_ida();

        (*IDA_mem).ida_yy = yy;
        (*IDA_mem).ida_yp = yp;

        let mut ida_yy = adapter::vector_view_dynamic((*IDA_mem).ida_yy);
        let mut ida_yp = adapter::vector_view_dynamic((*IDA_mem).ida_yp);

        let ida_nrtfn = &mut (*IDA_mem).ida_nrtfn;
        let ida_gactive = from_raw_parts_mut((*IDA_mem).ida_gactive, *ida_nrtfn as usize);
        let ida_ghi = from_raw_parts_mut((*IDA_mem).ida_ghi, *ida_nrtfn as usize);
        let ida_rootdir = from_raw_parts_mut((*IDA_mem).ida_rootdir, *ida_nrtfn as usize);
        let ida_glo = from_raw_parts_mut((*IDA_mem).ida_glo, *ida_nrtfn as usize);
        let ida_trout = &mut (*IDA_mem).ida_trout;
        let ida_thi = &mut (*IDA_mem).ida_thi;
        let ida_grout = from_raw_parts_mut((*IDA_mem).ida_grout, *ida_nrtfn as usize);
        let ida_iroots = from_raw_parts_mut((*IDA_mem).ida_iroots, *ida_nrtfn as usize);
        let ida_tlo = &mut (*IDA_mem).ida_tlo;
        let ida_ttol = &mut (*IDA_mem).ida_ttol;
        let ida_nge = &mut (*IDA_mem).ida_nge;
        let ida_taskc = &mut (*IDA_mem).ida_taskc;
        let ida_tn = &mut (*IDA_mem).ida_tn;
        let ida_hh = &mut (*IDA_mem).ida_hh;
        let ida_toutc = &mut (*IDA_mem).ida_toutc;
        let ida_kused = &mut (*IDA_mem).ida_kused;
        let ida_psi = (*IDA_mem).ida_psi.as_mut_slice();
        let ida_cvals = (*IDA_mem).ida_cvals.as_mut_slice();
        let ida_dvals = (*IDA_mem).ida_dvals.as_mut_slice();
        let mut phi_views = (&(*IDA_mem).ida_phi)
            .iter()
            .map(|row| adapter::vector_view_dynamic(*row))
            .collect::<Vec<_>>();
        let ida_hused = &mut (*IDA_mem).ida_hused;
        let ida_nst = &mut (*IDA_mem).ida_nst;

        *ida_nrtfn = ida.roots.num_roots() as _;
        ida_gactive.copy_from_slice(
            nalgebra::convert_ref::<_, Vector<i32, Dyn, _>>(&ida.roots.ida_gactive).as_slice(),
        );
        ida_ghi.copy_from_slice(&ida.roots.ida_ghi.as_slice());
        ida_rootdir.copy_from_slice(
            nalgebra::convert_ref::<_, Vector<i32, Dyn, _>>(&ida.roots.ida_rootdir).as_slice(),
        );
        ida_glo.copy_from_slice(&ida.roots.ida_glo.as_slice());
        *ida_trout = ida.roots.ida_trout;
        *ida_thi = ida.roots.ida_thi;
        ida_grout.copy_from_slice(&ida.roots.ida_grout.as_slice());
        ida_iroots.copy_from_slice(
            nalgebra::convert_ref::<_, Vector<i32, Dyn, _>>(&ida.roots.ida_iroots).as_slice(),
        );
        *ida_tlo = ida.roots.ida_tlo;
        *ida_ttol = ida.roots.ida_ttol;
        *ida_nge = ida.roots.ida_nge as _;
        *ida_taskc = match ida.roots.ida_taskc {
            crate::IdaTask::Normal => sundials_sys::IDA_NORMAL,
            crate::IdaTask::OneStep => sundials_sys::IDA_ONE_STEP,
        };
        *ida_tn = ida.nlp.ida_tn;
        *ida_hh = ida.ida_hh;
        *ida_toutc = ida.roots.ida_toutc;
        *ida_kused = ida.ida_kused as _;
        ida_psi.copy_from_slice(&ida.ida_psi.as_slice());
        ida_cvals.copy_from_slice(&ida.ida_cvals.as_slice());
        ida_dvals.copy_from_slice(&ida.ida_dvals.as_slice());

        for (mut phi_view, phi) in phi_views.iter_mut().zip(ida.ida_phi.column_iter()) {
            phi_view.copy_from(&phi);
        }
        *ida_hused = ida.ida_hused;
        *ida_nst = ida.counters.ida_nst as _;

        ida_yy.copy_from(&ida.nlp.ida_yy);
        ida_yp.copy_from(&ida.nlp.ida_yp);

        IDA_mem
    }
}

pub mod reimpl {
    use super::*;

    pub const RTFOUND: i32 = 1;
    pub unsafe extern "C" fn IDARcheck3(IDA_mem: sundials_sys::IDAMem) -> i32 {
        let ida_nrtfn = (*IDA_mem).ida_nrtfn as usize;
        let ida_gactive = std::slice::from_raw_parts_mut((*IDA_mem).ida_gactive, ida_nrtfn);
        let ida_ghi = std::slice::from_raw_parts_mut((*IDA_mem).ida_ghi, ida_nrtfn);
        let ida_rootdir = std::slice::from_raw_parts_mut((*IDA_mem).ida_rootdir, ida_nrtfn);
        let ida_glo = std::slice::from_raw_parts_mut((*IDA_mem).ida_glo, ida_nrtfn);
        let ida_trout = &mut (*IDA_mem).ida_trout;
        let ida_thi = &mut (*IDA_mem).ida_thi;
        let ida_grout = std::slice::from_raw_parts_mut((*IDA_mem).ida_grout, ida_nrtfn);
        let ida_iroots = std::slice::from_raw_parts_mut((*IDA_mem).ida_iroots, ida_nrtfn);
        let ida_tlo = &mut (*IDA_mem).ida_tlo;
        let ida_ttol = &mut (*IDA_mem).ida_ttol;
        let ida_nge = &mut (*IDA_mem).ida_nge;
        let ida_taskc = &mut (*IDA_mem).ida_taskc;
        let ida_tn = (*IDA_mem).ida_tn;
        let ida_hh = (*IDA_mem).ida_hh;
        let ida_toutc = (*IDA_mem).ida_toutc;
        let ida_yy = adapter::vector_view_dynamic((*IDA_mem).ida_yy);
        let ida_yp = adapter::vector_view_dynamic((*IDA_mem).ida_yp);

        /* Set thi = tn or tout, whichever comes first. */
        if (*ida_taskc == sundials_sys::IDA_ONE_STEP) {
            *ida_thi = ida_tn
        };
        if (*ida_taskc == sundials_sys::IDA_NORMAL) {
            *ida_thi = if ((ida_toutc - ida_tn) * ida_hh >= 0.0) {
                ida_tn
            } else {
                ida_toutc
            };
        }

        /* Get y and y' at thi. */
        IDAGetSolution(
            IDA_mem,
            (*IDA_mem).ida_thi,
            (*IDA_mem).ida_yy,
            (*IDA_mem).ida_yp,
        );

        /* Set ghi = g(thi) and call IDARootfind to search (tlo,thi) for roots. */
        let retval = grob(
            (*IDA_mem).ida_thi,
            (*IDA_mem).ida_yy,
            (*IDA_mem).ida_yp,
            (*IDA_mem).ida_ghi,
            std::ptr::null_mut(),
        );
        (*IDA_mem).ida_nge += 1;
        //if (retval != 0) return(IDA_RTFUNC_FAIL);

        *ida_ttol = (((ida_tn).abs() + (ida_hh).abs()) * f64::EPSILON * 100.0);
        let ier = IDARootfind(IDA_mem);
        //if (ier == IDA_RTFUNC_FAIL) return(IDA_RTFUNC_FAIL);
        for i in 0..ida_nrtfn {
            if (ida_gactive[i] == 0 && ida_grout[i] != 0.0) {
                ida_gactive[i] = 1;
            }
        }
        *ida_tlo = *ida_trout;
        for i in 0..ida_nrtfn {
            ida_glo[i] = ida_grout[i];
        }

        /* If no root found, return IDA_SUCCESS. */
        if (ier == sundials_sys::IDA_SUCCESS) {
            return (sundials_sys::IDA_SUCCESS);
        }

        /* If a root was found, interpolate to get y(trout) and return.  */
        IDAGetSolution(
            IDA_mem,
            (*IDA_mem).ida_trout,
            (*IDA_mem).ida_yy,
            (*IDA_mem).ida_yp,
        );
        return RTFOUND;
    }

    pub unsafe extern "C" fn IDARootfind(IDA_mem: sundials_sys::IDAMem) -> i32 {
        let ida_nrtfn = (*IDA_mem).ida_nrtfn as usize;
        let ida_gactive = std::slice::from_raw_parts_mut((*IDA_mem).ida_gactive, ida_nrtfn);
        let ida_ghi = std::slice::from_raw_parts_mut((*IDA_mem).ida_ghi, ida_nrtfn);
        let ida_rootdir = std::slice::from_raw_parts_mut((*IDA_mem).ida_rootdir, ida_nrtfn);
        let ida_glo = std::slice::from_raw_parts_mut((*IDA_mem).ida_glo, ida_nrtfn);
        let ida_trout = &mut (*IDA_mem).ida_trout;
        let ida_thi = &mut (*IDA_mem).ida_thi;
        let ida_grout = std::slice::from_raw_parts_mut((*IDA_mem).ida_grout, ida_nrtfn);
        let ida_iroots = std::slice::from_raw_parts_mut((*IDA_mem).ida_iroots, ida_nrtfn);
        let ida_tlo = &mut (*IDA_mem).ida_tlo;
        let ida_ttol = (*IDA_mem).ida_ttol;
        let ida_nge = &mut (*IDA_mem).ida_nge;

        /* First check for change in sign in ghi or for a zero in ghi. */
        let mut imax = 0;
        let mut maxfrac = 0.0;
        let mut zroot = false;
        let mut sgnchg = false;
        for i in 0..ida_nrtfn {
            if ida_gactive[i] == 0 {
                continue;
            }
            if ida_ghi[i].abs() == 0.0 {
                if ida_rootdir[i] as f64 * ida_glo[i] <= 0.0 {
                    zroot = true;
                }
            } else {
                if ida_glo[i] * ida_ghi[i] < 0.0 && (ida_rootdir[i] as f64 * ida_glo[i] <= 0.0) {
                    let gfrac = (ida_ghi[i] / (ida_ghi[i] - ida_glo[i])).abs();
                    if gfrac > maxfrac {
                        sgnchg = true;
                        maxfrac = gfrac;
                        imax = i;
                    }
                }
            }
        }

        /* If no sign change was found, reset trout and grout.  Then return
        IDA_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
        if !sgnchg {
            *ida_trout = *ida_thi;
            for i in 0..ida_nrtfn {
                ida_grout[i] = ida_ghi[i];
            }
            if !zroot {
                return sundials_sys::IDA_SUCCESS;
            }
            for i in 0..ida_nrtfn {
                ida_iroots[i] = 0;
                if ida_gactive[i] == 0 {
                    continue;
                }

                if ida_ghi[i].abs() == 0.0 && (ida_rootdir[i] as f64 * ida_glo[i] <= 0.0) {
                    ida_iroots[i] = if ida_glo[i] > 0.0 { -1 } else { 1 };
                }
            }
            return RTFOUND;
        }
        /* Initialize alph to avoid compiler warning */
        let mut alph = 1.0;

        /* A sign change was found.  Loop to locate nearest root. */

        let mut side = 0;
        let mut sideprev = -1;
        loop {
            /* Looping point */

            /* If interval size is already less than tolerance ttol, break. */
            if (*ida_thi - *ida_tlo).abs() <= ida_ttol {
                break;
            }

            /* Set weight alph.
            On the first two passes, set alph = 1.  Thereafter, reset alph
            according to the side (low vs high) of the subinterval in which
            the sign change was found in the previous two passes.
            If the sides were opposite, set alph = 1.
            If the sides were the same, then double alph (if high side),
            or halve alph (if low side).
            The next guess tmid is the secant method value if alph = 1, but
            is closer to tlo if alph < 1, and closer to thi if alph > 1.    */

            if sideprev == side {
                alph = if side == 2 { alph * 2.0 } else { alph * 0.5 };
            } else {
                alph = 1.0;
            }

            /* Set next root approximation tmid and get g(tmid).
            If tmid is too close to tlo or thi, adjust it inward,
            by a fractional distance that is between 0.1 and 0.5.  */
            let mut tmid = *ida_thi
                - (*ida_thi - *ida_tlo) * ida_ghi[imax] / (ida_ghi[imax] - alph * ida_glo[imax]);
            if (tmid - *ida_tlo).abs() < 0.5 * ida_ttol {
                let fracint = (*ida_thi - *ida_tlo).abs() / ida_ttol;
                let fracsub = if fracint > 5.0 { 0.1 } else { 0.5 / fracint };
                tmid = *ida_tlo + fracsub * (*ida_thi - *ida_tlo);
            }
            if (*ida_thi - tmid).abs() < 0.5 * ida_ttol {
                let fracint = (*ida_thi - *ida_tlo).abs() / ida_ttol;
                let fracsub = if fracint > 5.0 { 0.1 } else { 0.5 / fracint };
                tmid = *ida_thi - fracsub * (*ida_thi - *ida_tlo);
            }

            IDAGetSolution(IDA_mem, tmid, (*IDA_mem).ida_yy, (*IDA_mem).ida_yp);
            let retval = grob(
                tmid,
                (*IDA_mem).ida_yy,
                (*IDA_mem).ida_yp,
                (*IDA_mem).ida_grout,
                std::ptr::null_mut(),
            );
            *ida_nge += 1;
            //if (retval != 0) {return(IDA_RTFUNC_FAIL)};

            /* Check to see in which subinterval g changes sign, and reset imax.
            Set side = 1 if sign change is on low side, or 2 if on high side.  */
            maxfrac = 0.0;
            zroot = false;
            sgnchg = false;
            sideprev = side;
            for i in 0..ida_nrtfn {
                if ida_gactive[i] == 0 {
                    continue;
                }
                if ida_grout[i].abs() == 0.0 {
                    if ida_rootdir[i] as f64 * ida_glo[i] <= 0.0 {
                        zroot = true;
                    }
                } else {
                    if ida_glo[i] * ida_grout[i] < 0.0 && ida_rootdir[i] as f64 * ida_glo[i] <= 0.0
                    {
                        let gfrac = (ida_grout[i] / (ida_grout[i] - ida_glo[i])).abs();
                        if (gfrac > maxfrac) {
                            sgnchg = true;
                            maxfrac = gfrac;
                            imax = i;
                        }
                    }
                }
            }

            if sgnchg {
                /* Sign change found in (tlo,tmid); replace thi with tmid. */
                *ida_thi = tmid;
                for i in 0..ida_nrtfn {
                    ida_ghi[i] = ida_grout[i];
                }
                side = 1;
                /* Stop at root thi if converged; otherwise loop. */
                if (*ida_thi - *ida_tlo).abs() <= ida_ttol {
                    break;
                }
                continue; /* Return to looping point. */
            }

            if zroot {
                /* No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid. */
                *ida_thi = tmid;
                for i in 0..ida_nrtfn {
                    ida_ghi[i] = ida_grout[i];
                }
                break;
            }
            /* No sign change in (tlo,tmid), and no zero at tmid.  Sign change must be in (tmid,thi).  Replace tlo with tmid. */
            *ida_tlo = tmid;
            for i in 0..ida_nrtfn {
                ida_glo[i] = ida_grout[i];
            }
            side = 2;
            /* Stop at root thi if converged; otherwise loop back. */
            if (*ida_thi - *ida_tlo).abs() <= ida_ttol {
                break;
            }
        } /* End of root-search loop */

        /* Reset trout and grout, set iroots, and return RTFOUND. */
        *ida_trout = *ida_thi;
        for i in 0..ida_nrtfn {
            ida_grout[i] = ida_ghi[i];
            ida_iroots[i] = 0;
            if ida_gactive[i] == 0 {
                continue;
            }
            if (ida_ghi[i].abs() == 0.0) && (ida_rootdir[i] as f64 * ida_glo[i] <= 0.0) {
                ida_iroots[i] = if ida_glo[i] > 0.0 { -1 } else { 1 };
            }

            if ((ida_glo[i] * ida_ghi[i] < 0.0) && (ida_rootdir[i] as f64 * ida_glo[i] <= 0.0)) {
                ida_iroots[i] = if ida_glo[i] > 0.0 { -1 } else { 1 };
            }
        }

        return RTFOUND;
    }
}
