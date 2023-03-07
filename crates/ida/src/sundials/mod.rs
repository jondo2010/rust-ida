//! Build a `rust-ida` solver using a previously initialized solver from `sundials-sys`.

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameAdd, Matrix, Storage,
    StorageMut, Vector, U1,
};
use nonlinear::NLSolver;

use crate::{
    constants::{MAXNLSIT, MXORDP1},
    ida_ls::{IdaLProblem, IdaLProblemCounters},
    ida_nls::IdaNLProblem,
    tol_control::TolControl,
    traits::{Jacobian, Residual, Root},
    Ida, IdaCounters, IdaTask,
};

mod adapter;
mod roberts;

/// Implements `IdaProblem` for a solver initialized by the `sundials-sys` crate.
#[derive(Debug)]
struct SundialsProblem {}

impl<D: DimName> Residual<f64, D> for SundialsProblem {
    fn res<SA, SB, SC>(
        &self,
        tt: f64,
        yy: &Matrix<f64, D, U1, SA>,
        yp: &Matrix<f64, D, U1, SB>,
        rr: &mut Matrix<f64, D, U1, SC>,
    ) where
        SA: Storage<f64, D>,
        SB: Storage<f64, D>,
        SC: StorageMut<f64, D>,
    {
        todo!()
    }
}

impl<D: DimName> Jacobian<f64, D> for SundialsProblem {
    fn jac<SA, SB, SC, SD>(
        &self,
        tt: f64,
        cj: f64,
        yy: &Matrix<f64, D, U1, SA>,
        yp: &Matrix<f64, D, U1, SB>,
        rr: &Matrix<f64, D, U1, SC>,
        jac: &mut Matrix<f64, D, D, SD>,
    ) where
        SA: Storage<f64, D>,
        SB: Storage<f64, D>,
        SC: Storage<f64, D>,
        SD: StorageMut<f64, D, D>,
    {
        todo!()
    }
}

impl<D: DimName> Root<f64, D> for SundialsProblem {
    fn root<SA, SB, SC>(
        &self,
        t: f64,
        y: &Matrix<f64, D, U1, SA>,
        yp: &Matrix<f64, D, U1, SB>,
        gout: &mut Matrix<f64, D, U1, SC>,
    ) where
        SA: Storage<f64, D>,
        SB: Storage<f64, D>,
        SC: StorageMut<f64, D>,
    {
        todo!()
    }
}

const IDA_SS: i32 = 1;
const IDA_SV: i32 = 2;

impl<D> Ida<f64, D, SundialsProblem, linear::Dense<D>, nonlinear::Newton<f64, D>>
where
    D: DimName + DimNameAdd<D>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D, Const<MXORDP1>>
        + Allocator<f64, D>
        + Allocator<u8, D>
        + Allocator<usize, D>,
{
    pub fn from_sundials(mem: sundials_sys::IDAMem) -> Self {
        unsafe {
            let tol_control = tol_control(mem);

            let phi_views = (&(*mem).ida_phi)
                .iter()
                .map(|row| adapter::vector_view::<D>(*row))
                .collect::<Vec<_>>();

            let nlp = nlproblem(mem);

            let ida_constraints = ((*mem).ida_constraintsSet > 0)
                .then(|| adapter::vector_view::<D>((*mem).ida_constraints).clone_owned());

            Self {
                ida_setup_done: (*mem).ida_SetupDone > 0,
                tol_control,
                ida_suppressalg: (*mem).ida_suppressalg > 0,
                ida_phi: Matrix::from_columns(phi_views.as_slice()),
                ida_psi: Vector::from((*mem).ida_psi),
                ida_alpha: Vector::from((*mem).ida_alpha),
                ida_beta: Vector::from((*mem).ida_beta),
                ida_sigma: Vector::from((*mem).ida_sigma),
                ida_gamma: Vector::from((*mem).ida_gamma),
                ida_delta: adapter::vector_view::<D>((*mem).ida_delta).clone_owned(),
                ida_id: None,
                ida_constraints,
                ida_ee: adapter::vector_view::<D>((*mem).ida_ee).clone_owned(),
                ida_tstop: ((*mem).ida_tstopset > 0).then(|| (*mem).ida_tstop),
                ida_kk: (*mem).ida_kk as usize,
                ida_kused: (*mem).ida_kused as usize,
                ida_knew: (*mem).ida_knew as usize,
                ida_phase: (*mem).ida_phase as usize,
                ida_ns: (*mem).ida_ns as usize,
                ida_hin: (*mem).ida_hin,
                ida_h0u: (*mem).ida_h0u,
                ida_hh: (*mem).ida_hh,
                ida_hused: (*mem).ida_hused,
                ida_rr: (*mem).ida_rr,
                ida_tretlast: (*mem).ida_tretlast,
                ida_cjlast: (*mem).ida_cjlast,
                ida_eps_newt: (*mem).ida_epsNewt,
                ida_epcon: (*mem).ida_epcon,
                ida_maxncf: (*mem).ida_maxncf as u64,
                ida_maxnef: (*mem).ida_maxnef as u64,
                ida_maxord: (*mem).ida_maxord as usize,
                ida_mxstep: (*mem).ida_mxstep as u64,
                ida_hmax_inv: (*mem).ida_hmax_inv,
                counters: IdaCounters {
                    ida_nst: (*mem).ida_nst as usize,
                    ida_nre: (*mem).ida_nre as usize,
                    ida_ncfn: (*mem).ida_ncfn as usize,
                    ida_netf: (*mem).ida_netf as usize,
                    ida_nni: (*mem).ida_nni as usize,
                },
                ida_cvals: Vector::from((*mem).ida_cvals),
                ida_dvals: Vector::from((*mem).ida_dvals),
                ida_tolsf: (*mem).ida_tolsf,
                ida_tlo: (*mem).ida_tlo,
                ida_thi: (*mem).ida_thi,
                ida_trout: (*mem).ida_trout,
                ida_toutc: (*mem).ida_toutc,
                ida_ttol: (*mem).ida_ttol,
                ida_taskc: match (*mem).ida_taskc {
                    sundials_sys::IDA_NORMAL => IdaTask::Normal,
                    sundials_sys::IDA_ONE_STEP => IdaTask::OneStep,
                    _ => {
                        panic!("Unknown ida_taskc: {}", (*mem).ida_taskc);
                    }
                },
                ida_irfnd: (*mem).ida_irfnd > 0,
                ida_nge: (*mem).ida_nge as usize,
                ida_mxgnull: (*mem).ida_mxgnull as usize,
                nls: nonlinear::Newton::new(MAXNLSIT),
                nlp,
            }
        }
    }
}

fn nlproblem<D>(
    mem: *mut sundials_sys::IDAMemRec,
) -> IdaNLProblem<f64, D, SundialsProblem, linear::Dense<D>>
where
    D: DimName + DimNameAdd<D>,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D> + Allocator<usize, D>,
{
    assert_ne!(mem, std::ptr::null_mut());
    let lp = lproblem(mem);
    unsafe {
        IdaNLProblem {
            ida_yy: adapter::vector_view::<D>((*mem).ida_yy).clone_owned(),
            ida_yp: adapter::vector_view::<D>((*mem).ida_yp).clone_owned(),
            ida_yypredict: adapter::vector_view::<D>((*mem).ida_yypredict).clone_owned(),
            ida_yppredict: adapter::vector_view::<D>((*mem).ida_yppredict).clone_owned(),
            ida_ewt: adapter::vector_view::<D>((*mem).ida_ewt).clone_owned(),
            ida_savres: adapter::vector_view::<D>((*mem).ida_savres).clone_owned(),
            ida_tn: (*mem).ida_tn,
            ida_ss: (*mem).ida_ss,
            ida_oldnrm: (*mem).ida_oldnrm,
            ida_toldel: (*mem).ida_toldel,
            ida_nre: (*mem).ida_nre as usize,
            ida_nsetups: (*mem).ida_nsetups as usize,
            lp,
        }
    }
}

fn lproblem<D>(
    mem: *mut sundials_sys::IDAMemRec,
) -> IdaLProblem<f64, D, SundialsProblem, linear::Dense<D>>
where
    D: DimName + DimNameAdd<D>,
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D> + Allocator<usize, D>,
{
    assert_ne!(mem, std::ptr::null_mut());
    unsafe {
        let lmem = (*mem).ida_lmem as sundials_sys::IDALsMem;
        assert_ne!(lmem, std::ptr::null_mut());

        IdaLProblem {
            ls: linear::Dense::<D>::new(),
            mat_j: adapter::dense_matrix_view::<D, D>((*lmem).J).clone_owned(),
            x: adapter::vector_view::<D>((*lmem).x).clone_owned(),
            eplifac: (*lmem).eplifac,
            nrmfac: (*lmem).nrmfac,
            ida_cj: (*mem).ida_cj,
            ida_cjold: (*mem).ida_cjold,
            ida_cjratio: (*mem).ida_cjratio,
            problem: SundialsProblem {},
            counters: IdaLProblemCounters {
                nje: (*lmem).nje as usize,
                npe: (*lmem).npe as usize,
                nli: (*lmem).nli as usize,
                nps: (*lmem).nps as usize,
                ncfl: (*lmem).ncfl as usize,
                nre_dq: (*lmem).nreDQ as usize,
                njtsetup: (*lmem).njtsetup as usize,
                njtimes: (*lmem).njtimes as usize,
                nst0: (*lmem).nst0 as usize,
                nni0: (*lmem).nni0 as usize,
                ncfn0: (*lmem).ncfn0 as usize,
                ncfl0: (*lmem).ncfl0 as usize,
                nwarn: (*lmem).nwarn as usize,
            },
        }
    }
}

fn tol_control<D>(mem: *mut sundials_sys::IDAMemRec) -> TolControl<f64, D>
where
    D: DimName + DimNameAdd<D>,
    DefaultAllocator: Allocator<f64, D>,
{
    assert_ne!(mem, std::ptr::null_mut());
    unsafe {
        match (*mem).ida_itol {
            IDA_SS => {
                let rtol = (*mem).ida_rtol;
                let atol = (*mem).ida_Satol;
                TolControl::new_ss(rtol, atol)
            }
            IDA_SV => {
                let rtol = (*mem).ida_rtol;
                let atol = adapter::vector_view::<D>((*mem).ida_Vatol);
                TolControl::new_sv(rtol, atol.clone_owned())
            }
            _ => {
                panic!("Unknown itol: {}", (*mem).ida_itol);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Vector3, U3};

    use crate::sundials::adapter::vector_view;

    #[test]
    fn test_get_dky() {
        unsafe {
            let (yy, yp, mem) = super::roberts::build_ida();

            // Call IDASolve
            let tout1 = 0.4;
            let mut tret = 0.0;
            let ret = sundials_sys::IDASolve(
                mem as _,
                tout1,
                &mut tret,
                yy,
                yp,
                sundials_sys::IDA_NORMAL,
            );

            let ida = crate::Ida::<f64, U3, _, _, _>::from_sundials(mem);
            let dky_expect = sundials_sys::N_VNew_Serial(3);
            let mut dky = Vector3::zeros();

            for k in 0..=ida.ida_kused {
                sundials_sys::IDAGetDky(mem as _, tret, k as _, dky_expect);
                ida.get_dky(tret, k, &mut dky).expect("get_dky failed");

                assert_eq!(dky, vector_view::<U3>(dky_expect));
            }
        }
    }

    #[test]
    fn test_get_current_y() {
        unsafe {
            let (yy, yp, mem) = super::roberts::build_ida();

            // Call IDASolve
            let tout1 = 0.4;
            let mut tret = 0.0;
            let ret = sundials_sys::IDASolve(
                mem as _,
                tout1,
                &mut tret,
                yy,
                yp,
                sundials_sys::IDA_NORMAL,
            );

            let ida = crate::Ida::<f64, U3, _, _, _>::from_sundials(mem);
            let mut y = Vector3::zeros();

            //ida.get_current_y(&mut y).expect("get_current_y failed");

            assert_eq!(y, vector_view::<U3>(yy));
        }
    }
}
