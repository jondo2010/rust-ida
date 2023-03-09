//! Build a `rust-ida` solver using a previously initialized solver from `sundials-sys`.

use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, Dyn, Matrix, Storage, StorageMut,
    Vector, U1,
};
use serde::{Deserialize, Serialize};

use crate::{
    constants::{MAXNLSIT, MXORDP1},
    private::{IdaLProblem, IdaLProblemCounters, IdaNLProblem},
    tol_control::TolControl,
    Ida, IdaCounters, IdaLimits, IdaProblem, IdaRootData, IdaTask,
};

mod adapter;
mod roberts;

/// Implements `IdaProblem` for a solver initialized by the `sundials-sys` crate.
#[derive(Debug, Serialize, Deserialize)]
pub struct SundialsProblem {}

impl IdaProblem<f64> for SundialsProblem {
    type D = Dyn;
    type R = Dyn;

    fn res<SA, SB, SC>(
        &self,
        tt: f64,
        yy: &Matrix<f64, Self::D, U1, SA>,
        yp: &Matrix<f64, Self::D, U1, SB>,
        rr: &mut Matrix<f64, Self::D, U1, SC>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: StorageMut<f64, Self::D>,
    {
        rr[0] = -0.04 * yy[0] + 1.0e4 * yy[1] * yy[2];
        rr[1] = -rr[0] - 3.0e7 * yy[1] * yy[1] - yp[1];
        rr[0] -= yp[0];
        rr[2] = yy[0] + yy[1] + yy[2] - 1.0;
    }

    fn jac<SA, SB, SC, SD>(
        &self,
        tt: f64,
        cj: f64,
        yy: &Matrix<f64, Self::D, U1, SA>,
        yp: &Matrix<f64, Self::D, U1, SB>,
        rr: &Matrix<f64, Self::D, U1, SC>,
        jac: &mut Matrix<f64, Self::D, Self::D, SD>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: Storage<f64, Self::D>,
        SD: StorageMut<f64, Self::D, Self::D>,
    {
        jac[(0, 0)] = -0.04 - cj;
        jac[(0, 1)] = 1.0e4 * yy[2];
        jac[(0, 2)] = 1.0e4 * yy[1];

        jac[(1, 0)] = 0.04;
        jac[(1, 1)] = -1.0e4 * yy[2] - 6.0e7 * yy[1] - cj;
        jac[(1, 2)] = -1.0e4 * yy[1];

        jac[(2, 0)] = 1.0;
        jac[(2, 1)] = 1.0;
        jac[(2, 2)] = 1.0;
    }

    fn root<SA, SB, SC>(
        &self,
        _t: f64,
        y: &Matrix<f64, Self::D, U1, SA>,
        _yp: &Matrix<f64, Self::D, U1, SB>,
        gout: &mut Matrix<f64, Self::R, U1, SC>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: StorageMut<f64, Self::R>,
    {
        gout[0] = y[0] - 0.0001;
        gout[1] = y[2] - 0.01;
    }
}

extern "C" {
    fn IDAGetSolution(
        mem: sundials_sys::IDAMem,
        t: f64,
        yref: sundials_sys::N_Vector,
        ypret: sundials_sys::N_Vector,
    ) -> std::ffi::c_int;
}

const IDA_SS: i32 = 1;
const IDA_SV: i32 = 2;

impl Ida<f64, SundialsProblem, linear::Dense<Dyn>, nonlinear::Newton<f64, Dyn>>
where
    DefaultAllocator:
        Allocator<f64, Dyn, Dyn> + Allocator<f64, Dyn, Const<MXORDP1>> + Allocator<f64, Dyn>,
{
    pub fn from_sundials(mem: sundials_sys::IDAMem) -> Self {
        unsafe {
            let tol_control = tol_control(mem);

            let phi_views = (&(*mem).ida_phi)
                .iter()
                .map(|row| adapter::vector_view_dynamic(*row))
                .collect::<Vec<_>>();

            let nlp = nlproblem(mem);

            let ida_constraints = ((*mem).ida_constraintsSet > 0)
                .then(|| adapter::vector_view_dynamic((*mem).ida_constraints).clone_owned());

            //TODO: pull this data out of mem
            let neq = sundials_sys::N_VGetLength((*mem).ida_yy) as usize;
            let nls = nonlinear::Newton::new_dynamic(neq, MAXNLSIT);

            let limits = IdaLimits {
                ida_maxncf: (*mem).ida_maxncf as usize,
                ida_maxnef: (*mem).ida_maxnef as usize,
                ida_maxord: (*mem).ida_maxord as usize,
                ida_mxstep: (*mem).ida_mxstep as usize,
                ida_hmax_inv: (*mem).ida_hmax_inv,
            };
            let counters = IdaCounters {
                ida_nst: (*mem).ida_nst as usize,
                ida_nre: (*mem).ida_nre as usize,
                ida_ncfn: (*mem).ida_ncfn as usize,
                ida_netf: (*mem).ida_netf as usize,
                ida_nni: (*mem).ida_nni as usize,
            };
            let num_roots = (*mem).ida_nrtfn as usize;

            let iroots = DVector::from_column_slice(std::slice::from_raw_parts(
                (*mem).ida_iroots,
                num_roots,
            ));
            let rootdir = DVector::from_column_slice(std::slice::from_raw_parts(
                (*mem).ida_rootdir,
                num_roots,
            ));
            let glo =
                DVector::from_column_slice(std::slice::from_raw_parts((*mem).ida_glo, num_roots));
            let ghi =
                DVector::from_column_slice(std::slice::from_raw_parts((*mem).ida_ghi, num_roots));
            let grout =
                DVector::from_column_slice(std::slice::from_raw_parts((*mem).ida_grout, num_roots));
            let gactive = DVector::from_column_slice(std::slice::from_raw_parts(
                (*mem).ida_gactive,
                num_roots,
            ));

            let roots = IdaRootData::<f64, Dyn> {
                ida_iroots: nalgebra::convert(iroots),
                ida_rootdir: nalgebra::convert(rootdir),
                ida_tlo: (*mem).ida_tlo,
                ida_thi: (*mem).ida_thi,
                ida_trout: (*mem).ida_trout,
                ida_glo: glo,
                ida_ghi: ghi,
                ida_grout: grout,
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
                ida_gactive: nalgebra::convert(gactive),
                ida_mxgnull: (*mem).ida_mxgnull as usize,
            };
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
                ida_delta: adapter::vector_view_dynamic((*mem).ida_delta).clone_owned(),
                ida_id: None,
                ida_constraints,
                ida_ee: adapter::vector_view_dynamic((*mem).ida_ee).clone_owned(),
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
                limits,
                counters,
                ida_cvals: Vector::from((*mem).ida_cvals),
                ida_dvals: Vector::from((*mem).ida_dvals),
                ida_tolsf: (*mem).ida_tolsf,
                roots,
                nls,
                nlp,
            }
        }
    }
}

fn nlproblem(
    mem: *mut sundials_sys::IDAMemRec,
) -> IdaNLProblem<f64, SundialsProblem, linear::Dense<Dyn>>
where
    DefaultAllocator: Allocator<f64, Dyn> + Allocator<f64, Dyn, Dyn> + Allocator<usize, Dyn>,
{
    assert_ne!(mem, std::ptr::null_mut());
    let lp = lproblem(mem);
    unsafe {
        IdaNLProblem {
            ida_yy: adapter::vector_view_dynamic((*mem).ida_yy).clone_owned(),
            ida_yp: adapter::vector_view_dynamic((*mem).ida_yp).clone_owned(),
            ida_yypredict: adapter::vector_view_dynamic((*mem).ida_yypredict).clone_owned(),
            ida_yppredict: adapter::vector_view_dynamic((*mem).ida_yppredict).clone_owned(),
            ida_ewt: adapter::vector_view_dynamic((*mem).ida_ewt).clone_owned(),
            ida_savres: adapter::vector_view_dynamic((*mem).ida_savres).clone_owned(),
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

fn lproblem(
    mem: *mut sundials_sys::IDAMemRec,
) -> IdaLProblem<f64, SundialsProblem, linear::Dense<Dyn>>
where
    DefaultAllocator: Allocator<f64, Dyn> + Allocator<f64, Dyn, Dyn> + Allocator<usize, Dyn>,
{
    assert_ne!(mem, std::ptr::null_mut());
    unsafe {
        let lmem = (*mem).ida_lmem as sundials_sys::IDALsMem;
        assert_ne!(lmem, std::ptr::null_mut());

        //TODO: pull this out of the mem
        let neq = sundials_sys::N_VGetLength((*mem).ida_yy) as usize;
        let ls = linear::Dense::new_dynamic(neq);

        IdaLProblem {
            ls,
            mat_j: adapter::dense_matrix_view_dynamic((*lmem).J).clone_owned(),
            x: adapter::vector_view_dynamic((*lmem).x).clone_owned(),
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

fn tol_control(mem: *mut sundials_sys::IDAMemRec) -> TolControl<f64, Dyn>
where
    DefaultAllocator: Allocator<f64, Dyn>,
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
                let atol = adapter::vector_view_dynamic((*mem).ida_Vatol);
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
    use std::io::Write;

    use nalgebra::DVector;

    use crate::sundials::adapter::{self, vector_view_dynamic};

    #[test]
    fn test_get_solution() {
        unsafe {
            let (yy, yp, mem) = super::roberts::build_ida();
            let mut tout1 = 0.4;
            let mut tret = 0.0;
            for _ in 0..50 {
                let ret = sundials_sys::IDASolve(
                    mem as _,
                    tout1,
                    &mut tret,
                    yy,
                    yp,
                    sundials_sys::IDA_NORMAL,
                );
                tout1 += 0.2;

                let mut ida = crate::Ida::from_sundials(mem);
                assert_eq!(
                    super::IDAGetSolution(mem as _, tout1, yy, yp),
                    sundials_sys::IDA_SUCCESS
                );

                ida.get_solution(tout1).unwrap();

                assert_eq!(
                    ida.nlp.ida_yy,
                    vector_view_dynamic(yy),
                    "test get_solution({tout1})"
                );
            }
        }
    }

    #[test]
    fn test_rcheck() {
        unsafe {
            let (yy, yp, mem) = super::roberts::build_ida();

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
            dbg!(tret);
            assert_eq!(ret, 666);
            let ida = crate::Ida::from_sundials(mem);

            let mut file = std::fs::File::create("src/tests/data/rcheck3_pre.json").unwrap();
            file.write(serde_json::to_string_pretty(&ida).unwrap().as_bytes())
                .unwrap();
        }
    }

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
            assert_eq!(ret, sundials_sys::IDA_ROOT_RETURN);

            let ida = crate::Ida::from_sundials(mem);
            let dky_expect = sundials_sys::N_VNew_Serial(3);
            let mut dky = DVector::zeros(3);

            for k in 0..=ida.ida_kused {
                sundials_sys::IDAGetDky(mem as _, tret, k as _, dky_expect);
                ida.get_dky(tret, k, &mut dky).expect("get_dky failed");
                assert_eq!(dky, vector_view_dynamic(dky_expect));
            }
        }
    }

    #[test]
    #[cfg(feature = "disabled")]
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

            ida.get_current_y(&mut y).expect("get_current_y failed");

            assert_eq!(y, vector_view::<U3>(yy));
        }
    }
}
