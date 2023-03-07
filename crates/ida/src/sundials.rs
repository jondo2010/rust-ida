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
        + Allocator<bool, D>
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

mod adapter {
    use nalgebra::{
        allocator::Allocator, DefaultAllocator, DimName, Matrix, VectorViewMut, ViewStorageMut, U1,
    };

    /// Create an nalgebra mutable view of a SUNDIALS dense matrix
    pub unsafe fn dense_matrix_view<'a, R: DimName, C: DimName>(
        m: sundials_sys::SUNMatrix,
    ) -> Matrix<f64, R, C, ViewStorageMut<'a, f64, R, C, U1, C>>
    where
        DefaultAllocator: Allocator<f64, R, C>,
    {
        let rows = sundials_sys::SUNDenseMatrix_Rows(m) as usize;
        let cols = sundials_sys::SUNDenseMatrix_Columns(m) as usize;

        assert_eq!(R::dim(), rows, "Matrix row count mismatch");
        assert_eq!(C::dim(), cols, "Matrix column count mismatch");

        Matrix::from_data(ViewStorageMut::from_raw_parts(
            sundials_sys::SUNDenseMatrix_Data(m),
            (R::name(), C::name()),
            (U1, C::name()),
        ))
    }

    /// Create an nalgebra mutable view of a SUNDIALS vector
    pub unsafe fn vector_view<'a, D: DimName>(
        v: sundials_sys::N_Vector,
    ) -> VectorViewMut<'a, f64, D> {
        assert_eq!(
            D::dim(),
            sundials_sys::N_VGetLength(v) as usize,
            "Vector length mismatch"
        );

        let data = sundials_sys::N_VGetArrayPointer(v);
        Matrix::from_data(ViewStorageMut::from_raw_parts(
            data,
            (D::name(), U1),
            (U1, D::name()),
        ))
    }

    #[test]
    fn test_mat() {
        use nalgebra::*;

        let a = unsafe {
            let a = sundials_sys::SUNDenseMatrix(3, 3);
            sundials_sys::SUNMatZero(a);
            sundials_sys::SUNMatScaleAddI(1.0, a);
            a
        };

        let mut m = unsafe { dense_matrix_view::<U3, U3>(a) };

        m[(2, 0)] = 2.0;

        unsafe {
            //sundials_sys::SUNDenseMatrix_Print(a, libc_stdhandle::stdout() as *mut _);
        }

        println!("{m}");
    }
}

mod roberts {

    use std::ffi::{c_int, c_void};

    use super::adapter::{dense_matrix_view, vector_view};
    use nalgebra::*;

    unsafe extern "C" fn res(
        _t: f64,
        _yy: sundials_sys::N_Vector,
        _yp: sundials_sys::N_Vector,
        _resval: sundials_sys::N_Vector,
        _user_data: *mut c_void,
    ) -> c_int {
        0
    }

    unsafe extern "C" fn g(
        _t: f64,
        yy: sundials_sys::N_Vector,
        _yp: sundials_sys::N_Vector,
        gout: *mut f64,
        _user_data: *mut c_void,
    ) -> c_int {
        let yval = std::slice::from_raw_parts(
            sundials_sys::N_VGetArrayPointer(yy),
            sundials_sys::N_VGetLength(yy) as usize,
        );
        let gout = std::slice::from_raw_parts_mut(gout, 3);
        gout[0] = yval[0] - 0.0001;
        gout[1] = yval[2] - 0.01;
        0
    }

    unsafe extern "C" fn jac(
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

    pub unsafe fn build() -> sundials_sys::IDAMem {
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
        let tout1 = 0.4;
        let mut tret = 0.0;

        let mem = sundials_sys::IDACreate();
        assert_eq!(sundials_sys::IDAInit(mem, Some(res), t0, yy, yp), 0);

        // Set tolerances
        assert_eq!(sundials_sys::IDASVtolerances(mem, rtol, avtol), 0);

        // Call IDARootInit to specify the root function g with 2 components
        assert_eq!(sundials_sys::IDARootInit(mem, 2, Some(g)), 0);

        // Create dense SUNMatrix for use in linear solves
        let A = sundials_sys::SUNDenseMatrix(3, 3);

        // Create dense SUNLinearSolver object
        let LS = sundials_sys::SUNDenseLinearSolver(yy, A);

        // Attach the matrix and linear solver
        assert_eq!(sundials_sys::IDASetLinearSolver(mem, LS, A), 0);

        // Set the user-supplied Jacobian routine
        assert_eq!(sundials_sys::IDASetJacFn(mem, Some(jac)), 0);

        // Create Newton SUNNonlinearSolver object. IDA uses a Newton SUNNonlinearSolver by default, so it is unecessary to create it and attach it. It is done in this example code solely for demonstration purposes.
        let NLS = sundials_sys::SUNNonlinSol_Newton(yy);

        // Attach the nonlinear solver
        assert_eq!(sundials_sys::IDASetNonlinearSolver(mem, NLS), 0);

        // Call IDASolve
        let ret = sundials_sys::IDASolve(mem, tout1, &mut tret, yy, yp, sundials_sys::IDA_NORMAL);
        println!("tret: {tret}");

        //dbg!(sundials_sys::idaNlsInit(mem as sundials_sys::IDAMem));
        let mem = mem as sundials_sys::IDAMem;

        mem
    }

    #[test]
    fn test() {
        unsafe {
            let mem = build();

            crate::Ida::<f64, U3, _, _, _>::from_sundials(mem);
        }
    }
}
