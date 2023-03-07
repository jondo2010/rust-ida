use crate::{
    constants::*,
    ida_nls::{self, IdaNLProblem},
    tol_control::{self, TolControl},
    traits::{Jacobian, Residual, Root},
    Error, Ida, IdaCounters, IdaTask,
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, DimName, DimNameAdd, Dyn, Matrix, OMatrix,
    OVector, Storage, StorageMut, Vector, U1, U6,
};
use nonlinear::NLSolver;

use crate::traits::{IdaProblem, IdaReal};

impl<T, D, P, LS, NLS> Ida<T, D, P, LS, NLS>
where
    T: IdaReal,
    D: Dim + DimName,
    P: IdaProblem<T, D>,
    LS: linear::LSolver<T, D>,
    NLS: nonlinear::NLSolver<T, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D> + Allocator<u8, D>,
{
    /// Creates a new Ida solver given initial Arrays of `yy0` and `yyp`.
    ///
    /// *Panics" if ModelSpec::Scalar is unable to convert any constant initialization value.
    pub fn new<SA, SB>(
        problem: P,
        ls: LS,
        yy0: &Matrix<T, D, U1, SA>,
        yp0: &Matrix<T, D, U1, SB>,
        tol_control: TolControl<T, D>,
    ) -> Self
    where
        SA: Storage<T, D, U1>,
        SB: Storage<T, D, U1>,
    {
        // Initialize the phi array
        let mut ida_phi = OMatrix::<T, D, Const<MXORDP1>>::zeros();
        ida_phi.set_column(0, yy0);
        ida_phi.set_column(1, yp0);

        #[cfg(feature = "data_trace")]
        {
            let mut data_trace = std::fs::File::create("roberts_rs.json").unwrap();
            data_trace.write_all(b"{\"data\":[\n").unwrap();
        }

        //IDAResFn res, realtype t0, N_Vector yy0, N_Vector yp0
        Self {
            ida_setup_done: false,

            tol_control,

            // Set default values for integrator optional inputs
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as u64,
            ida_hmax_inv: T::from(HMAX_INV_DEFAULT).unwrap(),
            ida_hin: T::zero(),
            ida_eps_newt: T::zero(),
            ida_epcon: T::from(EPCON).unwrap(),
            ida_maxnef: MXNEF as u64,
            ida_maxncf: MXNCF as u64,
            ida_suppressalg: false,
            //ida_id          = NULL;
            ida_constraints: None,

            ida_cjlast: T::zero(),

            // set the saved value maxord_alloc
            //ida_maxord_alloc = MAXORD_DEFAULT;

            // Set default values for IC optional inputs
            //ida_epiccon : T::from(0.01 * EPCON).unwrap(),
            //ida_maxnh   : MAXNH,
            //ida_maxnj   = MAXNJ;
            //ida_maxnit  = MAXNI;
            //ida_maxbacks  = MAXBACKS;
            //ida_lsoff   = SUNFALSE;
            //ida_steptol = SUNRpowerR(self.ida_uround, TWOTHIRDS);
            ida_phi,

            ida_psi: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_alpha: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_beta: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_sigma: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_gamma: OVector::<T, Const<MXORDP1>>::zeros(),

            ida_delta: OVector::<T, D>::zeros(),
            ida_id: None,

            // Initialize all the counters and other optional output values
            counters: IdaCounters {
                ida_nst: 0,
                ida_ncfn: 0,
                ida_netf: 0,
                ida_nni: 0,

                ida_nre: 0,
            },

            ida_kused: 0,
            ida_hused: T::zero(),
            ida_tolsf: T::one(),
            ida_nge: 0,

            // Initialize root-finding variables
            ida_irfnd: false,
            //ida_glo: Array::zeros(problem.num_roots()),
            //ida_ghi: Array::zeros(problem.num_roots()),
            //ida_grout: Array::zeros(problem.num_roots()),
            //ida_iroots: Array::zeros(problem.num_roots()),
            //ida_rootdir: Array::zeros(problem.num_roots()),
            //ida_nrtfn: problem.num_roots(),
            ida_mxgnull: 1,
            ida_tlo: T::zero(),
            ida_ttol: T::zero(),
            //ida_gactive: Array::from_elem(problem.num_roots(), false),
            ida_taskc: IdaTask::Normal,
            ida_toutc: T::zero(),
            ida_thi: T::zero(),
            ida_trout: T::zero(),

            // Not from ida.c...
            ida_ee: OVector::<T, D>::zeros(),

            ida_tstop: None,

            ida_kk: 0,
            ida_knew: 0,
            ida_phase: 0,
            ida_ns: 0,

            ida_rr: T::zero(),
            ida_tretlast: T::zero(),
            ida_h0u: T::zero(),
            ida_hh: T::zero(),

            ida_cvals: OVector::<T, Const<MXORDP1>>::zeros(),
            ida_dvals: OVector::<T, Const<MAXORD_DEFAULT>>::zeros(),

            //ida_zvecs: Array::zeros((MXORDP1, yy0.shape()[0])),

            // Initialize nonlinear solver
            nls: NLS::new(MAXNLSIT),
            nlp: ida_nls::IdaNLProblem::new(problem, ls, yy0, yp0),

            #[cfg(feature = "data_trace")]
            data_trace,
        }
    }

    /// IDAInitialSetup
    ///
    /// This routine is called by `solve` once at the first step. It performs all checks on optional inputs and inputs
    /// to `init`/`reinit` that could not be done before.
    ///
    /// If no error is encountered, IDAInitialSetup returns IDA_SUCCESS. Otherwise, it returns an error flag and
    /// reported to the error handler function.
    fn initial_setup(&mut self) {
        //booleantype conOK;
        //int ier;

        // Initial error weight vector
        self.tol_control
            .ewt_set(&self.ida_phi.column(0), &mut self.nlp.ida_ewt);

        /*
        // Check to see if y0 satisfies constraints.
        if (IDA_mem->ida_constraintsSet) {
          conOK = N_VConstrMask(IDA_mem->ida_constraints, IDA_mem->ida_phi[0], IDA_mem->ida_tempv2);
          if (!conOK) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_Y0_FAIL_CONSTR);
            return(IDA_ILL_INPUT);
          }
        }

        // Call linit function if it exists.
        if (IDA_mem->ida_linit != NULL) {
          ier = IDA_mem->ida_linit(IDA_mem);
          if (ier != 0) {
            IDAProcessError(IDA_mem, IDA_ILL_INPUT, "IDA", "IDAInitialSetup", MSG_LINIT_FAIL);
            return(IDA_LINIT_FAIL);
          }
        }
        */

        //return(IDA_SUCCESS);
    }
}
