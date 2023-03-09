use crate::{
    constants::*, private::IdaNLProblem, tol_control::TolControl, Ida, IdaCounters, IdaLimits,
    IdaRootData,
};
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, OMatrix, OVector, Storage, Vector,
};

use crate::traits::{IdaProblem, IdaReal};

impl<T, P, LS, NLS> Ida<T, P, LS, NLS>
where
    T: IdaReal,
    P: IdaProblem<T>,
    LS: linear::LSolver<T, P::D>,
    NLS: nonlinear::NLSolver<T, P::D>,
    DefaultAllocator: Allocator<T, P::D>
        + Allocator<T, P::R>
        + Allocator<i8, P::R>
        + Allocator<T, P::D, P::D>
        + Allocator<T, P::D, Const<MXORDP1>>,
{
    /// Creates a new Ida solver given initial Arrays of `yy0` and `yyp`.
    pub fn new<SA, SB>(
        problem: P,
        ls: LS,
        nls: NLS,
        tol_control: TolControl<T, P::D>,
        t0: T,
        yy0: &Vector<T, P::D, SA>,
        yp0: &Vector<T, P::D, SB>,
    ) -> Self
    where
        SA: Storage<T, P::D>,
        SB: Storage<T, P::D>,
        P::D: DimName,
        P::R: DimName,
    {
        // Initialize the phi array
        let mut ida_phi = OMatrix::<T, P::D, Const<MXORDP1>>::zeros();
        ida_phi.set_column(0, yy0);
        ida_phi.set_column(1, yp0);

        // Initialize all the counters and other optional output values
        let limits = IdaLimits {
            ida_maxncf: MXNCF as usize,
            ida_maxnef: MXNEF as usize,
            ida_maxord: MAXORD_DEFAULT as usize,
            ida_mxstep: MXSTEP_DEFAULT as usize,
            ida_hmax_inv: T::from(HMAX_INV_DEFAULT).unwrap(),
        };

        let counters = IdaCounters {
            ida_nst: 0,
            ida_ncfn: 0,
            ida_netf: 0,
            ida_nni: 0,
            ida_nre: 0,
        };

        Self {
            ida_setup_done: false,

            tol_control,

            ida_hin: T::zero(),
            ida_eps_newt: T::zero(),
            ida_epcon: T::from(EPCON).unwrap(),
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

            ida_delta: OVector::<T, P::D>::zeros(),
            ida_id: None,

            limits,
            counters,

            ida_kused: 0,
            ida_hused: T::zero(),
            ida_tolsf: T::one(),

            // Not from ida.c...
            ida_ee: OVector::<T, P::D>::zeros(),

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

            // Initialize root-finding variables
            roots: IdaRootData::new(),

            nls: nls,
            nlp: IdaNLProblem::new(problem, ls, yy0, yp0),
        }
    }
}

#[cfg(feature = "disabled")]
impl<T, P, LS, NLS> Ida<T, P, LS, NLS>
where
    T: IdaReal,
    P: IdaProblem<T>,
    LS: linear::LSolver<T, Dyn>,
    NLS: nonlinear::NLSolver<T, Dyn>,
    DefaultAllocator:
        Allocator<T, Dyn, Dyn> + Allocator<T, Dyn, Const<MXORDP1>> + Allocator<T, Dyn>,
{
    pub fn new_dynamic<SA, SB>(
        problem: P,
        ls: LS,
        nls: NLS,
        yy0: &Vector<T, Dyn, SA>,
        yp0: &Vector<T, Dyn, SB>,
        tol_control: TolControl<T, Dyn>,
    ) -> Self
    where
        SA: Storage<T, Dyn>,
        SB: Storage<T, Dyn>,
    {
    }
}
