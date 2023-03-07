use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, DimName, OMatrix, OVector};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

pub(crate) mod constants;
use constants::*;
mod error;
pub use error::Error;
pub mod ida_ls;
pub mod ida_nls;
mod impl_new;
pub mod sundials;
pub use impl_new::*;
pub mod tol_control;
pub mod traits;
use tol_control::TolControl;
use traits::{IdaProblem, IdaReal};
#[cfg(test)]
mod tests;

/// Counters
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct IdaCounters {
    /// number of internal steps taken
    ida_nst: usize,
    /// number of function (res) calls
    ida_nre: usize,
    /// number of corrector convergence failures
    ida_ncfn: usize,
    /// number of error test failures
    ida_netf: usize,
    /// number of Newton iterations performed
    ida_nni: usize,
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub enum IdaTask {
    Normal,
    OneStep,
}

#[derive(PartialEq, Debug)]
pub enum IdaSolveStatus {
    ContinueSteps,
    Success,
    TStop,
    Root,
}

enum IdaConverged {
    Converged,
    NotConverged,
}

/// This structure contains fields to keep track of problem state.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, D>: Serialize, OMatrix<T, D, Const<MXORDP1>>: Serialize, OVector<bool, D>: Serialize, ida_nls::IdaNLProblem<T, D, P, LS>: Serialize, NLS: Serialize, LS: Serialize, P: Serialize"
    ))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "T: Deserialize<'de>, OVector<T, D>: Deserialize<'de>, OMatrix<T, D, Const<MXORDP1>>: Deserialize<'de>, OVector<bool, D>: Deserialize<'de>, ida_nls::IdaNLProblem<T, D, P, LS>: Deserialize<'de>, NLS: Deserialize<'de>, LS: Deserialize<'de>, P: Deserialize<'de>"
    ))
)]
#[derive(Debug)]
pub struct Ida<T, D, P, LS, NLS>
where
    T: IdaReal,
    D: DimName,
    P: IdaProblem<T, D>,
    LS: linear::LSolver<T, D>,
    NLS: nonlinear::NLSolver<T, D>,
    DefaultAllocator:
        Allocator<T, D, D> + Allocator<T, D, Const<MXORDP1>> + Allocator<T, D> + Allocator<bool, D>,
{
    ida_setup_done: bool,
    tol_control: TolControl<T, D>,

    /// SUNTRUE means suppress algebraic vars in local error tests
    ida_suppressalg: bool,

    // Divided differences array and associated minor arrays
    /// phi = (maxord+1) arrays of divided differences
    ida_phi: OMatrix<T, D, Const<MXORDP1>>,
    /// differences in t (sums of recent step sizes)
    ida_psi: OVector<T, Const<MXORDP1>>,
    /// ratios of current stepsize to psi values
    ida_alpha: OVector<T, Const<MXORDP1>>,
    /// ratios of current to previous product of psi's
    ida_beta: OVector<T, Const<MXORDP1>>,
    /// product successive alpha values and factorial
    ida_sigma: OVector<T, Const<MXORDP1>>,
    /// sum of reciprocals of psi values
    ida_gamma: OVector<T, Const<MXORDP1>>,

    // Vectors
    /// residual vector
    ida_delta: OVector<T, D>,
    /// bit vector for diff./algebraic components
    ida_id: Option<OVector<bool, D>>,
    /// vector of inequality constraint options
    ida_constraints: Option<OVector<T, D>>,
    /// accumulated corrections to y vector, but set equal to estimated local errors upon successful return
    ida_ee: OVector<T, D>,

    //ida_mm;          /* mask vector in constraints tests (= tempv2)    */
    //ida_tempv1;      /* work space vector                              */
    //ida_tempv2;      /* work space vector                              */
    //ida_tempv3;      /* work space vector                              */
    //ida_ynew;        /* work vector for y in IDACalcIC (= tempv2)      */
    //ida_ypnew;       /* work vector for yp in IDACalcIC (= ee)         */
    //ida_delnew;      /* work vector for delta in IDACalcIC (= phi[2])  */
    //ida_dtemp;       /* work vector in IDACalcIC (= phi[3])            */

    // Tstop information
    ida_tstop: Option<T>,

    // Step Data
    /// current BDF method order
    ida_kk: usize,
    /// method order used on last successful step
    ida_kused: usize,
    /// order for next step from order decrease decision
    ida_knew: usize,
    /// flag to trigger step doubling in first few steps
    ida_phase: usize,
    /// counts steps at fixed stepsize and order
    ida_ns: usize,

    /// initial step
    ida_hin: T,
    /// actual initial stepsize
    ida_h0u: T,
    /// current step size h
    ida_hh: T,
    /// step size used on last successful step
    ida_hused: T,
    /// rr = hnext / hused
    ida_rr: T,

    /// value of tret previously returned by IDASolve
    ida_tretlast: T,
    /// cj value saved from last successful step
    ida_cjlast: T,

    /// test constant in Newton convergence test
    ida_eps_newt: T,

    /// coeficient of the Newton covergence test
    ida_epcon: T,

    // Limits
    /// max numer of convergence failures
    ida_maxncf: u64,
    /// max number of error test failures
    ida_maxnef: u64,
    /// max value of method order k:
    ida_maxord: usize,
    /// max number of internal steps for one user call
    ida_mxstep: u64,
    /// inverse of max. step size hmax (default = 0.0)
    ida_hmax_inv: T,

    //// Counters
    counters: IdaCounters,

    // Arrays for Fused Vector Operations
    ida_cvals: OVector<T, Const<MXORDP1>>,
    ida_dvals: OVector<T, Const<MAXORD_DEFAULT>>,

    /// tolerance scale factor (saved value)
    ida_tolsf: T,

    // Rootfinding Data
    /// number of components of g
    //ida_nrtfn: usize,
    /// array for root information
    //ida_iroots: OVector<T, Const<{ P::NROOTS }>>,
    /// array specifying direction of zero-crossing
    //ida_rootdir: Array1<u8>,
    /// nearest endpoint of interval in root search
    ida_tlo: T,
    /// farthest endpoint of interval in root search
    ida_thi: T,
    /// t return value from rootfinder routine
    ida_trout: T,

    /// saved array of g values at t = tlo
    //ida_glo: Array1<T>,
    /// saved array of g values at t = thi
    //ida_ghi: Array1<T>,
    /// array of g values at t = trout
    //ida_grout: Array1<T>,
    /// copy of tout (if NORMAL mode)
    ida_toutc: T,
    /// tolerance on root location
    ida_ttol: T,
    /// copy of parameter itask
    ida_taskc: IdaTask,
    /// flag showing whether last step had a root
    ida_irfnd: bool,
    /// counter for g evaluations
    ida_nge: usize,

    /// array with active/inactive event functions
    //ida_gactive: Array1<bool>,
    /// number of warning messages about possible g==0
    ida_mxgnull: usize,

    // Arrays for Fused Vector Operations
    //ida_zvecs: Array<T, Ix2>,
    /// Nonlinear Solver
    nls: NLS,

    /// Nonlinear problem
    nlp: ida_nls::IdaNLProblem<T, D, P, LS>,

    #[cfg(feature = "data_trace")]
    data_trace: std::fs::File,
}
