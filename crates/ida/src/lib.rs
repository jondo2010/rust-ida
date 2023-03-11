//! The `ida` crate is a pure Rust port of the Implicit Differential-Algebraic solver from the Sundials suite.
//!
//! IDA is a general purpose solver for the initial value problem (IVP) for systems of differential-algebraic equations
//! (DAEs). The name IDA stands for Implicit Differential-Algebraic solver.

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, Dim, DimName, OMatrix, OVector, Storage,
    StorageMut, Vector, U1,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "serde-serialize")]
mod sundials;

mod constants;
mod error;
mod ida_io;
mod impl_new;
mod private;
#[cfg(test)]
mod tests;
mod tol_control;
mod traits;

use private::IdaNLProblem;

// Re-exports
pub use constants::*;
pub use error::Error;
pub use impl_new::*;
pub use linear;
pub use nonlinear;
pub use tol_control::TolControl;
pub use traits::{IdaProblem, IdaReal};

/// Counters
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
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
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize, PartialEq))]
#[serde(tag = "type")]
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

#[allow(dead_code)]
enum IdaConverged {
    Converged,
    NotConverged,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq)]
struct IdaLimits<T> {
    /// max numer of convergence failures
    ida_maxncf: usize,
    /// max number of error test failures
    ida_maxnef: usize,
    /// max value of method order k:
    ida_maxord: usize,
    /// max number of internal steps for one user call
    ida_mxstep: usize,
    /// inverse of max. step size hmax (default = 0.0)
    ida_hmax_inv: T,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, R>: Serialize, OVector<i8, R>: Serialize",
        deserialize = "T: Deserialize<'de>, OVector<T, R>: Deserialize<'de>, OVector<i8, R>: Deserialize<'de>"
    ))
)]
#[derive(Debug, PartialEq)]
struct IdaRootData<T, R: Dim>
where
    DefaultAllocator: Allocator<T, R> + Allocator<i8, R>,
    OVector<T, R>: PartialEq,
{
    /// number of components of g
    //ida_nrtfn: usize,
    /// array for root information
    ida_iroots: OVector<i8, R>,
    /// array specifying direction of zero-crossing
    ida_rootdir: OVector<i8, R>,
    /// nearest endpoint of interval in root search
    ida_tlo: T,
    /// farthest endpoint of interval in root search
    ida_thi: T,
    /// t return value from rootfinder routine
    ida_trout: T,
    /// saved array of g values at t = tlo
    ida_glo: OVector<T, R>,
    /// saved array of g values at t = thi
    ida_ghi: OVector<T, R>,
    /// array of g values at t = trout
    ida_grout: OVector<T, R>,
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
    ida_gactive: OVector<i8, R>,
    /// number of warning messages about possible g==0
    ida_mxgnull: usize,
}

impl<T, R: Dim> IdaRootData<T, R>
where
    DefaultAllocator: Allocator<T, R> + Allocator<i8, R>,
    OVector<T, R>: PartialEq,
{
    pub fn new() -> Self
    where
        T: IdaReal,
        R: DimName,
    {
        IdaRootData {
            ida_iroots: OVector::<i8, R>::zeros(),
            // Set default values for rootdir (both directions)
            ida_rootdir: OVector::<i8, R>::zeros(),
            ida_tlo: T::zero(),
            ida_thi: T::zero(),
            ida_trout: T::zero(),
            ida_glo: OVector::<T, R>::zeros(),
            ida_ghi: OVector::<T, R>::zeros(),
            ida_grout: OVector::<T, R>::zeros(),
            ida_toutc: T::zero(),
            ida_ttol: T::zero(),
            ida_taskc: IdaTask::Normal,
            ida_irfnd: false,
            ida_nge: 0,
            // Set default values for gactive (all active)
            ida_gactive: OVector::<i8, R>::identity(),
            ida_mxgnull: 1,
        }
    }

    pub fn num_roots(&self) -> usize {
        self.ida_iroots.len()
    }
}

/// This structure contains fields to keep track of problem state.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize,
            OVector<T, P::D>: Serialize,
            OVector<T, P::R>: Serialize,
            OVector<i8, P::R>: Serialize,
            OMatrix<T, P::D, Const<MXORDP1>>: Serialize,
            IdaNLProblem<T, P, LS>: Serialize,
            NLS: Serialize,
            LS: Serialize,
            P: Serialize",
        deserialize = "T: Deserialize<'de>,
            OVector<T, P::D>: Deserialize<'de>,
            OVector<T, P::R>: Deserialize<'de>,
            OVector<i8, P::R>: Deserialize<'de>,
            OMatrix<T, P::D, Const<MXORDP1>>: Deserialize<'de>,
            IdaNLProblem<T, P, LS>: Deserialize<'de>,
            NLS: Deserialize<'de>,
            LS: Deserialize<'de>,
            P: Deserialize<'de>"
    ))
)]
#[derive(Debug, PartialEq)]
pub struct Ida<T, P, LS, NLS>
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
    ida_setup_done: bool,
    tol_control: TolControl<T, P::D>,

    /// Suppress algebraic vars in local error tests
    ida_suppressalg: bool,

    // # Divided differences array and associated minor arrays
    /// phi = (maxord+1) arrays of divided differences
    ida_phi: OMatrix<T, P::D, Const<MXORDP1>>,
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

    // # Vectors
    /// residual vector
    ida_delta: OVector<T, P::D>,
    /// bit vector for diff./algebraic components
    ida_id: Option<OVector<T, P::D>>,
    /// vector of inequality constraint options
    ida_constraints: Option<OVector<T, P::D>>,
    /// accumulated corrections to y vector, but set equal to estimated local errors upon successful return
    ida_ee: OVector<T, P::D>,

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

    // # Step Data
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
    limits: IdaLimits<T>,

    //// Counters
    counters: IdaCounters,

    // Arrays for Fused Vector Operations
    ida_cvals: OVector<T, Const<MXORDP1>>,
    ida_dvals: OVector<T, Const<MAXORD_DEFAULT>>,

    /// tolerance scale factor (saved value)
    ida_tolsf: T,

    // Rootfinding Data
    roots: IdaRootData<T, P::R>,

    /// Nonlinear Solver
    nls: NLS,

    /// Nonlinear problem
    nlp: IdaNLProblem<T, P, LS>,

    #[cfg(feature = "data_trace")]
    data_trace: std::fs::File,
}

/// # Interpolated output and extraction functions
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
    /// IDAGetDky
    ///
    /// This routine evaluates the k-th derivative of `y(t)` as the value of the k-th derivative of the interpolating
    /// polynomial at the independent variable `t`, and stores the results in the vector `dky`. It uses the current
    /// independent variable value, `tn`, and the method order last used, `kused`.
    ///
    /// # Arguments
    /// * `t` - the independent variable value at which to evaluate the derivative
    /// * `k` - the order of the derivative to evaluate
    /// * `dky` - the vector in which to store the result
    ///
    /// # Return values
    /// * `Ok(())` - if successful
    /// * `Err(Error::BadTimeValue)` - if `t` is not within the interval of the last step taken
    /// * `Err(Error::BadK)` - if the requested `k` is not in the range `[0,order used]`
    pub fn get_dky<S>(&self, t: T, k: usize, dky: &mut Vector<T, P::D, S>) -> Result<(), Error>
    where
        S: StorageMut<T, P::D, U1>,
    {
        if k > self.ida_kused {
            Err(Error::BadK {
                kused: self.ida_kused,
            })?
        }

        self.check_t(t)?;

        // Initialize the c_j^(k) and c_k^(k-1)
        let mut cjk = OVector::<T, Const<MXORDP1>>::zeros();
        let mut cjk_1 = OVector::<T, Const<MXORDP1>>::zeros();

        let delt = t - self.nlp.ida_tn;
        let mut psij_1 = T::zero();

        for i in 0..k + 1 {
            let scalar_i = T::from(i as f64).unwrap();
            // The below reccurence is used to compute the k-th derivative of the solution:
            //    c_j^(k) = ( k * c_{j-1}^(k-1) + c_{j-1}^{k} (Delta+psi_{j-1}) ) / psi_j
            //
            //    Translated in indexes notation:
            //    cjk[j] = ( k*cjk_1[j-1] + cjk[j-1]*(delt+psi[j-2]) ) / psi[j-1]
            //
            //    For k=0, j=1: c_1 = c_0^(-1) + (delt+psi[-1]) / psi[0]
            //
            //    In order to be able to deal with k=0 in the same way as for k>0, the
            //    following conventions were adopted:
            //      - c_0(t) = 1 , c_0^(-1)(t)=0
            //      - psij_1 stands for psi[-1]=0 when j=1
            //                      for psi[j-2]  when j>1
            if i == 0 {
                cjk[i] = T::one();
            } else {
                //                                                i       i-1          1
                // c_i^(i) can be always updated since c_i^(i) = -----  --------  ... -----
                //                                               psi_j  psi_{j-1}     psi_1
                cjk[i] = cjk[i - 1] * scalar_i / self.ida_psi[i - 1];
                psij_1 = self.ida_psi[i - 1];
            }

            // update c_j^(i)
            //j does not need to go till kused
            for j in i + 1..=self.ida_kused - k {
                cjk[j] =
                    (scalar_i * cjk_1[j - 1] + cjk[j - 1] * (delt + psij_1)) / self.ida_psi[j - 1];
                psij_1 = self.ida_psi[j - 1];
            }

            // save existing c_j^(i)'s
            for j in i + 1..=self.ida_kused - k {
                cjk_1[j] = cjk[j];
            }
        }

        // Slice phi from k..kused+1
        let phi = self.ida_phi.columns(k, self.ida_kused + 1);

        // Slice cjk from k..kused+1
        let cvals = cjk.rows(k, self.ida_kused + 1);

        // Compute sum (c_j(t) * phi(t)) from j=k to j<=kused
        phi.mul_to(&cvals, dky);

        Ok(())
    }

    /// Check `t` for legality. Here tn - hused is t_{n-1}.
    fn check_t(&self, t: T) -> Result<(), Error> {
        let tfuzz = T::hundred()
            * T::epsilon()
            * (self.nlp.ida_tn.abs() + self.ida_hh.abs())
            * self.ida_hh.signum();
        let tp = self.nlp.ida_tn - self.ida_hused - tfuzz;
        Ok(if (t - tp) * self.ida_hh < T::zero() {
            Err(Error::BadTimeValue {
                t: t.to_f64().unwrap(),
                tdiff: (self.nlp.ida_tn - self.ida_hused).to_f64().unwrap(),
                tcurr: self.nlp.ida_tn.to_f64().unwrap(),
            })?
        })
    }

    /// Computes y based on the current prediction and given correction `ycor`.
    ///
    /// # Arguments
    /// * `ycor` - the correction to the predicted value of `y`
    /// * `y` - the vector in which to store the result
    pub fn compute_y<SA, SB>(&self, ycor: &Vector<T, P::D, SA>, y: &mut Vector<T, P::D, SB>)
    where
        SA: Storage<T, P::D>,
        SB: StorageMut<T, P::D>,
    {
        y.copy_from(&self.nlp.ida_yypredict);
        y.axpy(T::one(), ycor, T::one());
    }

    /// Computes y' based on the current prediction and given correction `ycor`.
    ///
    /// # Arguments
    /// * `ycor` - the correction to the predicted value of `y`
    /// * `yp` - the vector in which to store the result
    pub fn compute_yp<SA, SB>(&self, ycor: &Vector<T, P::D, SA>, yp: &mut Vector<T, P::D, SB>)
    where
        SA: Storage<T, P::D>,
        SB: StorageMut<T, P::D>,
    {
        yp.copy_from(&self.nlp.ida_yppredict);
        yp.axpy(T::one(), ycor, T::one());
    }
}
