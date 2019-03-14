//! This simple example problem for IDA, due to Robertson, is from chemical kinetics, and consists
//! of the following three equations:
//!
//!      dy1/dt = -.04*y1 + 1.e4*y2*y3
//!      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
//!         0   = y1 + y2 + y3 - 1
//!
//! on the interval from t = 0.0 to t = 4.e10, with initial conditions: y1 = 1, y2 = y3 = 0.
//!
//! While integrating the system, we also use the rootfinding feature to find the points at which
//! y1 = 1e-4 or at which y3 = 0.01.
//!
//! The problem is solved with IDA using the DENSE linear solver, with a user-supplied Jacobian.
//! Output is printed at t = .4, 4, 40, ..., 4e10.

#[feature(test)]
use ida::*;

use ndarray::{array, prelude::*};

#[derive(Clone, Copy, Debug)]
struct Roberts {}

impl ModelSpec for Roberts {
    type Scalar = f64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        3
    }
}

impl Residual for Roberts {
    fn res<S1, S2, S3>(
        &self,
        _tres: Self::Scalar,
        yy: &ArrayBase<S1, Ix1>,
        yp: &ArrayBase<S2, Ix1>,
        resval: &mut ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {
        resval[0] = -0.04 * yy[0] + 1.0e4 * yy[1] * yy[2];
        resval[1] = -resval[0] - 3.0e7 * yy[1] * yy[1] - yp[1];
        resval[0] -= yp[0];
        resval[2] = yy[0] + yy[1] + yy[2] - 1.0;
    }
}

impl Jacobian for Roberts {
    fn jac<S1, S2, S3, S4>(
        &self,
        tt: Self::Scalar,
        cj: Self::Scalar,
        size: Self::Scalar,
        yy: &ArrayBase<S1, Ix1>,
        yp: &ArrayBase<S2, Ix1>,
        rr: &ArrayBase<S3, Ix1>,
        j: &mut ArrayBase<S4, Ix2>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::Data<Elem = Self::Scalar>,
        S4: ndarray::DataMut<Elem = Self::Scalar>,
    {
    }
}

#[test]
fn test_dense() {
    pretty_env_logger::init();

    let problem = Roberts {};

    let yy0 = array![1.0, 0.0, 0.0];
    let yp0 = array![-0.04, 0.04, 0.0];
    let t0 = 0.0;
    let mut tout = 0.4;
    let mut tret = 0.0;
    let mut yy = ndarray::Array::zeros(problem.model_size());
    let mut yp = ndarray::Array::zeros(problem.model_size());

    let mut ida: Ida<_, Dense<_>, Newton<_>> = Ida::new(problem, yy0, yp0);

    ida.solve(tout, &mut tret, &mut yy.view_mut(), &mut yp.view_mut(), IdaTask::Normal);
}
