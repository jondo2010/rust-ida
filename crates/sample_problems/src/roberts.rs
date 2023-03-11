use ida::IdaProblem;
use nalgebra::{Matrix, Storage, StorageMut, U1, U2, U3};

#[cfg(feature = "data_trace")]
use serde::Serialize;

/// This simple example problem for IDA, due to Robertson, is from chemical kinetics, and consists
/// of the following three equations:
///
/// ```math
/// dy1/dt = -.04*y1 + 1.e4*y2*y3
/// dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
/// 0      = y1 + y2 + y3 - 1
/// ```
///
/// on the interval from `t = 0.0..=4.e10`, with initial conditions: `y1 = 1`, `y2 = y3 = 0`.
///
/// While integrating the system, we also use the rootfinding feature to find the points at which
/// `y1 = 1e-4` or at which `y3 = 0.01`.
///
/// The problem is solved with IDA using the DENSE linear solver, with a user-supplied Jacobian.
/// Output is printed at `t = .4, 4, 40, ..., 4e10`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
pub struct Roberts {}

impl IdaProblem<f64> for Roberts {
    type D = U3;
    type R = U2;

    fn res<SA, SB, SC>(
        &self,
        _tt: f64,
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
        _tt: f64,
        cj: f64,
        yy: &Matrix<f64, Self::D, U1, SA>,
        _yp: &Matrix<f64, Self::D, U1, SB>,
        _rr: &Matrix<f64, Self::D, U1, SC>,
        jac: &mut Matrix<f64, Self::D, Self::D, SD>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: Storage<f64, Self::D>,
        SD: StorageMut<f64, Self::D, Self::D>,
    {
        // (row, col)
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
