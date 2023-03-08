use nalgebra::*;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::IdaProblem;

mod complete_step;
mod get_dky;
mod get_solution;
mod nonlinear_solve;
mod predict;
mod restore;
mod set_coeffs;
mod test_error;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct Dummy {}

impl IdaProblem<f64> for Dummy {
    type D = U3;
    type R = U3;

    fn res<SA, SB, SC>(
        &self,
        tt: f64,
        yy: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        rr: &mut Matrix<f64, Const<3>, U1, SC>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: StorageMut<f64, Const<3>>,
    {
    }

    fn jac<SA, SB, SC, SD>(
        &self,
        tt: f64,
        cj: f64,
        yy: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        rr: &Matrix<f64, Const<3>, U1, SC>,
        jac: &mut Matrix<f64, Const<3>, Const<3>, SD>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: Storage<f64, Const<3>>,
        SD: StorageMut<f64, Const<3>, Const<3>>,
    {
    }

    fn root<SA, SB, SC>(
        &self,
        t: f64,
        y: &Matrix<f64, Const<3>, U1, SA>,
        yp: &Matrix<f64, Const<3>, U1, SB>,
        gout: &mut Matrix<f64, Const<3>, U1, SC>,
    ) where
        SA: Storage<f64, Const<3>>,
        SB: Storage<f64, Const<3>>,
        SC: StorageMut<f64, Const<3>>,
    {
    }
}
