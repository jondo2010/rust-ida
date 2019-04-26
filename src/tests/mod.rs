use super::*;

use ndarray::array;
use nearly_eq::*;
use tol_control::*;

mod set_coeffs;
mod test_error;
mod predict;
mod restore;
mod complete_step;
mod get_solution;
mod nonlinear_solve;

#[derive(Clone, Copy, Debug)]
struct Dummy {}

impl ModelSpec for Dummy {
    type Scalar = f64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        3
    }
}

impl Residual for Dummy {
    fn res<S1, S2, S3>(
        &self,
        _tres: Self::Scalar,
        _yy: ArrayBase<S1, Ix1>,
        _yp: ArrayBase<S2, Ix1>,
        mut _resval: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {
    }
}

impl Jacobian for Dummy {
    fn jac<S1, S2, S3, S4>(
        &self,
        _tt: Self::Scalar,
        _cj: Self::Scalar,
        _yy: ArrayBase<S1, Ix1>,
        _yp: ArrayBase<S2, Ix1>,
        _rr: ArrayBase<S3, Ix1>,
        mut _j: ArrayBase<S4, Ix2>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::Data<Elem = Self::Scalar>,
        S4: ndarray::DataMut<Elem = Self::Scalar>,
    {
    }
}
