use crate::{Jacobian, ModelSpec, Residual};
use ndarray::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Roberts {}

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
        yy: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        mut resval: ArrayBase<S3, Ix1>,
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
        _tt: Self::Scalar,
        cj: Self::Scalar,
        yy: ArrayBase<S1, Ix1>,
        _yp: ArrayBase<S2, Ix1>,
        _rr: ArrayBase<S3, Ix1>,
        mut jac: ArrayBase<S4, Ix2>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::Data<Elem = Self::Scalar>,
        S4: ndarray::DataMut<Elem = Self::Scalar>,
    {
        jac[[0, 0]] = -0.04 - cj;
        jac[[0, 1]] = 1.0e4 * yy[2];
        jac[[0, 2]] = 1.0e4 * yy[1];

        jac[[1, 0]] = 0.04;
        jac[[1, 1]] = -1.0e4 * yy[2] - 6.0e7 * yy[1] - cj;
        jac[[1, 2]] = -1.0e4 * yy[1];

        jac[[2, 0]] = 1.0;
        jac[[2, 1]] = 1.0;
        jac[[2, 2]] = 1.0;
    }
}
