/// SUNDIALS Copyright Start
/// Copyright (c) 2002-2019, Lawrence Livermore National Security
/// and Southern Methodist University.
/// All rights reserved.
///
/// See the top-level LICENSE and NOTICE files for details.
///
/// SPDX-License-Identifier: BSD-3-Clause
/// SUNDIALS Copyright End

use crate::{Jacobian, ModelSpec, Residual, Root};
use ndarray::prelude::*;

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

impl Root for Roberts {
    fn num_roots(&self) -> usize {
        2
    }

    /// Root function routine. Compute functions g_i(t,y) for i = 0,1.
    fn root<S1, S2, S3>(
        &self,
        _t: Self::Scalar,
        y: ArrayBase<S1, Ix1>,
        _yp: ArrayBase<S2, Ix1>,
        mut gout: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {
        gout[0] = y[0] - 0.0001;
        gout[1] = y[2] - 0.01;
    }
}
