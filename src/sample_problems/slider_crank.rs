//! Simulation of a slider-crank mechanism modelled with 3 generalized coordinates: crank angle,
//! connecting bar angle, and slider location. The mechanism moves under the action of a constant
//! horizontal force applied to the connecting rod and a spring-damper connecting the crank and
//! connecting rod.
//!
//! The equations of motion are formulated as a system of stabilized index-2 DAEs
//! (Gear-Gupta-Leimkuhler formulation).
//!
//! Original Author: Radu Serban @ LLNL
//! -----------------------------------------------------------------
//! SUNDIALS Copyright Start
//! Copyright (c) 2002-2019, Lawrence Livermore National Security
//! and Southern Methodist University.
//! All rights reserved.
//!
//! See the top-level LICENSE and NOTICE files for details.
//!
//! SPDX-License-Identifier: BSD-3-Clause
//! SUNDIALS Copyright End

use crate::{ModelSpec, Residual};
use ndarray::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct SlCrank {
    /// half-length of crank
    a: f64,
    /// crank moment of inertia
    J1: f64,
    /// moment of inertia of connecting rod
    J2: f64,
    /// mass of connecting rod
    m2: f64,
    /// spring constant
    k: f64,
    /// damper constant
    c: f64,
    /// spring free length
    l0: f64,
    /// external constant force
    F: f64,
}

impl SlCrank {
    fn force<S1, S2>(&self, yy: ArrayBase<S1, Ix1>, mut Q: ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = f64>,
        S2: ndarray::DataMut<Elem = f64>,
    {
        let q = &yy[0];
        let x = &yy[1];
        let p = &yy[2];

        let qd = &yy[3];
        let xd = &yy[4];
        let pd = &yy[5];

        let s1 = q.sin();
        let c1 = q.cos();
        let s2 = p.sin();
        let c2 = p.cos();
        let s21 = s2 * c1 - c2 * s1;
        let c21 = c2 * c1 + s2 * s1;

        let l2 =
            x.powi(2) - x * (c2 + self.a * c1) + (1.0 + self.a.powi(2)) / 4.0 + self.a * c21 / 2.0;
        let l = l2.sqrt();
        let mut ld = 2.0 * x * xd - xd * (c2 + self.a * c1) + x * (s2 * pd + self.a * s1 * qd)
            - self.a * s21 * (pd - qd) / 2.0;
        ld /= 2.0 * l;

        let f = self.k * (l - self.l0) + self.c * ld;
        let fl = f / l;

        Q[0] = -fl * self.a * (s21 / 2.0 + x * s1) / 2.0;
        Q[1] = fl * (c2 / 2.0 - x + self.a * c1 / 2.0) + self.F;
        Q[2] = -fl * (x * s2 - self.a * s21 / 2.0) / 2.0 - self.F * s2;
    }
}

impl Default for SlCrank {
    fn default() -> Self {
        Self {
            a: 0.5,
            J1: 1.0,
            m2: 1.0,
            J2: 2.0,
            k: 1.0,
            c: 1.0,
            l0: 1.0,
            F: 1.0,
        }
    }
}

impl ModelSpec for SlCrank {
    type Scalar = f64;
    type Dim = Ix1;
    fn model_size(&self) -> usize {
        10
    }
}

impl Residual for SlCrank {
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
