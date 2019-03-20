//! Lorenz three-variables system
//! https://en.wikipedia.org/wiki/Lorenz_system
//!

//#[feature(test)]
use ida::{linear::*, nonlinear::*, traits::*, *};

use ndarray::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Lorenz63 {
    pub p: f64,
    pub r: f64,
    pub b: f64,
}

impl Default for Lorenz63 {
    fn default() -> Self {
        Lorenz63 {
            p: 10.0,
            r: 28.0,
            b: 8.0 / 3.0,
        }
    }
}

impl Lorenz63 {
    pub fn new(p: f64, r: f64, b: f64) -> Self {
        Lorenz63 { p: p, r: r, b: b }
    }
}

impl ModelSpec for Lorenz63 {
    type Scalar = f64;
    type Dim = Ix1;

    fn model_size(&self) -> usize {
        3
    }
}

impl Residual for Lorenz63 {
    //fn residual<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    //where
    //    S: DataMut<Elem = Self::Scalar>,
    //{
    //    let x = v[0];
    //    let y = v[1];
    //    let z = v[2];
    //    v[0] = self.p * (y - x);
    //    v[1] = x * (self.r - z) - y;
    //    v[2] = x * y - self.b * z;
    //    v
    //}

    fn res<S1, S2, S3>(
        &self,
        tres: Self::Scalar,
        yy: &ArrayBase<S1, Ix1>,
        yp: &ArrayBase<S2, Ix1>,
        resval: &mut ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {

    }
}

impl Jacobian for Lorenz63 {
    fn jac<S1, S2, S3, S4>(
        &self,
        tt: Self::Scalar,
        cj: Self::Scalar,
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
        unimplemented!();
    }
}
