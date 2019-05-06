//! Basic traits for problem specification

use ndarray::prelude::*;
use serde::Serialize;

/// Model specification
pub trait ModelSpec: Clone {
    type Scalar: num_traits::Float + Serialize;
    type Dim: Dimension;
    fn model_size(&self) -> <Ix1 as Dimension>::Pattern;
}

pub trait Residual: ModelSpec {
    /// Nonlinear residual function
    //fn residual<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    //where
    //S: ndarray::DataMut<Elem = Self::Scalar>;

    /// This function computes the problem residual for given values of the independent variable t, state vector y, and derivative ˙y. Arguments tt is the current value of the independent variable.
    ///
    /// # Arguments
    ///
    /// * `tt` is the current value of the independent variable.
    /// * `yy` is the current value of the dependent variable vector, `y(t)`.
    /// * `yp` is the current value of `y'(t)`.
    /// * `rr` is the output residual vector `F(t, y, y')`.
    ///
    /// An IDAResFn function type should return a value of 0 if successful, a positive value if a recoverable error occurred (e.g. yy has an illegal value), or a negative value if a nonrecoverable error occurred. In the last case, the integrator halts. If a recoverable error occurred, the integrator will attempt to correct and retry.
    fn res<S1, S2, S3>(
        &self,
        tt: Self::Scalar,
        yy: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        rr: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>;
}

pub trait Jacobian: ModelSpec {
    /// This function computes the Jacobian matrix J of the DAE system (or an approximation to it)
    ///
    /// # Arguments
    ///
    /// * `tt` is the current value of the independent variable `t`.
    /// * `cj` is the scalar in the system Jacobian, proportional to the inverse of the step size (α in Eq. (2.5)).
    /// * `yy` is the current value of the dependent variable vector, `y(t)`.
    /// * `yp` is the current value of `y'(t)`.
    /// * `rr` is the current value of the residual vector `F(t, y, y')`.
    /// * `jac` is the output (approximate) Jacobian matrix, `J = ∂F/∂y + cj ∂F/∂y'`.
    ///
    /// # Return value
    ///
    /// Should return 0 if successful, a positive value if a recoverable error occurred, or a
    /// negative value if a nonrecoverable error occurred. In the case of a recoverable eror
    /// return, the integrator will attempt to recover by reducing the stepsize, and hence changing α in
    fn jac<S1, S2, S3, S4>(
        &self,
        tt: Self::Scalar,
        cj: Self::Scalar,
        yy: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        rr: ArrayBase<S3, Ix1>,
        jac: ArrayBase<S4, Ix2>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::Data<Elem = Self::Scalar>,
        S4: ndarray::DataMut<Elem = Self::Scalar>;
}

pub trait Root: ModelSpec {
    fn num_roots(&self) -> usize {
        0
    }

    fn root<S1, S2, S3>(
        &self,
        t: Self::Scalar,
        y: ArrayBase<S1, Ix1>,
        yp: ArrayBase<S2, Ix1>,
        gout: ArrayBase<S3, Ix1>,
    ) where
        S1: ndarray::Data<Elem = Self::Scalar>,
        S2: ndarray::Data<Elem = Self::Scalar>,
        S3: ndarray::DataMut<Elem = Self::Scalar>,
    {
    }
}

/// Core implementation for explicit schemes
pub trait IdaProblem: Residual + Jacobian + Root {}

impl<T> IdaProblem for T where T: Residual + Jacobian + Root {}
