//! Basic traits for problem specification

use nalgebra::{Const, Dim, Matrix, RealField, Scalar, Storage, StorageMut, U1};
use num_traits::NumCast;

/// Core implementation for explicit schemes
pub trait IdaProblem<T> {
    /// The dimension of the problem
    type D: Dim;
    /// The number of roots
    type R: Dim;

    /// This function computes the problem residual for given values of the independent variable `tt`, state vector `yy`, and derivative `yp`.
    ///
    /// # Arguments
    ///
    /// * `tt` is the current value of the independent variable `t`.
    /// * `yy` is the current value of the dependent variable vector, `y(t)`.
    /// * `yp` is the current value of `y'(t)`.
    /// * `rr` is the output residual vector `F(t, y, y')`.
    ///
    /// An IDAResFn function type should return a value of 0 if successful, a positive value if a recoverable error occurred (e.g. yy has an illegal value), or a negative value if a nonrecoverable error occurred. In the last case, the integrator halts. If a recoverable error occurred, the integrator will attempt to correct and retry.
    fn res<SA, SB, SC>(
        &self,
        tt: T,
        yy: &Matrix<T, Self::D, U1, SA>,
        yp: &Matrix<T, Self::D, U1, SB>,
        rr: &mut Matrix<T, Self::D, U1, SC>,
    ) where
        SA: Storage<T, Self::D>,
        SB: Storage<T, Self::D>,
        SC: StorageMut<T, Self::D>;

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
    fn jac<SA, SB, SC, SD>(
        &self,
        tt: T,
        cj: T,
        yy: &Matrix<T, Self::D, U1, SA>,
        yp: &Matrix<T, Self::D, U1, SB>,
        rr: &Matrix<T, Self::D, U1, SC>,
        jac: &mut Matrix<T, Self::D, Self::D, SD>,
    ) where
        SA: Storage<T, Self::D>,
        SB: Storage<T, Self::D>,
        SC: Storage<T, Self::D>,
        SD: StorageMut<T, Self::D, Self::D>;

    fn root<SA, SB, SC>(
        &self,
        t: T,
        y: &Matrix<T, Self::D, U1, SA>,
        yp: &Matrix<T, Self::D, U1, SB>,
        gout: &mut Matrix<T, Self::R, U1, SC>,
    ) where
        SA: Storage<T, Self::D>,
        SB: Storage<T, Self::D>,
        SC: StorageMut<T, Self::R>;
}

/// Trait for real numbers in IDA
pub trait IdaReal: Scalar + RealField + Copy + NumCast {
    fn half() -> Self;
    fn quarter() -> Self;
    fn twothirds() -> Self;
    fn onept5() -> Self;
    fn two() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn twelve() -> Self;
    fn twenty() -> Self;
    fn hundred() -> Self;
    fn pt9() -> Self;
    fn pt99() -> Self;
    fn pt1() -> Self;
    fn pt05() -> Self;
    fn pt01() -> Self;
    fn pt001() -> Self;
    fn pt0001() -> Self;
    fn epsilon() -> Self;
}

impl IdaReal for f64 {
    fn half() -> Self {
        0.5
    }
    fn quarter() -> Self {
        0.25
    }
    fn twothirds() -> Self {
        0.667
    }
    fn onept5() -> Self {
        1.5
    }
    fn two() -> Self {
        2.0
    }
    fn four() -> Self {
        4.0
    }
    fn five() -> Self {
        5.0
    }
    fn ten() -> Self {
        10.0
    }
    fn twelve() -> Self {
        12.0
    }
    fn twenty() -> Self {
        20.0
    }
    fn hundred() -> Self {
        100.
    }
    fn pt9() -> Self {
        0.9
    }
    fn pt99() -> Self {
        0.99
    }
    fn pt1() -> Self {
        0.1
    }
    fn pt05() -> Self {
        0.05
    }
    fn pt01() -> Self {
        0.01
    }
    fn pt001() -> Self {
        0.001
    }
    fn pt0001() -> Self {
        0.0001
    }
    fn epsilon() -> Self {
        f64::EPSILON
    }
}
