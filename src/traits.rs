//! Basic traits for problem specification

use ndarray::prelude::*;

/// Model specification
pub trait ModelSpec: Clone {
    type Scalar: num_traits::Float;
    type Dim: Dimension;
    fn model_size(&self) -> <Ix1 as Dimension>::Pattern;
}

pub trait Residual: ModelSpec {
    /// Nonlinear residual function
    fn residual<'a, S>(&mut self, v: &'a mut ArrayBase<S, Ix1>) -> &'a mut ArrayBase<S, Ix1>
    where
        S: ndarray::DataMut<Elem = Self::Scalar>;

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
        yy: &ArrayBase<S1, Ix1>,
        yp: &ArrayBase<S2, Ix1>,
        rr: &mut ArrayBase<S3, Ix1>,
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
    /// * `cj` is the scalar in the system Jacobian, proportional to the inverse of the step.
    /// * `size` (α in Eq. (2.5)).
    /// * `yy` is the current value of the dependent variable vector, `y(t)`.
    /// * `yp` is the current value of `y'(t)`.
    /// * `rr` is the current value of the residual vector `F(t, y, y')`.
    /// * `j` is the output (approximate) Jacobian matrix, `J = ∂F/∂y + cj ∂F/∂y'`.
    ///
    /// # Return value
    ///
    /// Should return 0 if successful, a positive value if a recoverable error occurred, or a negative value if a nonrecoverable error occurred. In the case of a recoverable eror return, the integrator will attempt to recover by reducing the stepsize, and hence changing α in
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
        S4: ndarray::DataMut<Elem = Self::Scalar>;
}

/// Core implementation for explicit schemes
pub trait IdaProblem: Residual + Jacobian {}

impl<T> IdaProblem for T where T: Residual + Jacobian {}

/// Constants for Ida
pub trait IdaConst {
    type Scalar: num_traits::Float;
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
    fn pt01() -> Self;
    fn pt001() -> Self;
    fn pt0001() -> Self;
}

impl IdaConst for f64 {
    type Scalar = Self;
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
    fn pt01() -> Self {
        0.01
    }
    fn pt001() -> Self {
        0.001
    }
    fn pt0001() -> Self {
        0.0001
    }
}

pub trait NormRms<A, S, D>
where
    A: num_traits::float::Float,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
{
    /// Weighted root-mean-square norm
    fn norm_wrms(&self, w: &ArrayBase<S, D>) -> A;
}

pub trait NormRmsMasked<A, S, D, B>
where
    A: num_traits::float::Float,
    S: ndarray::Data<Elem = A>,
    D: Dimension,
    B: ndarray::Data<Elem = bool>,
{
    /// Weighted, masked root-mean-square norm
    fn norm_wrms_masked(&self, w: &ArrayBase<S, D>, id: &ArrayBase<B, D>) -> A;
}

impl<A, S1, S2, D> NormRms<A, S1, D> for ArrayBase<S2, D>
where
    A: num_traits::float::Float,
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    D: Dimension,
{
    fn norm_wrms(&self, w: &ArrayBase<S1, D>) -> A {
        ((self * w)
            .iter()
            .map(|x| x.powi(2))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(self.len()).unwrap())
        .sqrt()
    }
}

impl<A, S1, S2, D, B> NormRmsMasked<A, S1, D, B> for ArrayBase<S2, D>
where
    A: num_traits::float::Float,
    S1: ndarray::Data<Elem = A>,
    S2: ndarray::Data<Elem = A>,
    D: Dimension,
    B: ndarray::Data<Elem = bool>,
{
    fn norm_wrms_masked(&self, w: &ArrayBase<S1, D>, id: &ArrayBase<B, D>) -> A {
        let mask = id.map(|x| if *x { A::one() } else { A::zero() });
        ((self * w * mask)
            .iter()
            .map(|x| x.powi(2))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(self.len()).unwrap())
        .sqrt()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_norm_wrms() {
        const LENGTH: usize = 32;
        let x = Array::from_elem(LENGTH, -0.5);
        let w = Array::from_elem(LENGTH, 0.5);
        assert_eq!(x.norm_wrms(&w), 0.25);
    }

    #[test]
    fn test_norm_wrms_masked() {
        const LENGTH: usize = 32;
        //fac = SUNRsqrt((realtype) (global_length - 1)/(global_length));
        let fac = (((LENGTH - 1) as f64) / (LENGTH as f64)).sqrt();

        let x = Array::from_elem(LENGTH, -0.5);
        let w = Array::from_elem(LENGTH, 0.5);
        // use all elements except one
        let mut id = Array::from_elem(LENGTH, true);
        id[LENGTH - 1] = false;

        // ans equals 1/4 (same as wrms norm)
        assert_eq!(x.norm_wrms_masked(&w, &id), fac * 0.5 * 0.5);
    }
}
