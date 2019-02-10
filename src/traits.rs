//! Basic traits for problem specification

use ndarray::*;
//use ndarray_linalg::*;

//use num_traits::real::Real as Scalar;
//num_traits::float::Float + num_traits::float::FloatConst + num_traits::NumAssignRef + ScalarOperand

/// Model specification
pub trait ModelSpec: Clone {
    type Scalar: num_traits::float::Float;
    type Dim: Dimension;
    fn model_size(&self) -> <Ix1 as Dimension>::Pattern;
}

/// Core implementation for explicit schemes
pub trait IdaModel: ModelSpec {
    /// Calculate right hand side (rhs) of Explicit from current state
    fn residual<'a, S>(
        &mut self,
        v: &'a mut ArrayBase<S, Ix1>,
    ) -> &'a mut ArrayBase<S, Ix1>
    where
        S: DataMut<Elem = Self::Scalar>;

    /// Calculate the Jacobian
    fn jacobian<S>(
        &mut self,
        cj: Self::Scalar,
        yy: &ArrayView<S, Ix1>,
        yp: &ArrayView<S, Ix1>,
    ) -> ()
    where
        S: DataMut<Elem = Self::Scalar>;
}

pub trait NormRms<A, S, D>
where
    A: num_traits::float::Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    /// Weighted root-mean-square norm
    fn norm_wrms(&self, w: &ArrayBase<S, D>) -> A;
}

pub trait NormRmsMasked<A, S, D, B>
where
    A: num_traits::float::Float,
    S: Data<Elem = A>,
    D: Dimension,
    B: Data<Elem = bool>,
{
    /// Weighted, masked root-mean-square norm
    fn norm_wrms_masked(&self, w: &ArrayBase<S, D>, id: &ArrayBase<B, D>) -> A;
}

impl<A, S, D> NormRms<A, S, D> for ArrayBase<S, D>
where
    A: num_traits::float::Float,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn norm_wrms(&self, w: &ArrayBase<S, D>) -> A {
        ((self * w)
            .iter()
            .map(|x| x.powi(2))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(self.len()).unwrap())
        .sqrt()
    }
}

impl<A, S, D, B> NormRmsMasked<A, S, D, B> for ArrayBase<S, D>
where
    A: num_traits::float::Float,
    S: Data<Elem = A>,
    D: Dimension,
    B: Data<Elem = bool>,
{
    fn norm_wrms_masked(&self, w: &ArrayBase<S, D>, id: &ArrayBase<B, D>) -> A {
        let mask = id.map(|x| if *x { A::one() } else { A::zero() });
        ((self * w * mask)
            .iter()
            .map(|x| x.powi(2))
            .fold(A::zero(), |acc, x| acc + x)
            / A::from(self.len()).unwrap())
        .sqrt()
    }
}

pub fn N_VLinearCombination<A, S, SM, D>(
    c: &ArrayBase<S, D::Smaller>,
    x: &ArrayBase<S, D>,
    z: &mut ArrayBase<SM, D::Smaller>,
) where
    A: num_traits::float::Float + ScalarOperand + std::fmt::Debug,
    S: Data<Elem = A>,
    SM: DataMut<Elem = A>,
    D: Dimension + RemoveAxis,
{
    ndarray::Zip::from(z)
        .and(c)
        .and(x.gencolumns())
        .apply(|z, c, row| *z = row.sum());
}

pub fn N_VScale<A, S, SM, D>(c: A, x: &ArrayBase<S, D>, z: &mut ArrayBase<SM, D>)
where
    A: num_traits::float::Float + ScalarOperand,
    S: Data<Elem = A>,
    SM: DataMut<Elem = A>,
    D: Dimension,
{
    z.zip_mut_with(x, |z, xn| *z = c * (*xn));
}

#[test]
fn test_N_VLinearCombination() {
    let c: ArrayBase<_, Ix1> = array![2., 1., 1.];
    let x: ArrayBase<_, Ix2> = array![[1., 2., 3.], [4., 5., 6.], [5., 6., 7.]];
    let mut z = Array::zeros(c.raw_dim());

    N_VLinearCombination(&c, &x, &mut z);

    assert_eq!(z, aview1(&[11., 15., 19.]));
}

#[test]
fn test_N_VScale() //(N_Vector X, N_Vector Z, sunindextype local_length, int myid)
{
    const LENGTH: usize = 32;
    // Case 1: x = cx, VScaleBy

    // fill vector data
    let mut x = Array::from_elem(LENGTH, 0.5);

    //N_VScale(2.0, &mut x, &mut x);

    // X should be vector of +1
    //assert_eq!(x, Array::from_elem(LENGTH, 1.0));

    // Case 2: z = x, VCopy

    // fill vector data
    let mut x = Array::from_elem(LENGTH, -1.0);
    let mut z = Array::from_elem(LENGTH, 0.0);

    N_VScale(1.0, &x, &mut z);

    // Z should be vector of -1
    assert_eq!(z, Array::from_elem(LENGTH, -1.0));

    // Case 3: z = -x, VNeg

    // fill vector data
    let mut x = Array::from_elem(LENGTH, -1.0);
    let mut z = Array::from_elem(LENGTH, 0.0);

    N_VScale(-1.0, &x, &mut z);

    // Z should be vector of +1
    assert_eq!(z, Array::from_elem(LENGTH, 1.0));

    // Case 4: z = cx, All other cases

    // fill vector data
    let mut x = Array::from_elem(LENGTH, -0.5);
    let mut z = Array::from_elem(LENGTH, 0.0);

    N_VScale(2.0, &x, &mut z);

    /* Z should be vector of -1 */
    assert_eq!(z, Array::from_elem(LENGTH, -1.0));
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
