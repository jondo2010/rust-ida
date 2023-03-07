use nalgebra::{
    allocator::SameShapeAllocator,
    constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    DefaultAllocator, Dim, Matrix, RealField, Scalar, Storage, U1,
};
use num_traits::NumCast;

pub trait NormWRMS<T, R1: Dim, C1: Dim, SA> {
    fn norm_wrms<R2, SB>(&self, rhs: &Matrix<T, R2, U1, SB>) -> T
    where
        T: RealField,
        R2: Dim,
        SB: Storage<T, R2, U1>,
        DefaultAllocator: SameShapeAllocator<T, R1, C1, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, U1>;
}

impl<T, R1: Dim, C1: Dim, SA> NormWRMS<T, R1, C1, SA> for Matrix<T, R1, C1, SA>
where
    T: Scalar + RealField + NumCast + Copy,
    SA: Storage<T, R1, C1>,
{
    fn norm_wrms<R2, SB>(&self, w: &Matrix<T, R2, U1, SB>) -> T
    where
        R2: Dim,
        SB: Storage<T, R2, U1>,
        DefaultAllocator: SameShapeAllocator<T, R1, C1, R2, U1>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, U1>,
    {
        let len = T::from(self.nrows()).unwrap();
        (self
            .component_mul(&w)
            .iter()
            .map(|x| x.powi(2))
            .fold(T::zero(), |acc, x| acc + x)
            / len)
            .sqrt()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_norm_wrms() {
        const LENGTH: usize = 32;

        let x = DVector::from_element(LENGTH, -0.5);
        let w = DVector::from_element(LENGTH, 0.5);

        let val = x.norm_wrms(&w);
        assert_eq!(val, 0.25);
    }

    #[test]
    #[cfg(feature = "disable")]
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
