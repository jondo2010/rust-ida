use ndarray::prelude::*;

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
