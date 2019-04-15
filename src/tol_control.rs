use ndarray::prelude::*;

pub trait TolControl<Scalar> {
    fn ewt_set<S1, S2>(&self, ycur: ArrayBase<S1, Ix1>, mut ewt: ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>;
}

/// specifies scalar relative and absolute tolerances.
#[derive(Clone, Debug)]
pub struct TolControlSS<Scalar> {
    /// relative tolerance
    ida_rtol: Scalar,
    /// scalar absolute tolerance
    ida_atol: Scalar,
}

impl<Scalar> TolControlSS<Scalar> {
    pub fn new(rtol: Scalar, atol: Scalar) -> Self {
        Self {
            ida_rtol: rtol,
            ida_atol: atol,
        }
    }
}

impl<Scalar> TolControl<Scalar> for TolControlSS<Scalar>
where
    Scalar: num_traits::Float,
{
    fn ewt_set<S1, S2>(&self, ycur: ArrayBase<S1, Ix1>, mut ewt: ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
    {
        ndarray::Zip::from(&mut ewt).and(&ycur).apply(|ewt, ycur| {
            *ewt = (self.ida_rtol * ycur.abs() + self.ida_atol).recip();
        });
    }
}

/// specifies scalar relative tolerance and a vector absolute tolerance (a potentially different
/// absolute tolerance for each vector component).
#[derive(Clone, Debug)]
pub struct TolControlSV<Scalar> {
    /// relative tolerance
    ida_rtol: Scalar,
    /// vector absolute tolerance
    ida_atol: Array1<Scalar>,
}

impl<Scalar> TolControlSV<Scalar> {
    pub fn new(rtol: Scalar, atol: Array1<Scalar>) -> Self {
        Self {
            ida_rtol: rtol,
            ida_atol: atol,
        }
    }
}

impl<Scalar> TolControl<Scalar> for TolControlSV<Scalar>
where
    Scalar: num_traits::Float,
{
    fn ewt_set<S1, S2>(&self, ycur: ArrayBase<S1, Ix1>, mut ewt: ArrayBase<S2, Ix1>)
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
    {
        ndarray::Zip::from(&mut ewt)
            .and(&ycur)
            .and(&self.ida_atol)
            .apply(|ewt, ycur, atol| {
                *ewt = (self.ida_rtol * ycur.abs() + *atol).recip();
            });
    }
}