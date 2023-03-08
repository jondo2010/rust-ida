use ida::IdaProblem;
use nalgebra::{Matrix, Storage, StorageMut, U1, U3};

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug)]
pub struct Dummy {}

impl IdaProblem<f64> for Dummy {
    type D = U3;

    type R = U3;

    fn res<SA, SB, SC>(
        &self,
        tt: f64,
        yy: &Matrix<f64, Self::D, U1, SA>,
        yp: &Matrix<f64, Self::D, U1, SB>,
        rr: &mut Matrix<f64, Self::D, U1, SC>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: StorageMut<f64, Self::D>,
    {
        todo!()
    }

    fn jac<SA, SB, SC, SD>(
        &self,
        tt: f64,
        cj: f64,
        yy: &Matrix<f64, Self::D, U1, SA>,
        yp: &Matrix<f64, Self::D, U1, SB>,
        rr: &Matrix<f64, Self::D, U1, SC>,
        jac: &mut Matrix<f64, Self::D, Self::D, SD>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: Storage<f64, Self::D>,
        SD: StorageMut<f64, Self::D, Self::D>,
    {
        todo!()
    }

    fn root<SA, SB, SC>(
        &self,
        t: f64,
        y: &Matrix<f64, Self::D, U1, SA>,
        yp: &Matrix<f64, Self::D, U1, SB>,
        gout: &mut Matrix<f64, Self::R, U1, SC>,
    ) where
        SA: Storage<f64, Self::D>,
        SB: Storage<f64, Self::D>,
        SC: StorageMut<f64, Self::R>,
    {
        todo!()
    }
}
