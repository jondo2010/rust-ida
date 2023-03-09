use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, Matrix, OVector, Storage, StorageMut, U1,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::traits::IdaReal;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        serialize = "T: Serialize, OVector<T, D>: Serialize",
        deserialize = "T: Deserialize<'de>, OVector<T, D>: Deserialize<'de>"
    ))
)]
//#[serde(tag = "type")]
#[derive(Clone, Debug, PartialEq)]
pub enum TolControl<T, D>
where
    D: Dim,
    DefaultAllocator: Allocator<T, D>,
    OVector<T, D>: PartialEq,
{
    /// Specifies scalar relative and absolute tolerances.
    ///
    /// The error weight vector `ewt` is set according to:
    ///
    /// ```math
    /// ewt[i] = 1 / (rtol * abs(ycur[i]) + atol), i=0,...,Neq-1
    /// ```
    SS {
        /// relative tolerance
        ida_rtol: T,
        /// scalar absolute tolerance
        ida_atol: T,
    },
    /// Specifies vector relative and absolute tolerances.
    ///
    /// The error weight vector `ewt` is set according to:
    ///
    /// ```math
    /// ewt[i] = 1 / (rtol[i] * abs(ycur[i]) + atol[i]), i=0,...,Neq-1
    /// ```
    SV {
        /// relative tolerance
        ida_rtol: T,
        /// vector absolute tolerance
        ida_atol: OVector<T, D>,
    },
}

impl<T, D> TolControl<T, D>
where
    T: IdaReal,
    D: Dim,
    DefaultAllocator: Allocator<T, D>,
{
    pub fn new_ss(rtol: T, atol: T) -> Self {
        Self::SS {
            ida_rtol: rtol,
            ida_atol: atol,
        }
    }

    pub fn new_sv(rtol: T, atol: OVector<T, D>) -> Self {
        Self::SV {
            ida_rtol: rtol,
            ida_atol: atol,
        }
    }

    /// This routine is responsible for loading the error weight vector `ewt`, according to itol, as follows:
    pub fn ewt_set<SA, SB>(&self, ycur: &Matrix<T, D, U1, SA>, ewt: &mut Matrix<T, D, U1, SB>)
    where
        SA: Storage<T, D>,
        SB: StorageMut<T, D>,
    {
        match self {
            Self::SS { ida_rtol, ida_atol } => {
                ycur.iter().zip(ewt.iter_mut()).for_each(|(ycur, ewt)| {
                    *ewt = (*ida_rtol * ycur.abs() + *ida_atol).recip();
                });
            }
            Self::SV { ida_rtol, ida_atol } => {
                ycur.iter()
                    .zip(ewt.iter_mut())
                    .zip(ida_atol.iter())
                    .for_each(|((ycur, ewt), atol)| {
                        *ewt = (*ida_rtol * ycur.abs() + *atol).recip();
                    });
            }
        }
    }
}
