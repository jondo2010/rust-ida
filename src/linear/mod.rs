use ndarray::prelude::*;

use crate::traits::ModelSpec;

pub trait LSolver<M: ModelSpec> {
    //IDA_mem->ida_linit  = idaLsInitialize;
    fn new() -> Self;

    /// idaLsSetup
    ///
    /// This calls the Jacobian evaluation routine, updates counters, and calls the LS 'setup' routine
    /// to prepare for subsequent calls to the LS 'solve' routine.
    fn ls_setup<S1, S2, S3>(
        &self,
        y: &ArrayBase<S1, Ix1>,
        yp: &ArrayBase<S2, Ix1>,
        r: &ArrayBase<S3, Ix1>,
        //N_Vector vt1,
        //N_Vector vt2,
        //N_Vector vt3
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {
        Ok(())
    }

    /// idaLsSolve
    ///
    /// This routine interfaces between IDA and the generic
    /// SUNLinearSolver object LS, by setting the appropriate tolerance
    /// and scaling vectors, calling the solver, accumulating
    /// statistics from the solve for use/reporting by IDA, and scaling
    /// the result if using a non-NULL SUNMatrix and cjratio does not
    /// equal one.
    fn ls_solve<S1, S2, S3>(
        &self,
        b: &mut ArrayBase<S1, Ix1>,
        weight: &ArrayBase<S2, Ix1>,
        ycur: &ArrayBase<S3, Ix1>,
        ypcur: &ArrayBase<S3, Ix1>,
        rescur: &ArrayBase<S3, Ix1>,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::DataMut<Elem = M::Scalar>,
        S2: ndarray::Data<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {
        Ok(())
    }
}

#[derive(Debug)]
pub struct Dense<M: ModelSpec> {
    x: M::Scalar,
}

impl<M> LSolver<M> for Dense<M>
where
    M: ModelSpec,
    M::Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef + num_traits::Zero,
{
    fn new() -> Self {
        use num_traits::identities::Zero;
        Dense {
            x: M::Scalar::zero(),
        }
    }
}
