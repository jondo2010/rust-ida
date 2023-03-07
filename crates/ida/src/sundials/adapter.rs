//! Adapter methods for SUNDIALS types to nalgebra types

use nalgebra::{
    allocator::Allocator, DMatrixViewMut, DVectorViewMut, DefaultAllocator, DimName, Dyn, Matrix,
    VectorViewMut, ViewStorageMut, U1,
};

/// Create an nalgebra mutable view of a SUNDIALS dense matrix
pub unsafe fn dense_matrix_view<'a, R: DimName, C: DimName>(
    m: sundials_sys::SUNMatrix,
) -> Matrix<f64, R, C, ViewStorageMut<'a, f64, R, C, U1, C>>
where
    DefaultAllocator: Allocator<f64, R, C>,
{
    let rows = sundials_sys::SUNDenseMatrix_Rows(m) as usize;
    let cols = sundials_sys::SUNDenseMatrix_Columns(m) as usize;

    assert_eq!(R::dim(), rows, "Matrix row count mismatch");
    assert_eq!(C::dim(), cols, "Matrix column count mismatch");

    Matrix::from_data(ViewStorageMut::from_raw_parts(
        sundials_sys::SUNDenseMatrix_Data(m),
        (R::name(), C::name()),
        (U1, C::name()),
    ))
}

pub unsafe fn dense_matrix_view_dynamic<'a>(m: sundials_sys::SUNMatrix) -> DMatrixViewMut<'a, f64> {
    let rows = sundials_sys::SUNDenseMatrix_Rows(m) as usize;
    let cols = sundials_sys::SUNDenseMatrix_Columns(m) as usize;

    Matrix::from_data(ViewStorageMut::from_raw_parts(
        sundials_sys::SUNDenseMatrix_Data(m),
        (Dyn(rows), Dyn(cols)),
        (U1, Dyn(cols)),
    ))
}

/// Create an nalgebra mutable view of a SUNDIALS vector
pub unsafe fn vector_view<'a, D: DimName>(v: sundials_sys::N_Vector) -> VectorViewMut<'a, f64, D> {
    let len = sundials_sys::N_VGetLength(v) as usize;

    assert_eq!(D::dim(), len, "Vector length mismatch");

    let data = sundials_sys::N_VGetArrayPointer(v);
    Matrix::from_data(ViewStorageMut::from_raw_parts(
        data,
        (D::name(), U1),
        (U1, D::name()),
    ))
}

/// Creates an nalgebra mutable view of a SUNDIALS vector with runtime-dynamic size
pub unsafe fn vector_view_dynamic<'a>(v: sundials_sys::N_Vector) -> DVectorViewMut<'a, f64> {
    let len = sundials_sys::N_VGetLength(v) as usize;
    let data = sundials_sys::N_VGetArrayPointer(v);
    Matrix::from_data(ViewStorageMut::from_raw_parts(
        data,
        (Dyn(len), U1),
        (U1, Dyn(len)),
    ))
}

#[test]
fn test_mat() {
    use nalgebra::*;

    let a = unsafe {
        let a = sundials_sys::SUNDenseMatrix(3, 3);
        sundials_sys::SUNMatZero(a);
        sundials_sys::SUNMatScaleAddI(1.0, a);
        a
    };

    let mut m = unsafe { dense_matrix_view::<U3, U3>(a) };

    m[(2, 0)] = 2.0;

    unsafe {
        //sundials_sys::SUNDenseMatrix_Print(a, libc_stdhandle::stdout() as *mut _);
    }

    println!("{m}");
}
