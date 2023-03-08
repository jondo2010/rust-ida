//! Linear solver for dense matrices, ported from the SUNDIALS suite.
//!
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimName, Dyn, Matrix, OVector, RealField, Scalar,
    Storage, StorageMut, U1,
};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

use crate::{Error, LSolver, LSolverType};

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(serialize = "OVector<usize, D>: Serialize"))
)]
#[cfg_attr(
    feature = "serde-serialize",
    serde(bound(
        deserialize = "OVector<usize, D>: Deserialize<'de>, DefaultAllocator: Allocator<usize, D>"
    ))
)]
#[derive(Clone, Debug)]
pub struct Dense<D: Dim>
where
    DefaultAllocator: Allocator<usize, D>,
{
    pivots: OVector<usize, D>,
}

impl<D> Dense<D>
where
    D: Dim,
    DefaultAllocator: Allocator<usize, D>,
{
    /// Creates a new dense linear solver.
    pub fn new() -> Self
    where
        D: DimName,
    {
        Dense {
            pivots: OVector::<usize, D>::zeros(),
        }
    }
}

impl Dense<Dyn> {
    pub fn new_dynamic(dim: usize) -> Self {
        Dense {
            pivots: OVector::<usize, Dyn>::zeros(dim),
        }
    }
}

impl<T, D> LSolver<T, D> for Dense<D>
where
    T: RealField + Copy,
    D: Dim,
    DefaultAllocator: Allocator<T, D> + Allocator<usize, D>,
{
    fn get_type(&self) -> LSolverType {
        LSolverType::Direct
    }

    fn setup<S>(&mut self, mat_a: &mut Matrix<T, D, D, S>) -> Result<(), Error>
    where
        S: StorageMut<T, D, D>,
    {
        // perform LU factorization of input matrix
        dense_get_rf(mat_a, &mut self.pivots).map_err(|e| Error::LUFactFail { col: e })
    }

    fn solve<SA, SB, SC>(
        &self,
        mat_a: &Matrix<T, D, D, SA>,
        x: &mut Matrix<T, D, U1, SB>,
        b: &Matrix<T, D, U1, SC>,
        _tol: T,
    ) -> Result<(), Error>
    where
        SA: StorageMut<T, D, D>,
        SB: StorageMut<T, D>,
        SC: StorageMut<T, D>,
    {
        // copy b into x
        x.set_column(0, &b);

        dense_get_rs(mat_a, &self.pivots, x);
        Ok(())
    }
}

/// Performs the LU factorization of the M by N dense matrix A.
///
/// This is done using standard Gaussian elimination with partial (row) pivoting. Note that this
/// applies only to matrices with M >= N and full column rank.
///
/// A successful LU factorization leaves the matrix A and the pivot array p with the following
/// information:
///
/// 1. p[k] contains the row number of the pivot element chosen at the beginning of elimination
///     step k, k=0, 1, ..., N-1.
///
/// 2. If the unique LU factorization of A is given by PA = LU, where P is a permutation matrix,
///     L is a lower trapezoidal matrix with all 1's on the diagonal, and U is an upper triangular
///     matrix, then the upper triangular part of A (including its diagonal) contains U and the
///     strictly lower trapezoidal part of A contains the multipliers, I-L.
///
/// For square matrices (M = N), L is unit lower triangular.
///
/// returns `Ok` if successful. Otherwise it encountered a zero diagonal element during the factorization. In this case
/// it returns the column index (numbered from one) at which it encountered the zero.
fn dense_get_rf<T, R, C, SA, SB>(
    mat_a: &mut Matrix<T, R, C, SA>,
    pivot: &mut Matrix<usize, C, U1, SB>,
) -> Result<(), usize>
where
    T: Scalar + RealField + Copy,
    R: Dim,
    C: Dim,
    SA: StorageMut<T, R, C>,
    SB: StorageMut<usize, C>,
    DefaultAllocator: Allocator<usize, C>,
{
    assert!(
        R::try_to_usize().unwrap() >= C::try_to_usize().unwrap(),
        "Number of rows must be >= number of columns"
    );

    let m = mat_a.nrows();
    let n = mat_a.ncols();

    // k-th elimination step number
    for k in 0..n {
        let col_k = mat_a.column(k);

        // find l = pivot row number
        let mut l = k;
        for i in (k + 1)..m {
            if col_k[i].abs() > col_k[l].abs() {
                l = i;
            }
        }
        pivot[k] = l;

        // check for zero pivot element
        if col_k[l] == T::zero() {
            return Err(k + 1);
        }

        // swap a(k,1:n) and a(l,1:n) if necessary

        if l != k {
            for i in 0..n {
                mat_a.swap((k, i), (l, i));
            }
        }

        // Scale the elements below the diagonal in column k by 1.0 / a[k,k]. After the above swap a[k,k] holds the pivot
        // element. This scaling stores the pivot row multipliers a(i,k)/a(k,k) in a(i,k), i=k+1, ..., m-1.
        let mult = mat_a[(k, k)].recip();
        for i in (k + 1)..m {
            mat_a[(i, k)] *= mult;
        }

        // row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1
        // row k is the pivot row after swapping with row l.
        // The computation is done one column at a time, column j=k+1, ..., n-1.
        for j in (k + 1)..n {
            let a_kj = mat_a[(k, j)];

            // a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)
            // a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k)

            if a_kj != T::zero() {
                for i in (k + 1)..m {
                    let a_ik = mat_a[(i, k)];
                    mat_a[(i, j)] -= a_kj * a_ik;
                }
            }
        }
    }

    Ok(())
}

/// `dense_get_rs` solves the N-dimensional system A x = b using the LU factorization in A and the
/// pivot information in p computed in `dense_get_rf`. The solution x is returned in b. This routine
/// cannot fail if the corresponding call to `dense_get_rf` did not fail.
///
/// Does NOT check for a square matrix!
fn dense_get_rs<T, R, C, SA, SB, SC>(
    mat_a: &Matrix<T, R, C, SA>,
    pivot: &Matrix<usize, C, U1, SB>,
    b: &mut Matrix<T, C, U1, SC>,
) where
    T: Scalar + RealField + Copy,
    R: Dim,
    C: Dim,
    SA: Storage<T, R, C>,
    SB: Storage<usize, C>,
    SC: StorageMut<T, C>,
    DefaultAllocator: Allocator<usize, R>,
{
    let n = mat_a.ncols();

    // Permute b, based on pivot information in p
    for (k, &pk) in pivot.iter().enumerate().take(n) {
        if pk != k {
            b.swap((k, 0), (pk, 0));
        }
    }

    // Solve Ly = b, store solution y in b
    for k in 0..(n - 1) {
        let col_k = mat_a.column(k);
        let bk = b[k];
        for i in (k + 1)..n {
            b[i] -= col_k[i] * bk;
        }
    }

    // Solve Ux = y, store solution x in b
    for k in (1..n).rev() {
        let col_k = mat_a.column(k);
        b[k] /= col_k[k];
        let bk = b[k];
        for i in 0..k {
            b[i] -= col_k[i] * bk;
        }
    }
    b[0] /= mat_a[(0, 0)];
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{matrix, vector, Vector4};

    use super::*;

    #[test]
    fn test_get_rs1() {
        let mat_a = matrix![
            1.0, 0.040000000000000001, -0.040655973218655501;
            1.0, -9562.0329139608493, -0.99881984364015208;
            1.0, -0.041880782326080723, 0.00070539909027303449;
        ]
        .transpose();

        let mut b = vector![
            -0.00000018658722011386564,
            0.0000001791760359416981,
            0.000000000000015432100042289676
        ];

        let pivot = vector![2, 1, 2];
        dense_get_rs(&mat_a, &pivot, &mut b);

        let expect = vector![
            0.000010806109402745275,
            0.000000000028591564117644602,
            -0.000010806137978877292
        ];
        assert_eq!(b, expect);
    }

    #[test]
    fn test_get_rs2() {
        let mat_a = matrix![
            1.0, 0.040000000000000001, -0.041180751793579905;
            1.0, -9376.8756693193609, -0.99825358822328103;
            1.0, -0.04272931434962135, 0.0012553747713712066;
        ]
        .transpose();

        let mut b = vector![
            -0.00000092446647014019954,
            0.0000009098297931611867,
            0.000000000000010769163338864018
        ];

        let pivot = vector![2, 1, 2];
        dense_get_rs(&mat_a, &pivot, &mut b);

        let expect = vector![
            0.000012924954909363613,
            -0.000000000038131780122501411,
            -0.000012924916766814327
        ];
        assert_eq!(b, expect);
    }

    #[test]
    fn test_get_rf1() {
        let mut mat_a = matrix![
            -0.09593473862037126, 0.040000000000000001, 1.0;
            5274.5976183265557, -5485.2758397300222, 1.0;
            0.035103714444140913, -0.035103714444140913, 1.0;
        ]
        .transpose();

        let mut pivot = vector![0, 0, 0];
        dense_get_rf(&mut mat_a, &mut pivot).unwrap();

        let expect = matrix![
            1.0, 0.040000000000000001, -0.09593473862037126;
            1.0, -5485.3158397300222, -0.96160252338811314;
            1.0, -0.075103714444140907, 0.058818531739205995;
        ]
        .transpose();

        assert_eq!(mat_a, expect);
        assert_eq!(pivot, vector![2, 1, 2]);
    }

    #[test]
    fn test_get_rf2() {
        let mut mat_a = matrix![
            -0.042361503587159809, 0.040000000000000001, 1.0;
            9313.8399601148321, -9331.507477848012, 1.0;
            0.0029441927049318833, -0.0029441927049318833, 1.0;
        ]
        .transpose();

        let mut pivot = vector![0, 0, 0];
        dense_get_rf(&mut mat_a, &mut pivot).unwrap();

        let expect = matrix![
            1.0, 0.040000000000000001, -0.042361503587159809;
            1.0, -9331.5474778480129, -0.99810694246891751;
            1.0, -0.042944192704931883, 0.0024427994145761397;
        ]
        .transpose();

        assert_eq!(mat_a, expect);
        assert_eq!(pivot, vector![2, 1, 2]);
    }

    #[test]
    fn test_dense1() {
        let mut mat_a = matrix![
            5.0, 0.0, 0.0, 1.0;
            2.0, 2.0, 2.0, 1.0;
            4.0, 5.0, 5.0, 5.0;
            1.0, 6.0, 4.0, 5.0;
        ];
        let b = vector![9.0, 16.0, 49.0, 45.0];
        let expected = vector![1.0, 2.0, 3.0, 4.0];
        let mut dense = Dense::new();
        let mut x = Vector4::zeros();
        dense.setup(&mut mat_a).unwrap();
        dense.solve(&mat_a, &mut x, &b, 0.0).unwrap();
        assert_relative_eq!(x, expected, max_relative = 1e-9);
    }
}
