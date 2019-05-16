use super::*;

#[cfg(feature = "data_trace")]
use serde::Serialize;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "data_trace", derive(Serialize))]
pub struct Dense<Scalar> {
    x: Scalar,

    pivots: Vec<usize>,
    last_flag: usize,
}

impl<Scalar> LSolver<Scalar> for Dense<Scalar>
where
    Scalar: num_traits::Float
        + num_traits::NumRef
        + num_traits::NumAssignRef
        + num_traits::Zero
        + std::fmt::Debug,
{
    fn new(size: usize) -> Self {
        Dense {
            x: Scalar::zero(),

            pivots: vec![0; size],
            last_flag: 0,
        }
    }

    fn get_type(&self) -> LSolverType {
        LSolverType::Direct
    }

    fn setup<S1>(&mut self, mat_a: ArrayBase<S1, Ix2>) -> Result<(), failure::Error>
    where
        S1: ndarray::DataMut<Elem = Scalar>,
    {
        use failure::format_err;
        // perform LU factorization of input matrix
        self.last_flag = dense_get_rf(mat_a, &mut self.pivots);

        if self.last_flag > 0 {
            Err(format_err!("LUFACT_FAIL"))
        } else {
            Ok(())
        }
    }

    fn solve<S1, S2, S3>(
        &self,
        mat_a: ArrayBase<S1, Ix2>,
        mut x: ArrayBase<S2, Ix1>,
        b: ArrayBase<S3, Ix1>,
        _tol: Scalar,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>,
    {
        // copy b into x
        x.assign(&b);

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
/// returns 0 if successful. Otherwise it encountered a zero diagonal element during the
/// factorization. In this case it returns the column index (numbered from one) at which it
/// encountered the zero.
fn dense_get_rf<Scalar, S1>(mut mat_a: ArrayBase<S1, Ix2>, p: &mut [usize]) -> usize
where
    Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef + num_traits::Zero,
    S1: ndarray::DataMut<Elem = Scalar>,
{
    let m = mat_a.rows();
    let n = mat_a.cols();

    assert!(m >= n, "Number of rows must be >= number of columns");
    assert!(
        p.len() == n,
        "Partition slice length must be equal to the number of columns"
    );

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
        p[k] = l;

        // check for zero pivot element
        if col_k[l] == Scalar::zero() {
            return k + 1;
        }

        // swap a(k,1:n) and a(l,1:n) if necessary

        if l != k {
            for i in 0..n {
                mat_a.swap([k, i], [l, i]);
            }
        }

        // Scale the elements below the diagonal in
        // column k by 1.0/a(k,k). After the above swap
        // a(k,k) holds the pivot element. This scaling
        // stores the pivot row multipliers a(i,k)/a(k,k)
        // in a(i,k), i=k+1, ..., m-1.

        let mult = mat_a[[k, k]].recip();
        for i in (k + 1)..m {
            mat_a[[i, k]] *= mult;
        }

        // row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1
        // row k is the pivot row after swapping with row l.
        // The computation is done one column at a time, column j=k+1, ..., n-1.
        for j in (k + 1)..n {
            let a_kj = mat_a[[k, j]];

            // a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)
            // a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k)

            if a_kj != Scalar::zero() {
                for i in (k + 1)..m {
                    let a_ik = mat_a[[i, k]];
                    mat_a[[i, j]] -= a_kj * a_ik;
                }
            }
        }
    }

    // return 0 to indicate success
    0
}

/// `dense_get_rs` solves the N-dimensional system A x = b using the LU factorization in A and the
/// pivot information in p computed in `dense_get_rf`. The solution x is returned in b. This routine
/// cannot fail if the corresponding call to `dense_get_rf` did not fail.
///
/// Does NOT check for a square matrix!
fn dense_get_rs<Scalar, S1, S2>(mat_a: ArrayBase<S1, Ix2>, p: &[usize], mut b: ArrayBase<S2, Ix1>)
where
    Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef,
    S1: ndarray::Data<Elem = Scalar>,
    S2: ndarray::DataMut<Elem = Scalar>,
{
    let n = mat_a.cols();

    // Permute b, based on pivot information in p
    for (k, &pk) in p.iter().enumerate().take(n) {
        if pk != k {
            b.swap([k], [pk]);
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
    b[0] /= mat_a[[0, 0]];
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use nearly_eq::assert_nearly_eq;

    #[test]
    fn test_get_rs1() {
        let mat_a = array![
            [1.0, 0.040000000000000001, -0.040655973218655501],
            [1.0, -9562.0329139608493, -0.99881984364015208],
            [1.0, -0.041880782326080723, 0.00070539909027303449]
        ]
        .reversed_axes();
        let mut b = array![
            -0.00000018658722011386564,
            0.0000001791760359416981,
            0.000000000000015432100042289676
        ];
        dense_get_rs(mat_a, &vec![2, 1, 2], b.view_mut());
        let expect = array![
            0.000010806109402745275,
            0.000000000028591564117644602,
            -0.000010806137978877292
        ];
        assert_eq!(b, expect);
    }

    #[test]
    fn test_get_rs2() {
        let mat_a = array![
            [1.0, 0.040000000000000001, -0.041180751793579905],
            [1.0, -9376.8756693193609, -0.99825358822328103],
            [1.0, -0.04272931434962135, 0.0012553747713712066],
        ]
        .reversed_axes();
        let mut b = array![
            -0.00000092446647014019954,
            0.0000009098297931611867,
            0.000000000000010769163338864018
        ];
        dense_get_rs(mat_a, &vec![2, 1, 2], b.view_mut());
        let expect = array![
            0.000012924954909363613,
            -0.000000000038131780122501411,
            -0.000012924916766814327
        ];
        assert_eq!(b, expect);
    }

    #[test]
    fn test_get_rf1() {
        let mut mat_a = array![
            [-0.09593473862037126, 0.040000000000000001, 1.0],
            [5274.5976183265557, -5485.2758397300222, 1.0],
            [0.035103714444140913, -0.035103714444140913, 1.0]
        ]
        .reversed_axes();

        let mut pivot = vec![0, 0, 0];
        let ret = dense_get_rf(mat_a.view_mut(), &mut pivot);

        let expect = array![
            [1.0, 0.040000000000000001, -0.09593473862037126],
            [1.0, -5485.3158397300222, -0.96160252338811314],
            [1.0, -0.075103714444140907, 0.058818531739205995]
        ]
        .reversed_axes();

        assert_eq!(mat_a, expect);
        assert_eq!(pivot, vec![2, 1, 2]);
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_get_rf2() {
        let mut mat_a = array![
            [-0.042361503587159809, 0.040000000000000001, 1.0],
            [9313.8399601148321, -9331.507477848012, 1.0],
            [0.0029441927049318833, -0.0029441927049318833, 1.0],
        ]
        .reversed_axes();

        let mut pivot = vec![0, 0, 0];
        let ret = dense_get_rf(mat_a.view_mut(), &mut pivot);

        let expect = array![
            [1.0, 0.040000000000000001, -0.042361503587159809],
            [1.0, -9331.5474778480129, -0.99810694246891751],
            [1.0, -0.042944192704931883, 0.0024427994145761397]
        ]
        .reversed_axes();

        assert_eq!(mat_a, expect);
        assert_eq!(pivot, vec![2, 1, 2]);
        assert_eq!(ret, 0);
    }

    #[test]
    fn test_dense1() {
        let mut mat_a = array![
            [5.0, 0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0, 1.0],
            [4.0, 5.0, 5.0, 5.0],
            [1.0, 6.0, 4.0, 5.0]
        ];
        let b = array![9.0, 16.0, 49.0, 45.0];
        let expected = array![1.0, 2.0, 3.0, 4.0];
        let mut dense = Dense::new(4);
        let mut x = Array1::zeros(4);
        dense.setup(mat_a.view_mut()).unwrap();
        dense.solve(mat_a, x.view_mut(), b, 0.0).unwrap();
        assert_nearly_eq!(x, expected);
    }
}
