use super::*;

use log::trace;

#[derive(Clone, Debug)]
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
        // find l = pivot row number
        let mut l = k;
        for i in (k + 1)..m {
            if mat_a[[i, k]].abs() > mat_a[[i, l]].abs() {
                l = i;
            }
        }
        p[k] = l;

        // check for zero pivot element
        if mat_a[[l, k]] == Scalar::zero() {
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

    return 0;
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
    for k in 0..n {
        let pk = p[k];
        if pk != k {
            b.swap([k], [pk]);
        }
    }

    // Solve Ly = b, store solution y in b
    for k in 0..(n - 1) {
        let bk = b[k];
        for i in (k + 1)..n {
            b[i] -= mat_a[[i, k]] * bk;
        }
    }

    // Solve Ux = y, store solution x in b
    for k in (1..n).rev() {
        b[k] /= mat_a[[k, k]];
        let bk = b[k];
        for i in 0..k {
            b[i] -= mat_a[[i, k]] * bk;
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
    fn test_get_rs() {
        // 2nd test
        let mat_a_decomp = array![
            [-46190.4, 0.0, 0.00865982],
            [-8.65981e-07, -46242.3, -0.00865981],
            [-2.16495e-05, -2.16252e-05, 1.0]
        ];
        let mut b = array![-3.4639284579585095e-08, 2.2532389959396826e-05, -0.0];
        dense_get_rs(mat_a_decomp, &vec![0, 1, 2], b.view_mut());
        let x_expect = array![
            7.5001558608301906e-13,
            -4.8726813621044346e-10,
            4.8651812062436036e-10
        ];
        dbg!(&b);
        assert_nearly_eq!(b, x_expect, 1e-15);
    }

    #[test]
    fn test_dense1() {
        let mut mat_a = array![
            [-46190.370416726822, 0.0, 0.0086598211441923072,],
            [0.04, -46242.289343591976, -0.0086598211441923072],
            [1.0, 1.0, 1.0]
        ];

        let mat_a_decomp = array![
            [-46190.370416726822, 0.0, 0.0086598211441923072],
            [
                -8.6598136449485772e-7,
                -46242.289343591976,
                -0.008659813644948576
            ],
            [
                -0.000021649534112371443,
                -0.000021625226912312786,
                1.00000000e+00
            ]
        ];

        let mut pivot = vec![0, 0, 0];
        let ret = dense_get_rf(mat_a.view_mut(), &mut pivot);

        assert_nearly_eq!(mat_a, mat_a_decomp, 1e-6);
        assert_eq!(pivot, vec![0, 1, 2]);
        assert_eq!(ret, 0);

        let mut b = array![-0.000000034639284579585095, 0.000022532389959396826, -0.0];
        let b_exp = array![
            7.5001558608301906e-13,
            -4.8726813621044346e-10,
            4.8651812062436036e-10,
        ];
        dense_get_rs(mat_a, &pivot, b.view_mut());
        assert_nearly_eq!(b, b_exp, 1e-9);

        // Now test using the LSolver interface
        let mut mat_a = array![[1., 1., 1.], [2., 1., -4.], [3., -4., 1.]];
        let mut dense = Dense::new(3);
        dense.setup(mat_a.view_mut()).unwrap();
        let mut x = Array::zeros(3);
        //dbg!(&mat_a);
        //dbg!(&dense);
        dense
            .solve(mat_a, x.view_mut(), array![0.25, 1.25, 1.0], 0.0)
            .unwrap();
        assert_eq!(x, array![0.375, 0., -0.125]);
    }

}
