use super::*;

#[derive(Clone, Debug)]
pub struct Dense<Scalar> {
    x: Scalar,

    pivots: Vec<usize>,
    last_flag: usize,
}

impl<Scalar> LSolver<Scalar> for Dense<Scalar>
where
    Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef + num_traits::Zero,
{
    fn new(size: usize) -> Self {
        use num_traits::identities::Zero;
        Dense {
            x: Scalar::zero(),

            pivots: Vec::new(),
            last_flag: 0,
        }
    }

    fn setup<S1>(&mut self, matA: &mut ArrayBase<S1, Ix2>) -> Result<(), failure::Error>
    where
        S1: ndarray::DataMut<Elem = Scalar>,
    {
        use failure::format_err;
        // perform LU factorization of input matrix
        self.last_flag = dense_get_rf(matA, &mut self.pivots);

        if self.last_flag > 0 {
            Err(format_err!("LUFACT_FAIL"))
        } else {
            Ok(())
        }
    }

    fn solve<S1, S2, S3>(
        &self,
        matA: &ArrayBase<S1, Ix2>,
        x: &mut ArrayBase<S2, Ix1>,
        b: &ArrayBase<S3, Ix1>,
        _tol: Scalar,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = Scalar>,
        S2: ndarray::DataMut<Elem = Scalar>,
        S3: ndarray::Data<Elem = Scalar>,
    {
        // copy b into x
        x.assign(&b);

        dense_get_rs(matA, &self.pivots, x);
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
fn dense_get_rf<Scalar, S1>(matA: &mut ArrayBase<S1, Ix2>, p: &mut Vec<usize>) -> usize
where
    Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef + num_traits::Zero,
    S1: ndarray::DataMut<Elem = Scalar>,
{
    use num_traits::{Float, Zero};
    let m = matA.rows();
    let n = matA.cols();

    // k-th elimination step number
    for col in 0..n {
        // find pivot = pivot row number
        let mut pivot = col;
        for row in (col + 1)..m {
            if matA[[row, col]].abs() > matA[[row, pivot]].abs() {
                pivot = row;
            }
        }
        p[col] = pivot;

        // check for zero pivot element
        if matA[[pivot, col]] == Scalar::zero() {
            return col + 1;
        }

        // swap a(k,1:n) and a(pivot,1:n) if necessary

        if pivot != col {
            for row in 0..m {
                matA.swap([row, col], [row, pivot]);
            }
        }

        // Scale the elements below the diagonal in
        // column k by 1.0/a(k,k). After the above swap
        // a(k,k) holds the pivot element. This scaling
        // stores the pivot row multipliers a(i,k)/a(k,k)
        // in a(i,k), i=k+1, ..., m-1.

        let mult = matA[[col, col]].recip();
        for row in (col + 1)..m {
            matA[[row, col]] *= mult;
        }

        // row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1
        // row k is the pivot row after swapping with row l.
        // The computation is done one column at a time, column j=k+1, ..., n-1.
        for j in (col + 1)..n {
            let a_kj = matA[[col, j]];

            // a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)
            // a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k)

            if a_kj != Scalar::zero() {
                for i in (col + 1)..m {
                    matA[[i, j]] -= a_kj * matA[[i, col]];
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
fn dense_get_rs<Scalar, S1, S2>(
    matA: &ArrayBase<S1, Ix2>,
    p: &Vec<usize>,
    b: &mut ArrayBase<S2, Ix1>,
) where
    Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef,
    S1: ndarray::Data<Elem = Scalar>,
    S2: ndarray::DataMut<Elem = Scalar>,
{
    let n = matA.cols();

    // Permute b, based on pivot information in p
    for k in 0..n {
        let pk = p[k];
        if pk != k {
            let tmp = b[k];
            b[k] = b[pk];
            b[pk] = tmp;
        }
    }

    // Solve Ly = b, store solution y in b
    for k in 0..(n - 1) {
        for i in (k + 1)..n {
            b[i] -= matA[[i, k]] * b[k];
        }
    }

    // Solve Ux = y, store solution x in b
    for k in (0..(n - 1)).rev() {
        b[k] /= matA[[k, k]];
        for i in 0..k {
            b[i] -= matA[[i, k]] * b[k];
        }
    }
    b[0] /= matA[[0, 0]];
}

#[test]
fn test_dense() {
    use ndarray::array;
    use nearly_eq::assert_nearly_eq;

    let mut a1 = array![
        [-46190.370416726822, 0.0, 0.0086598211441923072,],
        [0.04, -46242.289343591976, -0.0086598211441923072],
        [1.0, 1.0, 1.0]
    ];

    let mut a1_f = array![
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

    let mut p = vec![0, 0, 0];
    let ret = dense_get_rf(&mut a1, &mut p);

    assert_nearly_eq!(a1, a1_f, 1e-6);
    assert_eq!(p, vec![0, 1, 2]);
    assert_eq!(ret, 0);

    let mut b = array![-0.000000034639284579585095, 0.000022532389959396826, -0.0];
    let b_exp = array![
        7.5001558608301906e-13,
        -4.8726813621044346e-10,
        4.8651812062436036e-10,
    ];
    dense_get_rs(&a1, &p, &mut b);

    assert_nearly_eq!(b, b_exp, 1e-9);
}
