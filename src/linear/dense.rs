use super::*;

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

impl<M> LSolver2<M> for Dense<M>
where
    M: ModelSpec,
    M::Scalar: num_traits::Float + num_traits::NumRef + num_traits::NumAssignRef + num_traits::Zero,
{
    fn new(size: usize) -> Self {}

    fn setup<S1>(&mut self, matA: &ArrayBase<S1, Ix2>) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
    {

    }

    fn solve<S1, S2, S3>(
        &mut self,
        matA: &mut ArrayBase<S1, Ix2>,
        x: &mut ArrayBase<S2, Ix1>,
        b: &ArrayBase<S3, Ix1>,
        tol: M::Scalar,
    ) -> Result<(), failure::Error>
    where
        S1: ndarray::Data<Elem = M::Scalar>,
        S2: ndarray::DataMut<Elem = M::Scalar>,
        S3: ndarray::Data<Elem = M::Scalar>,
    {

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
/// 2. If the unique LU factorization of A is given by PA = LU, where P is a permutation matrix,
///     L is a lower trapezoidal matrix with all 1's on the diagonal, and U is an upper triangular
///     matrix, then the upper triangular part of A (including its diagonal) contains U and the
///     strictly lower trapezoidal part of A contains the multipliers, I-L.
///
/// For square matrices (M = N), L is unit lower triangular.
///
/// returns 0 if successful. Otherwise it encountered a zero diagonal element during the factorization. In this case it returns the column index (numbered from one) at which it encountered the zero.
fn dense_get_rf<M, S1>(matA: &mut ArrayBase<S1, Ix2>, p: &mut Vec<usize>)
where
    M: ModelSpec,
    S1: ndarray::DataMut<Elem = M::Scalar>,
{
    //sunindextype i, j, k, l;
    //realtype *col_j, *col_k;
    //realtype temp, mult, a_kj;

    let (m, n) = matA.shape();

    /* k-th elimination step number */
    /*for (k=0; k < n; k++)*/
    {
        col_k = a[k];

        /* find l = pivot row number */
        l = k;
        //for (i=k+1; i < m; i++)
        {
            if (SUNRabs(col_k[i]) > SUNRabs(col_k[l])) {
                l = i;
            }
        }
        p[k] = l;

        /* check for zero pivot element */
        if (col_k[l] == ZERO) {
            return (k + 1);
        }

        /* swap a(k,1:n) and a(l,1:n) if necessary */

        if (l != k) {
            /*for (i=0; i<n; i++) */
            {
                temp = a[i][l];
                a[i][l] = a[i][k];
                a[i][k] = temp;
            }
        }

        /* Scale the elements below the diagonal in
         * column k by 1.0/a(k,k). After the above swap
         * a(k,k) holds the pivot element. This scaling
         * stores the pivot row multipliers a(i,k)/a(k,k)
         * in a(i,k), i=k+1, ..., m-1.
         */
        mult = ONE / col_k[k];
        //for(i=k+1; i < m; i++) {col_k[i] *= mult;}

        /* row_i = row_i - [a(i,k)/a(k,k)] row_k, i=k+1, ..., m-1 */
        /* row k is the pivot row after swapping with row l.      */
        /* The computation is done one column at a time,          */
        /* column j=k+1, ..., n-1.                                */

        /*for (j=k+1; j < n; j++)*/
        {
            col_j = a[j];
            a_kj = col_j[k];

            /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j)  */
            /* a_kj = a(k,j), col_k[i] = - a(i,k)/a(k,k) */

            if (a_kj != ZERO) {
                //for (i=k+1; i < m; i++)
                {
                    col_j[i] -= a_kj * col_k[i];
                }
            }
        }
    }

    /* return 0 to indicate success */

    return (0);
}
