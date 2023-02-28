mod dense;
mod traits;

pub use dense::Dense;
pub use traits::{LProblem, LSolver};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("A singular matrix was encountered during a LU factorization (col {col})")]
    LUFactFail { col: usize },
}

#[derive(Debug)]
pub enum LSolverType {
    Direct,
    Iterative,
    MatrixIterative,
}
