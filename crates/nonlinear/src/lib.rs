use thiserror::Error;

pub mod newton;
pub mod norm_wrms;
pub mod traits;

pub use newton::Newton;
pub use traits::*;

#[derive(Debug, Error)]
pub enum Error {
    // Recoverable
    /// SUN_NLS_CONTINUE
    /// not converged, keep iterating
    #[error("")]
    Continue {},

    /// convergece failure, try to recover
    /// SUN_NLS_CONV_RECVR
    #[error("")]
    ConvergenceRecover {},

    // Unrecoverable
    /// illegal function input
    ///SUN_NLS_ILL_INPUT
    #[error("")]
    IllegalInput {},

    // failed NVector operation
    //SUN_NLS_VECTOROP_ERR
    #[error(transparent)]
    Linear(#[from] linear::Error),
}
