/// hmax_inv default value
pub const HMAX_INV_DEFAULT: f64 = 0.0;
/// maxord default value
pub const MAXORD_DEFAULT: usize = 5;
/// max. number of N_Vectors in phi
pub const MXORDP1: usize = 6;
/// mxstep default value
pub const MXSTEP_DEFAULT: usize = 500;

//Control constants for tolerances
//--------------------------------

#[derive(Debug, Clone)]
pub enum ToleranceType {
    TolNN,
    TolSS,
    TolSV,
    TolWF,
}

//Algorithmic constants
//---------------------

/// max number of convergence failures allowed
pub const MXNCF: u32 = 10;
/// max number of error test failures allowed
pub const MXNEF: u32 = 10;
/// max. number of h tries in IC calc.
pub const MAXNH: u32 = 5;
/// max. number of J tries in IC calc.
pub const MAXNJ: u32 = 4;
/// max. Newton iterations in IC calc.
pub const MAXNI: u32 = 10;
/// Newton convergence test constant
pub const EPCON: f64 = 0.33;
/// max backtracks per Newton step in IDACalcIC
pub const MAXBACKS: u32 = 100;
/// constant for updating Jacobian/preconditioner
pub const XRATE: f64 = 0.25;
