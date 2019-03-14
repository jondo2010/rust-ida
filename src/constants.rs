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

/// default max number of nonlinear iterations
pub const MAXNLSIT: usize = 4;

/// Constants for Ida
pub trait IdaConst {
    type Scalar: num_traits::Float;
    fn half() -> Self;
    fn quarter() -> Self;
    fn twothirds() -> Self;
    fn onept5() -> Self;
    fn two() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn twelve() -> Self;
    fn twenty() -> Self;
    fn hundred() -> Self;
    fn pt9() -> Self;
    fn pt99() -> Self;
    fn pt1() -> Self;
    fn pt01() -> Self;
    fn pt001() -> Self;
    fn pt0001() -> Self;
}

impl IdaConst for f64 {
    type Scalar = Self;
    fn half() -> Self {
        0.5
    }
    fn quarter() -> Self {
        0.25
    }
    fn twothirds() -> Self {
        0.667
    }
    fn onept5() -> Self {
        1.5
    }
    fn two() -> Self {
        2.0
    }
    fn four() -> Self {
        4.0
    }
    fn five() -> Self {
        5.0
    }
    fn ten() -> Self {
        10.0
    }
    fn twelve() -> Self {
        12.0
    }
    fn twenty() -> Self {
        20.0
    }
    fn hundred() -> Self {
        100.
    }
    fn pt9() -> Self {
        0.9
    }
    fn pt99() -> Self {
        0.99
    }
    fn pt1() -> Self {
        0.1
    }
    fn pt01() -> Self {
        0.01
    }
    fn pt001() -> Self {
        0.001
    }
    fn pt0001() -> Self {
        0.0001
    }
}
