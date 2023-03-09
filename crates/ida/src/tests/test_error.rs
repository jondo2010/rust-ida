use approx::assert_relative_eq;

use crate::{tol_control::TolControl, Ida};

use super::*;

#[test]
fn test1() {
    let ck = 1.091414141414142;
    let suppressalg = 0;
    let kk = 5;
    let ida_phi = matrix![
         3.634565317158998e-05, 1.453878335134203e-10, 0.9999636542014404;
         -6.530333550677049e-06, -2.612329458968465e-11, 6.530359673556191e-06;
         1.946442728026142e-06, 7.786687275994346e-12, -1.946450515496441e-06;
         -8.097632208221231e-07, -3.239585549038764e-12, 8.097664556005615e-07;
         3.718130977075839e-07, 1.487573462300438e-12, -3.71814615793545e-07;
         -3.24421895454213e-07, -1.297915245220823e-12, 3.244230624265827e-07;
    ]
    .transpose();
    let ida_ee = vector![
        2.65787533317467e-07,
        1.063275845801634e-12,
        -2.657884288386138e-07
    ];
    let ida_ewt = vector![73343005.56993243, 999999.985461217, 9901.346408259429];
    let ida_sigma = vector![
        1.0,
        0.6666666666666666,
        0.6666666666666666,
        0.888888888888889,
        1.422222222222222,
        2.585858585858586
    ];
    let knew = 4;
    let err_k = 29.10297975314245;
    let err_km1 = 3.531162835377502;

    let mut ida = Ida::new(
        Dummy {},
        linear::Dense::new(),
        nonlinear::Newton::new(0),
        TolControl::new_ss(1e-4, 1e-4),
        0.0,
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
    );

    // Set preconditions:
    ida.ida_kk = kk;
    ida.ida_suppressalg = suppressalg > 0;
    ida.ida_phi.copy_from(&ida_phi);
    ida.ida_ee.copy_from(&ida_ee);
    ida.nlp.ida_ewt.copy_from(&ida_ewt);
    ida.ida_sigma.copy_from(&ida_sigma);

    // Call the function under test
    let (err_k_new, err_km1_new, _) = ida.test_error(ck).expect_err("Should be TestFail");

    assert_eq!(ida.ida_knew, knew);
    assert_relative_eq!(err_k_new, err_k);
    assert_relative_eq!(err_km1_new, err_km1, epsilon = 1e-12);
}

#[test]
fn test2() {
    //--- IDATestError Before:
    let ck = 0.2025812352167927;
    let suppressalg = 0;
    let kk = 4;
    let ida_phi = matrix![
        3.051237735052657e-05, 1.220531905117091e-10, 0.9999694875005963;
        -2.513114849098281e-06, -1.005308974226734e-11, 2.513124902721765e-06;
        4.500284453718991e-07, 1.800291970640913e-12, -4.500302448499092e-07;
        -1.366709389821433e-07, -5.467603693902342e-13, 1.366714866794709e-07;
        7.278821769100639e-08, 2.911981566628798e-13, -7.278850816613011e-08;
        -8.304741244343501e-09, -3.324587131187576e-14, 8.304772990651073e-09;
    ]
    .transpose();
    let ida_ee = vector![
        -2.981302228744271e-08,
        -1.192712676406388e-13,
        2.981313872620108e-08
    ];
    let ida_ewt = vector![76621085.31777237, 999999.9877946811, 9901.289220872719];
    let ida_sigma = vector![
        1.0,
        0.5,
        0.3214285714285715,
        0.2396514200444849,
        0.1941955227762807,
        2.585858585858586
    ];
    //--- IDATestError After:
    let knew = 4;
    let err_k = 0.2561137489433976;
    let err_km1 = 0.455601916633899;
    let nflag = true;

    let mut ida = Ida::new(
        Dummy {},
        linear::Dense::new(),
        nonlinear::Newton::new(0),
        TolControl::new_ss(1e-4, 1e-4),
        0.0,
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
    );

    // Set preconditions:
    ida.ida_kk = kk;
    ida.ida_suppressalg = suppressalg > 0;
    ida.ida_phi.copy_from(&ida_phi);
    ida.ida_ee.copy_from(&ida_ee);
    ida.nlp.ida_ewt.copy_from(&ida_ewt);
    ida.ida_sigma.copy_from(&ida_sigma);

    // Call the function under test
    let (err_k_new, err_km1_new, nflag_new) = ida.test_error(ck).unwrap();

    assert_eq!(ida.ida_knew, knew);
    assert_relative_eq!(err_k_new, err_k);
    assert_relative_eq!(err_km1_new, err_km1);
    assert_eq!(nflag_new, nflag);
}
