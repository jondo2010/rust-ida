use approx::assert_relative_eq;

use crate::{tol_control::TolControl, Ida};

use super::*;

#[test]
fn test1() {
    //Before
    let ida_phi = matrix![
         4.1295003522440181e-07, 1.6518008147114031e-12, 9.9999958704831304e-01;
         -6.4049734044789205e-08, -2.5619916159829551e-13, 6.4049990326726996e-08;
         2.1135440604995772e-08, 8.4541889872000439e-14, -2.1135525197726480e-08;
         -2.2351400807868742e-08, -8.9405756031743853e-14, 2.2351489636470618e-08;
         1.8323105973439385e-08, 7.3292641194159994e-14, -1.8323176512520801e-08;
         -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10;
    ]
    .transpose();
    let ida_psi = vector![
        6.6874844417638421e+08,
        1.4118022710390334e+09,
        1.8407375671333179e+09,
        1.8153920670983608e+09,
        2.1446764804714236e+09,
        2.6020582487631597e+07
    ];
    let ida_alpha = vector![
        1.0000000000000000e+00,
        4.7368421052631576e-01,
        3.6330461012857090e-01,
        4.0930763129879277e-01,
        3.9999999999999997e-01,
        3.6363636363636365e-01
    ];
    let ida_beta = vector![
        1.0000000000000000e+00,
        9.0000000000000002e-01,
        1.0841585634594841e+00,
        3.5332089881864119e+00,
        7.1999999999999993e+00,
        1.0285714285714285e+01
    ];
    let ida_sigma = vector![
        1.0000000000000000e+00,
        4.7368421052631576e-01,
        3.4418331485864612e-01,
        7.2268199139687761e-01,
        1.4222222222222223e+00,
        2.5858585858585861e+00
    ];
    let ida_gamma = vector![
        0.0000000000000000e+00,
        1.4953305816383288e-09,
        2.2036450676775371e-09,
        2.8236868704168917e-09,
        3.0437121109953610e-09,
        3.1823098347208659e-07
    ];
    let kk = 2;
    let kused = 2;
    let ns = 1;
    let hh = 6.6874844417638421e+08;
    let hused = 6.6874844417638421e+08;
    let cj = 2.2429958724574930e-09;
    let cjlast = 2.4672954597032423e-09;

    let mut ida = Ida::<f64, U3, _, _, nonlinear::Newton<f64, _>>::new(
        Dummy {},
        linear::Dense::new(),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    // Set preconditions:
    ida.ida_hh = hh;
    ida.ida_hused = hused;
    ida.ida_ns = ns;
    ida.ida_kused = kused;
    ida.ida_kk = kk;
    ida.ida_beta.copy_from(&ida_beta);
    ida.ida_alpha.copy_from(&ida_alpha);
    ida.ida_gamma.copy_from(&ida_gamma);
    ida.ida_sigma.copy_from(&ida_sigma);
    ida.ida_phi.copy_from(&ida_phi);
    ida.ida_psi.copy_from(&ida_psi);
    ida.ida_cjlast = cjlast;
    ida.nlp.lp.ida_cj = cj;

    // Call the function under test
    let ck = ida.set_coeffs();

    //--- IDASetCoeffs After
    let ck_expect = 0.3214285714285713969;
    let ida_phi = matrix![
         4.1295003522440181e-07, 1.6518008147114031e-12, 9.9999958704831304e-01;
         -6.4049734044789205e-08, -2.5619916159829551e-13, 6.4049990326726996e-08;
         2.0023048994206519e-08, 8.0092316720842518e-14, -2.0023129134688242e-08;
         -2.2351400807868742e-08, -8.9405756031743853e-14, 2.2351489636470618e-08;
         1.8323105973439385e-08, 7.3292641194159994e-14, -1.8323176512520801e-08;
         -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10;
    ]
    .transpose();
    let ida_psi = vector![
        6.6874844417638421e+08,
        1.3374968883527684e+09,
        2.0805507152154176e+09,
        1.8153920670983608e+09,
        2.1446764804714236e+09,
        2.6020582487631597e+07
    ];
    let ida_alpha = vector![
        1.0000000000000000e+00,
        5.0000000000000000e-01,
        3.2142857142857140e-01,
        4.0930763129879277e-01,
        3.9999999999999997e-01,
        3.6363636363636365e-01
    ];
    let ida_beta = vector![
        1.0000000000000000e+00,
        1.0000000000000000e+00,
        9.4736842105263153e-01,
        3.5332089881864119e+00,
        7.1999999999999993e+00,
        1.0285714285714285e+01
    ];
    let ida_sigma = vector![
        1.0000000000000000e+00,
        5.0000000000000000e-01,
        3.2142857142857140e-01,
        7.2268199139687761e-01,
        1.4222222222222223e+00,
        2.5858585858585861e+00
    ];
    let ida_gamma = vector![
        0.0000000000000000e+00,
        1.4953305816383288e-09,
        2.2429958724574930e-09,
        2.8236868704168917e-09,
        3.0437121109953610e-09,
        3.1823098347208659e-07
    ];
    let kk = 2;
    let kused = 2;
    let ns = 2;
    let hh = 6.6874844417638421e+08;
    let hused = 6.6874844417638421e+08;
    let cj = 2.2429958724574930e-09;
    let cjlast = 2.2429958724574930e-09;

    assert_relative_eq!(ida.ida_hh, hh);
    assert_relative_eq!(ida.ida_hused, hused);
    assert_eq!(ida.ida_ns, ns);
    assert_eq!(ida.ida_kused, kused);
    assert_eq!(ida.ida_kk, kk);
    assert_relative_eq!(ida.ida_beta, ida_beta);
    assert_relative_eq!(ida.ida_alpha, ida_alpha);
    assert_relative_eq!(ida.ida_gamma, ida_gamma);
    assert_relative_eq!(ida.ida_sigma, ida_sigma);
    assert_relative_eq!(ida.ida_phi, ida_phi);
    assert_relative_eq!(ida.ida_psi, ida_psi);
    assert_relative_eq!(ida.ida_cjlast, cjlast);
    assert_relative_eq!(ida.nlp.lp.ida_cj, cj);
    assert_relative_eq!(ck, ck_expect);
}

#[test]
fn test2() {
    let problem = Dummy {};
    let mut ida = Ida::<f64, U3, _, _, nonlinear::Newton<f64, _>>::new(
        problem,
        linear::Dense::new(),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    // Set preconditions:
    {
        let ida_phi = matrix![
            9.9992400889930733e-01, 3.5884428024527148e-05, 4.0106672668125017e-05;
            -1.3748619452022122e-05, 1.1636437126348729e-06, 1.2584975739367733e-05;
            1.7125607629565644e-09, -1.3178687286728842e-06, 1.3161561679729596e-06;
            2.1033954646845001e-10, 1.0217905523752639e-06, -1.0220008918107099e-06;
            -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06;
            -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07;
        ]
        .transpose();
        let ida_psi = vector![
            3.4384304814216195e-04,
            6.8768609628432390e-04,
            1.0315291444264857e-03,
            7.7938390297730776e-04,
            3.4639284576769232e-04,
            0.0000000000000000e+00
        ];
        let ida_alpha = vector![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.3333333333333337e-01,
            4.4444444444444442e-01,
            5.0000000000000000e-01,
            0.0000000000000000e+00
        ];
        let ida_beta = vector![
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            4.8000000000000007e+00,
            1.5000000000000000e+01,
            0.0000000000000000e+00
        ];
        let ida_sigma = vector![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.3333333333333337e-01,
            8.8888888888888884e-01,
            2.4380952380952383e+00,
            0.0000000000000000e+00
        ];
        let ida_gamma = vector![
            0.0000000000000000e+00,
            2.9083036734439079e+03,
            4.3624555101658616e+03,
            6.2549405772650898e+03,
            1.6001650180080363e+04,
            0.0000000000000000e+00
        ];
        let kk = 2;
        let kused = 2;
        let ns = 2;
        let hh = 3.4384304814216195e-04;
        let hused = 3.4384304814216195e-04;
        let cj = 4.3624555101658616e+03;
        let cjlast = 4.3624555101658616e+03;

        ida.ida_hh = hh;
        ida.ida_hused = hused;
        ida.ida_ns = ns;
        ida.ida_kused = kused;
        ida.ida_kk = kk;
        ida.ida_beta.copy_from(&ida_beta);
        ida.ida_alpha.copy_from(&ida_alpha);
        ida.ida_gamma.copy_from(&ida_gamma);
        ida.ida_sigma.copy_from(&ida_sigma);
        ida.ida_phi.copy_from(&ida_phi);
        ida.ida_psi.copy_from(&ida_psi);
        ida.ida_cjlast = cjlast;
        ida.nlp.lp.ida_cj = cj;
    }

    // Call the function under test
    let ck = ida.set_coeffs();

    {
        let ck_expect = 0.3333333333333334814;
        let ida_phi = matrix![
            9.9992400889930733e-01, 3.5884428024527148e-05, 4.0106672668125017e-05;
            -1.3748619452022122e-05, 1.1636437126348729e-06, 1.2584975739367733e-05;
            1.7125607629565644e-09, -1.3178687286728842e-06, 1.3161561679729596e-06;
            2.1033954646845001e-10, 1.0217905523752639e-06, -1.0220008918107099e-06;
            -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06;
            -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07;
        ]
        .transpose();
        let ida_psi = vector![
            3.4384304814216195e-04,
            6.8768609628432390e-04,
            1.0315291444264857e-03,
            7.7938390297730776e-04,
            3.4639284576769232e-04,
            0.0000000000000000e+00
        ];
        let ida_alpha = vector![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.3333333333333337e-01,
            4.4444444444444442e-01,
            5.0000000000000000e-01,
            0.0000000000000000e+00
        ];
        let ida_beta = vector![
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            4.8000000000000007e+00,
            1.5000000000000000e+01,
            0.0000000000000000e+00
        ];
        let ida_sigma = vector![
            1.0000000000000000e+00,
            5.0000000000000000e-01,
            3.3333333333333337e-01,
            8.8888888888888884e-01,
            2.4380952380952383e+00,
            0.0000000000000000e+00
        ];
        let ida_gamma = vector![
            0.0000000000000000e+00,
            2.9083036734439079e+03,
            4.3624555101658616e+03,
            6.2549405772650898e+03,
            1.6001650180080363e+04,
            0.0000000000000000e+00
        ];
        let kk = 2;
        let kused = 2;
        let ns = 3;
        let hh = 3.4384304814216195e-04;
        let hused = 3.4384304814216195e-04;
        let cj = 4.3624555101658616e+03;
        let cjlast = 4.3624555101658616e+03;

        assert_relative_eq!(ida.ida_hh, hh);
        assert_relative_eq!(ida.ida_hused, hused);
        assert_eq!(ida.ida_ns, ns);
        assert_eq!(ida.ida_kused, kused);
        assert_eq!(ida.ida_kk, kk);
        assert_relative_eq!(ida.ida_beta, ida_beta);
        assert_relative_eq!(ida.ida_alpha, ida_alpha);
        assert_relative_eq!(ida.ida_gamma, ida_gamma);
        assert_relative_eq!(ida.ida_sigma, ida_sigma);
        assert_relative_eq!(ida.ida_phi, ida_phi);
        assert_relative_eq!(ida.ida_psi, ida_psi);
        assert_relative_eq!(ida.ida_cjlast, cjlast);
        assert_relative_eq!(ida.nlp.lp.ida_cj, cj);
        assert_relative_eq!(ck, ck_expect);
    }
}
