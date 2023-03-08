use crate::{tol_control::TolControl, Ida};

use super::*;

#[test]
fn test_kused1() {
    let mut ida = Ida::new(
        Dummy {},
        linear::Dense::new(),
        nonlinear::Newton::new(0),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    let err_k = 0.0001987733462845937;
    let err_km1 = 0.0;

    // Set preconditions:
    {
        let ida_phi = matrix![
            1.0, 0.0, 0.0;
            -8.6598211441923077e-07, 8.6598211441923077e-07, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
            0.0, 0.0, 0.0;
        ]
        .transpose();
        let ida_ee = vector![
            7.5001558608301906e-13,
            -4.8726813621044346e-10,
            4.8651812062436036e-10
        ];
        let ida_ewt = vector![
            9.9990000999900003e+03,
            1.0000000000000000e+06,
            1.0000000000000000e+06
        ];
        let kk = 1;
        let kused = 0;
        let knew = 1;
        let phase = 0;
        let hh = 2.1649552860480770e-05;
        let hused = 0.0000000000000000e+00;
        let rr = 0.0000000000000000e+00;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 0;
        let maxord = 5;

        ida.counters.ida_nst = nst;
        ida.ida_kk = kk;
        ida.ida_hh = hh;
        ida.ida_rr = rr;
        ida.ida_kused = kused;
        ida.ida_hused = hused;
        ida.ida_knew = knew;
        ida.limits.ida_maxord = maxord;
        ida.ida_phase = phase;
        ida.limits.ida_hmax_inv = hmax_inv;
        ida.ida_ee.copy_from(&ida_ee);
        ida.ida_phi.copy_from(&ida_phi);
        ida.nlp.ida_ewt.copy_from(&ida_ewt);
    }

    ida.complete_step(err_k, err_km1);

    let ida_phi = matrix![
         9.9999913401863560e-01, 8.6549484628302034e-07, 4.8651812062436036e-10;
         -8.6598136440364466e-07, 8.6549484628302034e-07, 4.8651812062436036e-10;
         7.5001558608301906e-13, -4.8726813621044346e-10, 4.8651812062436036e-10;
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
        0.0, 0.0, 0.0;
    ]
    .transpose();
    let ida_ee = vector![
        7.5001558608301906e-13,
        -4.8726813621044346e-10,
        4.8651812062436036e-10
    ];
    let ida_ewt = vector![
        9.9990000999900003e+03,
        1.0000000000000000e+06,
        1.0000000000000000e+06
    ];
    let kk = 1;
    let kused = 1;
    let knew = 1;
    let phase = 0;
    let hh = 2.1649552860480770e-05;
    let hused = 2.1649552860480770e-05;
    let rr = 0.0000000000000000e+00;
    let hmax_inv = 0.0000000000000000e+00;
    let nst = 1;
    let maxord = 5;

    assert_eq!(ida.counters.ida_nst, nst);
    assert_eq!(ida.ida_kk, kk);
    assert_eq!(ida.ida_hh, hh);
    assert_eq!(ida.ida_rr, rr);
    assert_eq!(ida.ida_kused, kused);
    assert_eq!(ida.ida_hused, hused);
    assert_eq!(ida.ida_knew, knew);
    assert_eq!(ida.limits.ida_maxord, maxord);
    assert_eq!(ida.ida_phase, phase);
    assert_eq!(ida.limits.ida_hmax_inv, hmax_inv);
    assert_eq!(ida.ida_ee, ida_ee);
    assert_eq!(ida.nlp.ida_ewt, ida_ewt);
    assert_eq!(ida.ida_phi, ida_phi);
}

#[test]
fn test_kused2() {
    let mut ida = Ida::new(
        Dummy {},
        linear::Dense::new(),
        nonlinear::Newton::new(0),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    let err_k = 0.001339349356604325;
    let err_km1 = 0.003720519687081918;

    // Set preconditions:
    {
        ida.ida_phi.copy_from(
            &matrix![
                9.9999826803802172e-01, 1.7295310279504897e-06, 2.4309503863111873e-09;
                -1.7319612278663124e-06, 1.7280723633349389e-06, 3.8888645313736536e-09;
                2.2514114651871690e-12, -4.3759938466525865e-09, 4.3737424351873994e-09;
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00;
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00;
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00;
            ]
            .transpose(),
        );
        ida.ida_ee.copy_from(&vector![
            -4.2122294839452673e-13,
            -2.4605442771223734e-09,
            2.4609655000707684e-09
        ]);
        ida.nlp.ida_ewt.copy_from(&vector![
            9.9990174161763662e+03,
            9.9982707680480811e+05,
            9.9999975690502045e+05
        ]);

        ida.ida_kk = 2;
        ida.ida_kused = 1;
        ida.ida_knew = 2;
        ida.ida_phase = 0;
        ida.ida_hh = 4.3299105720961540e-05;
        ida.ida_hused = 2.1649552860480770e-05;
        ida.ida_rr = 0.0000000000000000e+00;
        ida.limits.ida_hmax_inv = 0.0000000000000000e+00;
        ida.counters.ida_nst = 2;
        ida.limits.ida_maxord = 5;
    }

    ida.complete_step(err_k, err_km1);

    #[rustfmt::skip]
        let ida_phi = matrix![
            9.9999653607862404e-01, 3.4507668531616537e-06, 1.3154522852943008e-08;
            -1.7319593976777956e-06, 1.7212358252111640e-06, 1.0723572466631820e-08;
            1.8301885167926423e-12, -6.8365381237749594e-09, 6.8347079352581675e-09;
            -4.2122294839452673e-13, -2.4605442771223734e-09, 2.4609655000707684e-09;
            0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00;
            0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00;
        ].transpose();
    let ida_ee = vector![
        -4.2122294839452673e-13,
        -2.4605442771223734e-09,
        2.4609655000707684e-09
    ];
    let ida_ewt = vector![
        9.9990174161763662e+03,
        9.9982707680480811e+05,
        9.9999975690502045e+05
    ];
    let kk = 3;
    let kused = 2;
    let knew = 2;
    let phase = 0;
    let hh = 8.6598211441923079e-05;
    let hused = 4.3299105720961540e-05;
    let rr = 0.0000000000000000e+00;
    let hmax_inv = 0.0000000000000000e+00;
    let nst = 3;
    let maxord = 5;

    assert_eq!(ida.counters.ida_nst, nst);
    assert_eq!(ida.ida_kk, kk);
    assert_eq!(ida.ida_hh, hh);
    assert_eq!(ida.ida_rr, rr);
    assert_eq!(ida.ida_kused, kused);
    assert_eq!(ida.ida_hused, hused);
    assert_eq!(ida.ida_knew, knew);
    assert_eq!(ida.limits.ida_maxord, maxord);
    assert_eq!(ida.ida_phase, phase);
    assert_eq!(ida.limits.ida_hmax_inv, hmax_inv);
    assert_eq!(ida.ida_ee, ida_ee);
    assert_eq!(ida.ida_phi, ida_phi);
    assert_eq!(ida.nlp.ida_ewt, ida_ewt);
}

#[test]
fn test3() {
    let mut ida = Ida::new(
        Dummy {},
        linear::Dense::new(),
        nonlinear::Newton::new(0),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    let err_k = 0.04158869255496026;
    let err_km1 = 0.0434084688121079;
    // Set preconditions:
    {
        let ida_phi = matrix![
            9.9989651723441231e-01, 3.6526684773526134e-05, 6.6956080814055887e-05;
            -1.3744883185162313e-05, 1.6198423621611769e-07, 1.3582898948830698e-05;
            1.8985246772161757e-09, -3.1828827656675421e-07, 3.1638975173052525e-07;
            6.0782494622797157e-11, 3.6508292328524681e-07, -3.6514370600191418e-07;
            -1.3875550771817554e-10, 1.3559268269012917e-06, -1.3557880688400603e-06;
            -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07;
        ]
        .transpose();
        let ida_ee = vector![
            -3.0358000609489579e-11,
            -2.0478050395116282e-07,
            2.0481086265121282e-07
        ];
        let ida_ewt = vector![
            1.0000034827777174e+04,
            9.9636062495315843e+05,
            9.9334892491071229e+05
        ];
        let kk = 3;
        let kused = 2;
        let knew = 2;
        let phase = 1;
        let hh = 3.4384304814216195e-04;
        let hused = 3.4384304814216195e-04;
        let rr = 1.7379509697128959e+00;
        let hmax_inv = 0.0000000000000000e+00;
        let nst = 12;
        let maxord = 5;

        ida.counters.ida_nst = nst;
        ida.ida_kk = kk;
        ida.ida_hh = hh;
        ida.ida_rr = rr;
        ida.ida_kused = kused;
        ida.ida_hused = hused;
        ida.ida_knew = knew;
        ida.limits.ida_maxord = maxord;
        ida.ida_phase = phase;
        ida.limits.ida_hmax_inv = hmax_inv;
        ida.ida_ee.copy_from(&ida_ee);
        ida.ida_phi.copy_from(&ida_phi);
        ida.nlp.ida_ewt.copy_from(&ida_ewt);
    }

    ida.complete_step(err_k, err_km1);

    let ida_phi = matrix![
        9.9988277428017636e-01, 3.6530683152509582e-05, 8.0695036671266405e-05;
        -1.3742954235991083e-05, 3.9983789834474729e-09, 1.3738955857210521e-05;
        1.9289491712294831e-09, -1.5798585723267022e-07, 1.5605690837982388e-07;
        3.0424494013307578e-11, 1.6030241933408400e-07, -1.6033284335070136e-07;
        -3.0358000609489579e-11, -2.0478050395116282e-07, 2.0481086265121282e-07;
        -1.1465196356066767e-10, 2.0021935974335382e-07, -2.0010470777979317e-07;
    ]
    .transpose();
    let ida_ee = vector![
        -3.0358000609489579e-11,
        -2.0478050395116282e-07,
        2.0481086265121282e-07
    ];
    let ida_ewt = vector![
        1.0000034827777174e+04,
        9.9636062495315843e+05,
        9.9334892491071229e+05
    ];
    let kk = 2;
    let kused = 3;
    let knew = 2;
    let phase = 1;
    let hh = 6.8768609628432390e-04;
    let hused = 3.4384304814216195e-04;
    let rr = 2.2575213239991561e+00;
    let hmax_inv = 0.0000000000000000e+00;
    let nst = 13;
    let maxord = 5;

    assert_eq!(ida.counters.ida_nst, nst);
    assert_eq!(ida.ida_kk, kk);
    assert_eq!(ida.ida_hh, hh);
    assert_eq!(ida.ida_rr, rr);
    assert_eq!(ida.ida_kused, kused);
    assert_eq!(ida.ida_hused, hused);
    assert_eq!(ida.ida_knew, knew);
    assert_eq!(ida.limits.ida_maxord, maxord);
    assert_eq!(ida.ida_phase, phase);
    assert_eq!(ida.limits.ida_hmax_inv, hmax_inv);
    assert_eq!(ida.ida_ee, ida_ee);
    assert_eq!(ida.ida_phi, ida_phi);
    assert_eq!(ida.nlp.ida_ewt, ida_ewt);
}
