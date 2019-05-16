use super::*;

#[test]
fn test1() {
    #[rustfmt::skip]
        let ida_phi = array![
            [ 1.0570152037228958e-07, 4.2280612558303261e-13, 9.9999989429805680e-01, ],
            [ -3.3082196412696304e-08, -1.3232881828710420e-13, 3.3082328676061534e-08, ],
            [ 1.8675273859330434e-08, 7.4701128706323864e-14, -1.8675348801050254e-08, ],
            [ -1.9956501813542136e-08, -7.9826057803058290e-14, 1.9956580862443821e-08, ],
            [ 1.2851942479612096e-09, 5.1407743965993651e-15, -1.2851948368212051e-09, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
    let ida_gamma = array![
        0.0000000000000000e+00,
        2.6496925453439462e-10,
        3.8862188959925182e-10,
        8.0997073172076138e-10,
        3.0437121109953610e-09,
        3.1823098347208659e-07,
    ];
    let ida_yypredict = array![
        1.2565802218583172e-07,
        5.0263218338609083e-13,
        9.9999987434147597e-01,
    ];
    let ida_yppredict = array![
        1.5848602690328082e-18,
        6.3394566628399208e-24,
        -1.5848663595269871e-18,
    ];
    let kk = 2;

    let problem = Dummy {};
    let mut ida: Ida<_, linear::Dense<_>, nonlinear::Newton<_>, _> = Ida::new(
        problem,
        array![0., 0., 0.],
        array![0., 0., 0.],
        TolControlSS::new(1e-4, 1e-4),
    );

    // Set preconditions:
    ida.ida_kk = kk;
    ida.ida_phi.assign(&ida_phi);
    ida.ida_gamma.assign(&ida_gamma);
    ida.nlp.ida_yypredict.assign(&ida_yypredict);
    ida.nlp.ida_yppredict.assign(&ida_yppredict);

    // Call the function under test
    ida.predict();

    //--- IDAPredict After
    #[rustfmt::skip]
        let ida_phi = array![
            [ 1.0570152037228958e-07, 4.2280612558303261e-13, 9.9999989429805680e-01, ],
            [ -3.3082196412696304e-08, -1.3232881828710420e-13, 3.3082328676061534e-08, ],
            [ 1.8675273859330434e-08, 7.4701128706323864e-14, -1.8675348801050254e-08, ],
            [ -1.9956501813542136e-08, -7.9826057803058290e-14, 1.9956580862443821e-08, ],
            [ 1.2851942479612096e-09, 5.1407743965993651e-15, -1.2851948368212051e-09, ],
            [ -2.2423672161947766e-10, -8.9709159667337618e-16, 2.2422474012398869e-10, ],
        ];
    let ida_yypredict = array![
        9.1294597818923714e-08,
        3.6517843600225230e-13,
        9.9999990870503663e-01,
    ];
    let ida_yppredict = array![
        -1.5081447058360581e-18,
        -6.0325745419028739e-24,
        1.5081506275685795e-18,
    ];

    assert_eq!(ida.ida_kk, kk);
    assert_nearly_eq!(ida.ida_phi, ida_phi);
    assert_nearly_eq!(ida.nlp.ida_yypredict, ida_yypredict);
    assert_nearly_eq!(ida.nlp.ida_yppredict, ida_yppredict);
}
