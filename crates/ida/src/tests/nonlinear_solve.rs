use crate::{tol_control::TolControl, Ida};

use super::*;

#[ignore = "nonlinear_solve() depends on a valid problem, which Dummy doesn't implement"]
#[test_log::test]
fn test1() {
    let mut ida = Ida::<f64, U3, _, _, nonlinear::Newton<f64, _>>::new(
        Dummy {},
        linear::Dense::new(),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    // Set preconditions:
    {
        let ida_delta = vector![
            1.8377421825933786e-09,
            -6.8337119985200103e-07,
            6.8153345773243943e-07
        ];
        let ida_ee = vector![
            4.1727139878938076e-11,
            2.1149917627362784e-07,
            -2.1154090341350681e-07
        ];
        let ida_ewt = vector![
            9.9998973798771240e+03,
            9.9637670594659017e+05,
            9.9469101754210191e+05
        ];
        let ida_yy = vector![
            9.9991026211759748e-01,
            3.6364700537310025e-05,
            5.3373181865225196e-05
        ];
        let ida_yp = vector![
            -3.9977143388587585e-02,
            4.0305282775288977e-04,
            3.9574090561052919e-02
        ];
        let ida_yypredict = vector![
            9.9989651717362982e-01,
            3.6161601850240890e-05,
            6.7321224520057807e-05
        ];
        let ida_yppredict = vector![
            -3.9971798676247110e-02,
            -1.5843981431024555e-03,
            4.1556196819751090e-02
        ];
        let ida_cj = 4.3624555101658616e+03;
        let ida_cjold = 4.3624555101658616e+03;
        let ida_cjratio = 1.0000000000000000e+00;
        let ida_ss = 5.8988157110219739e-03;
        let ida_eps_newt = 3.3000000000000002e-01;
        let ida_nst = 11;

        ida.counters.ida_nst = ida_nst;
        ida.nlp.lp.ida_cjold = ida_cjold;
        ida.nlp.lp.ida_cj = ida_cj;
        ida.nlp.ida_ss = ida_ss;
        ida.nlp.lp.ida_cjratio = ida_cjratio;
        ida.ida_delta = ida_delta;
        ida.ida_ee = ida_ee;
        ida.nlp.ida_ewt = ida_ewt;
        ida.ida_eps_newt = ida_eps_newt;
        ida.nlp.ida_yy = ida_yy;
        ida.nlp.ida_yypredict = ida_yypredict;
        ida.nlp.ida_yp = ida_yp;
        ida.nlp.ida_yppredict = ida_yppredict;
    }

    ida.nonlinear_solve().unwrap();

    {
        let ida_yy = vector![
            9.9989651723441231e-01,
            3.6526684773526134e-05,
            6.6956080814055887e-05
        ];
        let ida_yp = vector![
            -3.9971533515318521e-02,
            8.2598672507300415e-06,
            3.9963273647500658e-02
        ];
        let ida_yypredict = vector![
            9.9989651717362982e-01,
            3.6161601850240890e-05,
            6.7321224520057807e-05
        ];
        let ida_yppredict = vector![
            -3.9971798676247110e-02,
            -1.5843981431024555e-03,
            4.1556196819751090e-02
        ];
        let ida_delta = vector![
            0.0000000000000000e+00,
            0.0000000000000000e+00,
            0.0000000000000000e+00
        ];
        let ida_ee = vector![
            6.0782494622797157e-11,
            3.6508292328524681e-07,
            -3.6514370600191418e-07
        ];
        let ida_ewt = vector![
            9.9998973798771240e+03,
            9.9637670594659017e+05,
            9.9469101754210191e+05
        ];
        let ida_cj = 4.3624555101658616e+03;
        let ida_cjold = 4.3624555101658616e+03;
        let ida_cjratio = 1.0000000000000000e+00;
        let ida_ss = 5.8988157110219739e-03;
        let ida_epsNewt = 3.3000000000000002e-01;
        let ida_nst = 11;

        assert_eq!(ida.counters.ida_nst, ida_nst);
        assert_eq!(ida.nlp.lp.ida_cjold, ida_cjold);
        assert_eq!(ida.nlp.lp.ida_cj, ida_cj);
        //assert_eq!(ida.nlp.ida_ss, ida_ss);
        assert_eq!(ida.nlp.lp.ida_cjratio, ida_cjratio);
        assert_eq!(ida.ida_delta, ida_delta);
        assert_eq!(ida.ida_ee, ida_ee);
        assert_eq!(ida.nlp.ida_ewt, ida_ewt);
        assert_eq!(ida.ida_eps_newt, ida_epsNewt);
        assert_eq!(ida.nlp.ida_yy, ida_yy);
        assert_eq!(ida.nlp.ida_yypredict, ida_yypredict);
        assert_eq!(ida.nlp.ida_yp, ida_yp);
        assert_eq!(ida.nlp.ida_yppredict, ida_yppredict);
    }
}
