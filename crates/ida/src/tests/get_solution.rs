use super::Dummy;
use crate::{tol_control::TolControl, Ida};
use approx::assert_relative_eq;
use nalgebra::{matrix, vector, Const, U3};

#[test]
fn test_get_solution() {
    // --- IDAGetSolution Before:
    let t = 3623118336.24244;
    let hh = 857870592.1885694;
    let tn = 3623118336.24244;
    let kused = 4;
    let hused = 428935296.0942847;

    let ida_phi = matrix![
        5.716499633245077e-07, 2.286601144610028e-12, 0.9999994283477499;
        -7.779233860067279e-08, -3.111697299545603e-13, 7.779264957586927e-08;
        2.339417551980491e-08, 9.35768837422748e-14, -2.33942692332846e-08;
        -9.503346432581604e-09, -3.801349575270522e-14, 9.503383895634436e-09;
        7.768373161310588e-09, 3.107357755532867e-14, -7.768407422476745e-09;
        -2.242367216194777e-10, -8.970915966733762e-16, 2.242247401239887e-10;
    ]
    .transpose();

    let ida_psi = vector![
        428935296.0942847,
        857870592.1885694,
        1072338240.235712,
        1286805888.282854,
        1501273536.329997,
        26020582.4876316
    ];

    //--- IDAGetSolution After:
    let yret_expect = vector![
        5.716499633245077e-07,
        2.286601144610028e-12,
        0.9999994283477499
    ];
    let ypret_expect = vector![
        -1.569167478317552e-16,
        -6.276676917262037e-22,
        1.569173718962504e-16
    ];

    let problem = Dummy {};
    let mut ida = Ida::<f64, U3, _, _, nonlinear::Newton<f64, _>>::new(
        problem,
        linear::Dense::new(),
        &vector![0., 0., 0.],
        &vector![0., 0., 0.],
        TolControl::new_ss(1e-4, 1e-4),
    );

    ida.ida_hh = hh;
    ida.nlp.ida_tn = tn;
    ida.ida_kused = kused;
    ida.ida_hused = hused;
    ida.ida_phi.copy_from(&ida_phi);
    ida.ida_psi.copy_from(&ida_psi);

    println!("{}", serde_json::to_string(&ida).unwrap());

    ida.get_solution(t).unwrap();

    assert_relative_eq!(ida.get_yy(), &yret_expect, epsilon = 1e-8);
    assert_relative_eq!(ida.get_yp(), &ypret_expect, epsilon = 1e-8);
}
