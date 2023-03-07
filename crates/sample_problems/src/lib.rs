//! Example problems for testing, benchmarks and demonstration

mod roberts;
//mod slider_crank;

pub use roberts::Roberts;
//pub use slider_crank::SlCrank;

pub mod dummy;

#[test]
fn test() {
    use nalgebra::*;
    use std::ops::Mul;

    let vec1 = Vector3::new(1.0, 2.0, 3.0);
    let vec2 = RowVector3::new(0.1, 0.2, 0.3);
    assert_eq!(vec1.tr_dot(&vec2), 1.4);

    let mat1 = Matrix2x3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let mat2 = Matrix3x2::new(0.1, 0.4, 0.2, 0.5, 0.3, 0.6);
    assert_eq!(mat1.tr_dot(&mat2), 9.1);

    let mat3 = matrix![
        1.0, 2.0, 3.0;
        4.0, 5.0, 6.0
    ];
    let vec3 = Vector2::new(1.0, 2.0);
    dbg!(&mat3);
    dbg!(&mat3.transpose());
    dbg!(mat3.tr_mul(&vec3));
}
