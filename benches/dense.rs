use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ida::linear::*;
use ndarray::{s, Array};
use ndarray_rand::RandomExt;
use nearly_eq::assert_nearly_eq;
use rand::distributions::Uniform;

fn bench_dense(bencher: &mut criterion::Bencher, cols: usize) {
    // Fill A matrix with uniform random data in [0,1/cols]
    // Add anti-identity to ensure the solver needs to do row-swapping
    let mut mat_a = Array::random((cols, cols), Uniform::new(0., 1.0 / (cols as f64))) + Array::eye(cols).slice_move(s![.., ..;-1]);

    // Fill x vector with uniform random data in [0,1]
    let mut x = Array::random(cols, Uniform::new(0.0, 1.0));
    let b = x.clone();

    let mut dense = Dense::new(cols);

    bencher.iter(|| dense.setup(mat_a.clone()).unwrap());
    dense.setup(mat_a.view_mut()).unwrap();
    bencher.iter(|| dense.solve(mat_a.view(), x.view_mut(), b.view(), 0.0).unwrap());

    /*
    let b_comp = mat_a_original.dot(&x);
    let norm = (b - b_comp)
        .iter()
        .map(|x| x.powi(2))
        .fold(0.0, |acc, x| acc + x)
        / (cols as f64).sqrt();

    assert_nearly_eq!(norm, 0.0);
    */

    //assert_nearly_eq!(&b_comp, &b, 1e-14);
    //println!("b (original) = {:#?}", &b);
    //println!("b (computed) = {:#?}", &b_comp);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Dense solver 5", |b| bench_dense(b, black_box(5)));
    c.bench_function("Dense solver 10", |b| bench_dense(b, black_box(10)));
    c.bench_function("Dense solver 50", |b| bench_dense(b, black_box(50)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
