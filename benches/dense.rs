use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_dense(cols: usize) {
    use ida::linear::*;
    use ndarray::{s, Array};
    use ndarray_rand::RandomExt;
    use nearly_eq::assert_nearly_eq;
    use rand::distributions::Uniform;

    // Fill A matrix with uniform random data in [0,1/cols]
    // Add anti-identity to ensure the solver needs to do row-swapping
    let mut mat_a = Array::random((cols, cols), Uniform::new(0., 1.0 / (cols as f64)));
    //+ Array::eye(COLS).slice_move(s![.., ..;-1]);
    let mat_a_original = mat_a.clone();

    // Fill x vector with uniform random data in [0,1]
    let mut x = Array::random(cols, Uniform::new(0.0, 1.0));
    let b = x.clone();

    let mut dense = Dense::new(cols);
    //println!("A (original) = {:#?}", &mat_a);
    dense.setup(mat_a.view_mut()).unwrap();
    //println!("A (factored) = {:#?}", &mat_a);

    //println!("x (original) = {:#?}", &x);
    dense.solve(mat_a, x.view_mut(), b.view(), 0.0).unwrap();
    //println!("x (computed) = {:#?}", &x);

    let b_comp = mat_a_original.dot(&x);

    let norm = (b - b_comp)
        .iter()
        .map(|x| x.powi(2))
        .fold(0.0, |acc, x| acc + x)
        / (cols as f64).sqrt();
    
    assert_nearly_eq!(norm, 0.0);

    //assert_nearly_eq!(&b_comp, &b, 1e-14);
    //println!("b (original) = {:#?}", &b);
    //println!("b (computed) = {:#?}", &b_comp);
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Dense solver 5", |b| b.iter(|| bench_dense(black_box(5))));
    c.bench_function("Dense solver 10", |b| b.iter(|| bench_dense(black_box(10))));
    c.bench_function("Dense solver 50", |b| b.iter(|| bench_dense(black_box(50))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
