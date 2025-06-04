use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_reduce::<BITS, LIMBS>(criterion);
        bench_add::<BITS, LIMBS>(criterion);
        bench_mul::<BITS, LIMBS>(criterion);
        bench_pow::<BITS, LIMBS>(criterion);
        bench_inv::<BITS, LIMBS>(criterion);
    });
}

fn bench_reduce<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("reduce_mod/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, m)| black_box(black_box(a).reduce_mod(black_box(m))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_add<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::arbitrary(),
        Uint::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("add_mod/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b, m)| black_box(black_box(a).add_mod(black_box(b), black_box(m))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_mul<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::arbitrary(),
        Uint::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("mul_mod/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b, m)| black_box(black_box(a).mul_mod(black_box(b), black_box(m))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_pow<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::arbitrary(),
        Uint::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("pow_mod/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b, m)| black_box(black_box(a).pow_mod(black_box(b), black_box(m))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_inv<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("inv_mod/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, m)| black_box(black_box(a).inv_mod(black_box(m))),
            BatchSize::SmallInput,
        );
    });
}
