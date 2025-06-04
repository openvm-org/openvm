use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_pow::<BITS, LIMBS>(criterion);
        bench_overflowing_pow::<BITS, LIMBS>(criterion);
    });
}

fn bench_pow<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::<BITS, LIMBS>::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("pow/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(b, e)| black_box(black_box(b).pow(black_box(e))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_overflowing_pow<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::<BITS, LIMBS>::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("overflowing_pow/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(b, e)| black_box(black_box(b).overflowing_pow(black_box(e))),
            BatchSize::SmallInput,
        );
    });
}
