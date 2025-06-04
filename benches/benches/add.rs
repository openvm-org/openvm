use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_neg::<BITS, LIMBS>(criterion);
        bench_add::<BITS, LIMBS>(criterion);
        bench_sub::<BITS, LIMBS>(criterion);
    });
}

fn bench_neg<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = Uint::<BITS, LIMBS>::arbitrary();
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("neg/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |a| black_box(-black_box(a)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_add<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("add/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b)| black_box(black_box(a) + black_box(b)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_sub<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("sub/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b)| black_box(black_box(a) - black_box(b)),
            BatchSize::SmallInput,
        );
    });
}
