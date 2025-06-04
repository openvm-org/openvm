use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_root::<BITS, LIMBS>(criterion, 2);
        bench_root::<BITS, LIMBS>(criterion, 3);
        bench_root::<BITS, LIMBS>(criterion, 5);
        bench_root::<BITS, LIMBS>(criterion, 127);
    });
}

fn bench_root<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion, degree: usize) {
    let input = Uint::<BITS, LIMBS>::arbitrary();
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("root/{degree}/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |value| black_box(black_box(value).root(black_box(degree))),
            BatchSize::SmallInput,
        );
    });
}
