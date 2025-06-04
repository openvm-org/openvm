use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_log::<BITS, LIMBS>(criterion);
    });
}

fn bench_log<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    if BITS < 7 {
        return;
    }
    let input = (Uint::<BITS, LIMBS>::arbitrary(), 2_u64..100);
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("log/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(n, b)| black_box(black_box(n).checked_log(Uint::<BITS, LIMBS>::from(b))),
            BatchSize::SmallInput,
        );
    });
}
