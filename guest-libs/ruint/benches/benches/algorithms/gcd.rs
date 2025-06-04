use crate::prelude::*;
use core::cmp::{max, min};
use ruint::algorithms::LehmerMatrix as Matrix;

pub fn group(criterion: &mut Criterion) {
    bench_from_u64(criterion);
    bench_from_u64_prefix(criterion);
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_apply::<BITS, LIMBS>(criterion);
    });
}

fn bench_from_u64(criterion: &mut Criterion) {
    let input = (u64::arbitrary(), u64::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function("algorithms/gcd/matrix/from_u64", move |bencher| {
        bencher.iter_batched(
            || {
                let (a, b) = input.new_tree(&mut runner).unwrap().current();
                (max(a, b), min(a, b))
            },
            |(a, b)| black_box(Matrix::from_u64(black_box(a), black_box(b))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_from_u64_prefix(criterion: &mut Criterion) {
    let input = (u64::arbitrary(), u64::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function("algorithms/gcd/matrix/from_u64_prefix", move |bencher| {
        bencher.iter_batched(
            || {
                let (a, b) = input.new_tree(&mut runner).unwrap().current();
                (max(a, b), min(a, b))
            },
            |(a, b)| black_box(Matrix::from_u64_prefix(black_box(a), black_box(b))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_apply<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (
        Uint::<BITS, LIMBS>::arbitrary(),
        Uint::<BITS, LIMBS>::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(
        &format!("algorithms/gcd/matrix/apply/{BITS}"),
        move |bencher| {
            bencher.iter_batched(
                || {
                    let (a, b) = input.new_tree(&mut runner).unwrap().current();
                    let (a, b) = (max(a, b), min(a, b));
                    let m = Matrix::from(a, b);
                    (a, b, m)
                },
                |(a, b, m)| black_box(m).apply(&mut black_box(a), &mut black_box(b)),
                BatchSize::SmallInput,
            );
        },
    );
}
