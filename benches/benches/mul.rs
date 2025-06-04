use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_mul::<BITS, LIMBS>(criterion);
    });
    const_for!(BITS_LHS in [64, 256,1024] {
        const LIMBS_LHS: usize = nlimbs(BITS_LHS);
        const_for!(BITS_RHS in [64, 256,1024] {
            const LIMBS_RHS: usize = nlimbs(BITS_RHS);
            const BITS_RES: usize = BITS_LHS + BITS_RHS;
            const LIMBS_RES: usize = nlimbs(BITS_RES);
            bench_widening_mul::<BITS_LHS, LIMBS_LHS, BITS_RHS, LIMBS_RHS, BITS_RES, LIMBS_RES>(criterion);
        });
    });
}

fn bench_mul<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("mul/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || input.new_tree(&mut runner).unwrap().current(),
            |(a, b)| black_box(black_box(a) * black_box(b)),
            BatchSize::SmallInput,
        );
    });
}

fn bench_widening_mul<
    const BITS_LHS: usize,
    const LIMBS_LHS: usize,
    const BITS_RHS: usize,
    const LIMBS_RHS: usize,
    const BITS_RES: usize,
    const LIMBS_RES: usize,
>(
    criterion: &mut Criterion,
) {
    let input = (
        Uint::<BITS_LHS, LIMBS_LHS>::arbitrary(),
        Uint::<BITS_RHS, LIMBS_RHS>::arbitrary(),
    );
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(
        &format!("widening_mul/{BITS_LHS}/{BITS_RHS}"),
        move |bencher| {
            bencher.iter_batched(
                || input.new_tree(&mut runner).unwrap().current(),
                |(a, b)| {
                    black_box(
                        black_box(a)
                            .widening_mul::<BITS_RHS, LIMBS_RHS, BITS_RES, LIMBS_RES>(black_box(b)),
                    )
                },
                BatchSize::SmallInput,
            );
        },
    );
}
