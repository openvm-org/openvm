use crate::prelude::*;

pub fn group(criterion: &mut Criterion) {
    const_for!(BITS in BENCH {
        const LIMBS: usize = nlimbs(BITS);
        bench_div_rem_small::<BITS, LIMBS>(criterion);
        bench_div_rem_half::<BITS, LIMBS>(criterion);
        bench_div_rem_full::<BITS, LIMBS>(criterion);
    });
}

fn bench_div_rem_small<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    if BITS == 0 {
        return;
    }
    let input = (Uint::<BITS, LIMBS>::arbitrary(), u64::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("div_rem/{BITS}/64"), move |bencher| {
        bencher.iter_batched(
            || {
                let (n, mut d) = input.new_tree(&mut runner).unwrap().current();
                if BITS < 64 {
                    d &= Uint::<BITS, LIMBS>::MASK;
                }
                if d == 0 {
                    d = 1;
                }
                (n, Uint::from(d))
            },
            |(a, b)| black_box(black_box(a).div_rem(black_box(b))),
            BatchSize::SmallInput,
        );
    });
}

fn bench_div_rem_half<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    if BITS == 0 {
        return;
    }
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(
        &format!("div_rem/{BITS}/{}", BITS - BITS / 2),
        move |bencher| {
            bencher.iter_batched(
                || {
                    let (n, mut d) = input.new_tree(&mut runner).unwrap().current();
                    d >>= BITS / 2; // make d half size
                    if d.is_zero() {
                        d = Uint::ONE;
                    }
                    (n, d)
                },
                |(a, b)| black_box(black_box(a).div_rem(black_box(b))),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_div_rem_full<const BITS: usize, const LIMBS: usize>(criterion: &mut Criterion) {
    if BITS == 0 {
        return;
    }
    let input = (Uint::<BITS, LIMBS>::arbitrary(), Uint::arbitrary());
    let mut runner = TestRunner::deterministic();
    criterion.bench_function(&format!("div_rem/{BITS}/{BITS}"), move |bencher| {
        bencher.iter_batched(
            || {
                let (n, mut d) = input.new_tree(&mut runner).unwrap().current();
                if d.is_zero() {
                    d = Uint::ONE;
                }
                (n, d)
            },
            |(a, b)| black_box(black_box(a).div_rem(black_box(b))),
            BatchSize::SmallInput,
        );
    });
}
