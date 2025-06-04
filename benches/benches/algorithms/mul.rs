use crate::prelude::*;
use ruint::algorithms::addmul_n;

pub fn group(criterion: &mut Criterion) {
    bench_addmul_nnn(criterion);
}

fn bench_addmul_nnn(criterion: &mut Criterion) {
    const_for!(SIZE in [0,1,2,3,4,5,6] {
        let mut rng = rng();
        criterion.bench_function(&format!("algo/addmul_n/{SIZE}"), move |bencher| {
            bencher.iter_batched(
                || (
                    rng.random::<[u64; SIZE]>(),
                    rng.random::<[u64; SIZE]>(),
                    rng.random::<[u64; SIZE]>(),
                ),
                |(mut lhs, a, b)| {
                    addmul_n(&mut lhs, &a, &b);
                    black_box(lhs)
                },
                BatchSize::SmallInput,
            );
        });
    });
}
